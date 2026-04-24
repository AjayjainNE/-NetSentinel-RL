"""
NetSentinel-RL — Detector Agent
PPO-based binary threat detector trained on the custom NetworkEnv.
"""

from __future__ import annotations
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
from pathlib import Path
import logging

log = logging.getLogger(__name__)


class AttentionFlowExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor with a lightweight self-attention mechanism
    over the sliding window of flow observations.

    This is the novel architectural contribution: treating the window
    as a sequence and applying scaled dot-product attention before
    feeding into the PPO MLP policy head.
    """

    def __init__(self, observation_space, n_features: int, window_size: int = 5, d_model: int = 64):
        features_dim = d_model
        super().__init__(observation_space, features_dim)

        self.n_features = n_features
        self.window_size = window_size
        self.d_model = d_model

        # Project each flow in window to d_model
        self.input_proj = nn.Linear(n_features, d_model)
        # Self-attention over window positions
        self.attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        # Layer norm + final projection
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        batch_size = obs.shape[0]
        # Reshape: (batch, window*features) -> (batch, window, features)
        x = obs.view(batch_size, self.window_size, self.n_features)
        # Project
        x = self.input_proj(x)
        # Self-attention (current flow attends to history)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(x + attn_out)
        # Pool: take representation of the most recent flow (last position)
        x = x[:, -1, :]
        return self.output_proj(x)


class ThreatDetectionCallback(BaseCallback):
    """
    Custom callback tracking F1, precision, recall from environment info.
    Logs to MLflow if available.
    """

    def __init__(self, eval_freq: int = 2000, verbose: int = 0):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.episode_f1s: list = []
        self.episode_fps: list = []
        self.episode_fns: list = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "f1" in info and info.get("step", 0) % 100 == 0:
                self.episode_f1s.append(info["f1"])
                self.episode_fps.append(info.get("fp", 0))
                self.episode_fns.append(info.get("fn", 0))

        if self.num_timesteps % self.eval_freq == 0 and self.episode_f1s:
            mean_f1 = np.mean(self.episode_f1s[-50:])
            mean_fp = np.mean(self.episode_fps[-50:])
            if self.verbose > 0:
                log.info(f"Step {self.num_timesteps:6d} | Mean F1: {mean_f1:.4f} | Mean FP: {mean_fp:.1f}")
            try:
                import mlflow
                mlflow.log_metric("detector_f1", mean_f1, step=self.num_timesteps)
                mlflow.log_metric("detector_fp_rate", mean_fp, step=self.num_timesteps)
            except Exception:
                pass
        return True


def build_detector_agent(
    env,
    n_features: int,
    window_size: int = 5,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    device: str = "auto",
) -> PPO:
    """
    Constructs the PPO detector agent with AttentionFlowExtractor.
    """
    policy_kwargs = dict(
        features_extractor_class=AttentionFlowExtractor,
        features_extractor_kwargs=dict(
            n_features=n_features,
            window_size=window_size,
            d_model=64,
        ),
        net_arch=[dict(pi=[128, 64], vf=[128, 64])],
        activation_fn=nn.ReLU,
    )

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,       # Entropy regularisation to prevent premature convergence
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=device,
    )
    return model


def train_detector(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    total_timesteps: int = 500_000,
    save_path: str = "models/detector",
    n_envs: int = 4,
) -> PPO:
    """Train the detector agent and return the trained model."""
    from environment.network_env import NetworkEnv

    n_features_raw = X_train.shape[1]
    window_size = 5

    def make_env():
        env = NetworkEnv(X_train, y_train, max_steps=1000, window_size=window_size)
        return Monitor(env)

    vec_env = DummyVecEnv([make_env] * n_envs)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = DummyVecEnv([lambda: Monitor(NetworkEnv(X_val, y_val, max_steps=500, window_size=window_size))])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    model = build_detector_agent(vec_env, n_features=n_features_raw, window_size=window_size)

    callbacks = [
        ThreatDetectionCallback(eval_freq=5000, verbose=1),
        EvalCallback(
            eval_env,
            best_model_save_path=save_path,
            log_path=save_path,
            eval_freq=10_000,
            deterministic=True,
            render=False,
        ),
    ]

    log.info(f"Training detector agent for {total_timesteps:,} timesteps...")
    model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=True)

    Path(save_path).mkdir(parents=True, exist_ok=True)
    model.save(f"{save_path}/final_model")
    vec_env.save(f"{save_path}/vec_normalize.pkl")

    log.info(f"Detector agent saved to {save_path}")
    return model
