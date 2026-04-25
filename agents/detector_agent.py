"""
NetSentinel-RL — Detector Agent
PPO-based binary threat detector with self-attention over sliding flow windows.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

log = logging.getLogger(__name__)


class AttentionFlowExtractor(BaseFeaturesExtractor):
    """
    Lightweight self-attention feature extractor.

    The sliding window of flow observations is treated as a short sequence.
    Scaled dot-product attention lets each position attend to all others
    before the most-recent position is projected to the policy head.
    This architecture allows the agent to exploit temporal correlations
    (e.g. a burst of SYN packets preceding a DDoS) without the cost of
    an LSTM or full Transformer.
    """

    def __init__(
        self,
        observation_space,
        n_features: int,
        window_size: int = 5,
        d_model: int = 64,
        n_heads: int = 4,
    ) -> None:
        super().__init__(observation_space, features_dim=d_model)

        self.n_features  = n_features
        self.window_size = window_size
        self.d_model     = d_model

        self.input_proj = nn.Linear(n_features, d_model)
        self.attn       = nn.MultiheadAttention(d_model, num_heads=n_heads, batch_first=True)
        self.norm       = nn.LayerNorm(d_model)
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (batch, window * features) → (batch, window, features)
        x = obs.view(obs.shape[0], self.window_size, self.n_features)
        x = self.input_proj(x)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(x + attn_out)
        # Use the most-recent flow position as the pooled representation
        return self.output_proj(x[:, -1, :])


class DetectorMetricsCallback(BaseCallback):
    """
    Tracks per-episode F1, false-positive rate, and false-negative rate;
    logs to MLflow when available.
    """

    def __init__(self, eval_freq: int = 2_000, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.eval_freq   = eval_freq
        self._f1_buf:    list[float] = []
        self._fp_buf:    list[float] = []
        self._fn_buf:    list[float] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "f1" in info:
                self._f1_buf.append(info["f1"])
                self._fp_buf.append(info.get("fp", 0))
                self._fn_buf.append(info.get("fn", 0))

        if self.num_timesteps % self.eval_freq == 0 and self._f1_buf:
            mean_f1 = float(np.mean(self._f1_buf[-50:]))
            mean_fp = float(np.mean(self._fp_buf[-50:]))
            if self.verbose > 0:
                log.info("step=%6d  F1=%.4f  FP_rate=%.2f", self.num_timesteps, mean_f1, mean_fp)
            try:
                import mlflow
                mlflow.log_metric("detector_f1",      mean_f1, step=self.num_timesteps)
                mlflow.log_metric("detector_fp_rate", mean_fp, step=self.num_timesteps)
            except Exception:
                pass
        return True


def build_detector(
    env,
    n_features: int,
    window_size: int = 5,
    learning_rate: float = 3e-4,
    n_steps: int = 2_048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    device: str = "auto",
) -> PPO:
    """Construct the PPO detector with AttentionFlowExtractor policy."""

    policy_kwargs = dict(
        features_extractor_class=AttentionFlowExtractor,
        features_extractor_kwargs=dict(
            n_features=n_features,
            window_size=window_size,
            d_model=64,
        ),
        net_arch=[{"pi": [128, 64], "vf": [128, 64]}],
        activation_fn=nn.ReLU,
    )

    return PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=device,
    )


def train_detector(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    total_timesteps: int = 500_000,
    save_path: str = "models/detector",
    n_envs: int = 4,
    window_size: int = 5,
) -> PPO:
    """Train the detector agent and persist model + normaliser."""

    from environment.network_env import NetworkEnv

    def make_env():
        return Monitor(NetworkEnv(X_train, y_train, max_steps=1_000, window_size=window_size))

    vec_env  = VecNormalize(DummyVecEnv([make_env] * n_envs), norm_obs=True, norm_reward=True, clip_obs=10.0)
    eval_env = VecNormalize(
        DummyVecEnv([lambda: Monitor(NetworkEnv(X_val, y_val, max_steps=500, window_size=window_size))]),
        norm_obs=True, norm_reward=False, clip_obs=10.0,
    )

    model = build_detector(vec_env, n_features=X_train.shape[1], window_size=window_size)

    callbacks = [
        DetectorMetricsCallback(eval_freq=5_000, verbose=1),
        EvalCallback(
            eval_env,
            best_model_save_path=save_path,
            log_path=save_path,
            eval_freq=10_000,
            deterministic=True,
            render=False,
        ),
    ]

    log.info("Training detector for %s timesteps …", f"{total_timesteps:,}")
    model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=True)

    out = Path(save_path)
    out.mkdir(parents=True, exist_ok=True)
    model.save(out / "final_model")
    vec_env.save(out / "vec_normalize.pkl")
    log.info("Detector saved → %s", save_path)
    return model


def load_detector(save_path: str = "models/detector") -> tuple[PPO, VecNormalize]:
    """Load a saved detector model and its observation normaliser."""
    model  = PPO.load(f"{save_path}/final_model")
    vec_env = VecNormalize.load(f"{save_path}/vec_normalize.pkl", DummyVecEnv([lambda: None]))
    return model, vec_env
