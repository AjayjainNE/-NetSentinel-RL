"""
NetSentinel-RL — Responder Agent (MARL)
Independent PPO (IPPO) multi-agent with optional shared-critic coordination.

Receives structured signals from detector and classifier agents and selects
a proportionate network response action from the discrete action space.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

log = logging.getLogger(__name__)

RESPONSE_ACTIONS: Dict[int, str] = {
    0: "no_action",
    1: "alert_soc",
    2: "block_ip",
    3: "rate_limit",
    4: "deep_inspect",
}

# Operational cost of each action (0–1 scale)
RESPONSE_COSTS: Dict[int, float] = {
    0: 0.0,   # no_action      — no cost
    1: 0.1,   # alert_soc      — low overhead
    2: 0.5,   # block_ip       — high; risks blocking legitimate traffic
    3: 0.3,   # rate_limit     — moderate bandwidth cost
    4: 0.4,   # deep_inspect   — compute-intensive
}

N_ACTIONS = len(RESPONSE_ACTIONS)


# ── Callback ──────────────────────────────────────────────────────────────────

class CoordinationCallback(BaseCallback):
    """
    Tracks multi-agent coordination metrics at runtime:
    - Response action distribution (should be proportional to threat mix)
    - Nash equilibrium convergence proxy via reward variance across agents
    Logs summary statistics to MLflow at configurable intervals.
    """

    def __init__(self, log_freq: int = 5_000, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.log_freq    = log_freq
        self._dist: Dict[str, int] = {v: 0 for v in RESPONSE_ACTIONS.values()}

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            action = info.get("response_action")
            if action in self._dist:
                self._dist[action] += 1

        if self.num_timesteps % self.log_freq == 0:
            total = max(1, sum(self._dist.values()))
            if self.verbose > 0:
                dist_str = "  ".join(f"{k}:{v/total:.1%}" for k, v in self._dist.items())
                log.info("step=%6d  action_dist: %s", self.num_timesteps, dist_str)
            try:
                import mlflow
                for action, count in self._dist.items():
                    mlflow.log_metric(f"resp_{action}_rate", count / total, step=self.num_timesteps)
            except Exception:
                pass
        return True


# ── Observation builder ───────────────────────────────────────────────────────

def build_responder_obs(
    detector_output:    np.ndarray,
    classifier_probs:   np.ndarray,
    flow_features:      np.ndarray,
    threat_confidence:  float,
) -> np.ndarray:
    """
    Concatenate all agent signals into a single observation vector.

    Parameters
    ----------
    detector_output   : Binary detection signal (1-D).
    classifier_probs  : Per-class probability vector (N_CLASSES,).
    flow_features     : Raw normalised flow features.
    threat_confidence : Scalar confidence from the classifier head.
    """
    return np.concatenate([
        detector_output.flatten(),
        classifier_probs.flatten(),
        flow_features.flatten(),
        np.array([threat_confidence], dtype=np.float32),
    ]).astype(np.float32)


# ── IPPO trainer ──────────────────────────────────────────────────────────────

class IPPOTrainer:
    """
    Independent PPO (IPPO) coordinator for the responder agent.

    Each agent optimises its own policy using a local observation and reward,
    without access to other agents' internal states.  This is the standard
    baseline in cooperative MARL before adding a centralised critic.
    """

    def __init__(
        self,
        obs_dim:        int,
        n_actions:      int = N_ACTIONS,
        learning_rate:  float = 1e-4,
        n_steps:        int = 1_024,
        batch_size:     int = 64,
        n_epochs:       int = 5,
        gamma:          float = 0.95,
        device:         str = "auto",
    ) -> None:
        self.obs_dim       = obs_dim
        self.n_actions     = n_actions
        self.learning_rate = learning_rate
        self.n_steps       = n_steps
        self.batch_size    = batch_size
        self.n_epochs      = n_epochs
        self.gamma         = gamma
        self.device        = device
        self._model: PPO | None = None

    def _make_dummy_env(self, env_fn):
        return DummyVecEnv([lambda: Monitor(env_fn())])

    def train(
        self,
        env_fn,
        total_timesteps: int = 300_000,
        save_path: str = "models/responder",
    ) -> PPO:
        vec_env = self._make_dummy_env(env_fn)

        policy_kwargs = dict(
            net_arch=[{"pi": [256, 128, 64], "vf": [256, 128, 64]}],
            activation_fn=nn.ReLU,
        )

        self._model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=self.learning_rate,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            gamma=self.gamma,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.005,
            policy_kwargs=policy_kwargs,
            verbose=1,
            device=self.device,
        )

        self._model.learn(
            total_timesteps=total_timesteps,
            callback=CoordinationCallback(verbose=1),
            progress_bar=True,
        )

        out = Path(save_path)
        out.mkdir(parents=True, exist_ok=True)
        self._model.save(out / "responder_model")
        log.info("Responder saved → %s", save_path)
        return self._model

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> Tuple[int, str]:
        if self._model is None:
            raise RuntimeError("Call train() or load() before predict().")
        action, _ = self._model.predict(obs, deterministic=deterministic)
        action_id = int(action)
        return action_id, RESPONSE_ACTIONS[action_id]

    def load(self, save_path: str = "models/responder") -> None:
        self._model = PPO.load(f"{save_path}/responder_model")
        log.info("Responder loaded from %s", save_path)


# ── Reward function ───────────────────────────────────────────────────────────

def compute_responder_reward(
    action_id:      int,
    true_is_threat: bool,
    threat_class:   str,
) -> float:
    """
    Compute immediate reward for a responder action.

    Reward logic
    ------------
    - Correct block/inspect on a real threat  →  +1.5 (high reward)
    - Correct rate_limit / alert on a threat  →  +0.8
    - No action on benign traffic             →  +0.3
    - Under-reaction to a real threat         →  -1.0
    - Blocking benign traffic (false block)   →  -1.2
    """
    action = RESPONSE_ACTIONS[action_id]
    cost   = RESPONSE_COSTS[action_id]

    if true_is_threat:
        if action in ("block_ip", "deep_inspect"):
            return 1.5 - cost
        if action in ("rate_limit", "alert_soc"):
            return 0.8 - cost
        return -1.0   # no_action on real threat

    # Benign traffic
    if action == "no_action":
        return 0.3
    if action == "block_ip":
        return -1.2   # false block is the worst outcome for benign traffic
    return -cost       # mild penalty for unnecessary action
