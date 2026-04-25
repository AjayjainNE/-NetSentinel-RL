"""
NetSentinel-RL — Custom Gymnasium Environment
Models network traffic monitoring as a sequential decision problem.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

log = logging.getLogger(__name__)

ATTACK_NAMES = {
    0: "Benign",
    1: "DoS",
    2: "DDoS",
    3: "PortScan",
    4: "BruteForce",
    5: "Botnet",
    6: "WebAttack",
    7: "Infiltration",
}


class NetworkEnv(gym.Env):
    """
    Gymnasium environment for network threat detection.

    Observation: Concatenated sliding window of normalised flow feature vectors.
    Action:      Discrete binary — {0: benign, 1: threat}.
    Reward:      F1-weighted scoring with asymmetric FP/FN penalties
                 reflecting SOC alert-fatigue economics.

    Design notes
    ------------
    - FN_COST > FP_COST: a missed attack is operationally worse than a
      spurious alert, which motivates the agent to prefer recall over precision.
    - TIME_BONUS incentivises early detection within the first 20 % of the
      episode, rewarding fast response to emerging attacks.
    - The sliding window provides temporal context so the agent can recognise
      sustained attack patterns rather than reacting to single flows.
    """

    metadata = {"render_modes": ["human"]}

    TP_REWARD  =  1.0
    TN_REWARD  =  0.3
    FP_COST    = -0.8   # alert-fatigue penalty
    FN_COST    = -2.0   # missed attack — highest cost
    TIME_BONUS =  0.1   # reward for early detection

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_classes: int = 8,
        max_steps: int = 1_000,
        window_size: int = 5,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.X          = X.astype(np.float32)
        self.y          = y.astype(np.int32)
        self.n_classes  = n_classes
        self.max_steps  = max_steps
        self.window_size = window_size
        self.render_mode = render_mode

        self.n_features = X.shape[1]
        self.n_samples  = X.shape[0]

        obs_size = self.n_features * self.window_size
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)  # 0 = benign, 1 = threat

        # Populated on reset
        self._window: list[np.ndarray] = []
        self._step:   int = 0
        self._idx:    int = 0
        self._tp = self._tn = self._fp = self._fn = 0

    # ── Gym interface ─────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        start = self.np_random.integers(0, max(1, self.n_samples - self.max_steps))
        self._idx  = int(start)
        self._step = 0
        self._tp = self._tn = self._fp = self._fn = 0

        # Pre-fill window with zeros so the first observation is well-formed
        self._window = [np.zeros(self.n_features, dtype=np.float32)] * self.window_size
        obs = self._get_obs()
        return obs, {}

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        true_label = int(self.y[self._idx % self.n_samples] > 0)  # binary
        reward, outcome = self._compute_reward(action, true_label)

        # Update confusion counters
        match outcome:
            case "tp": self._tp += 1
            case "tn": self._tn += 1
            case "fp": self._fp += 1
            case "fn": self._fn += 1

        # Advance window
        flow = self.X[self._idx % self.n_samples].copy()
        self._window = self._window[1:] + [flow]
        self._idx  = (self._idx + 1) % self.n_samples
        self._step += 1

        terminated = self._step >= self.max_steps
        obs = self._get_obs()
        info = self._build_info() if terminated else {}
        return obs, reward, terminated, False, info

    def render(self) -> None:
        if self.render_mode == "human":
            f1 = self._f1()
            log.info(
                "Step %4d | TP=%d TN=%d FP=%d FN=%d | F1=%.4f",
                self._step, self._tp, self._tn, self._fp, self._fn, f1,
            )

    # ── Internals ─────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        return np.concatenate(self._window).astype(np.float32)

    def _compute_reward(self, action: int, true_label: int) -> Tuple[float, str]:
        early = self._step < self.max_steps * 0.2

        if true_label == 1 and action == 1:           # TP
            bonus  = self.TIME_BONUS if early else 0.0
            return self.TP_REWARD + bonus, "tp"
        if true_label == 0 and action == 0:           # TN
            return self.TN_REWARD, "tn"
        if true_label == 0 and action == 1:           # FP
            return self.FP_COST, "fp"
        # true_label == 1 and action == 0             # FN
        return self.FN_COST, "fn"

    def _f1(self) -> float:
        denom = 2 * self._tp + self._fp + self._fn
        return (2 * self._tp) / denom if denom > 0 else 0.0

    def _build_info(self) -> Dict[str, Any]:
        precision = self._tp / max(1, self._tp + self._fp)
        recall    = self._tp / max(1, self._tp + self._fn)
        return {
            "f1":        self._f1(),
            "precision": precision,
            "recall":    recall,
            "tp":        self._tp,
            "tn":        self._tn,
            "fp":        self._fp,
            "fn":        self._fn,
        }
