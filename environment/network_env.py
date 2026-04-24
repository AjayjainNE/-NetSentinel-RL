"""
NetSentinel-RL — Custom OpenAI Gymnasium Environment
Models network traffic monitoring as a sequential decision problem.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any
import logging

log = logging.getLogger(__name__)


class NetworkEnv(gym.Env):
    """
    Custom Gymnasium environment for network threat detection.

    State space:  Continuous vector of network flow features (normalised),
                  concatenated over a sliding window for temporal context.
    Action space: Discrete — {0: benign, 1: threat}
    Reward:       F1-weighted scoring with asymmetric FP/FN costs
                  reflecting real-world SOC alert fatigue economics.

    Key design decisions:
    - FN_COST > FP_COST: missed attacks more damaging than false alarms
    - TIME_BONUS rewards fast early detection within first 20% of episode
    - Sliding window gives agent temporal context over recent flows
    """

    metadata = {"render_modes": ["human"]}

    TP_REWARD = 1.0
    TN_REWARD = 0.3
    FP_COST   = -0.8   # Alert fatigue penalty
    FN_COST   = -2.0   # Missed attack — highest cost
    TIME_BONUS = 0.1

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_classes: int = 8,
        max_steps: int = 1000,
        window_size: int = 5,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int32)
        self.n_classes = n_classes
        self.max_steps = max_steps
        self.window_size = window_size
        self.render_mode = render_mode

        self.n_features = X.shape[1]
        self.n_samples = X.shape[0]

        obs_size = self.n_features * self.window_size
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(obs_size,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)

        self._indices: np.ndarray = np.array([])
        self._step: int = 0
        self._window: list = []
        self._episode_tp = self._episode_fp = self._episode_fn = self._episode_tn = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._indices = self.np_random.permutation(self.n_samples)
        self._step = 0
        self._window = [np.zeros(self.n_features, dtype=np.float32)] * self.window_size
        self._episode_tp = self._episode_fp = self._episode_fn = self._episode_tn = 0
        return self._get_obs(), self._get_info()

    def step(self, action: int):
        idx = self._indices[self._step % self.n_samples]
        true_label = int(self.y[idx])
        is_attack = int(true_label != 0)

        reward = self._compute_reward(action, is_attack)

        if action == 1 and is_attack:    self._episode_tp += 1
        elif action == 1 and not is_attack: self._episode_fp += 1
        elif action == 0 and is_attack:  self._episode_fn += 1
        else:                            self._episode_tn += 1

        self._window.pop(0)
        self._window.append(self.X[idx])
        self._step += 1

        terminated = self._step >= self.max_steps
        return self._get_obs(), reward, terminated, False, self._get_info()

    def _compute_reward(self, action: int, is_attack: int) -> float:
        if action == 1 and is_attack:
            r = self.TP_REWARD
            if self._step < self.max_steps * 0.2:
                r += self.TIME_BONUS
        elif action == 1 and not is_attack:
            r = self.FP_COST
        elif action == 0 and is_attack:
            r = self.FN_COST
        else:
            r = self.TN_REWARD
        return float(r)

    def _get_obs(self) -> np.ndarray:
        return np.concatenate(self._window, axis=0).astype(np.float32)

    def _get_info(self) -> Dict[str, Any]:
        tp, fp, fn, tn = self._episode_tp, self._episode_fp, self._episode_fn, self._episode_tn
        precision = tp / (tp + fp + 1e-9)
        recall    = tp / (tp + fn + 1e-9)
        f1        = 2 * precision * recall / (precision + recall + 1e-9)
        return {
            "step": self._step, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4),
        }

    def render(self):
        if self.render_mode == "human":
            info = self._get_info()
            print(
                f"Step {self._step:4d} | F1:{info['f1']:.3f} "
                f"P:{info['precision']:.3f} R:{info['recall']:.3f} "
                f"TP:{info['tp']} FP:{info['fp']} FN:{info['fn']} TN:{info['tn']}"
            )


class MultiAgentNetworkEnv(NetworkEnv):
    """
    Extended environment for three-agent MARL.

    Agents:
      detector   — binary {benign, threat}
      classifier — fine-grained 8-class threat type
      responder  — {no_action, alert, block, rate_limit, deep_inspect}

    Each agent receives the same observation but earns separate rewards,
    enabling Independent PPO (IPPO) training with cooperative incentives.
    """

    RESPONSE_ACTIONS = {
        0: "no_action",
        1: "alert_soc",
        2: "block_ip",
        3: "rate_limit",
        4: "deep_inspect",
    }

    def __init__(self, X, y, n_classes=8, max_steps=1000, **kwargs):
        super().__init__(X, y, n_classes, max_steps, **kwargs)
        self.action_spaces = {
            "detector":   spaces.Discrete(2),
            "classifier": spaces.Discrete(n_classes),
            "responder":  spaces.Discrete(len(self.RESPONSE_ACTIONS)),
        }

    def step_multi(self, actions: Dict[str, int]):
        """Process one step with actions from all three agents."""
        idx = self._indices[self._step % self.n_samples]
        true_label = int(self.y[idx])
        is_attack = int(true_label != 0)

        det_action = actions.get("detector", 0)
        cls_action = actions.get("classifier", 0)
        rsp_action = actions.get("responder", 0)

        det_reward = self._compute_reward(det_action, is_attack)

        # Classifier: rewards correct fine-grained label
        if cls_action == true_label:
            cls_reward = 1.5
        elif cls_action == 0 and is_attack:
            cls_reward = -2.0
        else:
            cls_reward = -0.5

        # Responder: reward depends on both correct detection and proportionate action
        if is_attack:
            if rsp_action in [2, 3, 4]:
                rsp_reward = 1.0 + (0.5 if det_action == 1 else 0.0)
            elif rsp_action == 1:
                rsp_reward = 0.3
            else:
                rsp_reward = -1.5
        else:
            rsp_reward = 0.2 if rsp_action == 0 else -0.6

        if det_action == 1 and is_attack:    self._episode_tp += 1
        elif det_action == 1 and not is_attack: self._episode_fp += 1
        elif det_action == 0 and is_attack:  self._episode_fn += 1
        else:                                self._episode_tn += 1

        self._window.pop(0)
        self._window.append(self.X[idx])
        self._step += 1

        terminated = self._step >= self.max_steps
        obs = self._get_obs()
        info = self._get_info()
        info["response_action"] = self.RESPONSE_ACTIONS[rsp_action]
        info["true_class"] = true_label

        return obs, {"detector": det_reward, "classifier": cls_reward, "responder": rsp_reward}, terminated, False, info
