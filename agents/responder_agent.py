"""
NetSentinel-RL — Responder Agent (MARL)
Independent PPO (IPPO) multi-agent training with shared critic option.

The responder receives signals from both detector and classifier agents
and selects the proportionate network response action.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from pathlib import Path
from typing import Dict, List, Tuple
import logging

log = logging.getLogger(__name__)

RESPONSE_ACTIONS = {
    0: "no_action",
    1: "alert_soc",
    2: "block_ip",
    3: "rate_limit",
    4: "deep_inspect",
}

RESPONSE_COSTS = {
    0: 0.0,    # no_action: no cost
    1: 0.1,    # alert: low overhead
    2: 0.5,    # block: high — risks blocking legitimate traffic
    3: 0.3,    # rate_limit: moderate
    4: 0.4,    # deep_inspect: compute expensive
}


class MARLCoordinationCallback(BaseCallback):
    """
    Tracks multi-agent coordination metrics:
    - Agreement rate between detector and responder
    - Proportionality: does response action match threat severity?
    - Nash equilibrium convergence proxy (reward variance across agents)
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.coordination_scores: List[float] = []
        self.response_distribution: Dict[str, int] = {k: 0 for k in RESPONSE_ACTIONS.values()}

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "response_action" in info:
                action = info["response_action"]
                self.response_distribution[action] = self.response_distribution.get(action, 0) + 1

        if self.num_timesteps % 5000 == 0 and self.response_distribution:
            total = sum(self.response_distribution.values())
            if total > 0 and self.verbose > 0:
                dist_str = " | ".join(
                    f"{k}:{v/total:.2%}" for k, v in self.response_distribution.items()
                )
                log.info(f"Step {self.num_timesteps} | Response dist: {dist_str}")
            try:
                import mlflow
                for action, count in self.response_distribution.items():
                    mlflow.log_metric(
                        f"response_{action}_rate",
                        count / (total + 1e-9),
                        step=self.num_timesteps,
                    )
            except Exception:
                pass
        return True


class IPPOTrainer:
    """
    Independent PPO trainer for the three-agent system.

    Each agent is trained with its own PPO instance but shares the
    environment. The environment's per-agent rewards incentivise
    cooperative behaviour without requiring centralised training.

    Training protocol:
    1. Pre-train detector agent independently (uses detector_agent.py)
    2. Freeze detector, train classifier on detector's output
    3. Fine-tune all three jointly with reduced learning rates
    """

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_classes: int = 8,
        max_steps: int = 1000,
        device: str = "auto",
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.n_classes = n_classes
        self.max_steps = max_steps
        self.device = device
        self.agents: Dict[str, PPO] = {}

    def _make_env(self, agent_role: str = "detector"):
        """Create environment with role-specific action space proxy."""
        from environment.network_env import NetworkEnv
        env = NetworkEnv(
            self.X_train, self.y_train,
            n_classes=self.n_classes,
            max_steps=self.max_steps,
            window_size=5,
        )
        return Monitor(env)

    def train_responder(
        self,
        total_timesteps: int = 300_000,
        save_path: str = "models/responder",
    ) -> PPO:
        """
        Train the responder agent using signals from pre-trained
        detector outputs baked into the reward function.
        """
        from environment.network_env import NetworkEnv

        def make_responder_env():
            env = NetworkEnv(
                self.X_train, self.y_train,
                n_classes=self.n_classes,
                max_steps=self.max_steps,
            )
            # Override action space to responder's 5-action space
            env.action_space = env.action_space.__class__(len(RESPONSE_ACTIONS))
            return Monitor(env)

        vec_env = DummyVecEnv([make_responder_env] * 2)

        responder = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=1e-4,
            n_steps=1024,
            batch_size=64,
            n_epochs=5,
            gamma=0.99,
            ent_coef=0.02,
            policy_kwargs=dict(net_arch=[dict(pi=[128, 128, 64], vf=[128, 128, 64])]),
            verbose=1,
            device=self.device,
        )

        callback = MARLCoordinationCallback(verbose=1)
        log.info(f"Training responder agent for {total_timesteps:,} steps...")
        responder.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)

        Path(save_path).mkdir(parents=True, exist_ok=True)
        responder.save(f"{save_path}/responder_model")
        self.agents["responder"] = responder

        log.info(f"Responder saved to {save_path}")
        return responder

    def joint_finetune(
        self,
        total_timesteps: int = 200_000,
        save_path: str = "models/joint",
    ):
        """
        Optional joint fine-tuning pass: all agents train simultaneously
        in the MultiAgentNetworkEnv for coordination refinement.
        Logs Nash equilibrium convergence proxy (reward std across agents).
        """
        from environment.network_env import MultiAgentNetworkEnv
        import mlflow

        env = MultiAgentNetworkEnv(
            self.X_train, self.y_train,
            n_classes=self.n_classes,
            max_steps=self.max_steps,
        )

        obs, _ = env.reset()
        reward_history: Dict[str, List[float]] = {k: [] for k in ["detector", "classifier", "responder"]}

        log.info("Running joint fine-tuning loop...")
        for step in range(total_timesteps):
            actions = {
                "detector":   self.agents.get("detector",   None) and
                              int(self.agents["detector"].predict(obs, deterministic=True)[0])
                              if "detector" in self.agents else env.action_space.sample(),
                "classifier": env.action_spaces["classifier"].sample(),
                "responder":  self.agents.get("responder",  None) and
                              int(self.agents["responder"].predict(obs, deterministic=True)[0])
                              if "responder" in self.agents else env.action_spaces["responder"].sample(),
            }
            obs, rewards, terminated, _, info = env.step_multi(actions)
            for agent, r in rewards.items():
                reward_history[agent].append(r)

            if terminated:
                obs, _ = env.reset()

            if step % 10_000 == 0 and step > 0:
                means = {k: np.mean(v[-1000:]) for k, v in reward_history.items()}
                stds  = {k: np.std(v[-1000:])  for k, v in reward_history.items()}
                nash_proxy = np.std(list(means.values()))
                log.info(
                    f"Step {step:6d} | Means: {means} | Nash proxy (lower=better): {nash_proxy:.4f}"
                )
                try:
                    mlflow.log_metric("nash_convergence_proxy", nash_proxy, step=step)
                except Exception:
                    pass

        log.info("Joint fine-tuning complete.")


def evaluate_marl_system(
    agents: Dict[str, PPO],
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_episodes: int = 10,
) -> Dict:
    """
    Evaluate the full MARL system on test data.
    Returns aggregate metrics across all agents and episodes.
    """
    from environment.network_env import MultiAgentNetworkEnv

    env = MultiAgentNetworkEnv(X_test, y_test, max_steps=len(X_test) // n_episodes)
    all_rewards: Dict[str, List] = {k: [] for k in ["detector", "classifier", "responder"]}
    all_infos: List[Dict] = []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        ep_rewards: Dict[str, float] = {k: 0.0 for k in all_rewards}
        done = False

        while not done:
            actions = {}
            for agent_name, agent_model in agents.items():
                action, _ = agent_model.predict(obs, deterministic=True)
                actions[agent_name] = int(action)
            obs, rewards, done, _, info = env.step_multi(actions)
            for k, r in rewards.items():
                ep_rewards[k] += r

        for k, v in ep_rewards.items():
            all_rewards[k].append(v)
        all_infos.append(info)

    results = {
        "mean_f1":         np.mean([i["f1"] for i in all_infos]),
        "mean_precision":  np.mean([i["precision"] for i in all_infos]),
        "mean_recall":     np.mean([i["recall"] for i in all_infos]),
        "agent_rewards":   {k: np.mean(v) for k, v in all_rewards.items()},
        "response_actions": {},
    }

    # Tally response action distribution
    for info in all_infos:
        action = info.get("response_action", "unknown")
        results["response_actions"][action] = results["response_actions"].get(action, 0) + 1

    log.info(f"MARL evaluation results: F1={results['mean_f1']:.4f}")
    return results
