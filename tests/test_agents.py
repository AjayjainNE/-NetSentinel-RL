"""
NetSentinel-RL — Agent Test Suite
Unit and integration tests for the RL agents and environment.
"""

from __future__ import annotations

import numpy as np
import pytest

# ── Fixtures ──────────────────────────────────────────────────────────────────

N_SAMPLES  = 500
N_FEATURES = 21


@pytest.fixture(scope="module")
def synthetic_data():
    rng = np.random.default_rng(42)
    X   = rng.standard_normal((N_SAMPLES, N_FEATURES)).astype(np.float32)
    y   = rng.integers(0, 8, size=N_SAMPLES)
    return X, y


@pytest.fixture(scope="module")
def network_env(synthetic_data):
    from environment.network_env import NetworkEnv
    X, y = synthetic_data
    return NetworkEnv(X, y, max_steps=100, window_size=5)


# ── NetworkEnv tests ──────────────────────────────────────────────────────────

class TestNetworkEnv:

    def test_reset_returns_correct_shape(self, network_env):
        obs, info = network_env.reset(seed=0)
        expected  = N_FEATURES * network_env.window_size
        assert obs.shape == (expected,), f"Expected ({expected},), got {obs.shape}"
        assert isinstance(info, dict)

    def test_step_returns_valid_types(self, network_env):
        network_env.reset(seed=1)
        obs, reward, terminated, truncated, info = network_env.step(0)
        assert isinstance(obs,        np.ndarray)
        assert isinstance(reward,     float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated,  bool)
        assert isinstance(info,       dict)

    def test_reward_structure(self, network_env):
        """TP reward must exceed TN; FN cost must be the most negative."""
        env = network_env
        assert env.TP_REWARD > env.TN_REWARD > 0
        assert env.FN_COST   < env.FP_COST   < 0

    def test_episode_terminates(self, network_env):
        obs, _ = network_env.reset(seed=2)
        done    = False
        steps   = 0
        while not done:
            obs, _, terminated, truncated, _ = network_env.step(network_env.action_space.sample())
            done  = terminated or truncated
            steps += 1
        assert steps == network_env.max_steps

    def test_episode_info_contains_metrics(self, network_env):
        env = network_env
        env.reset(seed=3)
        done = False
        info = {}
        while not done:
            _, _, terminated, truncated, info = env.step(env.action_space.sample())
            done = terminated or truncated
        for key in ("f1", "precision", "recall", "tp", "tn", "fp", "fn"):
            assert key in info, f"Missing key: {key}"

    def test_observation_space_bounds(self, network_env):
        obs, _ = network_env.reset(seed=4)
        assert network_env.observation_space.contains(obs)

    def test_multiple_resets_are_independent(self, network_env):
        obs1, _ = network_env.reset(seed=10)
        obs2, _ = network_env.reset(seed=11)
        # Two different seeds should (almost certainly) yield different observations
        assert not np.allclose(obs1, obs2)


# ── Detector agent tests ──────────────────────────────────────────────────────

class TestDetectorAgent:

    def test_attention_extractor_output_shape(self, synthetic_data):
        import torch
        from agents.detector_agent import AttentionFlowExtractor
        import gymnasium as gym

        n_feat = N_FEATURES
        win    = 5
        obs_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_feat * win,), dtype=np.float32
        )
        extractor = AttentionFlowExtractor(obs_space, n_features=n_feat, window_size=win, d_model=64)
        x         = torch.randn(8, n_feat * win)
        out       = extractor(x)
        assert out.shape == (8, 64), f"Unexpected extractor output shape: {out.shape}"

    def test_build_detector_returns_ppo(self, synthetic_data):
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.monitor import Monitor
        from environment.network_env import NetworkEnv
        from agents.detector_agent import build_detector

        X, y = synthetic_data
        env   = DummyVecEnv([lambda: Monitor(NetworkEnv(X, y, max_steps=50, window_size=5))])
        model = build_detector(env, n_features=N_FEATURES, window_size=5)
        assert isinstance(model, PPO)


# ── Responder agent tests ─────────────────────────────────────────────────────

class TestResponderAgent:

    def test_reward_tp_exceeds_tn(self):
        from agents.responder_agent import compute_responder_reward
        r_block_threat  = compute_responder_reward(2, True,  "DDoS")
        r_noact_benign  = compute_responder_reward(0, False, "Benign")
        assert r_block_threat > r_noact_benign

    def test_false_block_is_worst_outcome(self):
        from agents.responder_agent import compute_responder_reward
        r_false_block  = compute_responder_reward(2, False, "Benign")   # block_ip on benign
        r_missed_alert = compute_responder_reward(0, True,  "DDoS")     # no_action on threat
        # Both bad; false block should be heavily penalised
        assert r_false_block < 0
        assert r_missed_alert < 0

    def test_response_actions_complete(self):
        from agents.responder_agent import RESPONSE_ACTIONS, N_ACTIONS
        assert len(RESPONSE_ACTIONS) == N_ACTIONS
        assert set(RESPONSE_ACTIONS.values()) == {
            "no_action", "alert_soc", "block_ip", "rate_limit", "deep_inspect"
        }


# ── Classifier agent tests ─────────────────────────────────────────────────────

class TestClassifierAgent:

    def test_attack_names_length(self):
        from agents.classifier_agent import ATTACK_NAMES, N_CLASSES
        assert len(ATTACK_NAMES) == N_CLASSES == 8

    def test_flow_text_dataset_length(self):
        from agents.classifier_agent import FlowTextDataset
        from transformers import AutoTokenizer

        texts     = ["syn_flags 5 | flow_duration 1.2s | pkt_rate 200pps"] * 10
        labels    = np.zeros(10, dtype=int)
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        ds        = FlowTextDataset(texts, labels, tokenizer, max_length=64)
        assert len(ds) == 10
        sample = ds[0]
        assert "input_ids" in sample
        assert "labels"    in sample


# ── Data preprocessing tests ──────────────────────────────────────────────────

class TestPreprocessing:

    def test_clip_outlier_transformer(self):
        from data.preprocess import ClipOutlierTransformer
        rng = np.random.default_rng(0)
        X   = rng.standard_normal((200, 5)).astype(np.float32)
        t   = ClipOutlierTransformer(clip_percentile=95.0)
        t.fit(X)
        out = t.transform(X)
        assert out.shape == X.shape
        assert out.dtype == np.float32

    def test_flow_text_encoder_output_type(self):
        from data.preprocess import FlowTextEncoder, FLOW_FEATURES
        rng  = np.random.default_rng(1)
        data = {feat: rng.exponential(1.0, size=5) for feat in FLOW_FEATURES}
        df   = __import__("pandas").DataFrame(data)
        enc  = FlowTextEncoder()
        texts = enc.transform(df)
        assert len(texts) == 5
        assert all(isinstance(t, str) for t in texts)
        assert all(len(t) > 0 for t in texts)


# ── Orchestrator tests ────────────────────────────────────────────────────────

class TestOrchestrator:

    def test_mock_verdict_structure(self):
        from orchestrator.llm_orchestrator import LLMOrchestrator, AgentSignal

        orch = LLMOrchestrator()   # uses mock mode when anthropic unavailable
        sig  = AgentSignal(
            agent_name="detector-1",
            action="block_ip",
            confidence=0.92,
            top_features={"SYN_Flag_Count": 0.31, "Flow_Packets/s": 0.22},
            flow_stats={"threat_type": "DDoS", "src_ip": "10.0.1.1"},
        )
        verdict = orch.synthesise([sig])
        assert hasattr(verdict, "threat_detected")
        assert hasattr(verdict, "severity")
        assert hasattr(verdict, "recommended_action")
        assert verdict.latency_ms >= 0

    def test_fast_path_skips_llm(self):
        from orchestrator.llm_orchestrator import LLMOrchestrator, AgentSignal

        orch = LLMOrchestrator()
        signals = [
            AgentSignal(
                agent_name=f"detector-{i}",
                action="block_ip",
                confidence=0.95,
                top_features={"SYN_Flag_Count": 0.4},
                flow_stats={"threat_type": "DoS"},
            )
            for i in range(2)
        ]
        assert orch._should_fast_path(signals)
        verdict = orch.synthesise(signals)
        assert "Fast path" in verdict.reasoning_chain
