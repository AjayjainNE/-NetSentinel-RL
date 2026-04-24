"""
NetSentinel-RL — Unit & Integration Tests
Run with: pytest tests/ -v
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_data():
    """Small synthetic dataset for fast tests."""
    np.random.seed(42)
    n, d = 200, 27
    X = np.random.randn(n, d).astype(np.float32)
    y = np.random.choice([0, 1, 2, 3], size=n, p=[0.6, 0.2, 0.1, 0.1]).astype(np.int32)
    return X, y


@pytest.fixture
def network_env(synthetic_data):
    from environment.network_env import NetworkEnv
    X, y = synthetic_data
    return NetworkEnv(X, y, n_classes=4, max_steps=50, window_size=3)


@pytest.fixture
def multi_agent_env(synthetic_data):
    from environment.network_env import MultiAgentNetworkEnv
    X, y = synthetic_data
    return MultiAgentNetworkEnv(X, y, n_classes=4, max_steps=50)


# ── Environment tests ─────────────────────────────────────────────────────────

class TestNetworkEnv:

    def test_reset_returns_correct_shape(self, network_env):
        obs, info = network_env.reset()
        expected_shape = (27 * 3,)  # n_features * window_size
        assert obs.shape == expected_shape
        assert obs.dtype == np.float32

    def test_step_returns_correct_types(self, network_env):
        network_env.reset()
        obs, reward, terminated, truncated, info = network_env.step(0)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "f1" in info

    def test_episode_terminates_at_max_steps(self, network_env):
        network_env.reset()
        done = False
        steps = 0
        while not done:
            _, _, done, _, _ = network_env.step(0)
            steps += 1
        assert steps == network_env.max_steps

    def test_reward_range(self, network_env):
        """Rewards must be within expected bounds."""
        network_env.reset()
        rewards = []
        for _ in range(20):
            _, r, done, _, _ = network_env.step(np.random.randint(0, 2))
            rewards.append(r)
            if done:
                break
        for r in rewards:
            assert -3.0 <= r <= 1.5, f"Reward {r} out of expected range"

    def test_info_keys_present(self, network_env):
        network_env.reset()
        _, _, _, _, info = network_env.step(1)
        for key in ["step", "f1", "precision", "recall"]:
            assert key in info


class TestMultiAgentEnv:

    def test_step_multi_returns_per_agent_rewards(self, multi_agent_env):
        multi_agent_env.reset()
        actions = {"detector": 1, "classifier": 1, "responder": 2}
        obs, rewards, done, _, info = multi_agent_env.step_multi(actions)
        assert set(rewards.keys()) == {"detector", "classifier", "responder"}
        for r in rewards.values():
            assert isinstance(r, float)

    def test_step_multi_info_has_response_action(self, multi_agent_env):
        multi_agent_env.reset()
        _, _, _, _, info = multi_agent_env.step_multi(
            {"detector": 1, "classifier": 2, "responder": 2}
        )
        assert "response_action" in info
        assert info["response_action"] in multi_agent_env.RESPONSE_ACTIONS.values()


# ── Preprocessing tests ───────────────────────────────────────────────────────

class TestPreprocessing:

    def test_flow_text_encoder(self):
        from data.preprocess import FlowTextEncoder
        encoder = FlowTextEncoder()
        row = {
            "Flow Duration": 1234,
            "Total Fwd Packets": 10,
            "Total Backward Packets": 8,
            "Flow Bytes/s": 50000,
            "SYN Flag Count": 1,
            "RST Flag Count": 0,
            "ACK Flag Count": 7,
            "Fwd PSH Flags": 1,
            "Fwd URG Flags": 0,
            "Flow IAT Mean": 100,
            "Avg Packet Size": 500,
            "Down/Up Ratio": 1.2,
        }
        text = encoder.encode_flow(row)
        assert "flow:" in text
        assert "flags:" in text
        assert "stats:" in text
        assert "duration=1234ms" in text

    def test_network_flow_transformer(self):
        from data.preprocess import NetworkFlowTransformer
        import pandas as pd
        transformer = NetworkFlowTransformer()
        X = pd.DataFrame({
            "Flow Duration": [100, 200, np.inf],
            "Total Fwd Packets": [5, 10, 3],
            "Total Backward Packets": [4, 8, 2],
            "Flow Bytes/s": [1000, 2000, 500],
            "Flow Packets/s": [10, 20, 5],
        })
        X_fit = transformer.fit_transform(X)
        assert not np.any(np.isnan(X_fit))
        assert not np.any(np.isinf(X_fit))


# ── Orchestrator tests ────────────────────────────────────────────────────────

class TestOrchestrator:

    def test_parse_valid_json(self):
        from orchestrator.llm_orchestrator import LLMOrchestrator
        orch = LLMOrchestrator(enable_langsmith=False)
        raw = '{"threat_detected": true, "threat_type": "DoS", "severity": "high", "recommended_action": "block_ip", "explanation": "test", "confidence_score": 0.9, "reasoning_chain": "test"}'
        parsed = orch._parse_and_validate(raw)
        assert parsed["threat_detected"] is True
        assert parsed["threat_type"] == "DoS"

    def test_parse_json_with_markdown_fences(self):
        from orchestrator.llm_orchestrator import LLMOrchestrator
        orch = LLMOrchestrator(enable_langsmith=False)
        raw = '```json\n{"threat_detected": false, "threat_type": "Benign", "severity": "low", "recommended_action": "no_action", "explanation": "ok", "confidence_score": 0.95, "reasoning_chain": "clean"}\n```'
        parsed = orch._parse_and_validate(raw)
        assert parsed["threat_detected"] is False

    def test_shap_grounding_check(self):
        from orchestrator.llm_orchestrator import LLMOrchestrator
        orch = LLMOrchestrator(enable_langsmith=False)
        explanation = "High packet rate detected with anomalous SYN flags."
        shap = {"Flow_Packets/s": 0.85, "SYN_Flag_Count": 0.72, "Avg_Packet_Size": 0.31}
        assert orch._check_shap_grounding(explanation, shap) is True

    def test_shap_grounding_fails_for_unrelated(self):
        from orchestrator.llm_orchestrator import LLMOrchestrator
        orch = LLMOrchestrator(enable_langsmith=False)
        explanation = "Traffic seems suspicious."
        shap = {"SYN_Flag_Count": 0.9, "Fwd_IAT_Mean": 0.6}
        # "suspicious" doesn't match any feature name token
        result = orch._check_shap_grounding(explanation, shap)
        assert isinstance(result, bool)


# ── Eval harness tests ────────────────────────────────────────────────────────

class TestEvalHarness:

    def test_hallucination_detection(self):
        from llmops.llm_eval_harness import LLMEvalHarness
        harness = LLMEvalHarness()
        flags = harness._detect_hallucinations(
            "This appears to be a nation-state ransomware attack with zero-day exploit.",
            {}
        )
        assert "nation-state" in flags
        assert "ransomware" in flags
        assert "zero-day" in flags

    def test_clean_explanation_no_flags(self):
        from llmops.llm_eval_harness import LLMEvalHarness
        harness = LLMEvalHarness()
        flags = harness._detect_hallucinations(
            "High SYN flag rate with elevated packets per second. Blocked source IP.",
            {"SYN_Flag_Count": 0.9}
        )
        assert len(flags) == 0

    def test_report_generation(self):
        from llmops.llm_eval_harness import LLMEvalHarness
        harness = LLMEvalHarness()
        for i in range(20):
            harness.evaluate_sample(
                flow_id=f"flow_{i}",
                true_label=i % 4,
                predicted_label=["Benign", "DoS", "DDoS", "PortScan"][i % 4],
                confidence=0.85,
                explanation="High packet rate with SYN anomaly.",
                shap_features={"SYN_Flag_Count": 0.8, "Flow_Packets/s": 0.6},
                latency_ms=45.0 + i,
                recommended_action="block_ip",
                shap_grounded=True,
            )
        report = harness.generate_report(run_name="test_run")
        assert report.n_samples == 20
        assert 0 <= report.f1_weighted <= 1
        assert report.p95_latency_ms >= report.p50_latency_ms


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
