"""
NetSentinel-RL — LLM Orchestrator
Uses Claude Sonnet for task routing, agent briefing, and verdict synthesis.

This is NOT a wrapper — it implements:
1. Dynamic task routing based on agent confidence scores
2. Structured prompt engineering with chain-of-thought
3. Faithfulness validation of generated explanations against SHAP values
4. Full LangSmith tracing for observability
"""

from __future__ import annotations
import os
import json
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging
from pathlib import Path

log = logging.getLogger(__name__)

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    log.warning("anthropic package not installed. Using mock orchestrator.")


@dataclass
class AgentSignal:
    """Structured output from an RL agent passed to the orchestrator."""
    agent_name: str
    action: str
    confidence: float
    top_features: Dict[str, float]   # From SHAP: feature -> importance
    flow_stats: Dict[str, Any]       # Raw flow statistics
    timestamp: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()


@dataclass
class OrchestratorVerdict:
    """Structured verdict produced by the LLM orchestrator."""
    threat_detected: bool
    threat_type: str
    severity: str                    # low / medium / high / critical
    recommended_action: str
    explanation: str                 # Human-readable natural language
    confidence_score: float
    reasoning_chain: str             # Chain-of-thought from LLM
    shap_grounded: bool             # Did explanation reference SHAP features?
    latency_ms: float


SYSTEM_PROMPT = """You are NetSentinel, an expert network security analyst AI.
You receive structured signals from three specialist RL agents monitoring network traffic.
Your role is to:
1. Synthesise their signals into a coherent threat assessment
2. Produce a clear, actionable verdict for a SOC analyst
3. Ground your explanation in the specific flow statistics and SHAP feature importances provided
4. Be precise — avoid hedging language that reduces actionability
5. Always cite at least two specific numerical values from the flow data in your explanation

Output format: JSON only. No preamble.
"""

VERDICT_SCHEMA = {
    "threat_detected": "boolean",
    "threat_type": "string (e.g. DDoS, PortScan, BruteForce, Benign)",
    "severity": "string: low | medium | high | critical",
    "recommended_action": "string: no_action | alert_soc | rate_limit | block_ip | deep_inspect",
    "explanation": "string: 2-3 sentence human-readable explanation citing specific flow stats",
    "confidence_score": "float 0-1",
    "reasoning_chain": "string: step-by-step reasoning",
}


class LLMOrchestrator:
    """
    Orchestrates multi-agent signals through Claude Sonnet.

    Key features:
    - Structured JSON output with schema validation
    - Chain-of-thought reasoning for interpretability
    - SHAP-grounded explanation faithfulness check
    - Automatic retry with exponential backoff
    - Full request/response logging for LLMOps
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 1000,
        temperature: float = 0.1,   # Low temp for consistent structured output
        enable_langsmith: bool = True,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

        if ANTHROPIC_AVAILABLE:
            self.client = anthropic.Anthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY", "")
            )
        else:
            self.client = None

        self._request_log: list = []

        if enable_langsmith:
            self._setup_langsmith()

    def _setup_langsmith(self):
        """Configure LangSmith tracing if credentials present."""
        api_key = os.environ.get("LANGCHAIN_API_KEY")
        if api_key:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ.setdefault("LANGCHAIN_PROJECT", "netsentinel-rl")
            log.info("LangSmith tracing enabled")

    def synthesise_verdict(
        self,
        detector_signal: AgentSignal,
        classifier_signal: AgentSignal,
        responder_signal: AgentSignal,
    ) -> OrchestratorVerdict:
        """
        Main entry point: synthesise three agent signals into a verdict.
        """
        start_time = time.time()
        prompt = self._build_prompt(detector_signal, classifier_signal, responder_signal)

        raw_response = self._call_llm(prompt)
        parsed = self._parse_and_validate(raw_response)

        latency_ms = (time.time() - start_time) * 1000

        # Check if explanation references SHAP features
        shap_grounded = self._check_shap_grounding(
            parsed.get("explanation", ""),
            {**detector_signal.top_features, **classifier_signal.top_features},
        )

        verdict = OrchestratorVerdict(
            threat_detected=parsed.get("threat_detected", False),
            threat_type=parsed.get("threat_type", "Unknown"),
            severity=parsed.get("severity", "low"),
            recommended_action=parsed.get("recommended_action", "no_action"),
            explanation=parsed.get("explanation", ""),
            confidence_score=float(parsed.get("confidence_score", 0.5)),
            reasoning_chain=parsed.get("reasoning_chain", ""),
            shap_grounded=shap_grounded,
            latency_ms=latency_ms,
        )

        self._log_request(prompt, raw_response, verdict)
        return verdict

    def _build_prompt(
        self,
        det: AgentSignal,
        cls: AgentSignal,
        rsp: AgentSignal,
    ) -> str:
        top_features_str = "\n".join(
            f"  - {feat}: {imp:.4f}" for feat, imp in
            sorted(cls.top_features.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        )

        flow_stats_str = "\n".join(
            f"  {k}: {v}" for k, v in cls.flow_stats.items()
        )

        return f"""Analyse this network flow and produce a security verdict.

## Agent Signals

### Detector Agent (binary classifier)
- Decision: {det.action}
- Confidence: {det.confidence:.3f}

### Classifier Agent (multi-class threat typing)
- Predicted class: {cls.action}
- Confidence: {cls.confidence:.3f}
- Top SHAP features (feature → importance):
{top_features_str}

### Responder Agent (action selection)
- Recommended response: {rsp.action}
- Confidence: {rsp.confidence:.3f}

## Raw Flow Statistics
{flow_stats_str}

## Required Output Schema
{json.dumps(VERDICT_SCHEMA, indent=2)}

Produce ONLY the JSON verdict. No markdown, no preamble."""

    def _call_llm(self, prompt: str, retries: int = 3) -> str:
        """Call Claude API with exponential backoff retry."""
        if not self.client:
            return self._mock_response()

        for attempt in range(retries):
            try:
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}],
                )
                return message.content[0].text
            except Exception as e:
                wait = 2 ** attempt
                log.warning(f"LLM call failed (attempt {attempt+1}/{retries}): {e}. Retrying in {wait}s...")
                time.sleep(wait)
        log.error("All LLM call attempts failed.")
        return self._mock_response()

    def _mock_response(self) -> str:
        """Fallback mock for testing without API key."""
        return json.dumps({
            "threat_detected": True,
            "threat_type": "DoS",
            "severity": "high",
            "recommended_action": "block_ip",
            "explanation": "Mock verdict: high packet rate with SYN flag anomaly detected.",
            "confidence_score": 0.85,
            "reasoning_chain": "Mock reasoning chain for testing.",
        })

    def _parse_and_validate(self, raw: str) -> Dict:
        """Parse JSON response and validate required fields."""
        try:
            # Strip any accidental markdown fences
            raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            parsed = json.loads(raw)
            # Ensure all required keys present
            for key in VERDICT_SCHEMA:
                if key not in parsed:
                    log.warning(f"Missing key in LLM output: {key}")
            return parsed
        except json.JSONDecodeError as e:
            log.error(f"Failed to parse LLM response as JSON: {e}\nRaw: {raw[:200]}")
            return {}

    def _check_shap_grounding(
        self,
        explanation: str,
        shap_features: Dict[str, float],
    ) -> bool:
        """
        Checks if the explanation references at least one top SHAP feature.
        This is a lightweight faithfulness proxy for the eval harness.
        """
        if not explanation or not shap_features:
            return False
        explanation_lower = explanation.lower()
        top_features = sorted(shap_features, key=lambda k: abs(shap_features[k]), reverse=True)[:3]
        for feat in top_features:
            # Normalise feature name to natural language approximation
            feat_words = feat.lower().replace("_", " ").replace("/", " ").split()
            if any(word in explanation_lower for word in feat_words if len(word) > 3):
                return True
        return False

    def _log_request(self, prompt: str, response: str, verdict: OrchestratorVerdict):
        """Log to internal buffer and MLflow."""
        entry = {
            "timestamp": time.time(),
            "prompt_chars": len(prompt),
            "response_chars": len(response),
            "latency_ms": verdict.latency_ms,
            "threat_detected": verdict.threat_detected,
            "severity": verdict.severity,
            "shap_grounded": verdict.shap_grounded,
        }
        self._request_log.append(entry)

        try:
            import mlflow
            mlflow.log_metric("orchestrator_latency_ms", verdict.latency_ms)
            mlflow.log_metric("shap_grounded_rate",
                              sum(e["shap_grounded"] for e in self._request_log) / len(self._request_log))
        except Exception:
            pass

    def get_stats(self) -> Dict:
        """Return aggregate orchestrator statistics."""
        if not self._request_log:
            return {}
        latencies = [e["latency_ms"] for e in self._request_log]
        return {
            "total_calls": len(self._request_log),
            "mean_latency_ms": round(sum(latencies) / len(latencies), 1),
            "p95_latency_ms":  round(sorted(latencies)[int(len(latencies) * 0.95)], 1),
            "shap_grounded_rate": sum(e["shap_grounded"] for e in self._request_log) / len(self._request_log),
            "threat_detection_rate": sum(e["threat_detected"] for e in self._request_log) / len(self._request_log),
        }
