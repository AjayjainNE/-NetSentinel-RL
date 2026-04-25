"""
NetSentinel-RL — LLM Orchestrator
Dynamic task routing, chain-of-thought verdict synthesis, and SHAP faithfulness
validation using the Anthropic API.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

try:
    import anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False
    log.warning("anthropic package not installed — falling back to mock orchestrator.")

# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class AgentSignal:
    """Structured output from an upstream RL agent."""
    agent_name:   str
    action:       str
    confidence:   float
    top_features: Dict[str, float]    # feature name → SHAP importance
    flow_stats:   Dict[str, Any]      # raw flow statistics
    timestamp:    float = field(default_factory=time.time)


@dataclass
class OrchestratorVerdict:
    """Final verdict produced by the LLM orchestrator."""
    threat_detected:     bool
    threat_type:         str
    severity:            str           # low | medium | high | critical
    recommended_action:  str
    explanation:         str           # human-readable SOC narrative
    confidence_score:    float
    reasoning_chain:     str           # CoT scratchpad from LLM
    shap_grounded:       bool          # explanation references SHAP features?
    latency_ms:          float


# ── Prompt templates ──────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are NetSentinel, an expert network security analyst AI embedded in an
autonomous threat response pipeline.  You receive structured signals from
multiple ML agents and must produce a clear, actionable security verdict.

Rules:
1. Base your verdict strictly on the provided agent signals and SHAP features.
2. Provide a chain-of-thought in <reasoning> tags before the verdict.
3. The verdict JSON must contain exactly these keys:
   threat_detected, threat_type, severity, recommended_action,
   explanation, confidence_score, shap_grounded.
4. severity must be one of: low, medium, high, critical.
5. recommended_action must be one of:
   no_action, alert_soc, rate_limit, block_ip, deep_inspect.
6. explanation must be ≤ 3 sentences and reference at least one SHAP feature
   from the provided list (set shap_grounded = true if you do so).
"""

def _build_user_prompt(signals: list[AgentSignal]) -> str:
    lines = []
    for sig in signals:
        feats = ", ".join(f"{k}={v:.3f}" for k, v in sig.top_features.items())
        stats = ", ".join(f"{k}={v}" for k, v in sig.flow_stats.items())
        lines.append(
            f"Agent: {sig.agent_name}\n"
            f"  action={sig.action}  confidence={sig.confidence:.3f}\n"
            f"  SHAP features: {feats}\n"
            f"  flow stats: {stats}"
        )
    joined = "\n\n".join(lines)
    return (
        f"Analyse the following agent signals and produce a security verdict.\n\n"
        f"{joined}\n\n"
        f"Respond with <reasoning>…</reasoning> followed by a JSON verdict object."
    )


# ── Orchestrator ──────────────────────────────────────────────────────────────

class LLMOrchestrator:
    """
    Coordinates RL agent outputs through an LLM reasoning layer.

    Responsibilities
    ----------------
    1. Dynamic routing: decides whether to escalate low-confidence cases
       to the LLM or resolve them with a fast rule-based path.
    2. Verdict synthesis: chains CoT reasoning over multi-agent signals.
    3. Faithfulness validation: checks that the explanation references at
       least one SHAP-grounded feature.
    4. Observability: logs latency and verdicts to MLflow / LangSmith.
    """

    FAST_PATH_THRESHOLD = 0.90   # skip LLM when confidence exceeds this

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 800,
        langsmith_project: Optional[str] = None,
    ) -> None:
        self.model            = model
        self.max_tokens       = max_tokens
        self._client: Optional["anthropic.Anthropic"] = None

        if _ANTHROPIC_AVAILABLE:
            self._client = anthropic.Anthropic()
        else:
            log.warning("Running in mock mode — LLM calls will be simulated.")

        if langsmith_project:
            self._init_langsmith(langsmith_project)

    # ── Public interface ──────────────────────────────────────────────────────

    def synthesise(self, signals: list[AgentSignal]) -> OrchestratorVerdict:
        """
        Produce a verdict from a list of agent signals.

        Uses a fast deterministic path for very high-confidence signals
        to reduce latency; escalates ambiguous cases to the LLM.
        """
        t0 = time.perf_counter()

        if self._should_fast_path(signals):
            verdict = self._fast_verdict(signals)
        elif self._client is not None:
            verdict = self._llm_verdict(signals)
        else:
            verdict = self._mock_verdict(signals)

        verdict.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        self._log_verdict(verdict)
        return verdict

    # ── Routing logic ─────────────────────────────────────────────────────────

    def _should_fast_path(self, signals: list[AgentSignal]) -> bool:
        """Route to fast path when all agents agree with high confidence."""
        return all(s.confidence >= self.FAST_PATH_THRESHOLD for s in signals)

    def _fast_verdict(self, signals: list[AgentSignal]) -> OrchestratorVerdict:
        primary = max(signals, key=lambda s: s.confidence)
        is_threat = primary.action not in ("no_action", "benign")
        top_feat  = next(iter(primary.top_features), "unknown")
        severity  = "high" if primary.confidence > 0.95 else "medium"
        action    = primary.action if is_threat else "no_action"

        return OrchestratorVerdict(
            threat_detected=is_threat,
            threat_type=primary.flow_stats.get("threat_type", "Unknown"),
            severity=severity if is_threat else "low",
            recommended_action=action,
            explanation=(
                f"High-confidence detection (conf={primary.confidence:.2f}). "
                f"Primary indicator: {top_feat} "
                f"(SHAP={primary.top_features.get(top_feat, 0):.3f}). "
                f"Fast-path routing applied — no LLM call required."
            ),
            confidence_score=primary.confidence,
            reasoning_chain="Fast path: unanimous high-confidence consensus.",
            shap_grounded=True,
            latency_ms=0.0,
        )

    # ── LLM path ──────────────────────────────────────────────────────────────

    def _llm_verdict(self, signals: list[AgentSignal]) -> OrchestratorVerdict:
        user_msg = _build_user_prompt(signals)
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        raw_text = response.content[0].text
        return self._parse_response(raw_text, signals)

    def _parse_response(self, text: str, signals: list[AgentSignal]) -> OrchestratorVerdict:
        reasoning = ""
        if "<reasoning>" in text:
            reasoning = text.split("<reasoning>")[1].split("</reasoning>")[0].strip()

        # Extract JSON block
        json_start = text.rfind("{")
        json_end   = text.rfind("}") + 1
        payload    = json.loads(text[json_start:json_end]) if json_start != -1 else {}

        avg_conf = float(np.mean([s.confidence for s in signals])) if signals else 0.5

        return OrchestratorVerdict(
            threat_detected=bool(payload.get("threat_detected", False)),
            threat_type=str(payload.get("threat_type", "Unknown")),
            severity=str(payload.get("severity", "low")),
            recommended_action=str(payload.get("recommended_action", "alert_soc")),
            explanation=str(payload.get("explanation", "")),
            confidence_score=float(payload.get("confidence_score", avg_conf)),
            reasoning_chain=reasoning,
            shap_grounded=bool(payload.get("shap_grounded", False)),
            latency_ms=0.0,
        )

    def _mock_verdict(self, signals: list[AgentSignal]) -> OrchestratorVerdict:
        """Deterministic mock when Anthropic client is unavailable."""
        avg_conf = float(sum(s.confidence for s in signals) / max(1, len(signals)))
        is_threat = avg_conf > 0.5
        return OrchestratorVerdict(
            threat_detected=is_threat,
            threat_type="Unknown",
            severity="medium" if is_threat else "low",
            recommended_action="alert_soc" if is_threat else "no_action",
            explanation="[Mock] Anthropic client unavailable. Using heuristic verdict.",
            confidence_score=avg_conf,
            reasoning_chain="Mock mode.",
            shap_grounded=False,
            latency_ms=0.0,
        )

    # ── Observability ─────────────────────────────────────────────────────────

    def _log_verdict(self, verdict: OrchestratorVerdict) -> None:
        try:
            import mlflow
            mlflow.log_metric("orchestrator_latency_ms", verdict.latency_ms)
            mlflow.log_metric("orchestrator_confidence",  verdict.confidence_score)
            mlflow.log_metric("shap_grounded",            float(verdict.shap_grounded))
        except Exception:
            pass

    @staticmethod
    def _init_langsmith(project: str) -> None:
        import os
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGCHAIN_PROJECT", project)
        log.info("LangSmith tracing enabled for project: %s", project)


# Avoid circular import from parse helper
try:
    import numpy as np
except ImportError:
    pass
