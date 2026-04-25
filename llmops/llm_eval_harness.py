"""
NetSentinel-RL — LLM Evaluation Harness
Evaluates the LLM orchestrator on faithfulness, hallucination rate, latency,
and BERTScore against gold-standard SOC analyst verdicts.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)

try:
    from bert_score import score as bert_score_fn
    _BERT_AVAILABLE = True
except ImportError:
    _BERT_AVAILABLE = False
    log.warning("bert-score not installed — BERTScore metrics will be skipped.")


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class EvalSample:
    """A single evaluation case with ground-truth labels."""
    sample_id:         str
    agent_signals:     List[Dict[str, Any]]
    gold_threat_type:  str
    gold_severity:     str
    gold_action:       str
    gold_explanation:  str
    shap_features:     List[str]           # features the explanation MUST reference


@dataclass
class EvalResult:
    """Evaluation outcome for one sample."""
    sample_id:          str
    threat_correct:     bool
    severity_correct:   bool
    action_correct:     bool
    shap_grounded:      bool
    hallucinated:       bool
    bert_score_f1:      float
    latency_ms:         float
    predicted_verdict:  Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalReport:
    """Aggregated metrics across all samples."""
    n_samples:          int
    accuracy:           float
    f1_weighted:        float
    precision:          float
    recall:             float
    false_positive_rate: float
    false_negative_rate: float
    mean_bert_score:    float
    shap_grounded_rate: float
    hallucination_rate: float
    mean_latency_ms:    float
    p95_latency_ms:     float
    per_class_f1:       Dict[str, float] = field(default_factory=dict)


# ── Metric helpers ────────────────────────────────────────────────────────────

def _is_hallucinated(explanation: str, shap_features: List[str]) -> bool:
    """Return True if the explanation mentions no SHAP-grounded feature."""
    lower = explanation.lower()
    return not any(f.lower().replace("_", " ") in lower for f in shap_features)


def _bert_score(hypothesis: str, reference: str) -> float:
    if not _BERT_AVAILABLE or not hypothesis or not reference:
        return 0.0
    _, _, f1 = bert_score_fn(
        [hypothesis], [reference], lang="en", verbose=False
    )
    return float(f1.mean())


# ── Harness ───────────────────────────────────────────────────────────────────

class LLMEvalHarness:
    """
    Runs systematic evaluation of the LLM orchestrator.

    Metrics
    -------
    - Threat type accuracy (classification correctness)
    - Severity accuracy
    - Action accuracy
    - SHAP groundedness rate (explanation references at least one SHAP feature)
    - Hallucination rate (complement of groundedness on threat samples)
    - BERTScore F1 vs gold explanations
    - Latency distribution (mean, P95)
    """

    def __init__(self, orchestrator, tracker=None) -> None:
        self._orchestrator = orchestrator
        self._tracker      = tracker

    def run(self, samples: List[EvalSample]) -> EvalReport:
        results = [self._eval_one(s) for s in samples]
        report  = self._aggregate(results)
        self._persist(report, results)
        return report

    # ── Per-sample evaluation ─────────────────────────────────────────────────

    def _eval_one(self, sample: EvalSample) -> EvalResult:
        from orchestrator.llm_orchestrator import AgentSignal

        signals = [
            AgentSignal(
                agent_name=sig["agent_name"],
                action=sig["action"],
                confidence=sig["confidence"],
                top_features=sig.get("top_features", {}),
                flow_stats=sig.get("flow_stats", {}),
            )
            for sig in sample.agent_signals
        ]

        verdict = self._orchestrator.synthesise(signals)

        threat_correct   = verdict.threat_type == sample.gold_threat_type
        severity_correct = verdict.severity    == sample.gold_severity
        action_correct   = verdict.recommended_action == sample.gold_action
        hallucinated     = _is_hallucinated(verdict.explanation, sample.shap_features)
        bs               = _bert_score(verdict.explanation, sample.gold_explanation)

        return EvalResult(
            sample_id=sample.sample_id,
            threat_correct=threat_correct,
            severity_correct=severity_correct,
            action_correct=action_correct,
            shap_grounded=verdict.shap_grounded,
            hallucinated=hallucinated,
            bert_score_f1=bs,
            latency_ms=verdict.latency_ms,
            predicted_verdict=asdict(verdict),
        )

    # ── Aggregation ───────────────────────────────────────────────────────────

    def _aggregate(self, results: List[EvalResult]) -> EvalReport:
        n = len(results)
        if n == 0:
            raise ValueError("No evaluation results to aggregate.")

        correct_threat  = sum(r.threat_correct   for r in results)
        correct_sev     = sum(r.severity_correct for r in results)
        correct_action  = sum(r.action_correct   for r in results)
        grounded        = sum(r.shap_grounded    for r in results)
        hallucinated    = sum(r.hallucinated     for r in results)
        latencies       = [r.latency_ms          for r in results]
        bert_scores     = [r.bert_score_f1       for r in results]

        tp = correct_threat
        fp = n - correct_action
        fn = n - correct_threat

        precision = tp / max(1, tp + fp)
        recall    = tp / max(1, tp + fn)
        f1        = 2 * precision * recall / max(1e-9, precision + recall)

        return EvalReport(
            n_samples=n,
            accuracy=correct_threat / n,
            f1_weighted=f1,
            precision=precision,
            recall=recall,
            false_positive_rate=fp / n,
            false_negative_rate=fn / n,
            mean_bert_score=float(np.mean(bert_scores)),
            shap_grounded_rate=grounded / n,
            hallucination_rate=hallucinated / n,
            mean_latency_ms=float(np.mean(latencies)),
            p95_latency_ms=float(np.percentile(latencies, 95)),
        )

    # ── Persistence ───────────────────────────────────────────────────────────

    def _persist(self, report: EvalReport, results: List[EvalResult]) -> None:
        out = Path("reports/eval_report.json")
        out.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "report":  asdict(report),
            "results": [asdict(r) for r in results],
        }
        out.write_text(json.dumps(payload, indent=2))
        log.info("Eval report saved → %s", out)

        if self._tracker is not None:
            self._tracker.log_metrics({
                "accuracy":           report.accuracy,
                "f1_weighted":        report.f1_weighted,
                "precision":          report.precision,
                "recall":             report.recall,
                "false_positive_rate": report.false_positive_rate,
                "false_negative_rate": report.false_negative_rate,
                "mean_bert_score":    report.mean_bert_score,
                "shap_grounded_rate": report.shap_grounded_rate,
                "hallucination_rate": report.hallucination_rate,
                "mean_latency_ms":    report.mean_latency_ms,
                "p95_latency_ms":     report.p95_latency_ms,
            })
            self._tracker.log_artifact(str(out))
