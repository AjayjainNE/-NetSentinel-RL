"""
NetSentinel-RL — LLMOps Evaluation Harness
Custom evaluation framework measuring:
  - LLM verdict faithfulness (SHAP grounding)
  - Hallucination detection
  - Latency P50/P95/P99
  - Classification precision/recall at the system level
  - Experiment comparison via MLflow
"""

from __future__ import annotations
import time
import json
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path
import logging

log = logging.getLogger(__name__)


@dataclass
class EvalSample:
    """Single evaluation sample with ground truth and prediction."""
    flow_id: str
    true_label: int
    true_label_name: str
    predicted_label: str
    confidence: float
    verdict_explanation: str
    shap_features: Dict[str, float]
    shap_grounded: bool
    latency_ms: float
    recommended_action: str
    hallucination_flags: List[str] = field(default_factory=list)
    bert_score: float = 0.0


@dataclass
class EvalReport:
    """Aggregate evaluation report across all samples."""
    n_samples: int
    accuracy: float
    f1_weighted: float
    precision: float
    recall: float
    false_positive_rate: float
    false_negative_rate: float
    mean_confidence: float
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    shap_grounded_rate: float
    hallucination_rate: float
    mean_bert_score: float
    per_class_f1: Dict[str, float] = field(default_factory=dict)
    action_distribution: Dict[str, int] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class LLMEvalHarness:
    """
    Comprehensive evaluation harness for the NetSentinel-RL system.

    Implements three evaluation dimensions:
    1. Operational: detection accuracy, F1, FP/FN rates
    2. Interpretability: SHAP grounding rate, explanation faithfulness
    3. LLM quality: BERTScore, hallucination detection, latency

    All results logged to MLflow for experiment comparison.
    """

    # Hallucination detection: claims that cannot be grounded in flow data
    HALLUCINATION_PATTERNS = [
        "ransomware",        # Not a detectable flow-level class
        "exfiltrated data",  # Too specific without evidence
        "nation-state",      # Unfounded attribution
        "zero-day",          # Cannot be determined from flow stats
        "malware payload",   # Payload not available at flow level
    ]

    ATTACK_NAMES = [
        "Benign", "DoS", "DDoS", "PortScan",
        "BruteForce", "Botnet", "WebAttack", "Infiltration"
    ]

    def __init__(
        self,
        experiment_name: str = "netsentinel-rl",
        tracking_uri: str = "./mlruns",
    ):
        self.experiment_name = experiment_name
        self._samples: List[EvalSample] = []
        self._setup_mlflow(tracking_uri)

    def _setup_mlflow(self, tracking_uri: str):
        try:
            import mlflow
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            self._mlflow_available = True
            log.info(f"MLflow tracking at {tracking_uri}")
        except ImportError:
            self._mlflow_available = False
            log.warning("MLflow not available. Metrics will not be tracked.")

    def evaluate_sample(
        self,
        flow_id: str,
        true_label: int,
        predicted_label: str,
        confidence: float,
        explanation: str,
        shap_features: Dict[str, float],
        latency_ms: float,
        recommended_action: str,
        shap_grounded: bool,
    ) -> EvalSample:
        """Evaluate a single verdict against ground truth."""
        true_label_name = self.ATTACK_NAMES[true_label] if true_label < len(self.ATTACK_NAMES) else "Unknown"
        hallucination_flags = self._detect_hallucinations(explanation, shap_features)
        bert_score = self._compute_bert_score_proxy(explanation, shap_features)

        sample = EvalSample(
            flow_id=flow_id,
            true_label=true_label,
            true_label_name=true_label_name,
            predicted_label=predicted_label,
            confidence=confidence,
            verdict_explanation=explanation,
            shap_features=shap_features,
            shap_grounded=shap_grounded,
            latency_ms=latency_ms,
            recommended_action=recommended_action,
            hallucination_flags=hallucination_flags,
            bert_score=bert_score,
        )
        self._samples.append(sample)
        return sample

    def _detect_hallucinations(
        self, explanation: str, shap_features: Dict[str, float]
    ) -> List[str]:
        """Flag explanations containing ungrounded claims."""
        flags = []
        exp_lower = explanation.lower()
        for pattern in self.HALLUCINATION_PATTERNS:
            if pattern in exp_lower:
                flags.append(pattern)
        return flags

    def _compute_bert_score_proxy(
        self, explanation: str, shap_features: Dict[str, float]
    ) -> float:
        """
        Lightweight BERTScore proxy (avoid full BERTScore for efficiency in eval loop).
        Measures token overlap between explanation and top SHAP feature names.
        Full BERTScore computed in notebook 08 for the evaluation set.
        """
        if not explanation or not shap_features:
            return 0.0
        exp_tokens = set(explanation.lower().split())
        feat_tokens = set()
        for feat in list(shap_features.keys())[:5]:
            feat_tokens.update(feat.lower().replace("_", " ").split())
        if not feat_tokens:
            return 0.0
        overlap = len(exp_tokens & feat_tokens)
        return round(min(overlap / len(feat_tokens), 1.0), 4)

    def generate_report(self, run_name: Optional[str] = None) -> EvalReport:
        """Compute and log the aggregate evaluation report."""
        if not self._samples:
            raise ValueError("No evaluation samples collected. Run evaluate_sample() first.")

        n = len(self._samples)
        latencies = [s.latency_ms for s in self._samples]
        latencies_sorted = sorted(latencies)

        # Classification metrics
        correct = sum(
            1 for s in self._samples
            if s.predicted_label == s.true_label_name
        )
        accuracy = correct / n

        # Binary threat detection metrics (benign vs any attack)
        tp = sum(1 for s in self._samples if s.true_label != 0 and s.predicted_label != "Benign")
        fp = sum(1 for s in self._samples if s.true_label == 0 and s.predicted_label != "Benign")
        fn = sum(1 for s in self._samples if s.true_label != 0 and s.predicted_label == "Benign")
        tn = sum(1 for s in self._samples if s.true_label == 0 and s.predicted_label == "Benign")

        precision = tp / (tp + fp + 1e-9)
        recall    = tp / (tp + fn + 1e-9)
        f1        = 2 * precision * recall / (precision + recall + 1e-9)
        fpr       = fp / (fp + tn + 1e-9)
        fnr       = fn / (fn + tp + 1e-9)

        # Per-class F1
        per_class_f1 = {}
        for cls_name in self.ATTACK_NAMES:
            cls_tp = sum(1 for s in self._samples if s.true_label_name == cls_name and s.predicted_label == cls_name)
            cls_fp = sum(1 for s in self._samples if s.true_label_name != cls_name and s.predicted_label == cls_name)
            cls_fn = sum(1 for s in self._samples if s.true_label_name == cls_name and s.predicted_label != cls_name)
            cls_p  = cls_tp / (cls_tp + cls_fp + 1e-9)
            cls_r  = cls_tp / (cls_tp + cls_fn + 1e-9)
            per_class_f1[cls_name] = round(2 * cls_p * cls_r / (cls_p + cls_r + 1e-9), 4)

        # Action distribution
        action_dist: Dict[str, int] = {}
        for s in self._samples:
            action_dist[s.recommended_action] = action_dist.get(s.recommended_action, 0) + 1

        report = EvalReport(
            n_samples=n,
            accuracy=round(accuracy, 4),
            f1_weighted=round(f1, 4),
            precision=round(precision, 4),
            recall=round(recall, 4),
            false_positive_rate=round(fpr, 4),
            false_negative_rate=round(fnr, 4),
            mean_confidence=round(np.mean([s.confidence for s in self._samples]), 4),
            mean_latency_ms=round(np.mean(latencies), 2),
            p50_latency_ms=round(latencies_sorted[int(n * 0.50)], 2),
            p95_latency_ms=round(latencies_sorted[int(n * 0.95)], 2),
            p99_latency_ms=round(latencies_sorted[min(int(n * 0.99), n-1)], 2),
            shap_grounded_rate=round(sum(s.shap_grounded for s in self._samples) / n, 4),
            hallucination_rate=round(sum(1 for s in self._samples if s.hallucination_flags) / n, 4),
            mean_bert_score=round(np.mean([s.bert_score for s in self._samples]), 4),
            per_class_f1=per_class_f1,
            action_distribution=action_dist,
        )

        self._log_to_mlflow(report, run_name)
        return report

    def _log_to_mlflow(self, report: EvalReport, run_name: Optional[str]):
        """Log all metrics to MLflow."""
        if not self._mlflow_available:
            return
        try:
            import mlflow
            with mlflow.start_run(run_name=run_name or f"eval_{int(time.time())}"):
                metrics = {
                    "accuracy":             report.accuracy,
                    "f1_weighted":          report.f1_weighted,
                    "precision":            report.precision,
                    "recall":               report.recall,
                    "false_positive_rate":  report.false_positive_rate,
                    "false_negative_rate":  report.false_negative_rate,
                    "mean_latency_ms":      report.mean_latency_ms,
                    "p95_latency_ms":       report.p95_latency_ms,
                    "shap_grounded_rate":   report.shap_grounded_rate,
                    "hallucination_rate":   report.hallucination_rate,
                    "mean_bert_score":      report.mean_bert_score,
                }
                for k, v in metrics.items():
                    mlflow.log_metric(k, v)
                for cls_name, f1 in report.per_class_f1.items():
                    mlflow.log_metric(f"f1_{cls_name}", f1)
                mlflow.log_dict(asdict(report), "eval_report.json")
            log.info(f"Evaluation report logged to MLflow: F1={report.f1_weighted:.4f}")
        except Exception as e:
            log.warning(f"MLflow logging failed: {e}")

    def save_report(self, path: str = "reports/eval_report.json"):
        """Save the latest report to disk."""
        report = self.generate_report()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(report), f, indent=2)
        log.info(f"Report saved to {path}")
        return report


class PrometheusMetrics:
    """
    Prometheus metrics for real-time production monitoring.
    Exposes metrics on a configurable port.
    """

    def __init__(self, port: int = 8000):
        self.port = port
        self._setup()

    def _setup(self):
        try:
            from prometheus_client import Counter, Histogram, Gauge, start_http_server
            self.flows_processed = Counter("netsentinel_flows_total", "Total flows processed")
            self.threats_detected = Counter("netsentinel_threats_total", "Total threats detected", ["threat_type"])
            self.latency_hist = Histogram(
                "netsentinel_verdict_latency_seconds", "Verdict latency",
                buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5]
            )
            self.confidence_gauge = Gauge("netsentinel_mean_confidence", "Running mean confidence")
            self.fp_rate_gauge    = Gauge("netsentinel_fp_rate", "False positive rate (rolling)")
            self._available = True
            start_http_server(self.port)
            log.info(f"Prometheus metrics serving on :{self.port}")
        except ImportError:
            self._available = False

    def record_verdict(self, threat_detected: bool, threat_type: str, latency_s: float, confidence: float):
        if not self._available:
            return
        self.flows_processed.inc()
        if threat_detected:
            self.threats_detected.labels(threat_type=threat_type).inc()
        self.latency_hist.observe(latency_s)
        self.confidence_gauge.set(confidence)
