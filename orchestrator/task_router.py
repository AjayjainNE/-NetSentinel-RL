"""Routes tasks between agents based on confidence and urgency."""
from typing import Dict, Tuple
import logging
log = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD_FAST = 0.95   # Skip LLM, act immediately
CONFIDENCE_THRESHOLD_LLM  = 0.60   # Send to LLM for synthesis
CONFIDENCE_THRESHOLD_LOG  = 0.40   # Log but no action

class TaskRouter:
    """
    Decides whether a flow needs LLM synthesis or can be fast-pathed.
    Fast-path: very high confidence threats/benign → skip LLM latency.
    LLM-path:  ambiguous, medium confidence → full orchestrator synthesis.
    """
    def __init__(self, fast_threshold=CONFIDENCE_THRESHOLD_FAST,
                 llm_threshold=CONFIDENCE_THRESHOLD_LLM):
        self.fast_threshold = fast_threshold
        self.llm_threshold = llm_threshold
        self._route_counts = {"fast_threat":0,"fast_benign":0,"llm":0,"log_only":0}

    def route(self, detector_conf: float, classifier_conf: float,
              detector_action: str) -> Tuple[str, str]:
        """Returns (route_type, reason)."""
        avg_conf = (detector_conf + classifier_conf) / 2
        if avg_conf >= self.fast_threshold:
            route = "fast_threat" if detector_action == "threat" else "fast_benign"
            reason = f"High confidence ({avg_conf:.2f}) — fast-path"
        elif avg_conf >= self.llm_threshold:
            route = "llm"
            reason = f"Medium confidence ({avg_conf:.2f}) — LLM synthesis"
        else:
            route = "log_only"
            reason = f"Low confidence ({avg_conf:.2f}) — log only"
        self._route_counts[route] = self._route_counts.get(route, 0) + 1
        log.debug(f"Route: {route} | {reason}")
        return route, reason

    def stats(self) -> Dict:
        total = sum(self._route_counts.values())
        return {k: {"count":v,"rate":round(v/max(1,total),3)} for k,v in self._route_counts.items()}
