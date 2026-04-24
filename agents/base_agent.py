"""Base agent interface for all NetSentinel-RL agents."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict
import numpy as np
import time

@dataclass
class AgentDecision:
    action: int
    action_name: str
    confidence: float
    latency_ms: float
    metadata: Dict[str, Any]

class BaseAgent(ABC):
    def __init__(self, name: str):
        self.name = name
        self._decisions = []

    @abstractmethod
    def predict(self, obs: np.ndarray, deterministic: bool = True) -> AgentDecision:
        pass

    def predict_timed(self, obs: np.ndarray, deterministic: bool = True) -> AgentDecision:
        t0 = time.time()
        decision = self.predict(obs, deterministic)
        decision.latency_ms = (time.time()-t0)*1000
        self._decisions.append(decision)
        return decision

    def decision_stats(self) -> Dict:
        if not self._decisions: return {}
        latencies = [d.latency_ms for d in self._decisions]
        return {
            "n_decisions": len(self._decisions),
            "mean_latency_ms": round(np.mean(latencies),2),
            "p95_latency_ms":  round(np.percentile(latencies,95),2),
            "mean_confidence": round(np.mean([d.confidence for d in self._decisions]),4),
        }
