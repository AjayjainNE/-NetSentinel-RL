"""LangSmith tracing integration for LLM call observability."""
import os, time, json
from typing import Dict, Any, Optional
import logging
log = logging.getLogger(__name__)

class LangSmithTracer:
    """Wraps LangSmith tracing for orchestrator calls."""
    def __init__(self):
        self._available = bool(os.environ.get("LANGCHAIN_API_KEY"))
        if self._available:
            os.environ.setdefault("LANGCHAIN_TRACING_V2","true")
            os.environ.setdefault("LANGCHAIN_PROJECT","netsentinel-rl")
            log.info("LangSmith tracing enabled")
        else:
            log.warning("LANGCHAIN_API_KEY not set — LangSmith tracing disabled")

    def trace_llm_call(self, prompt: str, response: str, metadata: Optional[Dict]=None):
        if not self._available: return
        try:
            from langsmith import Client
            client = Client()
            client.create_run(
                name="orchestrator_verdict",
                run_type="llm",
                inputs={"prompt": prompt[:2000]},
                outputs={"response": response[:2000]},
                extra={"metadata": metadata or {}},
            )
        except Exception as e:
            log.debug(f"LangSmith trace failed: {e}")

    def trace_agent_decision(self, agent: str, obs_shape: tuple, action: int,
                              reward: float, step: int):
        if not self._available: return
        try:
            from langsmith import Client
            Client().create_run(
                name=f"{agent}_decision",
                run_type="tool",
                inputs={"obs_shape":str(obs_shape),"step":step},
                outputs={"action":action,"reward":reward},
            )
        except Exception as e:
            log.debug(f"LangSmith agent trace failed: {e}")
