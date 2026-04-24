"""Natural language verdict generation grounded in SHAP values."""
from typing import Dict, List, Optional
import numpy as np

ATTACK_DESCRIPTIONS = {
    "Benign":"normal traffic","DoS":"denial-of-service attack",
    "DDoS":"distributed denial-of-service attack","PortScan":"network reconnaissance / port scan",
    "BruteForce":"credential brute-force attack","Botnet":"botnet C2 traffic",
    "WebAttack":"web application attack","Infiltration":"data exfiltration attempt",
}

SEVERITY_MAP = {"critical":0.90,"high":0.70,"medium":0.45,"low":0.0}

def get_severity(confidence: float) -> str:
    for sev, thresh in SEVERITY_MAP.items():
        if confidence >= thresh: return sev
    return "low"

def generate_verdict(predicted_class: str, confidence: float, shap_features: Dict[str,float],
                     flow_stats: Dict, action_taken: str, source_ip: Optional[str]=None) -> str:
    severity = get_severity(confidence)
    desc = ATTACK_DESCRIPTIONS.get(predicted_class, predicted_class)
    top2 = list(shap_features.items())[:2]
    src = f"from {source_ip}" if source_ip else "detected"
    action_phrase = {"block_ip":"Blocked","rate_limit":"Rate-limited","deep_inspect":"Flagged for deep inspection",
                     "alert_soc":"SOC alerted for","no_action":"Monitoring"}.get(action_taken,"Actioned")
    feat_citations = " and ".join(
        [f"{f.replace('_',' ').lower()} (SHAP:{v:+.3f})" for f,v in top2]
    ) or "flow anomalies"
    explanation = (
        f"{action_phrase} {src}: {severity.upper()} confidence {desc} "
        f"({confidence:.1%}). Primary indicators: {feat_citations}."
    )
    if flow_stats.get("Flow Packets/s",0) > 1000:
        explanation += f" Flow rate: {flow_stats['Flow Packets/s']:.0f} pkt/s."
    elif flow_stats.get("SYN Flag Count",0) > 0:
        explanation += " Anomalous SYN flag pattern detected."
    return explanation

def batch_report(verdicts: List[Dict], window_minutes: int=5) -> str:
    threats = [v for v in verdicts if v.get("threat_detected")]
    types: Dict[str,int] = {}
    for v in threats: types[v.get("threat_type","Unknown")] = types.get(v.get("threat_type","Unknown"),0)+1
    lines = [
        f"=== NetSentinel Report ({window_minutes}min window) ===",
        f"Total flows: {len(verdicts)} | Threats: {len(threats)} ({len(threats)/max(1,len(verdicts)):.1%})",
        "Breakdown: " + " | ".join(f"{k}:{v}" for k,v in sorted(types.items(),key=lambda x:x[1],reverse=True)),
    ]
    if threats:
        top = max(threats, key=lambda v: v.get("confidence_score",0))
        lines.append(f"Highest threat: {top.get('threat_type')} ({top.get('confidence_score',0):.1%}) → {top.get('recommended_action')}")
    return "\n".join(lines)
