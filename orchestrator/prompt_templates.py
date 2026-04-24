"""Prompt templates for the LLM orchestrator."""

SYSTEM_PROMPT = """You are NetSentinel, an expert network security AI analyst.
Synthesise RL agent signals into precise, actionable, SHAP-grounded verdicts.
Always cite at least two numerical flow values. Output JSON only."""

ZERO_SHOT_TEMPLATE = """Agent signals:
Detector: {detector_action} (conf:{detector_conf:.2f})
Classifier: {classifier_action} (conf:{classifier_conf:.2f})
Responder: {responder_action}
Top SHAP features: {shap_features}
Flow stats: {flow_stats}

Output JSON verdict matching schema: threat_detected, threat_type, severity,
recommended_action, explanation, confidence_score, reasoning_chain"""

FEW_SHOT_EXAMPLES = [
    {
        "input": "Detector:threat(0.97) Classifier:DDoS(0.94) Responder:block_ip SYN_Flag_Count:0.89 Flow_Packets/s:0.76",
        "output": '{"threat_detected":true,"threat_type":"DDoS","severity":"critical","recommended_action":"block_ip","explanation":"Blocked source IP: DDoS attack detected (94% confidence). SYN flag density 0.94 is 4.2σ above baseline; flow rate 8,470 pkt/s exceeds threshold by 12×. SHAP: SYN_Flag_Count(0.89) and Flow_Packets/s(0.76) are primary drivers.","confidence_score":0.94,"reasoning_chain":"1. Detector flagged threat(0.97). 2. Classifier confirms DDoS(0.94). 3. SYN flood pattern from SHAP. 4. Responder selected block. Verdict: block."}',
    },
    {
        "input": "Detector:benign(0.91) Classifier:Benign(0.89) Responder:no_action ACK_Flag_Count:0.12 Avg_Packet_Size:0.08",
        "output": '{"threat_detected":false,"threat_type":"Benign","severity":"low","recommended_action":"no_action","explanation":"Normal traffic confirmed (89% confidence). ACK-dominated flow with mean packet size 412B and IAT 320ms — consistent with established HTTP session.","confidence_score":0.89,"reasoning_chain":"1. Both detector and classifier agree: benign. 2. No anomalous flags. 3. No action required."}',
    },
]

COT_TEMPLATE = """Think step by step before producing the verdict:
1. Do the detector and classifier agree?
2. Which SHAP features are most significant and why?
3. Is the confidence high enough for a blocking action?
4. What is the proportionate response?

Agent signals:
Detector: {detector_action} (conf:{detector_conf:.2f})
Classifier: {classifier_action} (conf:{classifier_conf:.2f})  
Responder preferred action: {responder_action}
Top SHAP features (feature→importance): {shap_features}
Raw flow stats: {flow_stats}

Output JSON only."""
