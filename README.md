# NetSentinel-RL 

> **Autonomous Network Threat Response via Multi-Agent Reinforcement Learning with Interpretable LLM Orchestration**

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-orange)](https://huggingface.co)
[![MLflow](https://img.shields.io/badge/MLflow-tracking-blue)](https://mlflow.org)

A production-grade, interpretable multi-agent reinforcement learning system for autonomous network threat detection, classification, and response — combining Hugging Face NLP models, custom RL agents, and an LLM orchestrator with full LLMOps observability.

---

## Motivation

After years as a network engineer watching SOC analysts drown in false positives from rule-based SIEMs, I built the system I wish existed: one that detects threats with RL precision, explains every decision in plain English, and can be trusted in production because it shows its work.

---

## Architecture

```
Real Network Data (CICIDS-2017 / KDD Cup / Live pcap)
        │
        ▼
┌─────────────────────────────────┐
│     LLM Orchestrator (Claude)   │  ← task routing, briefing, verdict synthesis
└────────┬──────────┬─────────────┘
         │          │          │
    ┌────▼──┐  ┌────▼──┐  ┌───▼────┐
    │Detect │  │Classif│  │Respond │  ← three specialised RL agents
    │ Agent │  │ Agent │  │ Agent  │
    └────┬──┘  └────┬──┘  └───┬────┘
         └──────────┴──────────┘
                    │
        ┌───────────▼──────────┐
        │  Interpretability    │  ← SHAP + attention + NL verdicts
        └───────────┬──────────┘
                    │
        ┌───────────▼──────────┐
        │  LLMOps Pipeline     │  ← MLflow + LangSmith + Prometheus
        └──────────────────────┘
```

## Hugging Face Tasks Used

| Task | Usage |
|------|-------|
| Token Classification | Flow-as-text attack type labelling |
| Text Classification | Threat severity scoring |
| Feature Extraction | Flow embedding for similarity search |
| Zero-Shot Classification | Novel attack detection without retraining |
| Text Generation | LLM verdict + explanation synthesis |
| Sentence Similarity | Flow vs known attack signature matching |

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/AjayjainNE/-NetSentinel-RL
cd NetSentinel-RL

# 2. Environment
conda env create -f environment.yml
conda activate netsentinel

# 3. Set secrets
cp .env.example .env
# Edit .env: add your ANTHROPIC_API_KEY

# 4. Download data
bash data/download_data.sh

# 5. Run notebooks in order (01 → 08) or launch dashboard
streamlit run dashboard/app.py
```

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| 01_EDA_and_Visualisation | Flow distributions, t-SNE, protocol heatmaps |
| 02_Feature_Engineering | 40+ engineered features, sklearn pipeline |
| 03_Detector_Agent_Training | PPO on custom NetworkEnv |
| 04_Classifier_Agent_Training | HF DistilBERT + RL reward shaping |
| 05_MARL_Responder_Training | IPPO multi-agent coordination |
| 06_LLM_Orchestrator_Design | Claude routing + prompt engineering |
| 07_Interpretability_SHAP | SHAP + attention maps + NL verdicts |
| 08_LLMOps_Evaluation | Custom eval harness, MLflow comparison |

---

## Key Results

| Metric | Value | Baseline |
|--------|-------|----------|
| Detection F1 (CICIDS-2017) | 97.3% | 89.2% (RF) |
| False Positive Rate | 1.1% | 4.2% |
| Mean Response Latency | 47ms | 340ms (rule-based) |
| LLM Verdict Faithfulness | 0.91 BERTScore | N/A |
| Hallucination Rate | <2.3% | N/A |

---

## Tech Stack

- **RL**: Stable-Baselines3, RLlib, custom Gym environment
- **NLP/HF**: transformers, datasets, evaluate (DistilBERT, sentence-transformers)
- **LLM**: Anthropic Claude Sonnet via API
- **Interpretability**: SHAP, Captum, BERTViz
- **LLMOps**: MLflow, LangSmith, Prometheus
- **Data**: CICIDS-2017, KDD Cup 99, Scapy, Zeek
- **Serving**: Streamlit, Docker, FastAPI

---

## License

MIT

---

## 👤 Author

Researcher @ RHUL | London - MSc Applied Data Science (Deep Learning specialisation). Motivated by 6+ years of network engineering experience.
