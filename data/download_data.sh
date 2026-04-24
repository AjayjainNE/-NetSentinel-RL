#!/usr/bin/env bash
set -e

echo "=== NetSentinel-RL Data Downloader ==="
mkdir -p data/raw/cicids2017 data/raw/kddcup data/processed

# CICIDS-2017: Canadian Institute for Cybersecurity
echo "[1/3] Downloading CICIDS-2017 sample (Friday traffic - includes common attacks)..."
# Full dataset at: https://www.unb.ca/cic/datasets/ids-2017.html
# Using the publicly mirrored CSVs (MachineLearningCVE):
BASE="https://raw.githubusercontent.com/wesleykelsey/MachineLearningCVE/main/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv"
echo "  NOTE: For full dataset, register at https://www.unb.ca/cic/datasets/ids-2017.html"
echo "  Downloading sample file..."
curl -L --retry 3 -o data/raw/cicids2017/thursday_sample.csv \
  "https://raw.githubusercontent.com/PacktPublishing/Python-Machine-Learning-By-Example/master/Chapter09/KDD.csv" \
  2>/dev/null || echo "  Sample download failed - place CICIDS CSV files in data/raw/cicids2017/"

# KDD Cup 99 (via sklearn built-in or direct download)
echo "[2/3] Downloading KDD Cup 99 dataset..."
python3 - << 'PYEOF'
from sklearn.datasets import fetch_kddcup99
import pandas as pd, os
print("  Fetching KDD Cup 99 via sklearn...")
data = fetch_kddcup99(as_frame=True)
df = data.frame
df.to_parquet("data/raw/kddcup/kddcup99.parquet", index=False)
print(f"  Saved {len(df):,} rows to data/raw/kddcup/kddcup99.parquet")
PYEOF

echo "[3/3] Creating synthetic CICIDS-style dataset for development..."
python3 - << 'PYEOF'
import numpy as np, pandas as pd

np.random.seed(42)
n = 50000

# Simulate realistic network flow features
normal_flows = pd.DataFrame({
    'Flow Duration': np.random.exponential(1000, n//2),
    'Total Fwd Packets': np.random.poisson(5, n//2),
    'Total Backward Packets': np.random.poisson(4, n//2),
    'Total Length of Fwd Packets': np.random.exponential(500, n//2),
    'Fwd Packet Length Max': np.random.exponential(1000, n//2),
    'Fwd Packet Length Mean': np.random.exponential(200, n//2),
    'Bwd Packet Length Max': np.random.exponential(800, n//2),
    'Flow Bytes/s': np.random.exponential(10000, n//2),
    'Flow Packets/s': np.random.exponential(50, n//2),
    'Flow IAT Mean': np.random.exponential(500, n//2),
    'Flow IAT Std': np.random.exponential(300, n//2),
    'Fwd IAT Mean': np.random.exponential(800, n//2),
    'Bwd IAT Mean': np.random.exponential(700, n//2),
    'Fwd PSH Flags': np.random.binomial(1, 0.1, n//2),
    'Bwd PSH Flags': np.random.binomial(1, 0.1, n//2),
    'Fwd URG Flags': np.zeros(n//2),
    'SYN Flag Count': np.random.binomial(1, 0.05, n//2),
    'RST Flag Count': np.random.binomial(1, 0.02, n//2),
    'ACK Flag Count': np.random.poisson(4, n//2),
    'Down/Up Ratio': np.random.uniform(0.5, 2.0, n//2),
    'Avg Packet Size': np.random.normal(400, 100, n//2),
    'Label': 'BENIGN'
})

# Simulate attack flows
attack_types = ['DoS Hulk', 'PortScan', 'DDoS', 'FTP-Patator', 'SSH-Patator']
attack_dfs = []
per_attack = (n // 2) // len(attack_types)
for attack in attack_types:
    df_a = normal_flows.copy().iloc[:per_attack]
    if 'DoS' in attack or 'DDoS' in attack:
        df_a['Flow Packets/s'] *= 50
        df_a['SYN Flag Count'] = np.random.binomial(1, 0.9, per_attack)
        df_a['Flow Duration'] /= 10
    elif 'Scan' in attack:
        df_a['Total Fwd Packets'] = np.random.poisson(1, per_attack)
        df_a['Flow Duration'] = np.random.uniform(1, 50, per_attack)
        df_a['RST Flag Count'] = np.random.binomial(1, 0.7, per_attack)
    elif 'Patator' in attack:
        df_a['Flow Packets/s'] *= 3
        df_a['Fwd IAT Mean'] = np.random.exponential(100, per_attack)
    df_a['Label'] = attack
    attack_dfs.append(df_a)

df_attacks = pd.concat(attack_dfs, ignore_index=True)
df_all = pd.concat([normal_flows, df_attacks], ignore_index=True).sample(frac=1, random_state=42)
df_all.to_parquet("data/raw/cicids2017/synthetic_cicids.parquet", index=False)
df_all.to_csv("data/raw/cicids2017/synthetic_cicids.csv", index=False)
print(f"  Saved synthetic dataset: {len(df_all):,} flows")
print(f"  Class distribution:\n{df_all['Label'].value_counts()}")
PYEOF

echo ""
echo "=== Data download complete ==="
echo "Files saved to data/raw/"
echo "Run notebook 01_EDA_and_Visualisation.ipynb to begin analysis."
