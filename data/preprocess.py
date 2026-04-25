"""
NetSentinel-RL — Data Preprocessing Pipeline
Transforms raw CICIDS2017 / KDD Cup 99 CSVs into normalised train/val/test splits
and produces the flow-text representations used by the classifier agent.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Column / label constants ──────────────────────────────────────────────────

FLOW_FEATURES: List[str] = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Mean",
    "Bwd Packet Length Max",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Fwd IAT Mean",
    "Bwd IAT Mean",
    "Fwd PSH Flags",
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "SYN Flag Count",
    "RST Flag Count",
    "ACK Flag Count",
    "Down/Up Ratio",
    "Avg Packet Size",
]

ATTACK_LABELS: Dict[str, int] = {
    "BENIGN":                    0,
    "DoS Hulk":                  1,
    "DoS GoldenEye":             1,
    "DoS slowloris":             1,
    "DoS Slowhttptest":          1,
    "DDoS":                      2,
    "PortScan":                  3,
    "FTP-Patator":               4,
    "SSH-Patator":               4,
    "Bot":                       5,
    "Web Attack":                6,
    "Web Attack – Brute Force":  6,
    "Infiltration":              7,
}

ATTACK_NAMES: Dict[int, str] = {
    0: "Benign",
    1: "DoS",
    2: "DDoS",
    3: "PortScan",
    4: "BruteForce",
    5: "Botnet",
    6: "WebAttack",
    7: "Infiltration",
}


# ── Custom transformers ───────────────────────────────────────────────────────

class ClipOutlierTransformer(BaseEstimator, TransformerMixin):
    """Clips each feature to a fitted upper percentile, then standard-scales."""

    def __init__(self, clip_percentile: float = 99.5) -> None:
        self.clip_percentile = clip_percentile
        self._upper_bounds: np.ndarray = np.array([])
        self._scaler = StandardScaler()

    def fit(self, X: np.ndarray, y=None) -> "ClipOutlierTransformer":
        X = np.asarray(X, dtype=np.float64)
        self._upper_bounds = np.percentile(X, self.clip_percentile, axis=0)
        X_clipped = np.clip(X, a_min=None, a_max=self._upper_bounds)
        self._scaler.fit(X_clipped)
        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        X_clipped = np.clip(X, a_min=None, a_max=self._upper_bounds)
        return self._scaler.transform(X_clipped).astype(np.float32)


class FlowTextEncoder(BaseEstimator, TransformerMixin):
    """
    Converts a numeric flow feature vector to a natural-language string.

    Example output:
        "flow duration 1.24s | fwd_packets 12 | bwd_packets 8 | …"

    This representation is consumed by the DistilBERT-based classifier.
    The vocabulary mirrors how a human analyst would describe the flow,
    enabling pre-trained language-model knowledge to transfer.
    """

    _TEMPLATES = [
        ("Flow Duration",              "flow duration {:.2f}s"),
        ("Total Fwd Packets",          "fwd_packets {:.0f}"),
        ("Total Backward Packets",     "bwd_packets {:.0f}"),
        ("Flow Bytes/s",               "bytes_rate {:.0f}Bps"),
        ("Flow Packets/s",             "pkt_rate {:.1f}pps"),
        ("SYN Flag Count",             "syn_flags {:.0f}"),
        ("RST Flag Count",             "rst_flags {:.0f}"),
        ("ACK Flag Count",             "ack_flags {:.0f}"),
        ("Avg Packet Size",            "avg_pkt_size {:.1f}B"),
        ("Down/Up Ratio",              "down_up_ratio {:.2f}"),
        ("Flow IAT Mean",              "iat_mean {:.3f}"),
        ("Fwd Packet Length Mean",     "fwd_pkt_mean {:.1f}B"),
    ]

    def fit(self, X, y=None) -> "FlowTextEncoder":
        return self

    def transform(self, X: pd.DataFrame, y=None) -> List[str]:
        texts = []
        for _, row in X.iterrows():
            parts = []
            for col, template in self._TEMPLATES:
                if col in row and pd.notna(row[col]):
                    parts.append(template.format(row[col]))
            texts.append(" | ".join(parts))
        return texts


# ── Pipeline builder ──────────────────────────────────────────────────────────

def build_preprocessing_pipeline(clip_percentile: float = 99.5) -> Pipeline:
    return Pipeline([("clip_scale", ClipOutlierTransformer(clip_percentile))])


# ── Main preprocessing function ───────────────────────────────────────────────

def preprocess(
    csv_path: str,
    label_col: str = " Label",
    out_dir: str = "data/processed",
    val_size: float = 0.10,
    test_size: float = 0.15,
    random_state: int = 42,
    max_rows: Optional[int] = None,
) -> Tuple[np.ndarray, ...]:
    """
    Load a CICIDS2017-format CSV, clean it, and produce normalised splits.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    log.info("Loading %s …", csv_path)
    df = pd.read_csv(csv_path, nrows=max_rows)
    df.columns = df.columns.str.strip()

    # Map string labels to integers; drop rows with unknown labels
    label_col_clean = label_col.strip()
    df[label_col_clean] = df[label_col_clean].str.strip().map(ATTACK_LABELS)
    df = df.dropna(subset=[label_col_clean])
    df[label_col_clean] = df[label_col_clean].astype(int)

    # Select features present in this file
    available = [c for c in FLOW_FEATURES if c in df.columns]
    if len(available) < len(FLOW_FEATURES):
        missing = set(FLOW_FEATURES) - set(available)
        log.warning("Missing features (will be zero-padded): %s", missing)

    X_raw = df[available].fillna(0.0).replace([np.inf, -np.inf], 0.0).values
    y     = df[label_col_clean].values
    log.info("Shape after cleaning: X=%s  label distribution: %s",
             X_raw.shape, dict(zip(*np.unique(y, return_counts=True))))

    # Train / val / test split
    X_tv, X_test, y_tv, y_test = train_test_split(
        X_raw, y, test_size=test_size, stratify=y, random_state=random_state
    )
    adjusted_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=adjusted_val, stratify=y_tv, random_state=random_state
    )

    # Fit and apply preprocessing pipeline
    pipe = build_preprocessing_pipeline()
    X_train = pipe.fit_transform(X_train)
    X_val   = pipe.transform(X_val)
    X_test  = pipe.transform(X_test)

    # Persist
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "X_train.npy", X_train)
    np.save(out / "X_val.npy",   X_val)
    np.save(out / "X_test.npy",  X_test)
    np.save(out / "y_train.npy", y_train)
    np.save(out / "y_val.npy",   y_val)
    np.save(out / "y_test.npy",  y_test)
    with open(out / "pipeline.pkl", "wb") as f:
        pickle.dump(pipe, f)

    log.info("Saved splits to %s", out_dir)
    return X_train, X_val, X_test, y_train, y_val, y_test
