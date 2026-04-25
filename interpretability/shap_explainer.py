"""
NetSentinel-RL — SHAP Explainability Module
Computes SHAP values for tree-based models and exports summary artefacts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False
    log.warning("shap not installed — explainability features disabled.")


FEATURE_NAMES: List[str] = [
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


class SHAPExplainer:
    """
    Wraps shap.TreeExplainer (or KernelExplainer as fallback) for the
    NetSentinel classifier models.

    Usage
    -----
    explainer = SHAPExplainer(model)
    values = explainer.explain(X_sample)
    top = explainer.top_features(X_sample[0])
    """

    def __init__(
        self,
        model,
        feature_names: Optional[List[str]] = None,
        background_data: Optional[np.ndarray] = None,
        n_background: int = 100,
    ) -> None:
        if not _SHAP_AVAILABLE:
            raise ImportError("Install shap: pip install shap")

        self.model         = model
        self.feature_names = feature_names or FEATURE_NAMES

        try:
            self._explainer = shap.TreeExplainer(model)
            log.info("Using TreeExplainer.")
        except Exception:
            if background_data is None:
                raise ValueError("background_data required for KernelExplainer.")
            bg = shap.sample(background_data, n_background)
            self._explainer = shap.KernelExplainer(model.predict_proba, bg)
            log.info("Using KernelExplainer (slower).")

    def explain(self, X: np.ndarray, check_additivity: bool = False) -> np.ndarray:
        """
        Return raw SHAP values array.

        Shape for binary classification: (n_samples, n_features)
        Shape for multiclass:            (n_classes, n_samples, n_features)
        """
        values = self._explainer.shap_values(X, check_additivity=check_additivity)
        # For multiclass TreeExplainer returns a list; stack to 3-D array
        if isinstance(values, list):
            return np.stack(values, axis=0)
        return values

    def top_features(
        self, x: np.ndarray, n: int = 5, class_idx: int = 1
    ) -> Dict[str, float]:
        """
        Return the top-n most important features for a single flow vector.

        Parameters
        ----------
        x         : 1-D feature vector.
        n         : Number of top features to return.
        class_idx : Which class to explain for multiclass models.
        """
        values = self.explain(x.reshape(1, -1))

        if values.ndim == 3:
            row = values[class_idx, 0, :]
        else:
            row = values[0, :]

        order = np.argsort(np.abs(row))[::-1][:n]
        return {
            self.feature_names[i]: float(row[i])
            for i in order
            if i < len(self.feature_names)
        }

    def mean_importance(
        self, X: np.ndarray, class_idx: int = 1
    ) -> pd.Series:
        """
        Compute mean absolute SHAP importance across a sample set.

        Returns a pd.Series sorted descending, indexed by feature name.
        """
        values = self.explain(X)
        if values.ndim == 3:
            mat = np.abs(values[class_idx])
        else:
            mat = np.abs(values)

        means = mat.mean(axis=0)
        n     = min(len(means), len(self.feature_names))
        return (
            pd.Series(means[:n], index=self.feature_names[:n])
            .sort_values(ascending=False)
        )

    def summary_plot(
        self,
        X: np.ndarray,
        out_path: Optional[str] = None,
        plot_type: str = "bar",
        class_idx: int = 1,
    ) -> None:
        """
        Render and optionally save a SHAP summary plot.

        Parameters
        ----------
        X         : Feature matrix to explain.
        out_path  : If given, saves the figure to this path (PNG).
        plot_type : "bar" | "dot" | "violin"
        class_idx : Class index for multiclass models.
        """
        import matplotlib.pyplot as plt

        values = self.explain(X)
        if isinstance(values, np.ndarray) and values.ndim == 3:
            shap_vals = values[class_idx]
        else:
            shap_vals = values

        shap.summary_plot(
            shap_vals,
            X,
            feature_names=self.feature_names[: X.shape[1]],
            plot_type=plot_type,
            show=out_path is None,
        )

        if out_path:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close()
            log.info("SHAP summary saved → %s", out_path)

    def dependence_plot(
        self,
        feature: str,
        X: np.ndarray,
        out_path: Optional[str] = None,
        class_idx: int = 1,
    ) -> None:
        """Plot SHAP dependence for a single feature."""
        import matplotlib.pyplot as plt

        values = self.explain(X)
        shap_vals = values[class_idx] if values.ndim == 3 else values

        shap.dependence_plot(
            feature,
            shap_vals,
            X,
            feature_names=self.feature_names[: X.shape[1]],
            show=out_path is None,
        )

        if out_path:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close()
