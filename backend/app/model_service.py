"""Model loading, inference, and lifecycle management service."""

import logging
import time
from pathlib import Path
from typing import Any, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from app.config import settings
from app.features import FEATURE_NAMES

logger = logging.getLogger("fraud_shield.model_service")


class ModelService:
    """Manages the lifecycle, deserialization, and inference for the fraud model and scaler."""

    def __init__(self) -> None:
        self.model: Optional[Any] = None
        self.scaler: Optional[Any] = None
        self._is_loaded: bool = False
        self.loaded_at: Optional[float] = None
        self.model_name: str = "RandomForestClassifier"

    def load_artifacts(self) -> None:
        """Load and validate the model and scaler artifacts from disk.

        Raises:
            FileNotFoundError: If model.pkl or scaler.pkl cannot be found.
            RuntimeError: If artifacts fail validation or compatibility checks.
        """
        model_path = settings.resolved_model_path
        scaler_path = settings.resolved_scaler_path

        logger.info(f"Loading model artifact from: {model_path}")
        logger.info(f"Loading scaler artifact from: {scaler_path}")

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at '{model_path}'. Check MODEL_PATH configuration."
            )
        if not scaler_path.exists():
            raise FileNotFoundError(
                f"Scaler file not found at '{scaler_path}'. Check SCALER_PATH configuration."
            )

        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to unpickle model or scaler artifacts: {exc}"
            ) from exc

        # Compatibility verification
        if not hasattr(self.scaler, "n_features_in_") or self.scaler.n_features_in_ != len(FEATURE_NAMES):
            raise RuntimeError(
                f"Scaler expects {getattr(self.scaler, 'n_features_in_', 'unknown')} features, "
                f"but {len(FEATURE_NAMES)} are defined in feature specifications."
            )

        if not hasattr(self.model, "predict_proba"):
            raise RuntimeError(
                f"Loaded model of type {type(self.model)} lacks 'predict_proba' method."
            )

        self.model_name = type(self.model).__name__
        self._is_loaded = True
        self.loaded_at = time.time()

        # Warm up model with a zero-cost test inference
        try:
            dummy_input = pd.DataFrame(
                [[12.0, 1000.0, 5000.0, 4000.0, 2000.0, 3000.0, 0.0, 0.0, 1.0]],
                columns=FEATURE_NAMES,
            )
            dummy_scaled = self.scaler.transform(dummy_input)
            _ = self.model.predict_proba(dummy_scaled)
            logger.info("Model warmup successful. Artifacts ready for inference.")
        except Exception as exc:
            raise RuntimeError(f"Warmup inference failed on loaded artifacts: {exc}") from exc

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def compute_tree_contributions(self, scaled_features: np.ndarray) -> np.ndarray:
        """Compute exact instance feature contributions (Saabas/TreeInterpreter) for Random Forest.

        Decomposes the prediction into the exact delta each feature added along the decision paths
        across all 50 trees in the ensemble.
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded.")

        X_scaled_f32 = scaled_features.astype(np.float32)
        n_features = X_scaled_f32.shape[1]
        contributions = np.zeros(n_features)

        for estimator in self.model.estimators_:
            tree = estimator.tree_
            # Class 1 is fraud probability at each node
            values = tree.value[:, 0, :]
            node_probs = values[:, 1] / values.sum(axis=1)

            node_indicators = tree.decision_path(X_scaled_f32).indices
            for i in range(len(node_indicators) - 1):
                curr_node = node_indicators[i]
                next_node = node_indicators[i + 1]
                split_feat = tree.feature[curr_node]
                diff = node_probs[next_node] - node_probs[curr_node]
                contributions[split_feat] += diff

        n_trees = len(self.model.estimators_)
        contributions /= n_trees
        return contributions

    def predict_risk(self, features_df: pd.DataFrame) -> Tuple[float, np.ndarray]:
        """Predict fraud probability using the scaled feature vector.

        Args:
            features_df: DataFrame matching FEATURE_NAMES.

        Returns:
            Tuple of (fraud_probability: float, scaled_feature_array: np.ndarray)
        """
        if not self._is_loaded or self.model is None or self.scaler is None:
            raise RuntimeError("ModelService is not loaded. Call load_artifacts() first.")

        scaled = self.scaler.transform(features_df)
        probs = self.model.predict_proba(scaled)
        fraud_prob = float(probs[0, 1])
        return fraud_prob, scaled

    def predict_risk_and_contributions(
        self, features_df: pd.DataFrame
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """Predict fraud probability and calculate exact instance feature contributions.

        Returns:
            Tuple of (fraud_probability: float, scaled_features: np.ndarray, contributions: np.ndarray)
        """
        fraud_prob, scaled = self.predict_risk(features_df)
        contributions = self.compute_tree_contributions(scaled)
        return fraud_prob, scaled, contributions


# Global singleton instance
model_service = ModelService()
