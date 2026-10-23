"""
anomaly_detector.py — Tier 4 Unsupervised Anomaly Detection

Detects novel / zero-day BEC campaigns and sudden behavioral shifts:
  1. Isolation Forest (subspace tree partitioning)
  2. One-Class SVM (boundary learning around benign emails)
  3. PyTorch Deep Autoencoder (reconstruction error thresholding)
"""

import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:
    class DeepAutoencoder(nn.Module):
        """Autoencoder for feature reconstruction anomaly detection."""

        def __init__(self, input_dim: int = 25, hidden_dims: list = None):
            super().__init__()
            if hidden_dims is None:
                hidden_dims = [16, 8]

            # Encoder
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dims[0]),
                nn.BatchNorm1d(hidden_dims[0]),
                nn.ReLU(),
                nn.Linear(hidden_dims[0], hidden_dims[1]),
                nn.ReLU()
            )
            # Decoder
            self.decoder = nn.Sequential(
                nn.Linear(hidden_dims[1], hidden_dims[0]),
                nn.ReLU(),
                nn.Linear(hidden_dims[0], input_dim),
                nn.Sigmoid()
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            latent = self.encoder(x)
            recon = self.decoder(latent)
            return recon


class AnomalyDetectorSuite:
    """Suite containing Isolation Forest, One-Class SVM, and deep autoencoders."""

    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state

        self.iso_forest = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=150,
            n_jobs=-1
        )
        self.oc_svm = OneClassSVM(
            nu=contamination,
            kernel="rbf",
            gamma="scale"
        )
        self.ae_threshold: float = 0.1
        self.ae_model = None

    def fit_benign(self, X_benign: np.ndarray):
        """Fit unsupervised models solely on legitimate (benign) corporate emails."""
        self.iso_forest.fit(X_benign)
        self.oc_svm.fit(X_benign)

        if HAS_TORCH:
            input_dim = X_benign.shape[1]
            self.ae_model = DeepAutoencoder(input_dim=input_dim)
            optimizer = torch.optim.Adam(self.ae_model.parameters(), lr=1e-3)
            criterion = nn.MSELoss()

            # Normalize benign features to [0, 1] range for Sigmoid output
            mins = X_benign.min(axis=0)
            maxs = X_benign.max(axis=0) + 1e-6
            X_norm = (X_benign - mins) / (maxs - mins)
            tensor_x = torch.tensor(X_norm, dtype=torch.float32)

            self.ae_model.train()
            for _ in range(20):
                optimizer.zero_grad()
                recon = self.ae_model(tensor_x)
                loss = criterion(recon, tensor_x)
                loss.backward()
                optimizer.step()

            # Compute threshold as 95th percentile error
            self.ae_model.eval()
            with torch.no_grad():
                recon = self.ae_model(tensor_x)
                errors = torch.mean((recon - tensor_x) ** 2, dim=1).numpy()
                self.ae_threshold = float(np.percentile(errors, 95))

    def predict_anomaly_score(self, x_vec: np.ndarray) -> Dict[str, float]:
        """
        Return normalized anomaly risk scores [0.0 = completely normal, 1.0 = highly anomalous].
        """
        if x_vec.ndim == 1:
            x_mat = x_vec.reshape(1, -1)
        else:
            x_mat = x_vec

        # Isolation forest score (lower means more anomalous, decision_function is around [-0.5, 0.5])
        iso_raw = self.iso_forest.decision_function(x_mat)[0]
        # Invert and normalize to [0, 1]
        iso_score = float(np.clip(0.5 - iso_raw, 0.0, 1.0))

        # One-Class SVM score
        svm_raw = self.oc_svm.decision_function(x_mat)[0]
        svm_score = float(np.clip(0.5 - (svm_raw / 2.0), 0.0, 1.0))

        ae_score = 0.0
        if HAS_TORCH and self.ae_model is not None:
            self.ae_model.eval()
            with torch.no_grad():
                t = torch.tensor(x_mat, dtype=torch.float32)
                recon = self.ae_model(t)
                err = torch.mean((recon - t) ** 2).item()
                ae_score = float(np.clip(err / (self.ae_threshold * 2 + 1e-6), 0.0, 1.0))

        combined = round((iso_score * 0.4) + (svm_score * 0.3) + (ae_score * 0.3), 4)

        return {
            "isolation_forest_score": round(iso_score, 4),
            "one_class_svm_score": round(svm_score, 4),
            "autoencoder_score": round(ae_score, 4),
            "combined_anomaly_score": combined,
            "is_anomaly": bool(combined > 0.6)
        }

    def save(self, output_dir: str):
        """Save models to directory."""
        p = Path(output_dir)
        p.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.iso_forest, p / "isolation_forest.joblib")
        joblib.dump(self.oc_svm, p / "one_class_svm.joblib")
        if HAS_TORCH and self.ae_model is not None:
            torch.save(self.ae_model.state_dict(), p / "autoencoder.pt")

    def load(self, model_dir: str):
        """Load trained anomaly models."""
        p = Path(model_dir)
        if (p / "isolation_forest.joblib").exists():
            self.iso_forest = joblib.load(p / "isolation_forest.joblib")
        if (p / "one_class_svm.joblib").exists():
            self.oc_svm = joblib.load(p / "one_class_svm.joblib")
