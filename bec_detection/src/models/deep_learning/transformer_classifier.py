"""
transformer_classifier.py — BERT / RoBERTa / DistilBERT fine-tuning and inference for BEC detection.

Handles:
  - Tokenization via HuggingFace AutoTokenizer
  - AutoModelForSequenceClassification
  - Safe fallback if pretrained weights need to be downloaded or loaded locally
"""

from typing import List, Dict, Any, Optional
import numpy as np

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class TransformerBECClassifier:
    """HuggingFace Transformer wrapper for BEC classification."""

    def __init__(self, model_name: str = "distilbert-base-uncased"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if (HAS_TRANSFORMERS and torch.cuda.is_available()) else "cpu"
        self._initialized = False

    def load(self, model_path: Optional[str] = None):
        """Load transformer model and tokenizer."""
        if not HAS_TRANSFORMERS:
            return False
        try:
            target = model_path or self.model_name
            self.tokenizer = AutoTokenizer.from_pretrained(target)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                target, num_labels=2
            ).to(self.device)
            self.model.eval()
            self._initialized = True
            return True
        except Exception as e:
            # When offline, gracefully mark uninitialized
            self._initialized = False
            return False

    def predict_proba(self, text: str) -> float:
        """Predict BEC risk probability [0.0, 1.0]."""
        if not self._initialized or not HAS_TRANSFORMERS:
            # Fallback heuristic
            return 0.5

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            # Probability of class 1 (BEC)
            return float(probs[0, 1].item())
