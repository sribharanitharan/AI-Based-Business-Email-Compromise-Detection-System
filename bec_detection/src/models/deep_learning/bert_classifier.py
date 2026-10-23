"""
bert_classifier.py — BERT Sequence Classifier (Alias to transformer_classifier).
"""
from .transformer_classifier import TransformerBECClassifier

class BERTClassifier(TransformerBECClassifier):
    def __init__(self, model_name: str = "bert-base-uncased"):
        super().__init__(model_name=model_name)
