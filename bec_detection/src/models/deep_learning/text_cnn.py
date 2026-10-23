"""
text_cnn.py — Kim CNN (Convolutional Neural Networks for Sentence/Email Classification)

Architecture:
  Word Embeddings → Parallel 1D Convolutions (kernel sizes: 3, 4, 5) →
  Adaptive Max-Over-Time Pooling → Concatenation → Dropout → Linear → Sigmoid
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional
import numpy as np


class TextCNN(nn.Module):
    """1D CNN with multiple filter widths for catching n-gram BEC phrase patterns."""

    def __init__(
        self,
        vocab_size: int = 10000,
        embed_dim: int = 128,
        num_filters: int = 64,
        filter_sizes: List[int] = None,
        dropout: float = 0.3,
        num_classes: int = 1
    ):
        super(TextCNN, self).__init__()
        if filter_sizes is None:
            filter_sizes = [3, 4, 5]

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len]
        emb = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        emb = emb.permute(0, 2, 1)  # [batch_size, embed_dim, seq_len] for Conv1d

        pooled_outputs = []
        for conv in self.convs:
            # conv output: [batch_size, num_filters, seq_len - kernel_size + 1]
            c = F.relu(conv(emb))
            # Global max pool over time dimension
            p = F.adaptive_max_pool1d(c, 1).squeeze(2)
            pooled_outputs.append(p)

        cat = torch.cat(pooled_outputs, dim=1)  # [batch_size, num_filters * len(filter_sizes)]
        out = self.dropout(cat)
        logits = self.fc(out)
        return torch.sigmoid(logits).squeeze(-1)


class TextCNNClassifier:
    """Wrapper for training, saving, and predicting with TextCNN."""

    def __init__(self, vocab_size: int = 10000, max_seq_len: int = 256):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.word2idx = {"<pad>": 0, "<unk>": 1}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TextCNN(vocab_size=vocab_size).to(self.device)

    def build_vocab(self, texts: List[str]):
        """Build simple word index dictionary."""
        word_counts = {}
        for text in texts:
            for word in text.lower().split():
                word_counts[word] = word_counts.get(word, 0) + 1

        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        for i, (word, _) in enumerate(sorted_words[: self.vocab_size - 2]):
            self.word2idx[word] = i + 2

    def text_to_tensor(self, text: str) -> torch.Tensor:
        """Tokenize and pad text to tensor."""
        tokens = [self.word2idx.get(w, 1) for w in text.lower().split()[: self.max_seq_len]]
        if len(tokens) < self.max_seq_len:
            tokens = tokens + [0] * (self.max_seq_len - len(tokens))
        return torch.tensor(tokens, dtype=torch.long)

    def predict_proba(self, text: str) -> float:
        """Predict probability of BEC for single text."""
        self.model.eval()
        with torch.no_grad():
            tensor = self.text_to_tensor(text).unsqueeze(0).to(self.device)
            prob = self.model(tensor).item()
            return float(prob)
