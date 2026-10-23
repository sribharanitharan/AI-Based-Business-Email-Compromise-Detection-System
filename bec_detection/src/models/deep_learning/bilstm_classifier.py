"""
bilstm_classifier.py  —  Bidirectional LSTM with Attention for BEC Detection (Tier 3)

Architecture:
    Embedding → Bi-LSTM → Attention Pooling → MLP → Sigmoid

Trains on tokenized email text (subject + body).
Input: raw email text string (tokenized internally via a simple word vocab).
Output: probability of BEC (0–1).
"""

import re
import json
import numpy as np
import joblib
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, roc_auc_score

BASE_DIR = Path(__file__).parent.parent.parent  # bec_detection/

# ── Tokenizer ─────────────────────────────────────────────────────────────────

class SimpleTokenizer:
    """Word-level tokenizer with vocabulary built from training data."""

    PAD = 0
    UNK = 1

    def __init__(self, max_vocab: int = 10_000, max_len: int = 256):
        self.max_vocab = max_vocab
        self.max_len   = max_len
        self.word2idx  = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word  = {0: "<PAD>", 1: "<UNK>"}

    def _tokenize(self, text: str):
        return re.findall(r'\b\w+\b', text.lower())

    def fit(self, texts):
        counter = Counter()
        for t in texts:
            counter.update(self._tokenize(t))
        for word, _ in counter.most_common(self.max_vocab - 2):
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx]  = word
        return self

    def encode(self, text: str) -> list:
        tokens = self._tokenize(text)[: self.max_len]
        ids    = [self.word2idx.get(t, self.UNK) for t in tokens]
        # Pad to max_len
        ids += [self.PAD] * (self.max_len - len(ids))
        return ids

    def vocab_size(self) -> int:
        return len(self.word2idx)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.word2idx, f)

    @classmethod
    def load(cls, path: str, max_len: int = 256):
        obj = cls(max_len=max_len)
        with open(path) as f:
            obj.word2idx = json.load(f)
        obj.idx2word = {v: k for k, v in obj.word2idx.items()}
        return obj


# ── Dataset ───────────────────────────────────────────────────────────────────

class EmailTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer: SimpleTokenizer):
        self.encodings = [tokenizer.encode(t) for t in texts]
        self.labels    = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.encodings[idx], dtype=torch.long),
            torch.tensor(self.labels[idx],    dtype=torch.float32)
        )


# ── Model ─────────────────────────────────────────────────────────────────────

class BiLSTMAttention(nn.Module):
    """Bidirectional LSTM with self-attention pooling for text classification."""

    def __init__(self, vocab_size: int, embed_dim: int = 128,
                 hidden_dim: int = 256, n_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.attention  = nn.Linear(hidden_dim * 2, 1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded   = self.dropout(self.embedding(x))          # (B, seq, embed)
        lstm_out, _ = self.lstm(embedded)                      # (B, seq, hidden*2)
        attn_w     = torch.softmax(self.attention(lstm_out), dim=1)  # (B, seq, 1)
        context    = (lstm_out * attn_w).sum(dim=1)            # (B, hidden*2)
        return self.classifier(context).squeeze(1)             # (B,)


# ── Training ──────────────────────────────────────────────────────────────────

def train_bilstm(texts_train, y_train, texts_val, y_val,
                 epochs: int = 10, batch_size: int = 32,
                 embed_dim: int = 128, hidden_dim: int = 256,
                 max_vocab: int = 10_000, max_len: int = 256,
                 dropout: float = 0.3, lr: float = 1e-3):
    """
    Train the Bi-LSTM model on email text.

    Args:
        texts_train / texts_val  – raw email text strings
        y_train / y_val          – binary labels (0/1)

    Returns:
        model     – trained BiLSTMAttention
        tokenizer – fitted SimpleTokenizer
        val_f1    – F1 on validation set
        val_auc   – AUC on validation set
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    [BiLSTM] Device: {device}")

    tokenizer = SimpleTokenizer(max_vocab=max_vocab, max_len=max_len)
    tokenizer.fit(texts_train + texts_val)

    train_ds = EmailTextDataset(texts_train, y_train, tokenizer)
    val_ds   = EmailTextDataset(texts_val,   y_val,   tokenizer)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size)

    model = BiLSTMAttention(
        vocab_size=tokenizer.vocab_size(),
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)

    # Class weighting for imbalance
    pos_weight = torch.tensor(
        [(y_train == 0).sum() / max((y_train == 1).sum(), 1)],
        dtype=torch.float32
    ).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=2, factor=0.5
    )

    best_val_auc = 0.0
    best_state   = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb = xb.to(device)
                p  = model(xb).cpu().numpy()
                all_preds.extend(p.tolist())
                all_labels.extend(yb.numpy().tolist())

        try:
            auc = roc_auc_score(all_labels, all_preds)
            f1  = f1_score(all_labels, (np.array(all_preds) >= 0.5).astype(int), zero_division=0)
        except Exception:
            auc, f1 = 0.0, 0.0

        scheduler.step(1 - auc)
        if auc > best_val_auc:
            best_val_auc = auc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"    [BiLSTM] Epoch {epoch+1:02d}/{epochs}  "
              f"Loss={total_loss/len(train_dl):.4f}  F1={f1:.4f}  AUC={auc:.4f}")

    if best_state:
        model.load_state_dict(best_state)
    model.to("cpu")

    # Final eval
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in val_dl:
            p = model(xb).numpy()
            all_preds.extend(p.tolist())
            all_labels.extend(yb.numpy().tolist())
    val_f1  = f1_score(all_labels, (np.array(all_preds) >= 0.5).astype(int), zero_division=0)
    val_auc = roc_auc_score(all_labels, all_preds)

    print(f"    [BiLSTM] Best  F1={val_f1:.4f}  AUC={val_auc:.4f}")
    return model, tokenizer, val_f1, val_auc


def predict_bilstm(model: BiLSTMAttention, tokenizer: SimpleTokenizer,
                   texts) -> np.ndarray:
    """Return probability array for a list of text strings."""
    model.eval()
    ids  = [tokenizer.encode(t) for t in texts]
    xb   = torch.tensor(ids, dtype=torch.long)
    with torch.no_grad():
        return model(xb).numpy()


def save_bilstm(model, tokenizer, model_path: str = None, tok_path: str = None):
    model_path = model_path or str(BASE_DIR / "models" / "bilstm.pt")
    tok_path   = tok_path   or str(BASE_DIR / "models" / "bilstm_tokenizer.json")
    torch.save(model.state_dict(), model_path)
    tokenizer.save(tok_path)
    # Save config
    cfg = {
        "vocab_size": tokenizer.vocab_size(),
        "max_len":    tokenizer.max_len,
    }
    joblib.dump(cfg, str(BASE_DIR / "models" / "bilstm_config.pkl"))
    print(f"    [BiLSTM] Saved → {model_path}")


def load_bilstm(model_path: str = None, tok_path: str = None, cfg_path: str = None):
    model_path = model_path or str(BASE_DIR / "models" / "bilstm.pt")
    tok_path   = tok_path   or str(BASE_DIR / "models" / "bilstm_tokenizer.json")
    cfg_path   = cfg_path   or str(BASE_DIR / "models" / "bilstm_config.pkl")
    cfg        = joblib.load(cfg_path)
    tokenizer  = SimpleTokenizer.load(tok_path, max_len=cfg["max_len"])
    model      = BiLSTMAttention(vocab_size=cfg["vocab_size"])
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model, tokenizer
