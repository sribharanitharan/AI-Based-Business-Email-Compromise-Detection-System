"""
gnn_model.py — Graph Neural Network (GNN) for Node Embedding and Edge Classification in BEC.

Uses message-passing layers (GraphSAGE / GCN representation) over communication graphs.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class EmailGNN(nn.Module):
    """GNN layer mapping sender-recipient communication embeddings to anomaly scores."""
    def __init__(self, in_features: int = 16, hidden_dim: int = 32, num_classes: int = 2):
        super().__init__()
        self.fc_self = nn.Linear(in_features, hidden_dim)
        self.fc_neigh = nn.Linear(in_features, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, sender_feat: torch.Tensor, recipient_feat: torch.Tensor) -> torch.Tensor:
        # Aggregation of sender and recipient graph topological representations
        h_s = F.relu(self.fc_self(sender_feat))
        h_r = F.relu(self.fc_neigh(recipient_feat))
        edge_rep = torch.cat([h_s, h_r], dim=-1)
        logits = self.classifier(edge_rep)
        return torch.softmax(logits, dim=-1)
