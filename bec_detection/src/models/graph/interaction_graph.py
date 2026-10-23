"""
interaction_graph.py — Tier 5 Graph Analysis for Corporate Email Communication

Constructs an email interaction network (NetworkX directed multigraph) to analyze:
  - Node centrality (PageRank, In-Degree, Out-Degree)
  - Executive / VIP isolation metrics
  - Anomalous edges (first-time contact between employee and executive)
  - Communication bridges and lateral movement patterns
"""

import networkx as nx
from typing import Dict, Any, List, Tuple, Optional
import math


class EmailInteractionGraph:
    """Network-level social graph for detecting communication anomalies in BEC."""

    def __init__(self, vip_list: Optional[List[str]] = None):
        self.graph = nx.DiGraph()
        self.vip_list = set(v.lower() for v in (vip_list or ["ceo@company.com", "cfo@company.com", "finance@company.com"]))

    def add_interaction(self, sender: str, recipient: str, timestamp: Optional[float] = None, weight: float = 1.0):
        """Add an email transaction to the interaction graph."""
        s = sender.lower().strip()
        r = recipient.lower().strip()
        if not s or not r:
            return

        if self.graph.has_edge(s, r):
            self.graph[s][r]["weight"] += weight
            self.graph[s][r]["count"] += 1
        else:
            self.graph.add_edge(s, r, weight=weight, count=1, first_seen=timestamp)

    def build_from_logs(self, email_records: List[Dict[str, Any]]):
        """Build graph from a list of historical email logs."""
        for rec in email_records:
            sender = rec.get("sender", "")
            recipients = rec.get("recipients", [rec.get("to", "")])
            if isinstance(recipients, str):
                recipients = [recipients]
            for r in recipients:
                self.add_interaction(sender, r)

    def analyze_interaction(self, sender: str, recipient: str) -> Dict[str, Any]:
        """
        Evaluate graph topology anomaly score for a given sender -> recipient pair.
        """
        s = sender.lower().strip()
        r = recipient.lower().strip()

        is_new_edge = not self.graph.has_edge(s, r)
        is_sender_known = self.graph.has_node(s)
        is_recipient_vip = r in self.vip_list or any(auth in r for auth in ["ceo", "cfo", "director", "exec"])

        sender_in_degree = self.graph.in_degree(s) if is_sender_known else 0
        sender_out_degree = self.graph.out_degree(s) if is_sender_known else 0

        # Calculate anomaly score
        anomaly_score = 0.0

        if is_new_edge:
            anomaly_score += 0.35

        if is_new_edge and is_recipient_vip:
            # First time contacting a VIP is very suspicious in BEC (CEO fraud)
            anomaly_score += 0.40

        if not is_sender_known:
            # Completely external or unseen sender
            anomaly_score += 0.20

        # Reciprocity check: has recipient ever sent an email back to sender?
        reciprocal = self.graph.has_edge(r, s)
        if not reciprocal and is_recipient_vip:
            anomaly_score += 0.15

        risk_score = min(round(anomaly_score, 4), 1.0)

        return {
            "is_new_edge": is_new_edge,
            "is_sender_known": is_sender_known,
            "is_recipient_vip": is_recipient_vip,
            "sender_in_degree": sender_in_degree,
            "sender_out_degree": sender_out_degree,
            "has_reciprocity": reciprocal,
            "graph_anomaly_score": risk_score,
            "is_anomalous_interaction": bool(risk_score >= 0.55)
        }
