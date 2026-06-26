"""
top_nodes.py

Finds the nodes (addresses) with the highest degree in the zkSync
transaction network, for each log file separately.

Prints top N addresses ranked by: total degree, in-degree, out-degree.
"""

import pickle
import os
import networkx as nx
from collections import Counter
from typing import List, Dict

# ── Config ────────────────────────────────────────────────────────────────────
LOG_DIR = r"D:\zkSync_logs"
LOG_FILES = [
    "logs_69900000_70000000_final.pkl",
    "logs_14900000_15000000_final.pkl",
]
TOP_N = 20   # how many top nodes to show


# ── Load ──────────────────────────────────────────────────────────────────────
def load_logs(fname: str) -> List[Dict]:
    path = os.path.join(LOG_DIR, fname)
    with open(path, "rb") as f:
        data = pickle.load(f)
    logs = data["logs"]
    print(f"  Loaded {len(logs):,} logs")
    return logs


# ── Build directed graph ──────────────────────────────────────────────────────
def build_graph(logs: List[Dict]) -> nx.DiGraph:
    G = nx.DiGraph()
    for log in logs:
        src = log.get("from")
        dst = log.get("to")
        if src and dst and src != dst:
            if G.has_edge(src, dst):
                G[src][dst]["weight"] += 1
            else:
                G.add_edge(src, dst, weight=1)
    print(f"  Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    return G


# ── Print top nodes ───────────────────────────────────────────────────────────
def print_top(label: str, ranked: list, top_n: int):
    print(f"\n  Top {top_n} by {label}:")
    print(f"  {'Rank':<5} {'Degree':>8}  {'Address'}")
    print(f"  {'─'*5} {'─'*8}  {'─'*42}")
    for rank, (node, deg) in enumerate(ranked[:top_n], 1):
        print(f"  {rank:<5} {deg:>8,}  {node}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    for fname in LOG_FILES:
        label = fname.replace("logs_", "").replace("_final.pkl", "")
        print(f"\n{'='*60}")
        print(f"Dataset: {label}")
        print(f"{'='*60}")

        logs  = load_logs(fname)
        G     = build_graph(logs)

        # Sort by degree descending
        by_total  = sorted(G.degree(),    key=lambda x: x[1], reverse=True)
        by_in     = sorted(G.in_degree(), key=lambda x: x[1], reverse=True)
        by_out    = sorted(G.out_degree(),key=lambda x: x[1], reverse=True)

        print_top("total degree (in + out)", by_total, TOP_N)
        print_top("in-degree  (received)",   by_in,    TOP_N)
        print_top("out-degree (sent)",       by_out,   TOP_N)

    print(f"\n{'='*60}\nDone.\n")


if __name__ == "__main__":
    main()