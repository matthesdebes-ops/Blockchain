"""
analyze_shortest_path_excluding_top20.py

Same analysis as original pipeline, but removes the 20 highest-degree nodes
before extracting the largest connected component and running BFS analysis.

CHANGES:
  1. Findings (avg hop length, ln(N), ratio, diameter, etc.) are saved to
     JSON files, both per-file and combined.
  2. The order (node count) of the largest connected component (LCC) is
     saved explicitly and is what's used for the small-world ratio -
     NOT the node count of the original/full graph.
  3. Diameter of the LCC is computed and saved alongside the findings.
     Since exact diameter (nx.diameter) is O(N*E) and can be very slow on
     large graphs, we report an approximate diameter: the maximum shortest
     path length observed across all sampled-source BFS runs. This is a
     lower bound on the true diameter, and gets tighter as N_SOURCES grows.
     If you want the exact diameter and the graph is small enough, set
     EXACT_DIAMETER = True below.
"""

import pickle
import os
import json
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from typing import Iterator
import time
import random
import gc

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

LOG_DIR = r"D:\zkSync_logs"

LOG_FILES = [
    "eth_transfers_69900000_70000000.pkl",
    "eth_transfers_14900000_15000000.pkl",
]

N_SOURCES      = 5000
CHUNK_SIZE     = 50000
TOP_K_REMOVE   = 5
EDGE_SHARE_FRACTION = 0.5   # Part 2: remove the fewest highest-degree nodes covering this share of all edges

RESULTS_DIR    = r"D:\zkSync_logs\results"
EXACT_DIAMETER = False  # set True to force nx.diameter() instead of the BFS-sample approximation


# ----------------------------------------------------------------------
# Load logs in chunks
# ----------------------------------------------------------------------

def load_pickle_in_chunks(file_path: str, chunk_size: int = CHUNK_SIZE) -> Iterator:
    full_path = os.path.join(LOG_DIR, file_path)
    if not os.path.exists(full_path):
        print(f"  File not found: {full_path}")
        return

    print(f"  Loading {file_path} in chunks of {chunk_size:,}...")

    with open(full_path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict) and "logs" in data:
        logs = data["logs"]

        for i in range(0, len(logs), chunk_size):
            yield logs[i:i + chunk_size]
            gc.collect()


# ----------------------------------------------------------------------
# Build graph
# ----------------------------------------------------------------------

def build_graph_from_file(file_name: str) -> nx.Graph:
    print(f"\nBUILDING GRAPH: {file_name}")

    edge_set = set()
    total_logs = 0

    for chunk in load_pickle_in_chunks(file_name):
        for log in chunk:
            u = log.get("from")
            v = log.get("to")

            if u and v and u != v:
                edge_set.add(tuple(sorted((u, v))))

            total_logs += 1

        print(f"  logs={total_logs:,}, edges={len(edge_set):,}")

    G = nx.Graph()
    G.add_edges_from(edge_set)

    print(f"  Nodes={G.number_of_nodes():,}, Edges={G.number_of_edges():,}")
    return G


# ----------------------------------------------------------------------
# Remove top-K degree nodes
# ----------------------------------------------------------------------

def remove_top_k_hubs(G: nx.Graph, k: int) -> nx.Graph:
    print(f"\nRemoving top {k} degree nodes...")

    degree_sorted = sorted(G.degree(), key=lambda x: x[1], reverse=True)
    removed = {node for node, _ in degree_sorted[:k]}

    H = G.subgraph([n for n in G.nodes() if n not in removed]).copy()

    print(f"  Removed nodes: {len(removed):,}")
    print(f"  Remaining nodes: {H.number_of_nodes():,}")

    return H


# ----------------------------------------------------------------------
# Remove the fewest highest-degree nodes that cover X% of all edges
# ----------------------------------------------------------------------

def remove_hubs_by_edge_share(G: nx.Graph, fraction: float = EDGE_SHARE_FRACTION) -> nx.Graph:
    """
    Instead of removing a fixed count of top-degree nodes, remove the
    smallest possible set of highest-degree nodes such that the edges
    touching those nodes account for >= `fraction` of all edges in G.

    Nodes are taken in descending degree order, and after each node is
    added we track the union of edges incident to any removed node so far
    (an edge counts as "covered" if at least one endpoint is removed).
    We stop as soon as that covered-edge count crosses the target share.
    """
    print(f"\nRemoving fewest highest-degree nodes covering {fraction:.0%} of all edges...")

    total_edges = G.number_of_edges()
    target = fraction * total_edges

    degree_sorted = sorted(G.degree(), key=lambda x: x[1], reverse=True)

    covered_edges = set()
    removed = set()

    for node, _ in degree_sorted:
        if len(covered_edges) >= target:
            break
        removed.add(node)
        for nbr in G.neighbors(node):
            covered_edges.add(tuple(sorted((node, nbr))))

    actual_share = len(covered_edges) / total_edges if total_edges else 0.0

    H = G.subgraph([n for n in G.nodes() if n not in removed]).copy()

    print(f"  Total edges:          {total_edges:,}")
    print(f"  Nodes removed:        {len(removed):,}")
    print(f"  Edges covered:        {len(covered_edges):,} ({actual_share:.2%} of all edges)")
    print(f"  Remaining nodes:      {H.number_of_nodes():,}")

    return H, len(removed), actual_share


# ----------------------------------------------------------------------
# Largest connected component
# ----------------------------------------------------------------------

def get_largest_component(G: nx.Graph) -> nx.Graph:
    print("\nExtracting largest connected component...")

    lcc = max(nx.connected_components(G), key=len)
    H = G.subgraph(lcc).copy()

    print(f"  LCC size (order): {H.number_of_nodes():,}")
    return H


# ----------------------------------------------------------------------
# BFS shortest path analysis
# ----------------------------------------------------------------------

def analyze_shortest_paths(G: nx.Graph, label: str) -> dict:
    print(f"\nSHORTEST PATH ANALYSIS: {label}")

    # Order = node count of the LCC (G here IS the LCC). This is what the
    # small-world ratio is computed against, not the original full graph.
    order = G.number_of_nodes()
    n_src = min(N_SOURCES, order)

    sources = random.sample(list(G.nodes()), n_src)

    path_counts = Counter()
    total = 0
    sum_len = 0
    max_observed_dist = 0  # running max shortest-path length -> diameter lower bound

    start = time.time()

    for i, s in enumerate(sources):
        lengths = nx.single_source_shortest_path_length(G, s)

        for t, d in lengths.items():
            if s != t:
                path_counts[d] += 1
                total += 1
                sum_len += d
                if d > max_observed_dist:
                    max_observed_dist = d

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{n_src} sources processed")

    elapsed = time.time() - start

    avg = sum_len / total
    logn = np.log(order)
    ratio = avg / logn

    # ------------------------------------------------------------------
    # Diameter of the LCC
    # ------------------------------------------------------------------
    if EXACT_DIAMETER:
        print("  Computing EXACT diameter via nx.diameter() (can be slow)...")
        diameter = nx.diameter(G)
        diameter_is_exact = True
    else:
        diameter = max_observed_dist
        diameter_is_exact = False

    print("\nRESULTS")
    print(f"  LCC order:      {order:,}")
    print(f"  Avg hop length: {avg:.4f}")
    print(f"  ln(N):          {logn:.4f}")
    print(f"  Ratio:          {ratio:.4f}")
    print(f"  Diameter{'' if diameter_is_exact else ' (approx, lower bound)'}: {diameter}")

    findings = {
        "label": label,
        "lcc_order": order,
        "n_sources_sampled": n_src,
        "avg_hop_length": avg,
        "ln_lcc_order": logn,
        "small_world_ratio": ratio,
        "diameter": diameter,
        "diameter_is_exact": diameter_is_exact,
        "elapsed_seconds": elapsed,
    }

    return findings


# ----------------------------------------------------------------------
# Saving helpers
# ----------------------------------------------------------------------

def save_findings(findings: dict, label: str):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"{label}_findings.json")
    with open(out_path, "w") as f:
        json.dump(findings, f, indent=2)
    print(f"  Saved findings -> {out_path}")


def save_combined_findings(all_findings: list):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "combined_findings.json")
    with open(out_path, "w") as f:
        json.dump(all_findings, f, indent=2)
    print(f"\nSaved combined findings -> {out_path}")


# ----------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------

def main():
    print("=" * 60)
    print("SHORTEST PATH ANALYSIS")
    print("=" * 60)

    random.seed(42)

    all_findings = []

    for file_name in LOG_FILES:
        base_label = file_name.replace(".pkl", "")

        G_full = build_graph_from_file(file_name)

        # ------------------------------------------------------------
        # PART 1: remove a fixed count (TOP_K_REMOVE) of top-degree nodes
        # ------------------------------------------------------------
        label1 = base_label + f"_top{TOP_K_REMOVE}_removed"

        G1 = remove_top_k_hubs(G_full, TOP_K_REMOVE)
        G1 = get_largest_component(G1)

        findings1 = analyze_shortest_paths(G1, label1)
        findings1["removal_method"] = "top_k_count"
        findings1["nodes_removed"] = TOP_K_REMOVE
        save_findings(findings1, label1)
        all_findings.append(findings1)

        del G1
        gc.collect()

        # ------------------------------------------------------------
        # PART 2: remove the fewest highest-degree nodes covering
        # EDGE_SHARE_FRACTION (e.g. 50%) of all edges, instead of a
        # fixed node count
        # ------------------------------------------------------------
        label2 = base_label + f"_top_edge_share_{int(EDGE_SHARE_FRACTION * 100)}pct_removed"

        G2, n_removed_2, actual_share_2 = remove_hubs_by_edge_share(G_full, EDGE_SHARE_FRACTION)
        G2 = get_largest_component(G2)

        findings2 = analyze_shortest_paths(G2, label2)
        findings2["removal_method"] = "top_edge_share"
        findings2["target_edge_share"] = EDGE_SHARE_FRACTION
        findings2["actual_edge_share_covered"] = actual_share_2
        findings2["nodes_removed"] = n_removed_2
        save_findings(findings2, label2)
        all_findings.append(findings2)

        del G2
        gc.collect()

        del G_full
        gc.collect()

    save_combined_findings(all_findings)

    print("\nDONE")


if __name__ == "__main__":
    main()