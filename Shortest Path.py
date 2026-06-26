"""
analyze_shortest_path.py

Analyze shortest paths in the zkSync transaction network.
Replicates Section 3.5 of the paper.

Each log file is analyzed separately.
BFS runs on the FULL graph using N_SOURCES random source nodes.
Path lengths are counted exactly (no reservoir sampling).
"""

import pickle
import os
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
    "logs_69900000_70000000_final.pkl",
    "logs_14900000_15000000_final.pkl",
]

N_SOURCES  = 5000   # BFS source nodes; each reaches ALL nodes in the full graph
CHUNK_SIZE = 50000


# ----------------------------------------------------------------------
# 1. Load logs in chunks (memory efficient)
# ----------------------------------------------------------------------

def load_pickle_in_chunks(file_path: str, chunk_size: int = CHUNK_SIZE) -> Iterator:
    """Load a single pickle file in chunks."""
    full_path = os.path.join(LOG_DIR, file_path)
    if not os.path.exists(full_path):
        print(f"  File not found: {full_path}")
        return

    print(f"  Loading {file_path} in chunks of {chunk_size:,}...")

    try:
        with open(full_path, 'rb') as f:
            data = pickle.load(f)

        if isinstance(data, dict) and 'logs' in data:
            logs = data['logs']
            print(f"  Loaded {len(logs):,} logs")
            for i in range(0, len(logs), chunk_size):
                chunk = logs[i:i + chunk_size]
                print(f"    Chunk {i // chunk_size + 1}: logs {i:,} – {i + len(chunk):,}")
                yield chunk
                del chunk
                gc.collect()
        else:
            print(f"  Unexpected data format: {type(data)}")

    except MemoryError:
        print("  MemoryError: file too large. Pre-process or use more RAM.")
    except Exception as e:
        print(f"  Error loading file: {e}")


# ----------------------------------------------------------------------
# 2. Build graph from a single file
# ----------------------------------------------------------------------

def build_graph_from_file(file_name: str) -> nx.Graph | None:
    """Build an undirected graph from a single log file."""

    print(f"\n{'='*70}")
    print(f"BUILDING GRAPH: {file_name}")
    print(f"{'='*70}")

    edge_set = set()
    total_logs = 0

    for chunk in load_pickle_in_chunks(file_name):
        for log in chunk:
            sender   = log.get('from')
            receiver = log.get('to')
            if sender and receiver and sender != receiver:
                edge_set.add(tuple(sorted((sender, receiver))))
            total_logs += 1

        print(f"  {total_logs:,} logs processed, {len(edge_set):,} unique edges so far...")
        del chunk
        gc.collect()

    print(f"\n  Total logs:   {total_logs:,}")
    print(f"  Unique edges: {len(edge_set):,}")

    if not edge_set:
        print("  No edges found – skipping.")
        return None

    G = nx.Graph()
    G.add_edges_from(edge_set)
    del edge_set
    gc.collect()

    print(f"  Nodes: {G.number_of_nodes():,}  |  Edges: {G.number_of_edges():,}")
    return G


def get_largest_component(G: nx.Graph) -> nx.Graph:
    """Return the largest connected component of G."""
    print("\n  Finding largest connected component...")
    largest_nodes = max(nx.connected_components(G), key=len)
    print(f"  LCC size: {len(largest_nodes):,} nodes")
    return G.subgraph(largest_nodes).copy()


# ----------------------------------------------------------------------
# 3. Shortest path analysis
# ----------------------------------------------------------------------

def analyze_shortest_paths(G: nx.Graph, label: str = ""):
    """
    Analyze shortest paths on the FULL graph.
    - Randomly selects N_SOURCES source nodes
    - Each BFS traverses the entire graph (all nodes are targets)
    - Every path length is counted exactly via a Counter (no sampling loss)
    """

    print(f"\n{'='*70}")
    print(f"SHORTEST PATH ANALYSIS  –  {label}")
    print(f"{'='*70}")

    n_nodes  = G.number_of_nodes()
    n_src    = min(N_SOURCES, n_nodes)

    print(f"  Graph nodes:  {n_nodes:,}")
    print(f"  Graph edges:  {G.number_of_edges():,}")
    print(f"  BFS sources:  {n_src:,}  (each reaches all {n_nodes:,} nodes)")
    print(f"  Est. paths:   ~{n_src * (n_nodes - 1):,}")

    random.seed(42)
    source_nodes = random.sample(list(G.nodes()), n_src)

    print("\n  Computing shortest paths...")
    start_time = time.time()

    # Exact counter — no reservoir sampling, every length recorded
    path_counts  = Counter()
    total_paths  = 0
    sum_lengths  = 0

    for i, source in enumerate(source_nodes):
        try:
            lengths = nx.single_source_shortest_path_length(G, source)
            for target, length in lengths.items():
                if source != target:
                    path_counts[length] += 1
                    total_paths         += 1
                    sum_lengths         += length

        except MemoryError:
            print(f"  MemoryError at source {i}, continuing...")
            gc.collect()
            continue
        except Exception as e:
            print(f"  Warning at source {i}: {e}")
            continue

        if (i + 1) % 50 == 0:
            elapsed     = time.time() - start_time
            avg_so_far  = sum_lengths / total_paths if total_paths > 0 else 0
            print(f"    [{i+1}/{n_src}]  {elapsed:.1f}s  "
                  f"paths: {total_paths:,}  avg: {avg_so_far:.3f}")
            if i % 200 == 0:
                gc.collect()

    elapsed = time.time() - start_time

    if not path_counts:
        print("  No paths found!")
        return

    # Derive all statistics directly from the counter
    min_length        = min(path_counts.keys())
    max_length        = max(path_counts.keys())
    avg_length        = sum_lengths / total_paths
    variance          = sum((k - avg_length)**2 * v for k, v in path_counts.items()) / total_paths
    std_length        = variance ** 0.5
    log_n             = np.log(n_nodes)
    small_world_ratio = avg_length / log_n

    print(f"\n{'='*60}")
    print(f"RESULTS  –  {label}")
    print(f"{'='*60}")
    print(f"  Total nodes (LCC):    {n_nodes:,}")
    print(f"  BFS sources:          {n_src:,}")
    print(f"  Paths measured:       {total_paths:,}")
    print(f"  Average hopcount:     {avg_length:.4f}")
    print(f"  Diameter (sampled):   {max_length}")
    print(f"  Minimum distance:     {min_length}")
    print(f"  Std deviation:        {std_length:.4f}")
    print(f"\n  H̄  (avg path length) = {avg_length:.2f}")
    print(f"  ln(N)                = {log_n:.2f}")
    print(f"  Small-world ratio    = {small_world_ratio:.3f}")

    if avg_length < log_n:
        print("  ✓ Small-world property detected (H̄ < ln(N))")
    else:
        print("  × No small-world property")

    plot_distribution(path_counts, total_paths, avg_length, max_length, log_n, label)
    save_results(avg_length, max_length, n_nodes, small_world_ratio, label, n_src)


# ----------------------------------------------------------------------
# 4. Plot & save
# ----------------------------------------------------------------------

def plot_distribution(path_counts, total_paths, avg_length, diameter, log_n, label):
    """Plot path length distribution and CCDF."""

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Build probability array from counter (every value exactly represented)
    bin_centers = np.arange(1, diameter + 1)
    counts      = np.array([path_counts.get(k, 0) for k in bin_centers], dtype=float)
    counts     /= counts.sum()

    # Trim trailing zeros so x-axis only shows hops that actually exist
    last_nonzero = np.max(np.where(counts > 0))
    bin_centers  = bin_centers[:last_nonzero + 1]
    counts       = counts[:last_nonzero + 1]
    x_max        = bin_centers[-1] + 0.5

    # --- (a) Distribution ---
    ax1.bar(bin_centers, counts, width=0.8,
            color='#2E86AB', alpha=0.7, edgecolor='white', linewidth=1.5,
            label='Observed')
    ax1.axvline(avg_length, color='#D62828', linestyle='--',
                linewidth=2, alpha=0.8, label=f'Mean = {avg_length:.2f}')
    ax1.set_xlabel('Shortest Path Length (hops)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Probability',                 fontsize=12, fontweight='bold')
    ax1.set_title('(a) Path Length Distribution', fontsize=13, fontweight='bold')
    ax1.legend(frameon=True, facecolor='white', edgecolor='gray', fontsize=10)
    ax1.set_xlim(0.5, x_max)
    ax1.set_xticks(bin_centers)

    # --- (b) CCDF ---
    # Step drops at x + 0.5 (right edge of each bar) by shifting centers by 0.5
    ccdf = 1 - np.cumsum(counts)
    ax2.step(bin_centers + 0.5, ccdf, where='post',
             color='#A23B72', linewidth=2.5, label='CCDF')

    x_rand = np.linspace(1, bin_centers[-1], 100)
    y_rand = np.exp(-x_rand / avg_length)
    ax2.plot(x_rand, y_rand, 'g--', linewidth=2,
             label='Random expectation', alpha=0.7)

    ax2.set_xlabel('Shortest Path Length (hops)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('P(Length ≥ L)',               fontsize=12, fontweight='bold')
    ax2.set_title('(b) Complementary CDF',        fontsize=13, fontweight='bold')
    x_start = bin_centers[0] - 0.5
    ax2.set_xlim(x_start, x_max)
    ax2.set_xticks(bin_centers)
    ax2.grid(True, alpha=0.2, which='both')

    # Legend bottom-left
    ax2.legend(frameon=True, facecolor='white', edgecolor='gray',
               fontsize=10, loc="lower right")

    # Stats box bottom-left
    stats_text = (f"Small-World Check:\n"
                  f"H̄ = {avg_length:.2f}\n"
                  f"ln(N) = {log_n:.2f}\n"
                  f"Ratio = {avg_length/log_n:.3f}")
    bbox_props = dict(boxstyle="round,pad=0.5", facecolor='white', edgecolor='gray', alpha=0.9)
    ax2.text(0.02, 0.02, stats_text, transform=ax2.transAxes,
             verticalalignment='bottom', bbox=bbox_props, family='monospace', fontsize=10)

    fig.suptitle(label, fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout(pad=1.5)

    save_name = f"shortest_paths_{label}.png"
    plt.savefig(save_name, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n  ✓ Plot saved: {save_name}")
    plt.close()


def save_results(avg_length, diameter, n_nodes, small_world_ratio, label, num_sources):
    """Save analysis results to a text file."""

    filename = f"path_results_{label}.txt"
    with open(filename, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write(f"SHORTEST PATH ANALYSIS RESULTS – {label}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total nodes:               {n_nodes:,}\n")
        f.write(f"BFS sources:               {num_sources:,}\n")
        f.write(f"Average hopcount:          {avg_length:.4f}\n")
        f.write(f"Diameter (sampled):        {diameter}\n")
        f.write(f"Small-world ratio H/ln(N): {small_world_ratio:.4f}\n")
        f.write(f"Small-world property:      {'YES' if avg_length < np.log(n_nodes) else 'NO'}\n")

    print(f"  ✓ Results saved: {filename}")


# ----------------------------------------------------------------------
# 5. Main – one analysis per file
# ----------------------------------------------------------------------

def main():
    print("=" * 70)
    print("ZKSYNC SHORTEST PATH ANALYSIS  (per-file mode)")
    print("=" * 70)
    print(f"BFS sources  : {N_SOURCES:,} per file (full graph traversal)")
    print(f"Chunk size   : {CHUNK_SIZE:,}")
    print(f"Files        : {len(LOG_FILES)}")
    print("=" * 70)

    for file_name in LOG_FILES:
        label = file_name.replace("_final.pkl", "").replace("logs_", "blocks_")

        G = build_graph_from_file(file_name)
        if G is None:
            print(f"  Skipping {file_name} (no graph built).\n")
            continue

        lcc = get_largest_component(G)
        del G
        gc.collect()

        analyze_shortest_paths(lcc, label)

        del lcc
        gc.collect()

    print("\n" + "=" * 70)
    print("✓ ALL FILES PROCESSED")
    print("=" * 70)


if __name__ == "__main__":
    main()