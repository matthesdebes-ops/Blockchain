import pickle
import os
import networkx as nx
import numpy as np
from typing import List, Dict
import gc
import random
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import linregress

LOG_DIR = r"D:\zkSync_logs"

LOG_FILES = [
    "logs_69900000_70000000_final.pkl",
    "logs_14900000_15000000_final.pkl",
    "eth_transfers_69900000_70000000.pkl",
    "eth_transfers_14900000_15000000.pkl",
]

# Number of highest-degree nodes to remove for the "trimmed" analysis.
TOP_DEGREE_TRIM_N = 5

# For the power-law fitting: use more points, only exclude extreme tail
MIN_POINTS_IN_FIT = 10  # Minimum points to use for fitting
MAX_FIT_FRACTION = 0.85  # Use up to 85% of points (exclude the very end)
MIN_TAIL_POINTS = 1  # Ensure at least this many points are excluded as "tail"


def load_logs(file_path: str) -> List[Dict]:
    full_path = os.path.join(LOG_DIR, file_path)

    with open(full_path, 'rb') as f:
        data = pickle.load(f)

    logs = data['logs']
    print(f"  Loaded {len(logs):,} logs from {file_path}")
    return logs


def build_graph(logs: List[Dict]) -> nx.Graph:
    edges = []
    for log in logs:
        s = log.get("from")
        t = log.get("to")
        if s and t and s != t:
            edges.append((s, t))

    G = nx.Graph()
    G.add_edges_from(edges)

    print(f"  Nodes: {G.number_of_nodes():,}, Edges: {G.number_of_edges():,}")
    return G


def remove_top_degree_nodes(G: nx.Graph, n: int = TOP_DEGREE_TRIM_N) -> nx.Graph:
    """
    Return a copy of G with the n highest-degree nodes removed.
    """
    if G.number_of_nodes() == 0:
        return G.copy()

    degrees = dict(G.degree())
    top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:n]

    print(f"  Removing top {len(top_nodes)} highest-degree nodes:")
    for node in top_nodes:
        print(f"    {node}  (degree={degrees[node]:,})")

    G_trimmed = G.copy()
    G_trimmed.remove_nodes_from(top_nodes)

    print(f"  After trim -> Nodes: {G_trimmed.number_of_nodes():,}, "
          f"Edges: {G_trimmed.number_of_edges():,}")

    return G_trimmed


def fit_power_law_bulk(x: np.ndarray, y: np.ndarray):
    """
    Fit log(y) = intercept + slope * log(x) on the bulk of the distribution,
    excluding only the extreme tail points (last 15% or last 3 points,
    whichever is larger).

    Returns (slope, intercept, fit_indices) or (None, None, None).
    """
    if len(x) < MIN_POINTS_IN_FIT:
        print(f"  Skipping power-law fit: only {len(x)} points "
              f"(need >= {MIN_POINTS_IN_FIT}).")
        return None, None, None

    # Use most of the data, but exclude the extreme tail
    # Keep at least MIN_POINTS_IN_FIT points and exclude at least MIN_TAIL_POINTS
    n_fit = min(len(x) - MIN_TAIL_POINTS, int(len(x) * MAX_FIT_FRACTION))
    n_fit = max(n_fit, MIN_POINTS_IN_FIT)

    # Use the first n_fit points (small to medium component sizes)
    bx, by = x[:n_fit], y[:n_fit]

    # Remove any zero or negative values
    valid_mask = (bx > 0) & (by > 0)
    bx, by = bx[valid_mask], by[valid_mask]

    if len(bx) < 2:
        print("  Skipping power-law fit: not enough valid positive points.")
        return None, None, None

    try:
        slope, intercept, r, p, err = linregress(np.log(bx), np.log(by))

        tail_points = len(x) - len(bx)
        fit_fraction = len(bx) / len(x) * 100

        print(f"  power-law fit: slope={slope:.3f}, intercept={intercept:.3f}, "
              f"r^2={r ** 2:.3f}, n_points={len(bx)} ({fit_fraction:.0f}% of data)")
        return slope, intercept, list(range(len(bx)))
    except:
        print("  Power-law fit failed.")
        return None, None, None


def analyze_connected_components(G: nx.Graph, label: str):
    components = list(nx.connected_components(G))
    sizes = np.array([len(c) for c in components])

    if len(sizes) == 0:
        print(f"  No components to plot for {label}")
        return

    # Exact counts for every distinct component size
    vals, counts = np.unique(sizes, return_counts=True)
    x, y = vals.astype(float), counts.astype(float)

    # Power-law fit on the bulk (excluding heavy tail)
    slope, intercept, fit_indices = fit_power_law_bulk(x, y)

    # ---- PLOT ----
    plt.figure(figsize=(7, 5))

    plt.loglog(x, y, 'o', color='blue', label="CC distribution", markersize=4)

    if slope is not None and fit_indices is not None:
        # Plot the fit line over the range of the fitted points
        x_fit_min = x[fit_indices[0]]
        x_fit_max = x[fit_indices[-1]]
        x_fit = np.linspace(x_fit_min, x_fit_max, 100)
        y_fit = np.exp(intercept) * x_fit ** slope
        plt.loglog(x_fit, y_fit, '-', color='red', linewidth=2,
                   label=f"Power-law fit (slope={slope:.2f})")

    plt.title(f"Connected Component Distribution - {label}")
    plt.xlabel("Component size")
    plt.ylabel("Number of components")
    plt.legend()

    # Multiplicative padding for log scale
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    plt.xlim(x_min * 0.8, x_max * 1.2)
    plt.ylim(y_min * 0.8, y_max * 1.2)

    ax = plt.gca()
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    out = f"cc_distribution_{label}.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"  saved plot: {out}")


def main():
    random.seed(42)

    for file_name in LOG_FILES:
        label = file_name.replace(".pkl", "")

        logs = load_logs(file_name)
        G = build_graph(logs)

        del logs
        gc.collect()

        # ---- Original (full) graph analysis ----
        print(f"\n[{label}] Full graph:")
        analyze_connected_components(G, label)

        # ---- Trimmed graph: remove top-N highest-degree nodes ----
        print(f"\n[{label}] Trimmed graph (top {TOP_DEGREE_TRIM_N} degree nodes removed):")
        G_trimmed = remove_top_degree_nodes(G, TOP_DEGREE_TRIM_N)
        analyze_connected_components(G_trimmed, f"{label}_trimmed_top{TOP_DEGREE_TRIM_N}")

        del G_trimmed
        del G
        gc.collect()


if __name__ == "__main__":
    main()