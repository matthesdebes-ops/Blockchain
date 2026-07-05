import pickle
import os
import networkx as nx
import numpy as np
from typing import List, Dict
import gc
import random
import matplotlib.pyplot as plt
from scipy.stats import linregress

LOG_DIR = r"D:\zkSync_logs"

LOG_FILES = [
    #"logs_69900000_70000000_final.pkl",
    #"logs_14900000_15000000_final.pkl",
    "eth_transfers_69900000_70000000.pkl",
    "eth_transfers_14900000_15000000.pkl",
]


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


def analyze_connected_components(G: nx.Graph, label: str):

    components = list(nx.connected_components(G))
    sizes = np.array([len(c) for c in components])

    if len(sizes) == 0:
        return

    sizes = np.sort(sizes)

    # log bins over dataset only
    bins = np.logspace(np.log10(max(1, sizes.min())), np.log10(sizes.max()), 30)
    hist, edges = np.histogram(sizes, bins=bins)

    x = np.sqrt(edges[:-1] * edges[1:])
    y = hist

    mask = y > 0
    x, y = x[mask], y[mask]

    # ---- bulk only fit (ignore tails entirely) ----
    q1, q2 = np.percentile(sizes, [10, 90])
    bulk = sizes[(sizes >= q1) & (sizes <= q2)]

    slope = None
    intercept = None

    if len(bulk) > 5:
        bh, be = np.histogram(bulk, bins=12)
        bx = np.sqrt(be[:-1] * be[1:])
        by = bh

        m = by > 0
        bx, by = bx[m], by[m]

        slope, intercept, r, p, err = linregress(np.log(bx), np.log(by))
        print(f"  power-law (bulk only): slope={slope:.3f}, R2={r*r:.3f}")

    # ---- SINGLE PLOT PER DATASET (DATA ONLY) ----
    plt.figure(figsize=(7, 5))

    # empirical distribution ONLY
    plt.loglog(x, y, 'o', label="CC distribution (data)")

    plt.title(f"Connected Component Distribution - {label}")
    plt.xlabel("Component size")
    plt.ylabel("Number of components")

    plt.figure(figsize=(7, 5))

    plt.loglog(x, y, 'o', label="CC distribution")

    plt.title(f"Connected Component Distribution - {label}")
    plt.xlabel("Component size")
    plt.ylabel("Number of components")

    # --- POWER LAW (unchanged, optional) ---
    if slope is not None:
        x_fit = np.linspace(x.min(), x.max(), 200)
        y_fit = np.exp(intercept) * x_fit ** slope
        plt.loglog(x_fit, y_fit, '-', alpha=0.5, label="bulk fit")

    plt.legend()

    # ----------------------------
    # CONTROLLED AXIS PADDING (log-safe)
    # ----------------------------

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    # multiplicative padding is correct for log scale
    plt.xlim(x_min * 0.8, x_max * 1.2)
    plt.ylim(y_min * 0.8, y_max * 1.2)

    out = f"cc_distribution_{label}.png"
    import matplotlib.ticker as mticker

    ax = plt.gca()

    # force plain numbers instead of scientific notation
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())

    # force integer-style ticks (rounded, not 10^x labels)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
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

        analyze_connected_components(G, label)

        del G
        gc.collect()


if __name__ == "__main__":
    main()