"""
create_interactive_graph_full.py

Create an interactive HTML visualization of the zkSync transaction network.
Each log file is processed separately, producing one HTML file per dataset.
"""

import pickle
import os
import networkx as nx
import numpy as np
from typing import List, Dict
import gc

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

LOG_DIR = r"D:\zkSync_logs"

LOG_FILES = [
    "logs_69900000_70000000_final.pkl",
    "logs_14900000_15000000_final.pkl",
]

# ----------------------------------------------------------------------
# 1. Load logs
# ----------------------------------------------------------------------

def load_logs(file_path: str) -> List[Dict]:
    """Load logs from a single pickle file"""
    full_path = os.path.join(LOG_DIR, file_path)
    if not os.path.exists(full_path):
        print(f"  File not found: {full_path}")
        return []

    with open(full_path, 'rb') as f:
        data = pickle.load(f)

    logs = data['logs']
    print(f"  Loaded {len(logs):,} logs from {file_path}")
    return logs


# ----------------------------------------------------------------------
# 2. Build graph and get largest component
# ----------------------------------------------------------------------

def get_largest_component(logs: List[Dict]) -> nx.Graph:
    """Build graph from logs and return the largest connected component"""

    print("\n  Building graph...")
    edges = []
    for log in logs:
        sender   = log.get('from')
        receiver = log.get('to')
        if sender and receiver and sender != receiver:
            edges.append((sender, receiver))

    G = nx.Graph()
    G.add_edges_from(edges)
    print(f"  Total nodes: {G.number_of_nodes():,}")
    print(f"  Total edges: {G.number_of_edges():,}")

    print("\n  Finding largest connected component...")
    largest_nodes = max(nx.connected_components(G), key=len)
    print(f"  LCC size: {len(largest_nodes):,} nodes")

    lcc = G.subgraph(largest_nodes).copy()
    del G, edges
    gc.collect()

    return lcc


# ----------------------------------------------------------------------
# 3. Create interactive HTML
# ----------------------------------------------------------------------

def create_html_graph(G: nx.Graph, label: str = ""):
    """Create interactive HTML visualization for a single graph"""

    try:
        from pyvis.network import Network
    except ImportError:
        print("  Installing pyvis...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'pyvis'])
        from pyvis.network import Network

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    print(f"\n  Creating interactive HTML...")
    print(f"  Nodes: {n_nodes:,}  |  Edges: {n_edges:,}")

    net = Network(
        height="1200px",
        width="100%",
        bgcolor="#1a1a2e",
        font_color="white"
    )

    net.set_options("""
    {
        "physics": {
            "enabled": true,
            "stabilization": {
                "iterations": 500,
                "updateInterval": 50
            },
            "forceAtlas2Based": {
                "gravitationalConstant": -300,
                "centralGravity": 0.005,
                "springLength": 300,
                "springConstant": 0.05,
                "damping": 0.4
            },
            "solver": "forceAtlas2Based"
        },
        "nodes": {
            "size": 4,
            "shape": "dot",
            "font": { "size": 0 },
            "borderWidth": 0
        },
        "edges": {
            "color": { "color": "rgba(255,255,255,0.08)" },
            "width": 0.5,
            "smooth": { "enabled": false }
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 100,
            "navigationButtons": true,
            "keyboard": true,
            "zoomView": true,
            "dragView": true
        },
        "layout": { "randomSeed": 42 }
    }
    """)

    # Degree-based coloring and sizing
    print("  Computing degrees...")
    degrees  = dict(G.degree())
    max_deg  = max(degrees.values()) if degrees else 1

    print(f"  Adding {n_nodes:,} nodes...")
    for i, node in enumerate(G.nodes()):
        deg   = degrees[node]
        ratio = deg / max_deg if max_deg > 0 else 0

        # Blue (low degree) → Red (high degree)
        r     = int(30 + 225 * ratio)
        g     = int(30 + 200 * (1 - ratio))
        b     = int(150 + 100 * (1 - ratio))
        color = f"rgb({r},{g},{b})"

        size  = 2 + 10 * (np.log(deg + 1) / np.log(max_deg + 1)) if max_deg > 1 else 4

        net.add_node(
            node,
            label="",
            title=f"Address: {node[:16]}...\nDegree: {deg:,}",
            color=color,
            size=size
        )

        if (i + 1) % 10000 == 0:
            print(f"    {i+1:,} nodes added...")

    print(f"  Adding {n_edges:,} edges...")
    if n_edges > 200000:
        print(f"  WARNING: {n_edges:,} edges is very large — browser may be slow.")

    for i, (u, v) in enumerate(G.edges()):
        net.add_edge(u, v, color="rgba(255,255,255,0.05)", width=0.3)
        if (i + 1) % 50000 == 0:
            print(f"    {i+1:,} edges added...")

    html_path = f"zkSync_graph_{label}.html"
    print(f"\n  Saving to: {html_path}...")
    net.show(html_path)

    print(f"  ✓ Saved: {html_path}")
    print(f"  Nodes: {n_nodes:,}  |  Edges: {n_edges:,}")

    return html_path


# ----------------------------------------------------------------------
# 4. Main — one HTML per file
# ----------------------------------------------------------------------

def main():
    print("=" * 60)
    print("ZKSYNC INTERACTIVE GRAPH  (per-file mode)")
    print("=" * 60)

    for file_name in LOG_FILES:
        label = file_name.replace("_final.pkl", "").replace("logs_", "blocks_")

        print(f"\n{'='*60}")
        print(f"Processing: {file_name}")
        print(f"{'='*60}")

        logs = load_logs(file_name)
        if not logs:
            print(f"  No logs found, skipping.")
            continue

        lcc = get_largest_component(logs)
        del logs
        gc.collect()

        create_html_graph(lcc, label)

        del lcc
        gc.collect()

    print("\n" + "=" * 60)
    print("✓ ALL FILES PROCESSED")
    print("=" * 60)
    print("\nOpen the HTML files in Chrome with extra memory:")
    print('  "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe" --max-old-space-size=8192')


if __name__ == "__main__":
    main()