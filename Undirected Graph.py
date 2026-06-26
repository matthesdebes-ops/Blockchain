"""
create_interactive_graph_full.py

Create an interactive HTML visualization of the zkSync transaction network.
Each log file is processed separately, producing one HTML file per dataset.
Sampling ensures a maximum of 10,000 nodes for performance.
"""

import pickle
import os
import networkx as nx
import numpy as np
from typing import List, Dict
import gc
import random
from pyvis.network import Network

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

LOG_DIR = r"D:\zkSync_logs"

LOG_FILES = [
    "logs_69900000_70000000_final.pkl",
    "logs_14900000_15000000_final.pkl",
]

MAX_NODES = 10000  # Maximum nodes to display in the interactive graph

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
# 2. Build graph and sample nodes
# ----------------------------------------------------------------------

def build_sampled_graph(logs: List[Dict], max_nodes: int = 10000) -> nx.Graph:
    """Build graph from logs, sample to max_nodes, and return largest component"""

    print("\n  Building graph...")
    edges = []
    for log in logs:
        sender   = log.get('from')
        receiver = log.get('to')
        if sender and receiver and sender != receiver:
            edges.append((sender, receiver))

    # Count edges to build degree distribution
    G = nx.Graph()
    G.add_edges_from(edges)
    print(f"  Total nodes in full graph: {G.number_of_nodes():,}")
    print(f"  Total edges in full graph: {G.number_of_edges():,}")

    # If graph is already small enough, just find LCC
    if G.number_of_nodes() <= max_nodes:
        print("\n  Finding largest connected component...")
        largest_nodes = max(nx.connected_components(G), key=len)
        print(f"  LCC size: {len(largest_nodes):,} nodes")
        lcc = G.subgraph(largest_nodes).copy()
        del G, edges
        gc.collect()
        return lcc

    # Sample nodes based on degree
    print(f"\n  Sampling {max_nodes:,} nodes from full graph...")

    # Get degree for all nodes
    degrees = dict(G.degree())

    # Sort nodes by degree (high to low)
    sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)

    # Strategy: Take top high-degree nodes + random sample from rest
    high_degree_count = min(max_nodes // 2, len(sorted_nodes))
    high_degree_nodes = [node for node, deg in sorted_nodes[:high_degree_count]]

    # Remaining nodes to sample
    remaining_quota = max_nodes - len(high_degree_nodes)
    remaining_nodes = [node for node, deg in sorted_nodes[high_degree_count:]]

    if remaining_nodes and remaining_quota > 0:
        sample_size = min(remaining_quota, len(remaining_nodes))
        sampled_remaining = random.sample(remaining_nodes, sample_size)
    else:
        sampled_remaining = []

    # Combine selected nodes
    selected_nodes = set(high_degree_nodes + sampled_remaining)
    print(f"  Selected {len(selected_nodes):,} nodes for sampling")

    # Create subgraph with selected nodes
    sampled_G = G.subgraph(selected_nodes).copy()

    # If still too many, take top degree nodes
    if sampled_G.number_of_nodes() > max_nodes:
        print(f"  Still over limit, taking top {max_nodes:,} degree nodes...")
        top_nodes = sorted(sampled_G.degree(), key=lambda x: x[1], reverse=True)[:max_nodes]
        sampled_G = G.subgraph([node for node, deg in top_nodes]).copy()

    print(f"  Sampled graph nodes: {sampled_G.number_of_nodes():,}")
    print(f"  Sampled graph edges: {sampled_G.number_of_edges():,}")

    # Now find largest connected component in sampled graph
    print("\n  Finding largest connected component in sampled graph...")
    if sampled_G.number_of_nodes() > 0:
        largest_nodes = max(nx.connected_components(sampled_G), key=len)
        print(f"  LCC size: {len(largest_nodes):,} nodes")
        lcc = sampled_G.subgraph(largest_nodes).copy()
    else:
        lcc = sampled_G

    del G, sampled_G, edges
    gc.collect()

    return lcc


# ----------------------------------------------------------------------
# 3. Create interactive HTML
# ----------------------------------------------------------------------

def create_html_graph(G: nx.Graph, label: str = ""):
    """Create interactive HTML visualization for a single graph"""

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    print(f"\n  Creating interactive HTML...")
    print(f"  Nodes: {n_nodes:,}  |  Edges: {n_edges:,}")

    if n_nodes == 0:
        print("  WARNING: Graph is empty, skipping HTML generation")
        return None

    net = Network(
        height="1200px",
        width="100%",
        bgcolor="#1a1a2e",
        font_color="white"
    )

    # Adaptive physics settings based on graph size
    if n_nodes > 5000:
        physics_enabled = True
        iterations = 300
        spring_length = 250
        gravitational_constant = -200
    elif n_nodes > 2000:
        physics_enabled = True
        iterations = 400
        spring_length = 200
        gravitational_constant = -250
    else:
        physics_enabled = True
        iterations = 500
        spring_length = 150
        gravitational_constant = -300

    net.set_options(f"""
    {{
        "physics": {{
            "enabled": {str(physics_enabled).lower()},
            "stabilization": {{
                "iterations": {iterations},
                "updateInterval": 50
            }},
            "forceAtlas2Based": {{
                "gravitationalConstant": {gravitational_constant},
                "centralGravity": 0.005,
                "springLength": {spring_length},
                "springConstant": 0.05,
                "damping": 0.4
            }},
            "solver": "forceAtlas2Based"
        }},
        "nodes": {{
            "size": 4,
            "shape": "dot",
            "font": {{ "size": 0 }},
            "borderWidth": 0
        }},
        "edges": {{
            "color": {{ "color": "rgba(255,255,255,0.08)" }},
            "width": 0.5,
            "smooth": {{ "enabled": false }}
        }},
        "interaction": {{
            "hover": true,
            "tooltipDelay": 100,
            "navigationButtons": true,
            "keyboard": true,
            "zoomView": true,
            "dragView": true
        }},
        "layout": {{ "randomSeed": 42 }}
    }}
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
    edge_threshold = 10000  # Show all edges if under this threshold
    if n_edges > edge_threshold:
        print(f"  Note: {n_edges:,} edges - rendering may be slow")

    # Add edges with progress indicator
    for i, (u, v) in enumerate(G.edges()):
        # Edge transparency based on count (if too many, make them more transparent)
        if n_edges > 50000:
            alpha = 0.03
        elif n_edges > 20000:
            alpha = 0.05
        else:
            alpha = 0.08

        net.add_edge(u, v, color=f"rgba(255,255,255,{alpha})", width=0.3)
        if (i + 1) % 50000 == 0:
            print(f"    {i+1:,} edges added...")

    # Create output filename with node count
    html_path = f"zkSync_graph_{label}_{n_nodes}_nodes.html"
    print(f"\n  Saving to: {html_path}...")
    net.write_html(html_path)

    print(f"  ✓ Saved: {html_path}")
    print(f"  Nodes: {n_nodes:,}  |  Edges: {n_edges:,}")
    print(f"  File size: {os.path.getsize(html_path) / (1024*1024):.2f} MB")

    return html_path


# ----------------------------------------------------------------------
# 4. Main — one HTML per file
# ----------------------------------------------------------------------

def main():
    # Set random seed for reproducibility
    random.seed(42)

    print("=" * 60)
    print("ZKSYNC INTERACTIVE GRAPH  (per-file mode with sampling)")
    print("=" * 60)
    print(f"\nMax nodes per graph: {MAX_NODES:,}")

    for file_name in LOG_FILES:
        label = file_name.replace("_final.pkl", "").replace("logs_", "blocks_")

        print(f"\n{'='*60}")
        print(f"Processing: {file_name}")
        print(f"{'='*60}")

        logs = load_logs(file_name)
        if not logs:
            print(f"  No logs found, skipping.")
            continue

        lcc = build_sampled_graph(logs, MAX_NODES)
        del logs
        gc.collect()

        if lcc and lcc.number_of_nodes() > 0:
            create_html_graph(lcc, label)
        else:
            print("  Graph is empty, skipping HTML generation")

        del lcc
        gc.collect()

    print("\n" + "=" * 60)
    print("✓ ALL FILES PROCESSED")
    print("=" * 60)
    print("\nOpen the HTML files in Chrome with extra memory:")
    print('  "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe" --max-old-space-size=8192')
    print("\nAlternatively, use Firefox for better large graph performance.")


if __name__ == "__main__":
    main()