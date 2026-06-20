"""
analyze_zkSync_communities.py

Analyze downloaded zkSync logs to find connected communities and create visualizations.
Each community gets its own separate graph plot.
"""

import pickle
import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple
import time
from scipy import stats

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

# Path to downloaded logs
LOG_DIR = "zkSync_logs"

# Files to process
LOG_FILES = [
    "logs_10M_20M.pkl",
    "logs_60M_70M.pkl",
]

# Whether to combine both ranges or process separately
COMBINE_RANGES = True

# For plotting
FIGURE_SIZE = (10, 8)
NODE_SIZE = 50
EDGE_WIDTH = 0.5

# Maximum community size to plot individually (skip giant communities)
MAX_COMMUNITY_SIZE_TO_PLOT = 1000
MIN_COMMUNITY_SIZE_TO_PLOT = 3  # Skip tiny communities


# ----------------------------------------------------------------------
# 1. Load logs from files
# ----------------------------------------------------------------------

def load_logs(file_path: str) -> List[Dict]:
    """Load logs from a pickle file"""
    full_path = os.path.join(LOG_DIR, file_path)
    if not os.path.exists(full_path):
        print(f"File {full_path} not found!")
        return []

    with open(full_path, 'rb') as f:
        data = pickle.load(f)

    print(f"Loaded {data['num_logs']:,} logs from {file_path}")
    print(f"  Range: {data['block_range'][0]:,} - {data['block_range'][1]:,}")
    return data['logs']


def load_all_logs() -> Dict[str, List[Dict]]:
    """Load all available log files"""
    all_logs = {}

    for file_name in LOG_FILES:
        logs = load_logs(file_name)
        if logs:
            all_logs[file_name] = logs

    return all_logs


# ----------------------------------------------------------------------
# 2. Extract edges (sender -> receiver pairs)
# ----------------------------------------------------------------------

def extract_edges(logs: List[Dict]) -> List[Tuple[str, str]]:
    """Extract edges from logs (sender -> receiver), skipping self-transfers"""
    edges = []

    for log in logs:
        sender = log.get('from')
        receiver = log.get('to')

        if sender and receiver and sender != receiver:
            edges.append((sender, receiver))

    return edges


# ----------------------------------------------------------------------
# 3. Build graph
# ----------------------------------------------------------------------

def build_graph(edges: List[Tuple[str, str]]) -> nx.Graph:
    """Build undirected graph from edges"""
    G = nx.Graph()
    G.add_edges_from(edges)
    print(f"Graph built: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    return G


# ----------------------------------------------------------------------
# 4. Find connected communities (BFS)
# ----------------------------------------------------------------------

def find_communities_bfs(G: nx.Graph) -> List[List]:
    """
    Find connected components using BFS.
    """
    list0 = list(G.nodes())
    communities = []

    print(f"\nFinding connected communities...")
    print(f"Starting with {len(list0):,} nodes")

    step = 0
    while list0:
        step += 1
        start_node = list0.pop(0)
        community = [start_node]
        frontier = [start_node]

        while frontier:
            next_frontier = []
            for node in frontier:
                for neighbor in G.neighbors(node):
                    if neighbor in list0:
                        list0.remove(neighbor)
                        community.append(neighbor)
                        next_frontier.append(neighbor)
            frontier = next_frontier

        communities.append(community)

        if step % 1000 == 0:
            print(f"  Found {step} communities, {len(list0):,} nodes remaining")

    print(f"Found {len(communities)} communities")
    return communities


# ----------------------------------------------------------------------
# 5. Print community information
# ----------------------------------------------------------------------

def print_communities(communities: List[List], max_print=20):
    """Print community sizes"""
    print(f"\n{'='*70}")
    print(f"COMMUNITY SUMMARY")
    print(f"{'='*70}")

    sorted_communities = sorted(communities, key=len, reverse=True)

    for i, comm in enumerate(sorted_communities[:max_print]):
        print(f"Community {i+1}: size = {len(comm):,}")

    if len(communities) > max_print:
        print(f"\n... and {len(communities) - max_print} more communities")

    # Statistics
    sizes = [len(c) for c in communities]
    print(f"\n{'='*70}")
    print(f"COMMUNITY STATISTICS")
    print(f"{'='*70}")
    print(f"Total communities: {len(communities):,}")
    print(f"Largest community: {max(sizes):,} nodes")
    print(f"Smallest community: {min(sizes):,} nodes")
    print(f"Average size: {np.mean(sizes):.2f}")
    print(f"Median size: {np.median(sizes):.2f}")

    sorted_sizes = sorted(sizes, reverse=True)
    print(f"Top 5 community sizes: {sorted_sizes[:5]}")


# ----------------------------------------------------------------------
# 6. Plot each community as a separate graph
# ----------------------------------------------------------------------

def plot_individual_communities(G: nx.Graph, communities: List[List],
                                label: str = "",
                                output_dir: str = "community_graphs"):
    """
    Plot each community as a separate graph.
    Creates a folder with individual PNG files for each community.
    """

    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"\nPlotting individual communities...")
    print(f"  Output directory: {output_dir}/")

    # Sort communities by size (largest first)
    sorted_communities = sorted(communities, key=len, reverse=True)

    # Track which communities we plot
    plotted_count = 0
    skipped_small = 0
    skipped_large = 0

    for i, comm in enumerate(sorted_communities):
        community_size = len(comm)

        # Skip giant communities (too many nodes to visualize)
        if community_size > MAX_COMMUNITY_SIZE_TO_PLOT:
            skipped_large += 1
            continue

        # Skip tiny communities (too small to be interesting)
        if community_size < MIN_COMMUNITY_SIZE_TO_PLOT:
            skipped_small += 1
            continue

        # Create subgraph for this community
        subgraph = G.subgraph(comm)

        # Skip if no edges (shouldn't happen with connected components)
        if subgraph.number_of_edges() == 0:
            continue

        # Create plot
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)

        # Layout for this subgraph
        if len(comm) > 100:
            # Use faster layout for larger communities
            pos = nx.spring_layout(subgraph, k=0.3, iterations=20)
        else:
            pos = nx.spring_layout(subgraph, k=0.5, iterations=50)

        # Draw
        nx.draw_networkx_nodes(subgraph, pos,
                              node_size=NODE_SIZE,
                              node_color='lightblue',
                              edgecolors='darkblue',
                              linewidths=0.5,
                              ax=ax)

        nx.draw_networkx_edges(subgraph, pos,
                              alpha=0.5,
                              width=EDGE_WIDTH,
                              ax=ax)

        # Add labels for small communities
        if len(comm) <= 20:
            labels = {node: i+1 for i, node in enumerate(comm)}
            nx.draw_networkx_labels(subgraph, pos, labels, font_size=8, ax=ax)

        # Title
        ax.set_title(f"Community {i+1}: {len(comm)} nodes, {subgraph.number_of_edges()} edges",
                    fontsize=12)
        ax.axis('off')

        # Save
        filename = f"community_{i+1:05d}_size_{len(comm)}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)

        plotted_count += 1

        # Progress indicator
        if plotted_count % 100 == 0:
            print(f"  Plotted {plotted_count} communities...")

    print(f"\n{'='*70}")
    print(f"PLOTTING SUMMARY")
    print(f"{'='*70}")
    print(f"Total communities: {len(communities)}")
    print(f"Plotted: {plotted_count}")
    print(f"Skipped (too large > {MAX_COMMUNITY_SIZE_TO_PLOT}): {skipped_large}")
    print(f"Skipped (too small < {MIN_COMMUNITY_SIZE_TO_PLOT}): {skipped_small}")
    print(f"Plots saved in: {output_dir}/")


# ----------------------------------------------------------------------
# 7. Component size distribution (like Figure 4 in the paper)
# ----------------------------------------------------------------------

def plot_component_size_distribution(communities: List[List],
                                    save_path: str = "component_distribution.png"):
    """Plot component size distribution (log-log) with power-law fit"""

    sizes = np.array([len(c) for c in communities])

    # Calculate histogram
    unique_sizes, counts = np.unique(sizes, return_counts=True)
    probs = counts / counts.sum()

    # Filter valid values
    mask = (unique_sizes > 0) & (probs > 0)
    log_x = np.log10(unique_sizes[mask])
    log_y = np.log10(probs[mask])

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Distribution P(s) (like Figure 4 left)
    ax1 = axes[0]
    ax1.scatter(log_x, log_y, color='red', s=30, label='Data', alpha=0.6)

    # Power law fit (on middle range, excluding heavy tail)
    if len(log_x) > 3:
        # Use middle 60% of the data
        sorted_indices = np.argsort(log_x)
        n_points = len(log_x)
        start_idx = int(n_points * 0.1)
        end_idx = int(n_points * 0.8)

        if end_idx > start_idx + 2:
            middle_indices = sorted_indices[start_idx:end_idx]
            log_x_mid = log_x[middle_indices]
            log_y_mid = log_y[middle_indices]

            slope, intercept, r_value, p_value, _ = stats.linregress(log_x_mid, log_y_mid)
            alpha = -slope

            x_fit = np.linspace(log_x_mid.min(), log_x_mid.max(), 100)
            y_fit = slope * x_fit + intercept

            ax1.plot(x_fit, y_fit, 'b--', linewidth=2,
                    label=f'Power-law fit, α={alpha:.3f}')

            ax1.text(0.05, 0.95, f'α = {alpha:.3f}\nR² = {r_value**2:.3f}',
                    transform=ax1.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax1.set_xlabel('log10(component size)', fontsize=12)
    ax1.set_ylabel('log10(P(s))', fontsize=12)
    ax1.set_title('Component Size Distribution', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Complementary Cumulative Distribution (like Figure 4 right)
    ax2 = axes[1]

    # Calculate CCDF
    sorted_sizes = np.sort(sizes)[::-1]
    ccdf = np.arange(1, len(sorted_sizes) + 1) / len(sorted_sizes)
    log_sorted = np.log10(sorted_sizes)
    log_ccdf = np.log10(ccdf)

    ax2.scatter(log_sorted, log_ccdf, color='red', s=30, alpha=0.6, label='CCDF')

    # Power law fit for CCDF
    if len(log_sorted) > 3:
        tail_size = int(len(log_sorted) * 0.8)
        log_sorted_fit = log_sorted[:tail_size]
        log_ccdf_fit = log_ccdf[:tail_size]

        if len(log_sorted_fit) > 2:
            slope, intercept, r_value, _, _ = stats.linregress(log_sorted_fit, log_ccdf_fit)
            alpha_ccdf = -slope + 1

            x_fit = np.linspace(log_sorted_fit.min(), log_sorted_fit.max(), 100)
            y_fit = slope * x_fit + intercept

            ax2.plot(x_fit, y_fit, 'b--', linewidth=2,
                    label=f'Power-law fit, α={alpha_ccdf:.3f}')

            ax2.text(0.05, 0.95, f'α = {alpha_ccdf:.3f}\nR² = {r_value**2:.3f}',
                    transform=ax2.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax2.set_xlabel('log10(component size)', fontsize=12)
    ax2.set_ylabel('log10(P(s) > s)', fontsize=12)
    ax2.set_title('Complementary Cumulative Distribution', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Distribution plot saved to {save_path}")
    plt.close()


# ----------------------------------------------------------------------
# 8. Main processing function
# ----------------------------------------------------------------------

def process_logs(logs: List[Dict], label: str = ""):
    """Process a set of logs"""
    print(f"\n{'='*70}")
    print(f"PROCESSING {label}")
    print(f"{'='*70}")

    start_time = time.time()

    # Extract edges
    print("\nExtracting edges from logs...")
    edges = extract_edges(logs)
    print(f"Extracted {len(edges):,} transaction edges")

    # Build graph
    print("\nBuilding graph...")
    G = build_graph(edges)

    # Find communities
    communities = find_communities_bfs(G)

    # Print communities
    print_communities(communities)

    # Plot each community individually
    output_dir = f"community_graphs_{label}" if label else "community_graphs"
    plot_individual_communities(G, communities, label, output_dir)

    # Plot component size distribution
    plot_component_size_distribution(communities,
                                    save_path=f"component_distribution_{label}.png")

    elapsed = time.time() - start_time
    print(f"\nProcessing complete in {elapsed:.1f} seconds")


# ----------------------------------------------------------------------
# 9. Main
# ----------------------------------------------------------------------

def main():
    print("="*70)
    print("ZKSYNC TRANSACTION GRAPH ANALYSIS")
    print("="*70)
    print(f"Log directory: {LOG_DIR}")
    print(f"Files to process: {LOG_FILES}")
    print(f"Combine ranges: {COMBINE_RANGES}")
    print("="*70)

    # Load all logs
    all_logs = load_all_logs()

    if not all_logs:
        print("No logs found! Please run download_zkSync_logs.py first.")
        return

    if COMBINE_RANGES:
        # Combine all logs
        combined_logs = []
        for file_name, logs in all_logs.items():
            combined_logs.extend(logs)

        print(f"\nCombined {len(combined_logs):,} logs from {len(all_logs)} files")
        process_logs(combined_logs, "combined")
    else:
        # Process each range separately
        for file_name, logs in all_logs.items():
            label = file_name.replace('.pkl', '')
            process_logs(logs, label)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()