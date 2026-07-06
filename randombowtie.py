import networkx as nx
import numpy as np

from Directed_Graph import (
    bowtie_decomposition,
    plot_bowtie_summary
)

# -------------------------
# Graph generator
# -------------------------
def make_directed_graph(N=50_000, gamma=2.2, seed=41):
    rng = np.random.default_rng(seed)

    deg = rng.zipf(gamma, N)
    deg = np.clip(deg, 1, N - 1)

    if deg.sum() % 2:
        deg[0] += 1

    G = nx.configuration_model(deg, seed=seed)
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))

    D = nx.DiGraph()
    D.add_nodes_from(G.nodes())

    for u, v in G.edges():
        if rng.random() < 0.5:
            D.add_edge(u, v)
        else:
            D.add_edge(v, u)

    return D

# -------------------------
# EXPERIMENT 1
# -------------------------
print("\n=== Graph 1 ===")
G1 = make_directed_graph()
print("r =", nx.degree_assortativity_coefficient(G1))

plot_bowtie_summary(G1, label="unconstrained")
