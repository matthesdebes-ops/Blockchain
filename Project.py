"""
zksync_pipeline.py

Full pipeline:
  1. Download 5000 zkSync Era blocks starting at block 50000000
     (concurrency=10, all tasks fired at once)
  2. Build a weighted address-interaction graph
  3. Run weighted label-propagation clustering
  4. Plot the 10 largest clusters

Run:
    pip install aiohttp networkx matplotlib
    python zksync_pipeline.py
"""

import asyncio
import aiohttp
import random
import collections
import pickle
import os

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

RPC_URL       = "https://mainnet.era.zksync.io"
START_BLOCK   = 67670670
NUM_BLOCKS    = 5000
CONCURRENCY   = 10
LP_ITERATIONS = 30
CACHE_FILE    = "blocks_cache.pkl"

# ─────────────────────────────────────────────
# STEP 1 — DOWNLOAD BLOCKS
# ─────────────────────────────────────────────

async def rpc_call(session, method, params, request_id=1):
    body = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": params,
    }
    async with session.post(RPC_URL, json=body) as resp:
        data = await resp.json()
        return data.get("result")


async def download_block(session, semaphore, block_number):
    async with semaphore:
        block = await rpc_call(
            session,
            "eth_getBlockByNumber",
            [hex(block_number), True],
            request_id=block_number,
        )
    return block


async def download_all_blocks():
    """if os.path.exists(CACHE_FILE):
        print(f"[cache] Loading blocks from {CACHE_FILE}")
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)"""

    semaphore = asyncio.Semaphore(CONCURRENCY)
    blocks = []

    async with aiohttp.ClientSession() as session:
        block_numbers = list(range(START_BLOCK, START_BLOCK + NUM_BLOCKS))
        tasks = [download_block(session, semaphore, n) for n in block_numbers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for r in results:
            if isinstance(r, Exception):
                print(f"  [warn] request failed: {r}")
            elif r is not None:
                blocks.append(r)

    print(f"[done] downloaded {len(blocks)} blocks")

    with open(CACHE_FILE, "wb") as f:
        pickle.dump(blocks, f)

    return blocks


# ─────────────────────────────────────────────
# STEP 2 — PRINT BLOCK NUMBERS + TOP SENDERS/RECEIVERS
# ─────────────────────────────────────────────

def print_block_stats(blocks):
    """Print block numbers, top 10 senders, top 10 receivers."""

    send_count: dict[str, int] = collections.defaultdict(int)
    recv_count: dict[str, int] = collections.defaultdict(int)

    block_numbers = []
    for block in blocks:
        if block is None:
            continue
        block_numbers.append(int(block["number"], 16))
        for tx in block.get("transactions", []):
            sender   = tx.get("from")
            receiver = tx.get("to")
            if sender:
                send_count[sender.lower()] += 1
            if receiver:
                recv_count[receiver.lower()] += 1

    block_numbers.sort()
    print(f"\n{'─'*55}")
    print(f"  Block range:  {block_numbers[0]}  →  {block_numbers[-1]}")
    print(f"  Total blocks: {len(block_numbers)}")
    print(f"{'─'*55}")

    print("\nTop 10 Senders (by tx count):")
    for rank, (addr, cnt) in enumerate(
        sorted(send_count.items(), key=lambda x: -x[1])[:10], 1
    ):
        print(f"  {rank:>2}. {addr}  —  {cnt} txs sent")

    print("\nTop 10 Receivers (by tx count):")
    for rank, (addr, cnt) in enumerate(
        sorted(recv_count.items(), key=lambda x: -x[1])[:10], 1
    ):
        print(f"  {rank:>2}. {addr}  —  {cnt} txs received")

    print()


# ─────────────────────────────────────────────
# STEP 3 — BUILD WEIGHTED ADDRESS GRAPH
# ─────────────────────────────────────────────

def build_graph(blocks) -> nx.Graph:
    """
    Nodes  : full hex addresses (unique identity).
    Edges  : undirected edge between sender and receiver.
    Weight : number of transactions between the two nodes.
    """
    G = nx.Graph()
    edge_counts: dict[tuple, int] = collections.defaultdict(int)

    for block in blocks:
        if block is None:
            continue
        for tx in block.get("transactions", []):
            sender   = tx.get("from")
            receiver = tx.get("to")

            if sender is None or receiver is None:
                continue

            sender   = sender.lower()
            receiver = receiver.lower()

            if sender == receiver:
                continue

            key = (min(sender, receiver), max(sender, receiver))
            edge_counts[key] += 1

    for (u, v), w in edge_counts.items():
        G.add_edge(u, v, weight=w)

    print(f"[graph] nodes={G.number_of_nodes()}  edges={G.number_of_edges()}")
    return G


# ─────────────────────────────────────────────
# STEP 4 — WEIGHTED LABEL PROPAGATION
# ─────────────────────────────────────────────

def weighted_label_propagation(G: nx.Graph, iterations: int) -> dict:
    nodes = list(G.nodes())
    n     = len(nodes)

    # Unique random labels 1..N
    labels = {node: label for node, label in zip(nodes, random.sample(range(1, n + 1), n))}

    for iteration in range(1, iterations + 1):
        changed = 0
        random.shuffle(nodes)

        for v in nodes:
            neighbours = list(G.neighbors(v))
            if not neighbours:
                continue

            score: dict[int, float] = collections.defaultdict(float)
            for u in neighbours:
                w = G[v][u].get("weight", 1)
                score[labels[u]] += w

            """total_weight = sum(G[v][u].get("weight", 1) for u in neighbours)
            threshold = total_weight / 2.0"""

            best_label = min(score, key=lambda lbl: (-score[lbl], lbl))
            """best_score = score[best_label]"""

            if  best_label != labels[v]:
                labels[v] = best_label
                """make the one with the most connections first. Then random"""
                changed += 1

        print(f"  LP iter {iteration:3d}/{iterations}  changes={changed:6d}", end="\r")
        if changed == 0:
            print(f"\n  [LP] converged early at iteration {iteration}")
            break

    print()
    return labels


# ─────────────────────────────────────────────
# STEP 5 — PLOT TOP-10 CLUSTERS
# ─────────────────────────────────────────────

def plot_top10_clusters(G: nx.Graph, labels: dict):
    cluster_members: dict[int, list] = collections.defaultdict(list)
    for node, lbl in labels.items():
        cluster_members[lbl].append(node)

    top10 = sorted(cluster_members.items(), key=lambda x: -len(x[1]))[:10]

    print("\nTop 10 cluster sizes:", [len(v) for _, v in top10])

    # Print all addresses per cluster
    print()
    for idx, (lbl, members) in enumerate(top10):
        subG_tmp = G.subgraph(members)
        deg_tmp  = dict(subG_tmp.degree(weight="weight"))
        sep = chr(9472) * 60
        print(sep)
        print(f"  Cluster #{idx+1}  |  label={lbl}  |  {len(members)} nodes  |  {subG_tmp.number_of_edges()} edges")
        print(sep)
        for addr in sorted(members, key=lambda n: -deg_tmp.get(n, 0)):
            print(f"    {addr}  (weighted degree: {deg_tmp.get(addr,0)})")
    print()

    # ── What to look for guide ──────────────────────────────────────

    colormap = plt.colormaps["plasma"].resampled(10)
    saved_files = []

    for idx, (lbl, members) in enumerate(top10):
        fig, ax = plt.subplots(figsize=(16, 12))
        fig.patch.set_facecolor("#0f0f1a")
        ax.set_facecolor("#0f0f1a")

        subG = G.subgraph(members)
        color = colormap(idx)

        deg = dict(subG.degree(weight="weight"))
        top_node = max(deg, key=deg.get) if deg else None

        # Spread nodes out with large k
        if len(members) > 2:
            pos = nx.spring_layout(subG, seed=42 + idx, weight="weight",
                                   k=3.0 / max(len(members) ** 0.4, 1), iterations=100)
        else:
            pos = nx.circular_layout(subG)

        node_sz = max(300, min(2000, 30000 // max(len(members), 1)))

        weights = [subG[u][v].get("weight", 1) for u, v in subG.edges()]
        if weights:
            max_w = max(weights)
            edge_widths = [0.8 + 3.5 * (w / max_w) for w in weights]
        else:
            edge_widths = []

        def max_edge_weight(n):
            return max(subG[n][nb].get("weight", 1) for nb in subG.neighbors(n))

        peripheral = [n for n in subG.nodes()
                      if subG.degree(n) == 1 and max_edge_weight(n) == 1]
        low_weight = [n for n in subG.nodes()
                      if n not in peripheral and max_edge_weight(n) == 1]
        core       = [n for n in subG.nodes()
                      if n not in peripheral and n not in low_weight]

        if core:
            nx.draw_networkx_nodes(subG, pos, nodelist=core, ax=ax,
                                   node_size=node_sz, node_color=[color], alpha=0.9)
        if low_weight:
            nx.draw_networkx_nodes(subG, pos, nodelist=low_weight, ax=ax,
                                   node_size=node_sz * 0.9, node_color=[color], alpha=0.8)
        if peripheral:
            nx.draw_networkx_nodes(subG, pos, nodelist=peripheral, ax=ax,
                                   node_size=node_sz * 0.6, node_color=[color], alpha=0.2)

        nx.draw_networkx_edges(subG, pos, ax=ax, width=edge_widths,
                               edge_color="white", alpha=0.5)

        # Node labels = number of neighbours (degree in subgraph)
        font_sz = max(5, min(11, 180 // max(len(members), 1)))
        neighbour_labels = {n: str(subG.degree(n)) for n in subG.nodes()}

        if core:
            nx.draw_networkx_labels(subG, pos,
                                    labels={n: neighbour_labels[n] for n in core},
                                    ax=ax, font_size=font_sz, font_color="white",
                                    font_weight="bold")
        if low_weight:
            nx.draw_networkx_labels(subG, pos,
                                    labels={n: neighbour_labels[n] for n in low_weight},
                                    ax=ax, font_size=max(4, font_sz - 1),
                                    font_color="#9999bb", font_weight="bold")
        if peripheral:
            nx.draw_networkx_labels(subG, pos,
                                    labels={n: neighbour_labels[n] for n in peripheral},
                                    ax=ax, font_size=max(4, font_sz - 1),
                                    font_color="#666688", font_weight="bold")

        # Edge weight labels — bold, larger, with a dark background box to stand out
        edge_weight_labels = {(u, v): subG[u][v].get("weight", 1) for u, v in subG.edges()}
        nx.draw_networkx_edge_labels(
            subG, pos, edge_labels=edge_weight_labels, ax=ax,
            font_size=max(6, font_sz),
            font_color="#ffe066",
            font_weight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="#1a1a2e", ec="#ffe066",
                      lw=0.8, alpha=0.85),
        )

        top_note = f"  |  hub: {top_node}" if top_node else ""
        ax.set_title(
            f"Cluster #{idx + 1}  |  label={lbl}  |  {len(members)} nodes"
            f"  |  {subG.number_of_edges()} edges{top_note}\n"
            f"zkSync Era  blocks {START_BLOCK} – {START_BLOCK + NUM_BLOCKS - 1}",
            fontsize=10, color="white", pad=12,
        )
        ax.axis("off")

        plt.tight_layout()
        out_path = f"cluster_{idx+1:02d}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        saved_files.append(out_path)
        print(f"  saved -> {out_path}")

    print(f"[plot] {len(saved_files)} cluster images saved.")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

async def async_main():
    print("=" * 55)
    print(" zkSync Era — Block Graph Clustering Pipeline")
    print("=" * 55)

    print(f"\n[1/5] Downloading {NUM_BLOCKS} blocks starting at {START_BLOCK} …")
    blocks = await download_all_blocks()

    print("\n[2/5] Block numbers + top senders/receivers …")
    print_block_stats(blocks)

    print("[3/5] Building address-interaction graph …")
    G = build_graph(blocks)

    if G.number_of_nodes() == 0:
        print("[error] Graph is empty — check RPC connection and block range.")
        return

    print(f"\n[4/5] Running weighted label propagation ({LP_ITERATIONS} iterations) …")
    random.seed(0)
    labels = weighted_label_propagation(G, LP_ITERATIONS)

    n_clusters = len(set(labels.values()))
    print(f"       {n_clusters} clusters found across {G.number_of_nodes()} nodes")

    print("\n[5/5] Plotting top-10 clusters …")
    plot_top10_clusters(G, labels)


if __name__ == "__main__":
    asyncio.run(async_main())