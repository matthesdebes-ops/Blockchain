"""
analyze_zkSync_chunked.py  (revised)

Streaming pickle loader + full analysis suite:
  • Bowtie decomposition  (IN / SCC / OUT / Tubes / Tendrils / Disconnected)
  • Degree distributions vs Power-law & Poisson fits  (log-log)
  • Transaction-volume distribution vs Log-normal, Power-law, Exponential
  • Assortativity coefficient r
  • Volume-degree correlation kept SEPARATE from volume distribution

Edge weights represent USD value, not raw token units. Every ERC-20 Transfer
log carries its own token contract (`address`) and raw integer `value`. Since
different tokens have different decimals and different prices, raw values are
not comparable or summable across tokens. price_enrichment.py converts each
transfer to USD using the token's on-chain decimals() and DefiLlama's
historical price at the transfer's block timestamp, so `edge_volumes` /
edge['weight'] is now a USD amount. See price_enrichment.py for details and
caching behavior (token_decimals_cache.json, token_price_cache.json,
unpriced_tokens.json in PLOT_DIR).
"""

import pickle
import os
import networkx as nx
import numpy as np
from typing import Dict, Generator
import time
from scipy.special import erf as _erf
from collections import defaultdict, Counter
import gc
import warnings
warnings.filterwarnings('ignore')
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from price_enrichment import enrich_chunk_with_usd, unpriced_summary

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────

LOG_DIR   = r"D:\zkSync_logs"
LOG_FILES = [
    "logs_69900000_70000000_final.pkl",
    "logs_14900000_15000000_final.pkl",
    "eth_transfers_69900000_70000000.pkl",
    "eth_transfers_14900000_15000000.pkl",
]

COMBINE_RANGES = False
CHUNK_SIZE     = 10_000
TAIL_FRACTION  = 0.01      # fraction excluded as heavy tail when fitting
PLOT_DPI       = 150
MIN_VOLUME     = 0.01      # USD dust filter (was 0.0001 ETH; adjust to taste)
PLOT_DIR       = r"D:\zkSync_logs\plots"

os.makedirs(PLOT_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────
# 1.  Streaming pickle loader (unchanged)
# ──────────────────────────────────────────────────────────────────────

class StreamingPickleLoader:
    def __init__(self, file_path: str, chunk_size: int = CHUNK_SIZE):
        self.file_path  = os.path.join(LOG_DIR, file_path)
        self.chunk_size = chunk_size

    def stream_objects(self) -> Generator:
        if not os.path.exists(self.file_path):
            print(f"File {self.file_path} not found!")
            return

        print(f"\nStreaming {os.path.basename(self.file_path)}…")
        print(f"File size: {os.path.getsize(self.file_path)/(1024*1024):.1f} MB")

        try:
            with open(self.file_path, 'rb') as f:
                data = pickle.Unpickler(f).load()

            if isinstance(data, dict) and 'logs' in data:
                logs = data['logs']
                print(f"Found dict with {len(logs):,} logs")
                yield {'type': 'metadata',
                       'num_logs': data.get('num_logs', len(logs)),
                       'block_range': data.get('block_range', (0, 0))}
                for i in range(0, len(logs), self.chunk_size):
                    chunk = logs[i:i+self.chunk_size]
                    yield {'type': 'logs', 'chunk': chunk}
                    del chunk; gc.collect()
                del logs

            elif isinstance(data, list):
                print(f"Found list with {len(data):,} items")
                yield {'type': 'metadata', 'num_logs': len(data), 'block_range': (0, 0)}
                for i in range(0, len(data), self.chunk_size):
                    chunk = data[i:i+self.chunk_size]
                    yield {'type': 'logs', 'chunk': chunk}
                    del chunk; gc.collect()
            else:
                print(f"Unknown data type: {type(data)}")

        except Exception as e:
            print(f"Error: {e}")


# ──────────────────────────────────────────────────────────────────────
# 2.  Build graph
# ──────────────────────────────────────────────────────────────────────

def process_pickle_streaming(file_path: str):
    loader = StreamingPickleLoader(file_path)
    edge_volumes = defaultdict(float)
    total_logs, chunk_count, metadata = 0, 0, {}
    priced_logs, zero_price_logs = 0, 0

    for item in loader.stream_objects():
        if item['type'] == 'metadata':
            metadata = item
            print(f"  Total logs: {metadata['num_logs']:,}")
            print(f"  Block range: {metadata['block_range'][0]:,} – {metadata['block_range'][1]:,}")
        elif item['type'] == 'logs':
            chunk = item['chunk']
            chunk_count += 1
            total_logs += len(chunk)

            # Attach 'value_usd' to every log: (raw_value / 10**decimals) * price_at_day.
            # Decimals + historical price are fetched (and cached to disk) as needed.
            chunk = enrich_chunk_with_usd(chunk)

            for log in chunk:
                s = log.get('from'); r = log.get('to')
                raw_v = log.get('value', 0)
                usd_v = log.get('value_usd', 0.0)
                if s and r and s != r and raw_v > 0:
                    edge_volumes[(s, r)] += usd_v
                    if usd_v > 0:
                        priced_logs += 1
                    else:
                        zero_price_logs += 1

            if chunk_count % 5 == 0:
                print(f"  Processed {total_logs:,} logs, {len(edge_volumes):,} unique edges "
                      f"(priced={priced_logs:,}, unpriced={zero_price_logs:,})")
            del chunk; gc.collect()

    n_unpriced_pairs, n_unpriced_logs = unpriced_summary()
    print(f"\nTotal processed: {total_logs:,} logs | Unique edges: {len(edge_volumes):,}")
    print(f"  Priced transfers: {priced_logs:,}  |  Unpriced (excluded from $ volume): {zero_price_logs:,}")
    if n_unpriced_logs:
        print(f"  (Across all files so far: {n_unpriced_pairs:,} token/day pairs with no "
              f"price found, affecting {n_unpriced_logs:,} transfers — see unpriced_tokens.json)")
    sys.stdout.flush()
    return edge_volumes, total_logs, metadata


def build_graph_from_streaming(file_path: str) -> nx.DiGraph:
    t0 = time.time()
    edge_volumes, _, _ = process_pickle_streaming(file_path)
    if not edge_volumes:
        print("No edges found!")
        return nx.DiGraph()
    G = nx.DiGraph()
    for (s, r), v in edge_volumes.items():
        G.add_edge(s, r, weight=v)
    print(f"  Nodes: {G.number_of_nodes():,}  Edges: {G.number_of_edges():,}  "
          f"Total vol: ${sum(edge_volumes.values()):,.2f} USD")
    tag = os.path.basename(file_path).replace('.pkl', '')
    _save_graph(G, f"graph_{tag}.gpickle")
    del edge_volumes; gc.collect()
    print(f"Graph built in {time.time()-t0:.1f}s"); sys.stdout.flush()
    return G


def build_combined_graph_streaming() -> nx.DiGraph:
    t0 = time.time()
    edge_volumes = defaultdict(float)
    total_logs = 0
    for i, fn in enumerate(LOG_FILES, 1):
        print(f"\nFile {i}/{len(LOG_FILES)}: {fn}")
        for item in StreamingPickleLoader(fn).stream_objects():
            if item['type'] == 'logs':
                chunk = enrich_chunk_with_usd(item['chunk'])
                for log in chunk:
                    s = log.get('from'); r = log.get('to')
                    raw_v = log.get('value', 0)
                    usd_v = log.get('value_usd', 0.0)
                    if s and r and s != r and raw_v > 0:
                        edge_volumes[(s, r)] += usd_v
                total_logs += len(chunk)
                del chunk; gc.collect()
    G = nx.DiGraph()
    for (s, r), v in edge_volumes.items():
        G.add_edge(s, r, weight=v)
    print(f"Combined: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    _save_graph(G, "graph_combined.gpickle")
    del edge_volumes; gc.collect()
    print(f"Done in {time.time()-t0:.1f}s"); sys.stdout.flush()
    return G


def _save_graph(G: nx.DiGraph, filename: str):
    path = os.path.join(PLOT_DIR, filename)
    with open(path, 'wb') as f:
        pickle.dump(G, f)
    print(f"  Graph saved: {path}")


# ──────────────────────────────────────────────────────────────────────
# 3.  Distribution fitting helpers
# ──────────────────────────────────────────────────────────────────────

def _bulk_tail(data, tail_fraction=TAIL_FRACTION):
    data = np.asarray(data, float)
    data = data[data > 0]
    if len(data) == 0:
        return np.array([]), np.array([])
    s = np.sort(data)
    cut = int(len(s) * (1 - tail_fraction))
    return s[:cut], s[cut:]


def fit_power_law(x_data, tail_fraction=TAIL_FRACTION):
    bulk, _ = _bulk_tail(x_data, tail_fraction)
    if len(bulk) < 10:
        return None, None, None
    x_min = np.percentile(bulk, 5)
    xf = bulk[bulk >= x_min]
    if len(xf) < 10:
        return None, None, None
    n = len(xf)
    alpha = 1 + n / np.sum(np.log(xf / x_min))
    logL = n*np.log(alpha-1) - n*np.log(x_min) - alpha*np.sum(np.log(xf/x_min))
    return alpha, x_min, logL


def fit_exponential(x_data, tail_fraction=TAIL_FRACTION):
    bulk, _ = _bulk_tail(x_data, tail_fraction)
    if len(bulk) < 10:
        return None, None
    lam = 1.0 / np.mean(bulk)
    logL = len(bulk)*np.log(lam) - lam*np.sum(bulk)
    return lam, logL


def fit_log_normal(x_data, tail_fraction=TAIL_FRACTION):
    bulk, _ = _bulk_tail(x_data, tail_fraction)
    if len(bulk) < 10:
        return None, None, None
    lx = np.log(bulk)
    mu, sigma = np.mean(lx), np.std(lx, ddof=1)
    n = len(bulk)
    logL = (-n/2*np.log(2*np.pi) - n*np.log(sigma)
            - np.sum(lx) - np.sum((lx - mu)**2)/(2*sigma**2))
    return mu, sigma, logL


# ──────────────────────────────────────────────────────────────────────
# 4.  Bowtie decomposition
# ──────────────────────────────────────────────────────────────────────

def bowtie_decomposition(G: nx.DiGraph) -> Dict:
    """
    Classic bowtie:
      SCC  – nodes in the giant strongly connected component
      IN   – nodes that can reach SCC but SCC cannot reach them
      OUT  – nodes reachable from SCC but that cannot reach SCC
      Tubes – nodes reachable from IN and can reach OUT (bypass SCC)
      Tendrils – everything else connected to the giant WCC
      Disconnected – not in the giant WCC at all
    """
    # Giant WCC
    wccs = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
    giant_wcc = wccs[0] if wccs else set()
    non_giant = set(G.nodes()) - giant_wcc

    # Giant SCC
    sccs = sorted(nx.strongly_connected_components(G), key=len, reverse=True)
    scc = sccs[0] if sccs else set()

    # Forward reachable from SCC (in giant WCC)
    G_giant = G.subgraph(giant_wcc)
    scc_sample = next(iter(scc))
    fwd = nx.descendants(G_giant, scc_sample) | {scc_sample}
    bwd = nx.descendants(G_giant.reverse(copy=False), scc_sample) | {scc_sample}

    IN_nodes  = (bwd - scc) & giant_wcc
    OUT_nodes = (fwd - scc) & giant_wcc

    # Tubes: reachable from IN that can reach OUT, not through SCC
    # (approximate: IN ∩ bwd_from_OUT in subgraph without SCC)
    G_no_scc = G_giant.subgraph(giant_wcc - scc)
    tube_candidates = set()
    for n in IN_nodes:
        try:
            reachable = nx.descendants(G_no_scc, n)
            if reachable & OUT_nodes:
                tube_candidates.add(n)
                tube_candidates |= (reachable & OUT_nodes)
        except Exception:
            pass
    tubes    = tube_candidates - scc - IN_nodes - OUT_nodes
    tendrils = giant_wcc - scc - IN_nodes - OUT_nodes - tubes
    disconnected = non_giant

    return {
        'SCC':          scc,
        'IN':           IN_nodes,
        'OUT':          OUT_nodes,
        'Tubes':        tubes,
        'Tendrils':     tendrils,
        'Disconnected': disconnected,
    }


# ──────────────────────────────────────────────────────────────────────
# 5.  Plotting
# ──────────────────────────────────────────────────────────────────────

# ── 5a.  Bowtie summary ──────────────────────────────────────────────

def plot_bowtie_summary(G: nx.DiGraph, label: str = "", save_plots: bool = True):
    print("\n" + "="*70)
    print("BOWTIE DECOMPOSITION")
    print("="*70); sys.stdout.flush()

    bt = bowtie_decomposition(G)
    N  = G.number_of_nodes()

    names  = ['IN', 'SCC\n(Knot)', 'OUT', 'Tubes', 'Tendrils', 'Disconnected']
    keys   = ['IN', 'SCC', 'OUT', 'Tubes', 'Tendrils', 'Disconnected']
    counts = [len(bt[k]) for k in keys]
    pcts   = [c/N*100 for c in counts]
    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2', '#937860']

    for k, v in zip(keys, counts):
        print(f"  {k:15s}: {v:>8,} nodes  ({v/N*100:.1f}%)")
    sys.stdout.flush()

    fig, (ax_bow, ax_bar) = plt.subplots(1, 2, figsize=(16, 6),
                                          facecolor='#1a1a2e')
    fig.suptitle(f'Bowtie Structure – {label}' if label else 'Bowtie Structure',
                 fontsize=15, fontweight='bold', color='white')

    # ── Schematic ──
    ax_bow.set_facecolor('#1a1a2e')
    ax_bow.set_xlim(0, 10); ax_bow.set_ylim(0, 6); ax_bow.axis('off')

    rects = {
        'IN':  dict(xy=(0.3, 2.0), w=2.0, h=2.0, c='#4C72B0'),
        'SCC': dict(xy=(3.8, 1.5), w=2.4, h=3.0, c='#E07B39'),
        'OUT': dict(xy=(7.5, 2.0), w=2.0, h=2.0, c='#2e9e6b'),
    }
    for key, r in rects.items():
        ax_bow.add_patch(mpatches.FancyBboxPatch(
            r['xy'], r['w'], r['h'],
            boxstyle="round,pad=0.15", fc=r['c'], alpha=0.92,
            ec='white', lw=1.5))
        ax_bow.text(r['xy'][0] + r['w']/2, r['xy'][1] + r['h']/2,
                    f"{key}\n{len(bt[key]):,}\n({len(bt[key])/N*100:.1f}%)",
                    ha='center', va='center', fontsize=11,
                    fontweight='bold', color='white')

    # Arrows IN→SCC, SCC→OUT
    for x1, x2 in [(2.35, 3.78), (6.22, 7.48)]:
        ax_bow.annotate('', xy=(x2, 3.0), xytext=(x1, 3.0),
                        arrowprops=dict(arrowstyle='->', color='white',
                                        lw=2.5, mutation_scale=18))

    # Tube arc (bypass) — drawn as a Bezier curve above the boxes
    from matplotlib.patches import FancyArrowPatch
    tube_arrow = FancyArrowPatch(
        posA=(2.35, 1.8), posB=(7.48, 1.8),
        arrowstyle='->', color='#ff6b6b', lw=1.8, mutation_scale=15,
        connectionstyle='arc3,rad=0.45',   # positive = arc downward below boxes
        zorder=5)
    ax_bow.add_patch(tube_arrow)
    ax_bow.text(5.0, 0.30, f"Tubes: {len(bt['Tubes']):,}",
                ha='center', color='#ff6b6b', fontsize=10, fontweight='bold')
    ax_bow.text(5.0, 5.65,
                f"Tendrils: {len(bt['Tendrils']):,}       "
                f"Disconnected: {len(bt['Disconnected']):,}",
                ha='center', fontsize=9, color='#aaaaaa')

    # ── Bar chart ──
    ax_bar.set_facecolor('#1a1a2e')
    bars = ax_bar.barh(names, counts, color=colors, edgecolor='#1a1a2e', linewidth=0.5)
    ax_bar.set_xlabel('Number of nodes', color='white')
    ax_bar.set_xscale('log')
    ax_bar.set_title('Node counts by bowtie region', color='white')
    ax_bar.tick_params(colors='white')
    for sp in ax_bar.spines.values(): sp.set_color('#444')
    ax_bar.grid(axis='x', alpha=0.2, color='white')

    x_max = max(counts) if max(counts) > 0 else 1
    for bar, c, p in zip(bars, counts, pcts):
        w = bar.get_width() if bar.get_width() > 0 else 0.5
        ax_bar.text(w * 1.12, bar.get_y() + bar.get_height()/2,
                    f'{c:,}  ({p:.1f}%)',
                    va='center', fontsize=9, color='white')

    fig.subplots_adjust(left=0.05, right=0.90, top=0.88, bottom=0.08, wspace=0.35)
    _save_fig(fig, f"bowtie_{label}", save_plots)
    return bt


# ── 5b.  Degree distributions vs Power-law & Poisson ────────────────

def plot_degree_distributions(G: nx.DiGraph, label: str = "", save_plots: bool = True):
    print("\n" + "="*70)
    print("DEGREE DISTRIBUTIONS (log-log)")
    print("="*70); sys.stdout.flush()

    in_deg  = np.array([d for _, d in G.in_degree()])
    out_deg = np.array([d for _, d in G.out_degree()])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Degree Distributions – {label}' if label else 'Degree Distributions',
                 fontsize=15, fontweight='bold')

    for ax, deg, title, col in zip(
            axes,
            [in_deg, out_deg],
            ['In-degree', 'Out-degree'],
            ['#4C72B0', '#DD8452']):

        counts = Counter(deg)
        xs = np.array(sorted(counts.keys()))
        ys = np.array([counts[x] for x in xs])

        ax.loglog(xs, ys, 'o', color=col, alpha=0.7, ms=4, label='Empirical', zorder=3)

        # Data bounds — all fits clipped to this range so axes stay on the data
        x_lo, x_hi = xs.min(), xs.max()
        y_lo, y_hi = ys.min() * 0.5, ys.max() * 3.0

        # ── Power-law fit (MLE on bulk) ──
        bulk, _ = _bulk_tail(deg)
        if len(bulk) > 10:
            x_min  = max(np.percentile(bulk, 5), 1)
            xf     = bulk[bulk >= x_min]
            if len(xf) > 5:
                alpha_pl = 1 + len(xf) / np.sum(np.log(xf / x_min))
                x_fit = np.logspace(np.log10(x_min), np.log10(x_hi), 300)
                C = len(deg)
                y_pl = C * (alpha_pl - 1) * x_min**(alpha_pl-1) * x_fit**(-alpha_pl)
                mask = (y_pl >= y_lo) & (y_pl <= y_hi * 10)
                if mask.any():
                    ax.loglog(x_fit[mask], y_pl[mask], 'r-', lw=2,
                              label=f'Power law  α={alpha_pl:.2f}')

        # ── Poisson fit (log-stable, clipped to data range) ──
        lam = np.mean(deg)
        k_poi = np.arange(max(1, int(x_lo)), min(int(x_hi) + 1, 500))
        log_pmf = (k_poi * np.log(lam) - lam
                   - np.array([np.sum(np.log(np.arange(1, k+1))) for k in k_poi]))
        y_poi = len(deg) * np.exp(log_pmf)
        mask = (y_poi >= y_lo) & (y_poi <= y_hi * 10)
        if mask.any():
            ax.loglog(k_poi[mask], y_poi[mask], 'g--', lw=1.8,
                      label=f'Poisson  λ={lam:.1f}')

        # Lock axes to the empirical data range — no underflow tails
        ax.set_xlim(x_lo * 0.8, x_hi * 1.5)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel(title)
        ax.set_ylabel('Number of nodes')
        ax.set_title(f'{title} distribution\n'
                     f'mean={np.mean(deg):.2f}  max={deg.max()}  nodes={len(deg):,}')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    _save_fig(fig, f"degree_distributions_{label}", save_plots)


# ── 5c.  Transaction-volume distribution (standalone) ───────────────

def plot_volume_distribution(G: nx.DiGraph, label: str = "", save_plots: bool = True):
    print("\n" + "="*70)
    print("TRANSACTION VOLUME DISTRIBUTION (USD)")
    print("="*70); sys.stdout.flush()

    weights = np.array([d['weight'] for *_, d in G.edges(data=True)
                        if d['weight'] > MIN_VOLUME])
    if len(weights) == 0:
        print("No weight data!"); return

    bulk, tail = _bulk_tail(weights)
    print(f"  Edges: {len(weights):,}  |  bulk: {len(bulk):,}  tail: {len(tail):,}")
    print(f"  Min=${weights.min():.6f}  Max=${weights.max():.2f}  "
          f"Mean=${weights.mean():.4f}  Median=${np.median(weights):.4f}")

    # Fits
    alpha_pl, x_min_pl, logL_pl   = fit_power_law(weights)
    lam_exp,  logL_exp             = fit_exponential(weights)
    mu_ln, sigma_ln, logL_ln       = fit_log_normal(weights)

    if alpha_pl  is not None: print(f"  Power-law  α={alpha_pl:.4f}  x_min={x_min_pl:.6f}  logL={logL_pl:.1f}")
    if lam_exp   is not None: print(f"  Exponential  λ={lam_exp:.6f}  logL={logL_exp:.1f}")
    if mu_ln     is not None: print(f"  Log-normal  μ={mu_ln:.4f}  σ={sigma_ln:.4f}  logL={logL_ln:.1f}")
    sys.stdout.flush()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Transaction Volume Distribution (USD) – {label}' if label else 'Transaction Volume Distribution (USD)',
                 fontsize=15, fontweight='bold')

    # ── CCDF with all three fits ──────────────────────────────────────
    ax = axes[0]
    sw  = np.sort(weights)
    ccdf = 1 - np.arange(1, len(sw)+1) / len(sw)
    ax.loglog(sw, ccdf, 'k.', ms=2, alpha=0.4, label='Empirical CCDF', zorder=1)

    x_fit = np.logspace(np.log10(max(weights.min(), 1e-9)), np.log10(weights.max()), 500)

    # y-limits from the empirical data — fit lines clipped to this range
    y_lo = ccdf[ccdf > 0].min()
    y_hi = 1.0
    # Only plot fit curves where their value is within the data's y-range
    if alpha_pl is not None:
        surv_pl = (x_fit / x_min_pl)**(-(alpha_pl-1))
        surv_pl[x_fit < x_min_pl] = 1.0
        mask = surv_pl >= y_lo
        if mask.any():
            ax.loglog(x_fit[mask], surv_pl[mask], 'r-', lw=2, label=f'Power law α={alpha_pl:.2f}')

    if lam_exp is not None:
        surv_exp = np.exp(-lam_exp * x_fit)
        mask = surv_exp >= y_lo
        if mask.any():
            ax.loglog(x_fit[mask], surv_exp[mask], 'g-', lw=2, label=f'Exponential λ={lam_exp:.4f}')

    if mu_ln is not None:
        surv_ln = 0.5 * (1 - _erf((np.log(x_fit) - mu_ln) / (sigma_ln * np.sqrt(2))))
        mask = surv_ln >= y_lo
        if mask.any():
            ax.loglog(x_fit[mask], surv_ln[mask], 'b-', lw=2,
                      label=f'Log-normal μ={mu_ln:.2f} σ={sigma_ln:.2f}')

    ax.set_xlim(sw[0] * 0.9, sw[-1] * 1.1)
    ax.set_ylim(y_lo * 0.5, 2.0)
    ax.set_xlabel('Volume (USD)'); ax.set_ylabel('P(X ≥ x)')
    ax.set_title('CCDF  (log-log)'); ax.legend(fontsize=8); ax.grid(True, alpha=0.25)

    # ── PDF histogram with fits ───────────────────────────────────────
    ax2 = axes[1]
    log_bins = np.logspace(np.log10(max(weights.min(), 1e-9)), np.log10(weights.max()), 60)
    hist_vals, _, _ = ax2.hist(weights, bins=log_bins, density=True,
                                alpha=0.5, color='steelblue', label='Histogram')
    ax2.set_xscale('log'); ax2.set_yscale('log')
    pdf_lo = hist_vals[hist_vals > 0].min() * 0.1
    pdf_hi = hist_vals.max() * 10

    if alpha_pl is not None:
        y_pl = (alpha_pl-1) * x_min_pl**(alpha_pl-1) * x_fit**(-alpha_pl)
        y_pl[x_fit < x_min_pl] = np.nan
        mask = np.isfinite(y_pl) & (y_pl >= pdf_lo) & (y_pl <= pdf_hi * 100)
        if mask.any():
            ax2.loglog(x_fit[mask], y_pl[mask], 'r-', lw=2, label=f'Power law α={alpha_pl:.2f}')

    if lam_exp is not None:
        y_exp = lam_exp * np.exp(-lam_exp * x_fit)
        mask = (y_exp >= pdf_lo) & (y_exp <= pdf_hi * 100)
        if mask.any():
            ax2.loglog(x_fit[mask], y_exp[mask], 'g-', lw=2, label=f'Exponential λ={lam_exp:.4f}')

    if mu_ln is not None:
        y_ln = (1/(x_fit*sigma_ln*np.sqrt(2*np.pi)) *
                np.exp(-(np.log(x_fit)-mu_ln)**2/(2*sigma_ln**2)))
        mask = np.isfinite(y_ln) & (y_ln >= pdf_lo) & (y_ln <= pdf_hi * 100)
        if mask.any():
            ax2.loglog(x_fit[mask], y_ln[mask], 'b-', lw=2, label=f'Log-normal μ={mu_ln:.2f}')

    ax2.set_ylim(pdf_lo, pdf_hi)
    ax2.set_xlabel('Volume (USD)'); ax2.set_ylabel('PDF')
    ax2.set_title('PDF  (log-log)'); ax2.legend(fontsize=8); ax2.grid(True, alpha=0.25)

    plt.tight_layout()
    _save_fig(fig, f"volume_distribution_{label}", save_plots)


# ── 5d.  Network structure overview ──────────────────────────────────

def plot_network_overview(G: nx.DiGraph, label: str = "", save_plots: bool = True):
    print("\n" + "="*70)
    print("NETWORK STRUCTURE OVERVIEW")
    print("="*70); sys.stdout.flush()

    wccs = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
    sccs = sorted(nx.strongly_connected_components(G), key=len, reverse=True)

    wcc_sizes = [len(c) for c in wccs]
    scc_sizes = [len(c) for c in sccs]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Connected Components – {label}' if label else 'Connected Components',
                 fontsize=15, fontweight='bold')

    for ax, sizes, title, col in zip(
            axes,
            [wcc_sizes, scc_sizes],
            ['Weakly Connected Components', 'Strongly Connected Components'],
            ['#4C72B0', '#DD8452']):

        top = sizes[:20]
        ax.bar(range(1, len(top)+1), top, color=col, alpha=0.8, edgecolor='white')
        ax.set_xlabel('Component rank')
        ax.set_ylabel('Size (nodes, log scale)')
        ax.set_yscale('log')
        ax.set_title(f'{title}\n(total: {len(sizes):,} components)')
        ax.grid(axis='y', alpha=0.25)
        ax.set_xticks(range(1, len(top)+1))

    plt.tight_layout()
    _save_fig(fig, f"network_overview_{label}", save_plots)


# ── 5e.  Volume-degree correlation (separate, independent) ──────────

def plot_volume_degree_correlation(G: nx.DiGraph, label: str = "", save_plots: bool = True):
    print("\n" + "="*70)
    print("VOLUME-DEGREE CORRELATION (USD)")
    print("="*70); sys.stdout.flush()

    node_in_vol  = defaultdict(float)
    node_out_vol = defaultdict(float)

    for u, v, d in G.edges(data=True):
        w = d['weight']
        node_out_vol[u] += w
        node_in_vol[v]  += w

    # Aggregate per degree
    in_deg_vol  = defaultdict(list)
    out_deg_vol = defaultdict(list)

    for node in G.nodes():
        if node in node_in_vol:
            in_deg_vol[G.in_degree(node)].append(node_in_vol[node])
        if node in node_out_vol:
            out_deg_vol[G.out_degree(node)].append(node_out_vol[node])

    def _agg(d):
        xs = sorted(d.keys())
        ys = [np.median(d[x]) for x in xs]
        return np.array(xs), np.array(ys)

    x_in,  y_in  = _agg(in_deg_vol)
    x_out, y_out = _agg(out_deg_vol)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Volume vs Degree (correlation) – {label}' if label else 'Volume vs Degree',
                 fontsize=15, fontweight='bold')

    ax1.loglog(x_in,  y_in,  'o', color='#4C72B0', ms=4, alpha=0.7, label='Median vol per in-degree')
    ax1.set_xlabel('In-degree'); ax1.set_ylabel('Median incoming volume (USD)')
    ax1.set_title('In-degree vs Incoming Volume'); ax1.grid(True, alpha=0.25); ax1.legend()

    ax2.loglog(x_out, y_out, 'o', color='#DD8452', ms=4, alpha=0.7, label='Median vol per out-degree')
    ax2.set_xlabel('Out-degree'); ax2.set_ylabel('Median outgoing volume (USD)')
    ax2.set_title('Out-degree vs Outgoing Volume'); ax2.grid(True, alpha=0.25); ax2.legend()

    plt.tight_layout()
    _save_fig(fig, f"volume_degree_corr_{label}", save_plots)


# ──────────────────────────────────────────────────────────────────────
# 6.  Utility
# ──────────────────────────────────────────────────────────────────────

def _save_fig(fig, name: str, save: bool):
    if save:
        safe = name.replace(' ', '_').replace('/', '_')
        path = os.path.join(PLOT_DIR, f"{safe}.png")
        fig.savefig(path, dpi=PLOT_DPI, facecolor=fig.get_facecolor())
        print(f"  Plot saved: {path}")
    plt.close(fig)
    sys.stdout.flush()


# ──────────────────────────────────────────────────────────────────────
# 7.  Main analysis (all metrics + plots)
# ──────────────────────────────────────────────────────────────────────

def full_analysis(G: nx.DiGraph, label: str):
    if G.number_of_nodes() == 0:
        print("Empty graph – skipping analysis"); return

    print(f"\n{'#'*70}")
    print(f"  FULL ANALYSIS  –  {label}")
    print(f"{'#'*70}")
    print(f"  Nodes : {G.number_of_nodes():,}")
    print(f"  Edges : {G.number_of_edges():,}")
    print(f"  Density: {nx.density(G):.2e}")

    # ── Assortativity ──
    try:
        r = nx.degree_assortativity_coefficient(G)
        print(f"\n  Degree assortativity  r = {r:.4f}  "
              f"({'assortative' if r>0 else 'disassortative' if r<0 else 'neutral'})")
    except Exception as e:
        print(f"\n  Assortativity could not be computed: {e}")

    # ── Bowtie ──
    plot_bowtie_summary(G, label)

    # ── Degree distributions ──
    plot_degree_distributions(G, label)

    # ── Volume distribution (independent of degree) ──
    plot_volume_distribution(G, label)

    # ── Component overview ──
    plot_network_overview(G, label)

    print(f"\n  Analysis complete for: {label}")
    sys.stdout.flush()


# ──────────────────────────────────────────────────────────────────────
# 8.  Entry point
# ──────────────────────────────────────────────────────────────────────

def main():
    print("="*70)
    print("ZKSYNC STREAMING ANALYSIS")
    print("="*70)
    print(f"Log dir   : {LOG_DIR}")
    print(f"Files     : {LOG_FILES}")
    print(f"Combine   : {COMBINE_RANGES}")
    print(f"Chunk     : {CHUNK_SIZE:,}")
    print(f"Plot dir  : {PLOT_DIR}")
    print("="*70); sys.stdout.flush()

    if not os.path.exists(LOG_DIR):
        print(f"\nERROR: {LOG_DIR} does not exist!"); return

    for fn in LOG_FILES:
        fp = os.path.join(LOG_DIR, fn)
        if os.path.exists(fp):
            print(f"  Found: {fn}  ({os.path.getsize(fp)/(1024*1024):.1f} MB)")
        else:
            print(f"  MISSING: {fn}")
    sys.stdout.flush()

    try:
        if COMBINE_RANGES:
            G = build_combined_graph_streaming()
            full_analysis(G, "combined")
            del G; gc.collect()
        else:
            for fn in LOG_FILES:
                G = build_graph_from_streaming(fn)
                label = fn.replace('.pkl', '').replace('logs_', '')
                full_analysis(G, label)
                del G; gc.collect()
                print("\n" + "-"*70)

    except MemoryError:
        print("\nMemoryError – try reducing CHUNK_SIZE to 1000")
    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        import traceback
        print(f"\nError: {e}")
        traceback.print_exc()

    print("\n" + "="*70)
    print("DONE")
    print("="*70)


if __name__ == "__main__":
    main()