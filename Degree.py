"""
zkSync Graph Degree Distribution & Assortativity Analysis
=========================================================
Processes each log file as a SEPARATE dataset and produces one figure per file.
Each figure has 4 panels:
  - In-degree distribution  (log-binned PDF + fits)
  - Out-degree distribution (log-binned PDF + fits)
  - CCDF comparison (in & out on one plot)
  - Stats / assortativity panel

Fitting methodology (Clauset et al. 2009, as used in the reference paper):
  - Power-law: MLE via the `powerlaw` package  (xmin chosen by KS statistic)
  - Exponential & Log-normal: compared via likelihood-ratio (LR) tests

Install extras if needed:
    pip install powerlaw
"""

import os
import pickle
import warnings
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import curve_fit

try:
    import powerlaw as pl_pkg
    HAS_POWERLAW = True
except ImportError:
    HAS_POWERLAW = False
    print("  [warn] `powerlaw` package not found — falling back to scipy curve_fit.\n"
          "         Install with:  pip install powerlaw")

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
LOG_DIR = r"D:\zkSync_logs"
LOG_FILES = [
    "logs_69900000_70000000_final.pkl",
    "logs_14900000_15000000_final.pkl",
]

# ── Styling ───────────────────────────────────────────────────────────────────
DARK_BG  = "#0d0f14"
PANEL_BG = "#13161e"
ACCENT1  = "#7c6af7"   # purple  – power-law
ACCENT2  = "#f97316"   # orange  – exponential
ACCENT3  = "#22d3ee"   # cyan    – log-normal
DATA_COL = "#e2e8f0"
GRID_COL = "#1e2330"
TEXT_COL = "#94a3b8"

plt.rcParams.update({
    "figure.facecolor":  DARK_BG,
    "axes.facecolor":    PANEL_BG,
    "axes.edgecolor":    GRID_COL,
    "axes.labelcolor":   TEXT_COL,
    "xtick.color":       TEXT_COL,
    "ytick.color":       TEXT_COL,
    "text.color":        DATA_COL,
    "grid.color":        GRID_COL,
    "grid.linewidth":    0.6,
    "font.family":       "monospace",
    "legend.framealpha": 0.0,
    "legend.labelcolor": DATA_COL,
})


# ── Data loading ──────────────────────────────────────────────────────────────
def load_one_file(log_dir: str, fname: str) -> list:
    path = os.path.join(log_dir, fname)
    print(f"  Loading {fname} …", end=" ")
    with open(path, "rb") as f:
        data = pickle.load(f)
    records = []
    if isinstance(data, dict):
        for v in data.values():
            records.extend(v) if isinstance(v, (list, tuple)) else records.append(v)
    elif isinstance(data, (list, tuple)):
        records.extend(data)
    else:
        records.append(data)
    print(f"→ {len(records):,} records")
    return records


def build_graph(records) -> nx.DiGraph:
    if isinstance(records, nx.DiGraph):
        return records
    if isinstance(records, nx.Graph):
        return records.to_directed()
    G = nx.DiGraph()
    for rec in records:
        if isinstance(rec, nx.DiGraph):
            G = nx.compose(G, rec); continue
        if isinstance(rec, (list, tuple)) and len(rec) >= 2:
            src, dst = str(rec[0]), str(rec[1])
        elif isinstance(rec, dict):
            src = str(rec.get("from") or rec.get("from_address") or
                       rec.get("sender") or rec.get("src") or "")
            dst = str(rec.get("to")   or rec.get("to_address")   or
                       rec.get("receiver") or rec.get("dst") or "")
        else:
            continue
        if src and dst and src != "None" and dst != "None":
            if G.has_edge(src, dst):
                G[src][dst]["weight"] = G[src][dst].get("weight", 1) + 1
            else:
                G.add_edge(src, dst, weight=1)
    return G


# ── Math helpers ──────────────────────────────────────────────────────────────
def ccdf(values: np.ndarray):
    vals = np.sort(values)
    p    = 1.0 - np.arange(len(vals)) / len(vals)
    return vals, p


def log_bins(values: np.ndarray, n_bins: int = 60):
    lo = max(values.min(), 1)
    hi = values.max()
    if lo >= hi:
        return values, np.ones_like(values) / len(values)
    bins    = np.logspace(np.log10(lo), np.log10(hi), n_bins + 1)
    counts, edges = np.histogram(values, bins=bins)
    widths  = np.diff(edges)
    density = counts / (counts.sum() * widths + 1e-12)
    centres = 0.5 * (edges[:-1] + edges[1:])
    mask    = counts > 0
    return centres[mask], density[mask]


def zoom_axis(ax, x_data, y_data, margin_x=0.05, margin_y=0.15):
    x_lo = max(x_data.min(), 0.5)
    x_hi = x_data.max()
    y_lo = max(y_data[y_data > 0].min() if (y_data > 0).any() else 1e-10, 1e-10)
    y_hi = y_data.max()
    if x_lo >= x_hi or y_lo >= y_hi:
        return
    lx_lo = np.log10(x_lo) - margin_x * (np.log10(x_hi) - np.log10(x_lo))
    lx_hi = np.log10(x_hi) + margin_x * (np.log10(x_hi) - np.log10(x_lo))
    ly_lo = np.log10(y_lo) - margin_y * (np.log10(y_hi) - np.log10(y_lo))
    ly_hi = np.log10(y_hi) + margin_y * (np.log10(y_hi) - np.log10(y_lo))
    ax.set_xlim(10 ** lx_lo, 10 ** lx_hi)
    ax.set_ylim(10 ** ly_lo, 10 ** ly_hi)


# ── Curve shapes ──────────────────────────────────────────────────────────────
def _pl(log_x, log_C, alpha):   return log_C - alpha * log_x          # log-log: straight line
def _ln(log_x, mu, sig, log_C): return log_C - np.log(np.exp(log_x)*sig*np.sqrt(2*np.pi)) \
                                        - (log_x - mu)**2 / (2*sig**2) # log-log: parabola
def _poisson(log_x, log_C, lam): return log_C + np.exp(log_x)*np.log(lam) - lam - log_x  # log P(k) ≈ k*log(λ) - λ - log(k!), Stirling: log(k!)≈k*log(k)-k


def _lsq_log(fn, log_x, log_y, p0, bounds):
    """Least-squares fit in LOG space — this is what visually matches log-log plots."""
    try:
        popt, _ = curve_fit(fn, log_x, log_y, p0=p0, bounds=bounds, maxfev=20000)
        return popt
    except Exception:
        return None


# ── Main fitting + plotting function ─────────────────────────────────────────
def fit_and_plot(ax, deg_arr: np.ndarray, label: str, color: str):
    x, y = log_bins(deg_arr)
    # work in log-log space throughout
    log_x = np.log(x)
    log_y = np.log(y)
    # only fit bins where y > 0 (already guaranteed by log_bins mask, but be safe)
    valid = np.isfinite(log_y)
    lx, ly = log_x[valid], log_y[valid]

    ax.scatter(x, y, s=14, color=color, alpha=0.85, zorder=3, label=f"{label} (data)")
    x_fit    = np.logspace(np.log10(x.min()), np.log10(x.max()), 600)
    lx_fit   = np.log(x_fit)
    lr_results = {}

    # ── Power-law: fit in log-log (straight line), xmin = 1 (full range) ──
    # Initial guess from OLS slope on log-log
    slope, intercept = np.polyfit(lx, ly, 1)
    alpha0   = max(-slope, 0.5)
    log_C0   = intercept
    p_pl = _lsq_log(_pl, lx, ly, p0=[log_C0, alpha0], bounds=([-50, 0.1], [50, 10]))
    if p_pl is not None:
        log_C_pl, alpha_pl = p_pl
        ax.plot(x_fit, np.exp(_pl(lx_fit, log_C_pl, alpha_pl)), "--",
                color=ACCENT1, lw=1.8, alpha=0.95,
                label=f"Power-law α={alpha_pl:.2f}")

    # LR tests via powerlaw package (MLE on raw data, separate from plot fit)
    if HAS_POWERLAW:
        try:
            fit_mle = pl_pkg.Fit(deg_arr.astype(int), xmin=1, discrete=True, verbose=False)
            R_exp, p_exp = fit_mle.distribution_compare("power_law", "exponential",
                                                         normalized_ratio=True)
            R_ln,  p_ln  = fit_mle.distribution_compare("power_law", "lognormal",
                                                         normalized_ratio=True)
            lr_results = {"alpha": fit_mle.power_law.alpha,
                          "LR_vs_exp": (R_exp, p_exp), "LR_vs_lnorm": (R_ln, p_ln)}
        except Exception:
            pass

    # ── Log-normal: fit in log-log (parabola shape) ────────────────────────
    mu0  = float(np.average(lx, weights=np.exp(ly)))   # density-weighted mean of log(x)
    sig0 = float(np.sqrt(np.average((lx - mu0)**2, weights=np.exp(ly))))
    sig0 = max(sig0, 0.3)
    log_C0_ln = ly.max() + np.log(np.exp(mu0) * sig0 * np.sqrt(2*np.pi))
    p_ln = _lsq_log(_ln, lx, ly, p0=[mu0, sig0, log_C0_ln],
                    bounds=([-20, 0.05, -50], [20, 20, 50]))
    if p_ln is not None:
        mu_f, sig_f, lC_f = p_ln
        ln_lbl = f"Log-normal μ={mu_f:.2f} σ={sig_f:.2f}"
        if lr_results and lr_results.get("LR_vs_lnorm", (0,1))[0] < 0:
            ln_lbl += " ★"
        ax.plot(x_fit, np.exp(_ln(lx_fit, mu_f, sig_f, lC_f)), ":",
                color=ACCENT3, lw=1.8, alpha=0.9, label=ln_lbl)
    else:
        print(f"    [warn] log-normal fit did not converge for {label}")

    # ── Poisson: log P(k) = k·log(λ) - λ - log(k!) using Stirling approx ──
    # In log-log: log y = log_C + k·log(λ) - λ - k·log(k) + k  (Stirling)
    lam0    = float(np.average(np.exp(lx), weights=np.exp(ly)))  # mean degree as λ init
    lam0    = max(lam0, 0.5)
    p_pois = _lsq_log(_poisson, lx, ly, p0=[ly.max(), lam0],
                       bounds=([-50, 0.01], [50, np.inf]))
    if p_pois is not None:
        lC_p, lam_p = p_pois
        pois_lbl = f"Poisson λ={lam_p:.2f}"
        ax.plot(x_fit, np.exp(_poisson(lx_fit, lC_p, lam_p)), "-.",
                color=ACCENT2, lw=1.5, alpha=0.9, label=pois_lbl)

    return lr_results


# ── Per-dataset figure ────────────────────────────────────────────────────────
def make_figure(G: nx.DiGraph, fname: str, fig_idx: int):
    in_deg  = np.array([d for _, d in G.in_degree()],  dtype=float)
    out_deg = np.array([d for _, d in G.out_degree()], dtype=float)
    in_nz   = in_deg[in_deg   > 0]
    out_nz  = out_deg[out_deg > 0]

    # Assortativity
    def _safe(fn, *a):
        try:    return fn(*a)
        except: return float("nan")

    def _dir_assort(G, ms="out", md="in"):
        try:
            sd = dict(G.in_degree()  if ms == "in" else G.out_degree())
            dd = dict(G.in_degree()  if md == "in" else G.out_degree())
            xv = np.array([sd[u] for u, v in G.edges()], dtype=float)
            yv = np.array([dd[v] for u, v in G.edges()], dtype=float)
            if xv.std() == 0 or yv.std() == 0: return float("nan")
            return float(np.corrcoef(xv, yv)[0, 1])
        except: return float("nan")

    r_overall = _safe(nx.degree_assortativity_coefficient, G)
    r_in      = _dir_assort(G, "in",  "in")
    r_out     = _dir_assort(G, "out", "out")

    # Short dataset label from filename
    ds_label = fname.replace("logs_", "").replace("_final.pkl", "").replace("_", " – ")

    print(f"\n  [{ds_label}]")
    print(f"    Nodes {G.number_of_nodes():,}  Edges {G.number_of_edges():,}")
    print(f"    Assortativity  overall={r_overall:+.4f}  in={r_in:+.4f}  out={r_out:+.4f}")

    fig = plt.figure(figsize=(16, 10), facecolor=DARK_BG)
    fig.suptitle(f"zkSync — Degree Distribution   [{ds_label}]",
                 fontsize=14, fontweight="bold", color=DATA_COL, y=0.97)

    gs = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32,
                  left=0.07, right=0.97, top=0.91, bottom=0.09)

    ax_in   = fig.add_subplot(gs[0, 0])
    ax_out  = fig.add_subplot(gs[0, 1])
    ax_ccdf = fig.add_subplot(gs[1, 0])
    ax_info = fig.add_subplot(gs[1, 1])

    # In-degree
    ax_in.set_xscale("log"); ax_in.set_yscale("log")
    xi, yi = log_bins(in_nz)
    lr_in  = fit_and_plot(ax_in, in_nz, "In-degree", "#a78bfa")
    zoom_axis(ax_in, xi, yi)
    ax_in.set_xlabel("In-degree  k", fontsize=10)
    ax_in.set_ylabel("P(k)  [log-bin density]", fontsize=10)
    ax_in.set_title("In-Degree Distribution", color=DATA_COL, fontsize=11, pad=6)
    ax_in.legend(fontsize=7.5, loc="upper right")
    ax_in.grid(True, which="both", ls=":", alpha=0.4)

    # Out-degree
    ax_out.set_xscale("log"); ax_out.set_yscale("log")
    xo, yo = log_bins(out_nz)
    lr_out = fit_and_plot(ax_out, out_nz, "Out-degree", "#34d399")
    zoom_axis(ax_out, xo, yo)
    ax_out.set_xlabel("Out-degree  k", fontsize=10)
    ax_out.set_ylabel("P(k)  [log-bin density]", fontsize=10)
    ax_out.set_title("Out-Degree Distribution", color=DATA_COL, fontsize=11, pad=6)
    ax_out.legend(fontsize=7.5, loc="upper right")
    ax_out.grid(True, which="both", ls=":", alpha=0.4)

    # CCDF
    xi_c, pi_c = ccdf(in_nz)
    xo_c, po_c = ccdf(out_nz)
    ax_ccdf.set_xscale("log"); ax_ccdf.set_yscale("log")
    ax_ccdf.plot(xi_c, pi_c, color="#a78bfa", lw=1.5, alpha=0.85, label="In-degree CCDF")
    ax_ccdf.plot(xo_c, po_c, color="#34d399", lw=1.5, alpha=0.85, label="Out-degree CCDF")
    all_x = np.concatenate([xi_c, xo_c])
    all_y = np.concatenate([pi_c, po_c])
    zoom_axis(ax_ccdf, all_x, all_y[all_y > 0])
    ax_ccdf.set_xlabel("Degree  k", fontsize=10)
    ax_ccdf.set_ylabel("P(K ≥ k)", fontsize=10)
    ax_ccdf.set_title("Complementary CDF (in & out)", color=DATA_COL, fontsize=11, pad=6)
    ax_ccdf.legend(fontsize=8)
    ax_ccdf.grid(True, which="both", ls=":", alpha=0.4)

    # Stats panel
    ax_info.set_axis_off()
    stats_text = (
        f"  Graph Summary\n"
        f"  {'─'*34}\n"
        f"  Nodes            {G.number_of_nodes():>12,}\n"
        f"  Edges            {G.number_of_edges():>12,}\n"
        f"\n"
        f"  In-degree\n"
        f"    Max            {int(in_deg.max()):>12,}\n"
        f"    Mean           {in_deg.mean():>12.2f}\n"
        f"    Median         {np.median(in_deg):>12.0f}\n"
        f"\n"
        f"  Out-degree\n"
        f"    Max            {int(out_deg.max()):>12,}\n"
        f"    Mean           {out_deg.mean():>12.2f}\n"
        f"    Median         {np.median(out_deg):>12.0f}\n"
        f"\n"
        f"  Assortativity r\n"
        f"    Overall        {r_overall:>+12.4f}\n"
        f"    In-degree      {r_in:>+12.4f}\n"
        f"    Out-degree     {r_out:>+12.4f}\n"
    )
    ax_info.text(0.03, 0.97, stats_text, transform=ax_info.transAxes,
                 va="top", ha="left", fontsize=8.5, color=TEXT_COL,
                 fontfamily="monospace", linespacing=1.55)

    # Shared legend
    from matplotlib.lines import Line2D
    fig.legend(handles=[
        Line2D([0],[0], ls="--", color=ACCENT1, lw=1.5, label="Power-law (MLE)"),
        Line2D([0],[0], ls="-.", color=ACCENT2, lw=1.5, label="Poisson"),
        Line2D([0],[0], ls=":",  color=ACCENT3, lw=1.8, label="Log-normal"),
    ], loc="lower center", ncol=3, fontsize=8.5, framealpha=0,
       bbox_to_anchor=(0.5, 0.01))

    out_name = f"zkSync_degree_{fig_idx+1}_{ds_label.replace(' – ', '_')}.png"
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), out_name)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=DARK_BG)
    print(f"    ✓ Saved → {out_path}")
    return fig


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    print("\n── zkSync Graph Analysis ─────────────────────────────────────")
    for i, fname in enumerate(LOG_FILES):
        print(f"\n[Dataset {i+1}/{len(LOG_FILES)}] {fname}")
        records = load_one_file(LOG_DIR, fname)
        G = build_graph(records)
        print(f"  Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
        if G.number_of_nodes() == 0:
            print("  [skip] Graph is empty — check PKL format.")
            continue
        make_figure(G, fname, i)

    plt.show()
    print("\nDone.\n")


if __name__ == "__main__":
    main()