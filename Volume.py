"""
analyze_transaction_volume.py

Analyze transaction volume distribution in zkSync.
Replicates Section 3.6 of the paper.
"""

import pickle
import os
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import time
from scipy import stats
from collections import defaultdict

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
    """Load logs from a pickle file"""
    full_path = os.path.join(LOG_DIR, file_path)
    if not os.path.exists(full_path):
        print(f"File {full_path} not found!")
        return []

    with open(full_path, 'rb') as f:
        data = pickle.load(f)

    print(f"Loaded {data['num_logs']:,} logs from {file_path}")
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
# 2. Extract transaction volumes
# ----------------------------------------------------------------------

def extract_transaction_volumes(logs: List[Dict]) -> List[float]:
    """
    Extract transaction volumes (edge weights).
    Returns list of all transaction amounts.
    """
    print("\nExtracting transaction volumes...")

    volumes = []
    for log in logs:
        value = log.get('value', 0)
        if value > 0:
            volumes.append(value)

    print(f"  Extracted {len(volumes):,} transactions")
    print(f"  Total volume: {sum(volumes):,.2f} ETH")
    print(f"  Average: {np.mean(volumes):.2f} ETH")
    print(f"  Median: {np.median(volumes):.2f} ETH")
    print(f"  Max: {max(volumes):.2f} ETH")
    print(f"  Min: {min(volumes):.2f} ETH")

    return volumes


# ----------------------------------------------------------------------
# 3. Transaction volume analysis (Section 3.6)
# ----------------------------------------------------------------------

def analyze_transaction_volumes(volumes: List[float], label: str = ""):
    """
    Analyze transaction volume distribution.
    Replicates Section 3.6: Transaction Volume
    """

    print("\n" + "=" * 70)
    print("TRANSACTION VOLUME ANALYSIS (Section 3.6)")
    print("=" * 70)

    volumes = np.array(volumes)
    volumes = volumes[volumes > 0]  # Remove zero transactions

    if len(volumes) == 0:
        print("No transaction volumes found!")
        return

    # Basic statistics
    print("\n" + "-" * 50)
    print("VOLUME STATISTICS")
    print("-" * 50)
    print(f"Number of transactions: {len(volumes):,}")
    print(f"Total volume: {sum(volumes):,.2f} ETH")
    print(f"Mean: {np.mean(volumes):.2f} ETH")
    print(f"Median: {np.median(volumes):.2f} ETH")
    print(f"Max: {max(volumes):.2f} ETH")
    print(f"Min: {min(volumes):.2f} ETH")
    print(f"Std dev: {np.std(volumes):.2f} ETH")

    # Quantiles
    quantiles = [0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    print("\nQuantiles:")
    for q in quantiles:
        val = np.percentile(volumes, q * 100)
        print(f"  {q * 100:5.0f}th percentile: {val:,.2f} ETH")

    # Power law analysis
    print("\n" + "-" * 50)
    print("POWER LAW ANALYSIS")
    print("-" * 50)

    # Fit power law
    log_volumes = np.log10(volumes)

    # Try different xmin values
    best_alpha = 0
    best_xmin = 0
    best_ks = np.inf

    for percentile in [50, 60, 70, 80, 85, 90, 95]:
        xmin = np.percentile(volumes, percentile)
        data_fit = volumes[volumes >= xmin]

        if len(data_fit) < 10:
            continue

        # MLE for alpha
        alpha = 1 + len(data_fit) / np.sum(np.log(data_fit / xmin))

        # KS statistic
        sorted_data = np.sort(data_fit)
        cdf_emp = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        cdf_theory = 1 - (sorted_data / xmin) ** (-alpha + 1)
        ks_stat = np.max(np.abs(cdf_emp - cdf_theory))

        if ks_stat < best_ks:
            best_ks = ks_stat
            best_alpha = alpha
            best_xmin = xmin

    if best_alpha > 0:
        print(f"Power law fit:")
        print(f"  α = {best_alpha:.3f}")
        print(f"  xmin = {best_xmin:.2f} ETH")
        print(f"  KS statistic = {best_ks:.4f}")

        # Check if power law is good fit
        if best_ks < 0.1:
            print("  ✓ Good power law fit (KS < 0.1)")
        elif best_ks < 0.2:
            print("  ~ Moderate power law fit (KS between 0.1 and 0.2)")
        else:
            print("  ✗ Poor power law fit (KS > 0.2)")

    # Check for log-normal (as paper mentions)
    print("\nLog-normal fit:")
    log_data = np.log(volumes)
    mu = np.mean(log_data)
    sigma = np.std(log_data)
    print(f"  μ = {mu:.3f}")
    print(f"  σ = {sigma:.3f}")

    # Create plots (like paper's Figure 8)
    plot_volume_distribution(volumes, best_alpha, best_xmin, mu, sigma, label)

    # Compare distributions
    compare_distributions(volumes, label)

    return volumes


def plot_volume_distribution(volumes, alpha, xmin, mu, sigma, label):
    """Plot transaction volume distribution like Figure 8 in the paper"""

    # Filter positive values
    volumes = [v for v in volumes if v > 0]

    if not volumes:
        print("No positive volumes to plot!")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Distribution (log-log)
    ax1 = axes[0]

    # Log-binning
    log_volumes = np.log10(volumes)
    hist, bins = np.histogram(log_volumes, bins=50)
    centers = (bins[:-1] + bins[1:]) / 2
    valid = hist > 0

    ax1.scatter(centers[valid], np.log10(hist[valid]),
                color='red', s=30, alpha=0.6, label='Data')

    # Power law fit line
    if alpha > 0 and xmin > 0:
        x_fit = np.linspace(np.log10(xmin), max(centers[valid]), 100)
        # Normalization constant
        norm_const = len([v for v in volumes if v >= xmin]) * (alpha - 1) * xmin ** (alpha - 1)
        y_fit = np.log10(norm_const) - alpha * x_fit

        ax1.plot(x_fit, y_fit, 'b--', linewidth=2,
                 label=f'Power law: α={alpha:.3f}')

    # Log-normal fit (approximate)
    if mu and sigma:
        from scipy.stats import lognorm
        x_vals = np.logspace(min(centers[valid]), max(centers[valid]), 100)
        pdf = lognorm.pdf(x_vals, sigma, scale=np.exp(mu))
        y_vals = np.log10(
            pdf * len(volumes) * np.diff(np.logspace(min(centers[valid]), max(centers[valid]), 100)).mean())
        ax1.plot(np.log10(x_vals), y_vals, 'g--', linewidth=2,
                 label=f'Log-normal: μ={mu:.2f}, σ={sigma:.2f}')

    ax1.set_xlabel('log10(Transaction Volume in ETH)', fontsize=12)
    ax1.set_ylabel('log10(Count)', fontsize=12)
    ax1.set_title('Transaction Volume Distribution', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: CCDF (log-log)
    ax2 = axes[1]
    sorted_volumes = np.sort(volumes)[::-1]
    ccdf = np.arange(1, len(sorted_volumes) + 1) / len(sorted_volumes)

    log_sorted = np.log10(sorted_volumes)
    log_ccdf = np.log10(ccdf)

    ax2.scatter(log_sorted, log_ccdf, color='red', s=30, alpha=0.6, label='CCDF')

    # Power law fit for CCDF
    if alpha > 0 and xmin > 0:
        x_fit = np.linspace(np.log10(xmin), max(log_sorted), 100)
        y_fit = -(alpha - 1) * (x_fit - np.log10(xmin))  # CCDF exponent is alpha-1
        ax2.plot(x_fit, y_fit, 'b--', linewidth=2,
                 label=f'Power law: α={alpha:.3f}')

    # Log-normal CCDF
    if mu and sigma:
        from scipy.stats import lognorm
        x_vals = np.logspace(min(log_sorted), max(log_sorted), 100)
        ccdf_vals = 1 - lognorm.cdf(x_vals, sigma, scale=np.exp(mu))
        ax2.plot(np.log10(x_vals), np.log10(ccdf_vals + 1e-10), 'g--', linewidth=2,
                 label=f'Log-normal: μ={mu:.2f}, σ={sigma:.2f}')

    ax2.set_xlabel('log10(Transaction Volume in ETH)', fontsize=12)
    ax2.set_ylabel('log10(P(Volume > v))', fontsize=12)
    ax2.set_title('Complementary CDF of Transaction Volumes', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    save_path = f"transaction_volume_{label}.png" if label else "transaction_volume.png"
    plt.savefig(save_path, dpi=150)
    print(f"\n  Volume plot saved to: {save_path}")
    plt.close()


def compare_distributions(volumes, label):
    """Compare different distribution fits (like paper's LR tests)"""

    print("\n" + "-" * 50)
    print("DISTRIBUTION COMPARISON (Likelihood Ratio Tests)")
    print("-" * 50)

    volumes = np.array([v for v in volumes if v > 0])
    log_volumes = np.log(volumes)

    # Fit distributions
    # Power law
    alpha_pl, xmin_pl = fit_power_law(volumes)
    logL_pl = calculate_power_law_logL(volumes, alpha_pl, xmin_pl)

    # Exponential
    rate_exp = 1 / np.mean(volumes)
    logL_exp = len(volumes) * np.log(rate_exp) - rate_exp * np.sum(volumes)

    # Log-normal
    mu_ln = np.mean(log_volumes)
    sigma_ln = np.std(log_volumes)
    logL_ln = -len(volumes) / 2 * np.log(2 * np.pi) - len(volumes) * np.log(sigma_ln) - np.sum(
        (log_volumes - mu_ln) ** 2) / (2 * sigma_ln ** 2)

    print(f"\nLog-likelihoods:")
    print(f"  Power law:        {logL_pl:.2f}")
    print(f"  Exponential:      {logL_exp:.2f}")
    print(f"  Log-normal:       {logL_ln:.2f}")

    # Likelihood Ratio Tests
    print(f"\nLikelihood Ratio Tests (vs Power Law):")

    # Power Law vs Exponential
    LR_exp = 2 * (logL_pl - logL_exp)
    p_exp = 1 - stats.chi2.cdf(LR_exp, 1)
    print(f"\n  Exponential:")
    print(f"    LR statistic: {LR_exp:.3f}")
    print(f"    p-value: {p_exp:.4f}")
    print(f"    {'Better' if logL_pl > logL_exp else 'Worse'} than Power Law")

    # Power Law vs Log-normal
    LR_ln = 2 * (logL_pl - logL_ln)
    p_ln = 1 - stats.chi2.cdf(LR_ln, 2)
    print(f"\n  Log-normal:")
    print(f"    LR statistic: {LR_ln:.3f}")
    print(f"    p-value: {p_ln:.4f}")
    print(f"    {'Better' if logL_pl > logL_ln else 'Worse'} than Power Law")

    # Determine best fit
    logLs = {'Power Law': logL_pl, 'Exponential': logL_exp, 'Log-normal': logL_ln}
    best = max(logLs, key=logLs.get)
    print(f"\nBest fit: {best}")

    # Interpret (like paper)
    print("\nInterpretation (following paper):")
    if best == "Power Law":
        print("  Power law provides the best fit")
        print("  Consistent with paper's finding of heavy-tailed distribution")
    elif best == "Log-normal":
        print("  Log-normal provides the best fit")
        print("  Similar to paper's finding that log-normal outperforms pure power law")
    else:
        print("  Exponential provides the best fit")
        print("  Different from paper's findings")


def fit_power_law(data, xmin=None):
    """Fit power law using MLE"""
    data = np.array(data)
    data = data[data > 0]

    if xmin is None:
        # Try different xmin values
        best_alpha = 0
        best_xmin = np.min(data)
        best_ks = np.inf

        # Test percentiles
        for p in [0, 10, 20, 30, 40, 50, 60, 70, 80]:
            xmin_test = np.percentile(data, p)
            data_fit = data[data >= xmin_test]

            if len(data_fit) < 10:
                continue

            alpha_test = 1 + len(data_fit) / np.sum(np.log(data_fit / xmin_test))

            # KS test
            sorted_data = np.sort(data_fit)
            cdf_emp = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            cdf_theory = 1 - (sorted_data / xmin_test) ** (-alpha_test + 1)
            ks_stat = np.max(np.abs(cdf_emp - cdf_theory))

            if ks_stat < best_ks:
                best_ks = ks_stat
                best_alpha = alpha_test
                best_xmin = xmin_test

        return best_alpha, best_xmin

    else:
        data_fit = data[data >= xmin]
        if len(data_fit) < 10:
            return 0, xmin
        alpha = 1 + len(data_fit) / np.sum(np.log(data_fit / xmin))
        return alpha, xmin


def calculate_power_law_logL(data, alpha, xmin):
    """Calculate log-likelihood for power law"""
    data = np.array(data)
    data_fit = data[data >= xmin]
    if len(data_fit) < 2:
        return -np.inf
    logL = len(data_fit) * np.log(alpha - 1) - len(data_fit) * np.log(xmin) - alpha * np.sum(np.log(data_fit / xmin))
    return logL


# ----------------------------------------------------------------------
# 4. Main
# ----------------------------------------------------------------------

def main():
    print("=" * 70)
    print("ZKSYNC TRANSACTION VOLUME ANALYSIS")
    print("=" * 70)

    all_logs = load_all_logs()

    if not all_logs:
        print("No logs found!")
        return

    for filename, logs in all_logs.items():
        label = filename.replace('.pkl', '')
        volumes = extract_transaction_volumes(logs)
        analyze_transaction_volumes(volumes, label)

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()