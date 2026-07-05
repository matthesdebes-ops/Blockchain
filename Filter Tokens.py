"""
filter_eth_transfers.py

Filters native ETH transfers out of the log dumps produced by
download_zkSync_logs.py.

Background
----------
On zkSync Era, native ETH is NOT a plain balance update like on Ethereum L1.
It is implemented as an ERC-20-like system contract (the "L2 base token"),
deployed at the fixed system address:

    0x000000000000000000000000000000000000800A

This means every native ETH transfer on zkSync Era *does* emit a standard
ERC-20 `Transfer(address indexed from, address indexed to, uint256 value)`
event, just like any other token -- the only difference is the emitting
contract address. Since download_zkSync_logs.py already pulled *all*
Transfer-topic logs (regardless of token), all we need to do here is filter
those logs down to the ones coming from the ETH system contract.

This script:
  1. Loads the final .pkl log files produced for each block range.
  2. Filters logs where `address` == the ETH system contract address.
  3. Saves the filtered ETH-only logs to new .pkl and .csv files.
"""

import os
import glob
import pickle
import csv
from typing import List, Dict, Tuple

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

INPUT_DIR = r"D:\zkSync_logs"
OUTPUT_DIR = r"D:\zkSync_logs\eth_only"

# zkSync Era native ETH / L2 base token system contract address.
# All logs whose `address` field equals this are native ETH transfers.
ETH_ADDRESS = "0x000000000000000000000000000000000000800a"


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def ensure_output_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)


def find_final_log_files() -> List[str]:
    """Find all merged/final log files produced by download_zkSync_logs.py."""
    pattern = os.path.join(INPUT_DIR, "logs_*_final.pkl")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No final log files found in {INPUT_DIR} matching 'logs_*_final.pkl'")
    return files


def load_logs(path: str) -> Tuple[Tuple[int, int], List[Dict]]:
    """Load a final log file and return (block_range, logs)."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    return tuple(data["block_range"]), data["logs"]


def filter_eth_logs(logs: List[Dict]) -> List[Dict]:
    """Keep only logs emitted by the ETH system contract."""
    eth_addr = ETH_ADDRESS.lower()
    return [log for log in logs if log.get("address", "").lower() == eth_addr]


def save_pickle(logs: List[Dict], block_range: Tuple[int, int]):
    start, end = block_range
    path = os.path.join(OUTPUT_DIR, f"eth_transfers_{start}_{end}.pkl")
    with open(path, "wb") as f:
        pickle.dump(
            {"block_range": block_range, "num_logs": len(logs), "logs": logs},
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  Saved {len(logs):,} ETH transfers -> {os.path.basename(path)} ({size_mb:.1f} MB)")


def save_csv(logs: List[Dict], block_range: Tuple[int, int]):
    start, end = block_range
    path = os.path.join(OUTPUT_DIR, f"eth_transfers_{start}_{end}.csv")
    fieldnames = [
        "blockNumber", "timestamp", "datetime",
        "transactionHash", "from", "to", "value", "logIndex", "address",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for log in logs:
            writer.writerow(log)
    print(f"  Saved CSV -> {os.path.basename(path)}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    ensure_output_dir()

    final_files = find_final_log_files()
    if not final_files:
        return

    total_all = 0
    total_eth = 0

    for path in final_files:
        print(f"\nProcessing {os.path.basename(path)}...")
        block_range, logs = load_logs(path)
        print(f"  Loaded {len(logs):,} total Transfer logs "
              f"(blocks {block_range[0]:,}-{block_range[1]:,})")

        eth_logs = filter_eth_logs(logs)
        pct = (len(eth_logs) / len(logs) * 100) if logs else 0
        print(f"  Found {len(eth_logs):,} native ETH transfers ({pct:.2f}% of all logs)")

        save_pickle(eth_logs, block_range)
        save_csv(eth_logs, block_range)

        total_all += len(logs)
        total_eth += len(eth_logs)

    print("\n" + "=" * 60)
    print("DONE")
    print(f"  Total logs scanned:     {total_all:,}")
    print(f"  Total ETH transfers:    {total_eth:,}")
    print(f"  Output directory:       {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()