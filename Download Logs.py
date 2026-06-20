"""
download_zkSync_logs.py

Download ERC-20 Transfer logs from zkSync Era for specific block ranges
and save them to files for later use.

Block ranges:
- Range 1: 60,000,000 - 70,000,000
- Range 2: 10,000,000 - 20,000,000

Saves: Block number, timestamp, from, to, value, transaction hash, etc.
"""

import asyncio
import aiohttp
import json
import os
import time
from datetime import datetime
from web3 import Web3
from typing import List, Tuple, Dict, Any, Optional
import pickle

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

RPC_URL = "https://mainnet.era.zksync.io"

# Block ranges to download
BLOCK_RANGES = [
    (60_000_000, 70_000_000),
    (10_000_000, 20_000_000),
]

# Chunk size for eth_getLogs calls
LOG_CHUNK_SIZE = 2000

# Max concurrent requests
CONCURRENCY = 50

# Transfer topic (ERC-20 Transfer event)
TRANSFER_TOPIC = Web3.keccak(text="Transfer(address,address,uint256)").hex()

# Optional: Filter for specific tokens (None = all tokens)
TOKEN_ADDRESSES = None  # Set to list of addresses to filter

# Output directory
OUTPUT_DIR = "zkSync_logs"


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------

def ensure_output_dir():
    """Create output directory if it doesn't exist"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)


def get_file_path(block_range: Tuple[int, int]) -> str:
    """Get file path for a specific block range"""
    start, end = block_range
    return os.path.join(OUTPUT_DIR, f"logs_{start // 1_000_000}M_{end // 1_000_000}M.pkl")


def save_logs_to_file(logs: List[Dict], block_range: Tuple[int, int]):
    """Save logs to a file using pickle"""
    file_path = get_file_path(block_range)

    # Prepare data to save
    data = {
        'block_range': block_range,
        'timestamp': datetime.now().isoformat(),
        'num_logs': len(logs),
        'logs': logs
    }

    # Save with pickle
    with open(file_path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
    print(f"Saved {len(logs):,} logs to {file_path} ({file_size:.1f} MB)")


def load_logs_from_file(block_range: Tuple[int, int]) -> Optional[List[Dict]]:
    """Load logs from a file if it exists"""
    file_path = get_file_path(block_range)

    if not os.path.exists(file_path):
        return None

    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    print(f"Loaded {data['num_logs']:,} logs from {file_path}")
    print(f"  Range: {data['block_range'][0]:,} - {data['block_range'][1]:,}")
    print(f"  Saved: {data['timestamp']}")
    return data['logs']


def generate_chunks(start_block: int, end_block: int, chunk_size: int) -> List[Tuple[int, int]]:
    """Generate block ranges for chunked queries"""
    chunks = []
    current = start_block
    while current <= end_block:
        chunk_end = min(current + chunk_size - 1, end_block)
        chunks.append((current, chunk_end))
        current = chunk_end + 1
    return chunks


# ----------------------------------------------------------------------
# Async RPC functions
# ----------------------------------------------------------------------

async def rpc_call(session, semaphore, method, params, request_id, timeout=60):
    """Make a JSON-RPC call with timeout"""
    body = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
    async with semaphore:
        try:
            async with session.post(
                    RPC_URL,
                    json=body,
                    timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                data = await response.json()
                if "error" in data:
                    print(f"RPC error for {request_id}: {data['error']}")
                    return None
                return data["result"]
        except asyncio.TimeoutError:
            print(f"Timeout for request {request_id}")
            return None
        except Exception as e:
            print(f"Error for {request_id}: {e}")
            return None


async def fetch_log_chunk(session, semaphore, from_block: int, to_block: int,
                          chunk_id: int) -> List[Dict]:
    """Fetch a chunk of logs"""
    log_filter = {
        "fromBlock": hex(from_block),
        "toBlock": hex(to_block),
        "topics": [TRANSFER_TOPIC],
    }
    if TOKEN_ADDRESSES:
        log_filter["address"] = TOKEN_ADDRESSES

    result = await rpc_call(
        session, semaphore, "eth_getLogs", [log_filter], f"chunk-{chunk_id}"
    )

    if result is None:
        return []

    return result


async def fetch_block_timestamp(session, semaphore, block_number: int) -> Optional[int]:
    """Fetch timestamp for a specific block"""
    result = await rpc_call(
        session, semaphore, "eth_getBlockByNumber",
        [hex(block_number), False], f"block-{block_number}"
    )

    if result and "timestamp" in result:
        return int(result["timestamp"], 16)
    return None


# ----------------------------------------------------------------------
# Process and enrich logs
# ----------------------------------------------------------------------

def enrich_logs_with_info(logs: List[Dict]) -> List[Dict]:
    """Extract and organize important information from logs"""
    enriched = []

    for log in logs:
        topics = log.get("topics", [])
        if len(topics) < 3:
            continue

        try:
            # Extract address info
            from_addr = "0x" + topics[1][-40:]
            to_addr = "0x" + topics[2][-40:]

            # Extract value (amount)
            value_hex = log.get("data", "0x")
            value = int(value_hex, 16) if value_hex and value_hex != "0x" else 0

            # Get block number
            block_number = int(log.get("blockNumber", "0x0"), 16)

            # Get transaction hash
            tx_hash = log.get("transactionHash", "")

            # Create enriched log entry
            enriched_log = {
                "blockNumber": block_number,
                "transactionHash": tx_hash,
                "from": from_addr,
                "to": to_addr,
                "value": value,
                "logIndex": int(log.get("logIndex", "0x0"), 16),
                "address": log.get("address", ""),  # Token contract address
                # Keep raw data for reference
                "raw": log
            }

            enriched.append(enriched_log)

        except Exception as e:
            # Skip malformed logs
            continue

    return enriched


async def enrich_with_timestamps(session, semaphore, logs: List[Dict]) -> List[Dict]:
    """Add block timestamps to logs"""
    if not logs:
        return logs

    # Get unique block numbers
    block_numbers = set(log["blockNumber"] for log in logs)

    print(f"Fetching timestamps for {len(block_numbers)} unique blocks...")

    # Create tasks for fetching block timestamps
    tasks = [
        fetch_block_timestamp(session, semaphore, bn)
        for bn in block_numbers
    ]

    # Execute all tasks
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Map block numbers to timestamps
    timestamp_map = {}
    for bn, result in zip(block_numbers, results):
        if isinstance(result, Exception):
            print(f"  Failed to get timestamp for block {bn}: {result}")
            continue
        if result is not None:
            timestamp_map[bn] = result

    # Add timestamps to logs
    for log in logs:
        log["timestamp"] = timestamp_map.get(log["blockNumber"])
        if log["timestamp"]:
            log["datetime"] = datetime.fromtimestamp(log["timestamp"]).isoformat()

    print(f"Added timestamps to {len(timestamp_map)} blocks")

    return logs


# ----------------------------------------------------------------------
# Main download function
# ----------------------------------------------------------------------

async def download_range_async(block_range: Tuple[int, int]) -> List[Dict]:
    """Asynchronously download all logs for a block range"""
    start_block, end_block = block_range
    chunks = generate_chunks(start_block, end_block, LOG_CHUNK_SIZE)

    print(f"\n{'=' * 60}")
    print(f"Downloading blocks {start_block:,} to {end_block:,}")
    print(f"Total chunks: {len(chunks)} (size: {LOG_CHUNK_SIZE} blocks each)")
    print(f"Concurrency: {CONCURRENCY}")
    print(f"{'=' * 60}")

    start_time = time.time()

    semaphore = asyncio.Semaphore(CONCURRENCY)

    async with aiohttp.ClientSession() as session:
        # Download all log chunks
        log_tasks = [
            fetch_log_chunk(session, semaphore, from_b, to_b, i)
            for i, (from_b, to_b) in enumerate(chunks)
        ]

        log_results = await asyncio.gather(*log_tasks, return_exceptions=True)

        # Process log results
        all_logs = []
        failed_chunks = 0

        for i, result in enumerate(log_results):
            if isinstance(result, Exception):
                print(f"  Chunk {i} failed: {result}")
                failed_chunks += 1
                continue
            if result is None:
                print(f"  Chunk {i} returned None")
                failed_chunks += 1
                continue
            all_logs.extend(result)

        print(f"Downloaded {len(all_logs):,} raw logs")

        # Enrich logs with extracted information
        print("Enriching logs with address and value info...")
        enriched_logs = enrich_logs_with_info(all_logs)
        print(f"Enriched {len(enriched_logs):,} logs")

        # Add block timestamps
        print("Adding block timestamps...")
        enriched_logs = await enrich_with_timestamps(session, semaphore, enriched_logs)

        elapsed = time.time() - start_time

        print(f"\n{'=' * 60}")
        print(f"Download complete!")
        print(f"  Blocks: {start_block:,} - {end_block:,}")
        print(f"  Total logs: {len(enriched_logs):,}")
        print(f"  Successful chunks: {len(chunks) - failed_chunks}/{len(chunks)}")
        print(f"  Time: {elapsed:.1f} seconds ({elapsed / 60:.1f} minutes)")
        print(f"  Speed: {len(enriched_logs) / elapsed:.1f} logs/second")
        print(f"{'=' * 60}\n")

        return enriched_logs


async def download_all_ranges():
    """Download all block ranges and save to files"""
    ensure_output_dir()

    results = {}

    for block_range in BLOCK_RANGES:
        start_block, end_block = block_range
        file_path = get_file_path(block_range)

        # Check if file already exists
        if os.path.exists(file_path):
            print(f"\nFile {file_path} already exists.")
            print("Loading existing data...")
            logs = load_logs_from_file(block_range)
            if logs is not None:
                results[block_range] = logs
                continue
            else:
                print("Could not load existing file. Re-downloading...")

        # Download the range
        logs = await download_range_async(block_range)

        # Save to file
        save_logs_to_file(logs, block_range)

        results[block_range] = logs

    # Print summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    total_logs = 0
    for block_range, logs in results.items():
        start, end = block_range
        print(f"Range {start:,} - {end:,}: {len(logs):,} logs")
        total_logs += len(logs)

        # Show sample of enriched log
        if logs:
            sample = logs[0]
            print(f"  Sample log:")
            print(f"    Block: {sample.get('blockNumber', 'N/A')}")
            print(f"    Timestamp: {sample.get('datetime', 'N/A')}")
            print(f"    From: {sample.get('from', 'N/A')}")
            print(f"    To: {sample.get('to', 'N/A')}")
            print(f"    Value: {sample.get('value', 'N/A')}")
            print(f"    Tx: {sample.get('transactionHash', 'N/A')[:10]}...")
    print(f"Total logs downloaded: {total_logs:,}")
    print("=" * 60)


# ----------------------------------------------------------------------
# Main execution
# ----------------------------------------------------------------------

def main():
    print(f"Starting download at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"RPC URL: {RPC_URL}")
    print(f"Block ranges: {BLOCK_RANGES}")
    print(f"Concurrency: {CONCURRENCY}")
    print(f"Chunk size: {LOG_CHUNK_SIZE}")
    if TOKEN_ADDRESSES:
        print(f"Filtering tokens: {TOKEN_ADDRESSES}")
    else:
        print("No token filter (all ERC-20 transfers)")
    print()

    # Run the download
    asyncio.run(download_all_ranges())

    print(f"\nDownload complete at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Files saved in {OUTPUT_DIR}/ directory")
    print("\nTo load the data later use:")
    print("  import pickle")
    print("  with open('zkSync_logs/logs_10M_20M.pkl', 'rb') as f:")
    print("      data = pickle.load(f)")
    print("      logs = data['logs']")


if __name__ == "__main__":
    main()