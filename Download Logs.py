"""
download_zkSync_logs.py

Download ERC-20 Transfer logs from zkSync Era for specific block ranges
and save them to files for later use.

Block ranges:
- Range 1: 60,000,000 - 70,000,000
- Range 2: 10,000,000 - 20,000,000

Saves: Block number, timestamp, from, to, value, transaction hash, etc.

BATCH SAVING: Saves every BATCH_SAVE_BLOCKS blocks to disk so you never
lose progress and RAM stays bounded. Resumes from where it left off.

FAST TIMESTAMPS: Uses JSON-RPC batch requests to fetch up to 100 block
timestamps per HTTP round-trip instead of one request per block (~50x faster).
"""

import asyncio
import aiohttp
import json
import os
import re
import time
from datetime import datetime
from web3 import Web3
from typing import List, Tuple, Dict, Any, Optional
import pickle
import glob

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

RPC_URL = "https://mainnet.era.zksync.io"

# Block ranges to download
BLOCK_RANGES = [
    (69_000_000, 70_000_000),
    (14_000_000, 15_000_000),
]

# How many blocks to accumulate before flushing a batch file to disk.
# Lower = less RAM used, more small files on disk.
# 50_000 blocks ~ a few hundred MB RAM for dense ranges; tune down if needed.
BATCH_SAVE_BLOCKS = 50_000

# Initial chunk size for eth_getLogs calls
# Keep small — zkSync's 10k result cap triggers on dense ranges
LOG_CHUNK_SIZE = 200

# Max concurrent requests — public RPC is sensitive; keep this low
CONCURRENCY = 3

# Retry settings
MAX_RETRIES = 6          # attempts before giving up
BASE_BACKOFF = 2.0       # seconds; doubles each retry
MAX_BACKOFF = 60.0       # cap on wait time

# Transfer topic (ERC-20 Transfer event)
TRANSFER_TOPIC = Web3.keccak(text="Transfer(address,address,uint256)").hex()

# Optional: Filter for specific tokens (None = all tokens)
TOKEN_ADDRESSES = None  # Set to list of addresses to filter

# Output directory
OUTPUT_DIR = r"D:\zkSync_logs"

# How many block timestamp lookups to bundle into a single HTTP request.
# 100 is a safe default; lower if the RPC starts rejecting large batch payloads.
TIMESTAMP_BATCH_SIZE = 100

# Print progress every N completed chunks
PROGRESS_EVERY = 50


# ----------------------------------------------------------------------
# Progress state (updated by fetch_log_chunk)
# ----------------------------------------------------------------------

_progress_completed = 0
_progress_total = 0
_progress_logs = 0
_progress_start = 0.0


def _tick(logs_in_chunk: int):
    """Increment progress counters and print if milestone reached."""
    global _progress_completed, _progress_logs
    _progress_completed += 1
    _progress_logs += logs_in_chunk

    if _progress_completed % PROGRESS_EVERY == 0 or _progress_completed == _progress_total:
        elapsed = time.time() - _progress_start
        pct = _progress_completed / _progress_total * 100 if _progress_total else 0
        rate = _progress_completed / elapsed if elapsed > 0 else 0
        eta = (_progress_total - _progress_completed) / rate if rate > 0 else 0
        print(
            f"  [{_progress_completed:,}/{_progress_total:,} chunks | "
            f"{pct:.1f}% | "
            f"{rate:.1f} chunks/s | "
            f"ETA {eta / 60:.1f} min | "
            f"{_progress_logs:,} logs so far]"
        )


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------

def ensure_output_dir():
    """Create output directory if it doesn't exist"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)


def get_final_file_path(block_range: Tuple[int, int]) -> str:
    """Path for the merged final file for a block range."""
    start, end = block_range
    return os.path.join(OUTPUT_DIR, f"logs_{start}_{end}_final.pkl")


def get_batch_file_path(block_range: Tuple[int, int], batch_start: int, batch_end: int) -> str:
    """Path for an individual batch file within a range."""
    range_start, range_end = block_range
    return os.path.join(
        OUTPUT_DIR,
        f"batch_{range_start}_{range_end}__{batch_start}_{batch_end}.pkl"
    )


def get_progress_file_path(block_range: Tuple[int, int]) -> str:
    """Path for the progress/resume file for a block range."""
    start, end = block_range
    return os.path.join(OUTPUT_DIR, f"progress_{start}_{end}.json")


def save_batch(logs: List[Dict], block_range: Tuple[int, int],
               batch_start: int, batch_end: int):
    """Save a batch of logs to a numbered batch file."""
    file_path = get_batch_file_path(block_range, batch_start, batch_end)
    data = {
        'block_range': block_range,
        'batch_blocks': (batch_start, batch_end),
        'timestamp': datetime.now().isoformat(),
        'num_logs': len(logs),
        'logs': logs,
    }
    with open(file_path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"  💾 Batch saved: blocks {batch_start:,}–{batch_end:,} | "
          f"{len(logs):,} logs | {size_mb:.1f} MB → {os.path.basename(file_path)}")


def save_progress(block_range: Tuple[int, int], last_completed_block: int):
    """Save resume checkpoint so we can skip already-done batches."""
    path = get_progress_file_path(block_range)
    with open(path, 'w') as f:
        json.dump({
            'block_range': list(block_range),
            'last_completed_block': last_completed_block,
            'saved_at': datetime.now().isoformat(),
        }, f, indent=2)


def load_progress(block_range: Tuple[int, int]) -> Optional[int]:
    """Return the last completed block for this range, or None if starting fresh."""
    path = get_progress_file_path(block_range)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    last = data.get('last_completed_block')
    print(f"  ▶ Resuming from block {last:,} (progress file found)")
    return last


def list_batch_files(block_range: Tuple[int, int]) -> List[str]:
    """Return sorted list of batch files for a range."""
    start, end = block_range
    pattern = os.path.join(OUTPUT_DIR, f"batch_{start}_{end}__*.pkl")
    return sorted(glob.glob(pattern))


def merge_batches(block_range: Tuple[int, int]) -> List[Dict]:
    """Load and merge all batch files for a range into one list."""
    batch_files = list_batch_files(block_range)
    if not batch_files:
        return []

    print(f"\nMerging {len(batch_files)} batch files for range "
          f"{block_range[0]:,}–{block_range[1]:,}...")

    all_logs = []
    for path in batch_files:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        all_logs.extend(data['logs'])
        print(f"  Loaded {data['num_logs']:,} logs from {os.path.basename(path)}")

    print(f"  Total after merge: {len(all_logs):,} logs")
    return all_logs


def save_final(logs: List[Dict], block_range: Tuple[int, int]):
    """Save the merged final file and clean up batch files + progress file."""
    final_path = get_final_file_path(block_range)
    data = {
        'block_range': block_range,
        'timestamp': datetime.now().isoformat(),
        'num_logs': len(logs),
        'logs': logs,
    }
    with open(final_path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = os.path.getsize(final_path) / (1024 * 1024)
    print(f"\n✅ Final file saved: {final_path} ({size_mb:.1f} MB, {len(logs):,} logs)")

    # Clean up batch files
    batch_files = list_batch_files(block_range)
    for path in batch_files:
        os.remove(path)
    print(f"  Cleaned up {len(batch_files)} batch files")

    # Clean up progress file
    progress_path = get_progress_file_path(block_range)
    if os.path.exists(progress_path):
        os.remove(progress_path)


def load_final(block_range: Tuple[int, int]) -> Optional[List[Dict]]:
    """Load the final merged file if it exists."""
    path = get_final_file_path(block_range)
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded {data['num_logs']:,} logs from {path}")
    print(f"  Range: {data['block_range'][0]:,} – {data['block_range'][1]:,}")
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


def parse_suggested_range(message: str) -> Optional[Tuple[int, int]]:
    """
    Extract the suggested block range from a zkSync -32602 error message.
    Example: 'Try with this block range [0x395b1b0, 0x395b35b]'
    """
    match = re.search(r'\[(\s*0x[0-9a-fA-F]+)\s*,\s*(0x[0-9a-fA-F]+)\s*\]', message)
    if match:
        return int(match.group(1), 16), int(match.group(2), 16)
    return None


# ----------------------------------------------------------------------
# Async RPC functions
# ----------------------------------------------------------------------

async def rpc_call_raw(session: aiohttp.ClientSession, method: str, params: list,
                       request_id: str, timeout: int = 60) -> Dict:
    body = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
    async with session.post(
        RPC_URL,
        json=body,
        timeout=aiohttp.ClientTimeout(total=timeout)
    ) as response:
        if response.status == 429:
            retry_after = float(response.headers.get("Retry-After", 0))
            raise aiohttp.ClientResponseError(
                response.request_info, response.history,
                status=429, message=f"Rate limited (Retry-After: {retry_after}s)"
            )
        response.raise_for_status()
        return await response.json(content_type=None)


async def rpc_call(session: aiohttp.ClientSession, semaphore: asyncio.Semaphore,
                   method: str, params: list, request_id: str,
                   timeout: int = 60) -> Optional[Any]:
    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                data = await rpc_call_raw(session, method, params, request_id, timeout)

                if "error" in data:
                    return data  # Pass -32602 etc. back to caller

                return data["result"]

            except aiohttp.ClientResponseError as e:
                if e.status == 429:
                    wait = max(
                        BASE_BACKOFF * (2 ** attempt),
                        float(getattr(e, 'retry_after', 0) or 0)
                    )
                    wait = min(wait, MAX_BACKOFF)
                    print(f"  429 on {request_id} (attempt {attempt + 1}/{MAX_RETRIES}), "
                          f"waiting {wait:.1f}s...")
                    await asyncio.sleep(wait)
                    continue
                print(f"  HTTP {e.status} for {request_id}: {e}")
                return None

            except asyncio.TimeoutError:
                wait = BASE_BACKOFF * (2 ** attempt)
                print(f"  Timeout on {request_id} (attempt {attempt + 1}/{MAX_RETRIES}), "
                      f"waiting {wait:.1f}s...")
                await asyncio.sleep(min(wait, MAX_BACKOFF))

            except Exception as e:
                wait = BASE_BACKOFF * (2 ** attempt)
                print(f"  Error on {request_id}: {e} (attempt {attempt + 1}/{MAX_RETRIES})")
                await asyncio.sleep(min(wait, MAX_BACKOFF))

        print(f"  Giving up on {request_id} after {MAX_RETRIES} attempts")
        return None


async def fetch_log_chunk(session: aiohttp.ClientSession, semaphore: asyncio.Semaphore,
                          from_block: int, to_block: int, chunk_id: str,
                          depth: int = 0) -> List[Dict]:
    MAX_DEPTH = 20

    if depth > MAX_DEPTH:
        print(f"  Max recursion depth reached for chunk {chunk_id} — skipping")
        return []

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

    if isinstance(result, dict) and "error" in result:
        err = result["error"]
        code = err.get("code")
        message = err.get("message", "")

        if code == -32602 and "Try with this block range" in message:
            suggested = parse_suggested_range(message)
            split_at = suggested[1] if suggested else (from_block + to_block) // 2

            if split_at >= to_block or split_at < from_block:
                split_at = (from_block + to_block) // 2

            if from_block == to_block:
                print(f"  Single block {from_block} has >10k logs, skipping")
                return []

            left, right = await asyncio.gather(
                fetch_log_chunk(session, semaphore, from_block, split_at,
                                f"{chunk_id}L", depth + 1),
                fetch_log_chunk(session, semaphore, split_at + 1, to_block,
                                f"{chunk_id}R", depth + 1),
            )
            return left + right

        print(f"  RPC error for chunk-{chunk_id}: {err}")
        _tick(0)
        return []

    _tick(len(result))
    return result


# ----------------------------------------------------------------------
# Process and enrich logs
# ----------------------------------------------------------------------

def enrich_logs_with_info(logs: List[Dict]) -> List[Dict]:
    enriched = []
    for log in logs:
        topics = log.get("topics", [])
        if len(topics) < 3:
            continue
        try:
            from_addr = "0x" + topics[1][-40:]
            to_addr = "0x" + topics[2][-40:]
            value_hex = log.get("data", "0x")
            value = int(value_hex, 16) if value_hex and value_hex != "0x" else 0
            block_number = int(log.get("blockNumber", "0x0"), 16)
            tx_hash = log.get("transactionHash", "")
            enriched.append({
                "blockNumber": block_number,
                "transactionHash": tx_hash,
                "from": from_addr,
                "to": to_addr,
                "value": value,
                "logIndex": int(log.get("logIndex", "0x0"), 16),
                "address": log.get("address", ""),
                "raw": log,
            })
        except Exception:
            continue
    return enriched


async def fetch_timestamps_batch(session: aiohttp.ClientSession,
                                 semaphore: asyncio.Semaphore,
                                 block_numbers: List[int],
                                 batch_size: int = 100) -> Dict[int, int]:
    """
    Fetch timestamps for many blocks using JSON-RPC batch requests.
    Sends up to `batch_size` requests in a single HTTP call instead of
    one request per block — typically 10–50x faster than the naive approach.

    Returns a dict mapping block_number -> unix timestamp.
    """
    results: Dict[int, int] = {}
    block_numbers = list(block_numbers)
    total_batches = (len(block_numbers) + batch_size - 1) // batch_size

    for batch_idx in range(0, len(block_numbers), batch_size):
        chunk = block_numbers[batch_idx:batch_idx + batch_size]

        payload = [
            {
                "jsonrpc": "2.0",
                "id": bn,                        # use block number as id for easy mapping
                "method": "eth_getBlockByNumber",
                "params": [hex(bn), False],       # False = don't return full tx list
            }
            for bn in chunk
        ]

        # Retry loop (mirrors rpc_call retry logic)
        for attempt in range(MAX_RETRIES):
            try:
                async with semaphore:
                    async with session.post(
                        RPC_URL,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=60)
                    ) as response:
                        if response.status == 429:
                            wait = min(BASE_BACKOFF * (2 ** attempt), MAX_BACKOFF)
                            retry_after = float(response.headers.get("Retry-After", 0))
                            wait = max(wait, retry_after)
                            print(f"  429 on timestamp batch (attempt {attempt + 1}), "
                                  f"waiting {wait:.1f}s...")
                            await asyncio.sleep(wait)
                            continue
                        response.raise_for_status()
                        data = await response.json(content_type=None)

                # data is a list of RPC responses, one per request in the batch
                if isinstance(data, list):
                    for item in data:
                        if not isinstance(item, dict):
                            continue
                        if "error" in item:
                            continue   # skip failed individual block lookups
                        result = item.get("result")
                        if result and isinstance(result, dict) and "timestamp" in result:
                            bn = item["id"]
                            results[bn] = int(result["timestamp"], 16)
                break  # success — exit retry loop

            except asyncio.TimeoutError:
                wait = min(BASE_BACKOFF * (2 ** attempt), MAX_BACKOFF)
                print(f"  Timeout on timestamp batch {batch_idx // batch_size + 1} "
                      f"(attempt {attempt + 1}), waiting {wait:.1f}s...")
                await asyncio.sleep(wait)

            except Exception as e:
                wait = min(BASE_BACKOFF * (2 ** attempt), MAX_BACKOFF)
                print(f"  Error on timestamp batch {batch_idx // batch_size + 1}: "
                      f"{e} (attempt {attempt + 1})")
                await asyncio.sleep(wait)
        else:
            print(f"  Giving up on timestamp batch {batch_idx // batch_size + 1} "
                  f"after {MAX_RETRIES} attempts — timestamps will be missing for "
                  f"{len(chunk)} blocks")

    return results


async def enrich_with_timestamps(session: aiohttp.ClientSession,
                                 semaphore: asyncio.Semaphore,
                                 logs: List[Dict]) -> List[Dict]:
    """
    Add block timestamps to logs using batched RPC calls (100 blocks per request).
    Much faster than the previous one-request-per-block approach.
    """
    if not logs:
        return logs

    block_numbers = list(set(log["blockNumber"] for log in logs))
    n_batches = (len(block_numbers) + 99) // 100
    print(f"  Fetching timestamps for {len(block_numbers):,} unique blocks "
          f"({n_batches} batch requests of up to 100)...")

    t0 = time.time()
    timestamp_map = await fetch_timestamps_batch(session, semaphore, block_numbers,
                                                  batch_size=TIMESTAMP_BATCH_SIZE)
    elapsed = time.time() - t0

    for log in logs:
        log["timestamp"] = timestamp_map.get(log["blockNumber"])
        if log["timestamp"]:
            log["datetime"] = datetime.fromtimestamp(log["timestamp"]).isoformat()

    missing = len(block_numbers) - len(timestamp_map)
    print(f"  Timestamps done in {elapsed:.1f}s | "
          f"{len(timestamp_map):,} fetched"
          + (f" | {missing} missing" if missing else ""))

    return logs


# ----------------------------------------------------------------------
# Main download function — now with batch saving
# ----------------------------------------------------------------------

async def download_range_async(block_range: Tuple[int, int]):
    """
    Download all logs for a block range, saving to disk every BATCH_SAVE_BLOCKS.

    Strategy:
    - Splits the full range into sub-ranges of BATCH_SAVE_BLOCKS each.
    - Checks the progress file to skip already-completed sub-ranges.
    - After each sub-range, saves a batch file and updates the progress checkpoint.
    - At the end, merges all batches into one final file and cleans up.
    """
    start_block, end_block = block_range

    # Check if final file already exists
    if os.path.exists(get_final_file_path(block_range)):
        print(f"\nFinal file already exists for {start_block:,}–{end_block:,}, skipping.")
        return

    # Resume from checkpoint if available
    last_completed = load_progress(block_range)
    resume_from = (last_completed + 1) if last_completed is not None else start_block

    if resume_from > end_block:
        print(f"\nAll blocks already completed for {start_block:,}–{end_block:,}.")
        print("Merging and finalising...")
        logs = merge_batches(block_range)
        save_final(logs, block_range)
        return

    # Build list of sub-ranges to process
    sub_ranges = []
    current = resume_from
    while current <= end_block:
        batch_end = min(current + BATCH_SAVE_BLOCKS - 1, end_block)
        sub_ranges.append((current, batch_end))
        current = batch_end + 1

    print(f"\n{'=' * 60}")
    print(f"Downloading blocks {start_block:,} to {end_block:,}")
    print(f"Resuming from block {resume_from:,}")
    print(f"Sub-ranges (batches): {len(sub_ranges)} × {BATCH_SAVE_BLOCKS:,} blocks each")
    print(f"Chunk size: {LOG_CHUNK_SIZE} | Concurrency: {CONCURRENCY}")
    print(f"{'=' * 60}")

    semaphore = asyncio.Semaphore(CONCURRENCY)

    async with aiohttp.ClientSession() as session:
        for i, (batch_start, batch_end) in enumerate(sub_ranges):
            print(f"\n--- Batch {i + 1}/{len(sub_ranges)}: "
                  f"blocks {batch_start:,} – {batch_end:,} ---")

            chunks = generate_chunks(batch_start, batch_end, LOG_CHUNK_SIZE)

            # Reset progress counters for this batch
            global _progress_completed, _progress_total, _progress_logs, _progress_start
            _progress_completed = 0
            _progress_total = len(chunks)
            _progress_logs = 0
            _progress_start = time.time()

            # Fetch all chunks in this batch
            log_tasks = [
                fetch_log_chunk(session, semaphore, from_b, to_b, f"b{i}c{j}")
                for j, (from_b, to_b) in enumerate(chunks)
            ]
            log_results = await asyncio.gather(*log_tasks, return_exceptions=True)

            raw_logs = []
            for result in log_results:
                if isinstance(result, list):
                    raw_logs.extend(result)

            # Enrich
            enriched = enrich_logs_with_info(raw_logs)
            enriched = await enrich_with_timestamps(session, semaphore, enriched)

            # Save batch to disk immediately — RAM is now freed after this
            save_batch(enriched, block_range, batch_start, batch_end)
            save_progress(block_range, batch_end)

            elapsed = time.time() - _progress_start
            print(f"  Batch done: {len(enriched):,} logs in {elapsed:.1f}s")

            # Let the GC collect the now-unneeded list before next batch
            del raw_logs, enriched

        # All batches done — merge into final file
        print(f"\n{'=' * 60}")
        print("All batches complete. Merging into final file...")
        logs = merge_batches(block_range)
        save_final(logs, block_range)

        print(f"\n{'=' * 60}")
        print(f"Range {start_block:,}–{end_block:,} complete!")
        print(f"  Total logs: {len(logs):,}")
        print(f"{'=' * 60}\n")


async def download_all_ranges():
    ensure_output_dir()

    for block_range in BLOCK_RANGES:
        await download_range_async(block_range)

    print("\n" + "=" * 60)
    print("ALL RANGES COMPLETE")
    print("=" * 60)
    for block_range in BLOCK_RANGES:
        final_path = get_final_file_path(block_range)
        if os.path.exists(final_path):
            size_mb = os.path.getsize(final_path) / (1024 * 1024)
            print(f"  {os.path.basename(final_path)} — {size_mb:.1f} MB")
    print("=" * 60)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    print(f"Starting download at {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"RPC URL:        {RPC_URL}")
    print(f"Block ranges:   {BLOCK_RANGES}")
    print(f"Concurrency:    {CONCURRENCY}")
    print(f"Chunk size:     {LOG_CHUNK_SIZE}")
    print(f"Batch size:     {BATCH_SAVE_BLOCKS:,} blocks per batch file")
    print(f"Timestamp RPC:  {TIMESTAMP_BATCH_SIZE} blocks per batch request")
    print(f"Max retries:    {MAX_RETRIES}")
    if TOKEN_ADDRESSES:
        print(f"Token filter:   {TOKEN_ADDRESSES}")
    else:
        print("Token filter:   none (all ERC-20 transfers)")
    print()

    asyncio.run(download_all_ranges())

    print(f"\nDone at {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"Files saved in {OUTPUT_DIR}/")
    print("\nTo load later:")
    print("  import pickle")
    print("  with open(r'D:\\zkSync_logs\\logs_69900000_70000000_final.pkl', 'rb') as f:")
    print("      data = pickle.load(f)")
    print("      logs = data['logs']")


if __name__ == "__main__":
    main()