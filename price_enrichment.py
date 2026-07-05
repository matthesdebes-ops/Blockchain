"""
price_enrichment.py

Convert raw ERC-20 / native-token transfer amounts into USD so that the
weighted network's edge weight represents real economic value rather than
raw token units (which are not comparable across tokens with different
decimals and prices).

Two lookups are performed, both cached to disk as JSON so repeated runs
never re-fetch what's already known:

  1. Decimals — batched on-chain `eth_call` to each token contract's
     `decimals()` selector (0x313ce567), via the same zkSync Era RPC used
     to download the logs.

  2. Historical USD price — DefiLlama's free, keyless historical price API
     (https://coins.llama.fi/prices/historical/{timestamp}/{chain}:{address}).
     DefiLlama's zkSync Era chain slug is "era". Multiple tokens sharing the
     same day are batched into a single request to keep call counts low
     (bucketed to whole days, since millions of individual transactions
     would otherwise mean millions of price lookups).

If a token/day combination has no resolvable price (very new, illiquid, or
unlisted token), the affected transfers contribute $0 to volume and are
recorded in `unpriced_tokens.json` so you can see what got excluded. The
graph's structural edges (based on raw value > 0) are unaffected — only the
edge *weight* changes.
"""

import json
import os
import time
import requests
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

RPC_URL = "https://mainnet.era.zksync.io"      # same RPC used by the downloader
DEFILLAMA_CHAIN = "era"                        # DefiLlama chain slug for zkSync Era
DEFILLAMA_HIST_URL = "https://coins.llama.fi/prices/historical"

CACHE_DIR = r"D:\zkSync_logs\plots"
DECIMALS_CACHE_PATH = os.path.join(CACHE_DIR, "token_decimals_cache.json")
PRICE_CACHE_PATH    = os.path.join(CACHE_DIR, "token_price_cache.json")
UNPRICED_LOG_PATH   = os.path.join(CACHE_DIR, "unpriced_tokens.json")

PRICE_BUCKET_SECONDS   = 86_400   # cache/query prices once per calendar day per token
TOKENS_PER_PRICE_CALL  = 60       # cap tokens per request to keep the URL a sane length
DECIMALS_PER_RPC_BATCH = 100

REQUEST_TIMEOUT = 20
MAX_RETRIES  = 5
BASE_BACKOFF = 2.0
MAX_BACKOFF  = 30.0

DEFAULT_DECIMALS_FALLBACK = 18   # used only if decimals() itself can't be read

os.makedirs(CACHE_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# Disk-backed caches (loaded once, flushed after each enrichment call)
# ----------------------------------------------------------------------

def _load_json(path: str) -> dict:
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_json(path: str, data: dict):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, path)


_decimals_cache: Dict[str, int]   = _load_json(DECIMALS_CACHE_PATH)
_price_cache: Dict[str, float]    = _load_json(PRICE_CACHE_PATH)     # key "address:day_bucket"
_unpriced: Dict[str, int]         = _load_json(UNPRICED_LOG_PATH)    # key "address:day_bucket" -> skipped count


def save_caches():
    _save_json(DECIMALS_CACHE_PATH, _decimals_cache)
    _save_json(PRICE_CACHE_PATH, _price_cache)
    _save_json(UNPRICED_LOG_PATH, _unpriced)


# ----------------------------------------------------------------------
# Decimals via batched eth_call
# ----------------------------------------------------------------------

_DECIMALS_SELECTOR = "0x313ce567"   # keccak("decimals()")[:4]


def _retry_post(payload):
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(RPC_URL, json=payload, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 429:
                time.sleep(min(BASE_BACKOFF * (2 ** attempt), MAX_BACKOFF))
                continue
            resp.raise_for_status()
            return resp.json()
        except Exception:
            time.sleep(min(BASE_BACKOFF * (2 ** attempt), MAX_BACKOFF))
    return None


def fetch_missing_decimals(addresses: Set[str]):
    """Look up decimals() for any addresses not already cached, in RPC batches."""
    todo = [a for a in addresses if a and a.lower() not in _decimals_cache]
    if not todo:
        return

    for i in range(0, len(todo), DECIMALS_PER_RPC_BATCH):
        batch = todo[i:i + DECIMALS_PER_RPC_BATCH]
        payload = [
            {
                "jsonrpc": "2.0",
                "id": addr,
                "method": "eth_call",
                "params": [{"to": addr, "data": _DECIMALS_SELECTOR}, "latest"],
            }
            for addr in batch
        ]
        data = _retry_post(payload)
        if not isinstance(data, list):
            # Whole batch failed to even respond — fall back rather than stall the pipeline
            for addr in batch:
                _decimals_cache.setdefault(addr.lower(), DEFAULT_DECIMALS_FALLBACK)
            continue

        by_id = {item.get("id"): item for item in data if isinstance(item, dict)}
        for addr in batch:
            item = by_id.get(addr)
            result = item.get("result") if item else None
            if result and result not in ("0x", None):
                try:
                    _decimals_cache[addr.lower()] = int(result, 16)
                    continue
                except Exception:
                    pass
            # decimals() reverted / non-standard token — assume 18 (most common) and flag it
            _decimals_cache[addr.lower()] = DEFAULT_DECIMALS_FALLBACK

    save_caches()


def get_decimals(address: str) -> int:
    return _decimals_cache.get(address.lower(), DEFAULT_DECIMALS_FALLBACK)


# ----------------------------------------------------------------------
# Historical USD price via DefiLlama (free, no API key)
# ----------------------------------------------------------------------

def _day_bucket(unix_ts: int) -> int:
    return (int(unix_ts) // PRICE_BUCKET_SECONDS) * PRICE_BUCKET_SECONDS


def fetch_missing_prices(address_day_pairs: Set[Tuple[str, int]]):
    """
    address_day_pairs: set of (token_address, day_bucket_unix_ts).
    Batches tokens sharing the same day into a single DefiLlama request.
    """
    by_day: Dict[int, List[str]] = defaultdict(list)
    for addr, day in address_day_pairs:
        if not addr:
            continue
        key = f"{addr.lower()}:{day}"
        if key not in _price_cache and key not in _unpriced:
            by_day[day].append(addr.lower())

    for day, addrs in by_day.items():
        addrs = sorted(set(addrs))
        for i in range(0, len(addrs), TOKENS_PER_PRICE_CALL):
            batch = addrs[i:i + TOKENS_PER_PRICE_CALL]
            coins = ",".join(f"{DEFILLAMA_CHAIN}:{a}" for a in batch)
            url = f"{DEFILLAMA_HIST_URL}/{day}/{coins}"

            data = None
            for attempt in range(MAX_RETRIES):
                try:
                    resp = requests.get(url, timeout=REQUEST_TIMEOUT)
                    if resp.status_code == 429:
                        time.sleep(min(BASE_BACKOFF * (2 ** attempt), MAX_BACKOFF))
                        continue
                    resp.raise_for_status()
                    data = resp.json()
                    break
                except Exception:
                    time.sleep(min(BASE_BACKOFF * (2 ** attempt), MAX_BACKOFF))

            found = set()
            if data and "coins" in data:
                for coin_key, info in data["coins"].items():
                    addr = coin_key.split(":")[-1].lower()   # "era:0xabc..." -> "0xabc..."
                    price = info.get("price")
                    if price is not None:
                        _price_cache[f"{addr}:{day}"] = float(price)
                        found.add(addr)

            for addr in batch:
                if addr not in found:
                    key = f"{addr}:{day}"
                    _unpriced[key] = _unpriced.get(key, 0) + 1

    save_caches()


def get_price(address: str, unix_ts: int) -> Optional[float]:
    day = _day_bucket(unix_ts)
    return _price_cache.get(f"{address.lower()}:{day}")


# ----------------------------------------------------------------------
# Public entry point used by the analysis script
# ----------------------------------------------------------------------

def enrich_chunk_with_usd(chunk: List[dict]) -> List[dict]:
    """
    For a chunk of enriched logs (each with 'address', 'value', 'timestamp'),
    fetch any missing decimals/prices, then attach 'value_usd' to each log.
    Logs with no resolvable price get 'value_usd' = 0.0 (excluded downstream
    by the existing MIN_VOLUME filters) and are tallied in the unpriced report.
    """
    addresses = {log["address"] for log in chunk if log.get("address") and log.get("timestamp")}
    fetch_missing_decimals(addresses)

    day_pairs = {
        (log["address"], _day_bucket(log["timestamp"]))
        for log in chunk if log.get("address") and log.get("timestamp")
    }
    fetch_missing_prices(day_pairs)

    for log in chunk:
        addr = log.get("address")
        ts = log.get("timestamp")
        raw_value = log.get("value", 0)
        if not addr or not ts:
            log["value_usd"] = 0.0
            continue
        decimals = get_decimals(addr)
        price = get_price(addr, ts)
        if price is None:
            log["value_usd"] = 0.0
        else:
            token_amount = raw_value / (10 ** decimals)
            log["value_usd"] = token_amount * price

    return chunk


def unpriced_summary() -> Tuple[int, int]:
    """Returns (num_distinct_token_day_pairs_unpriced, num_logs_affected)."""
    return len(_unpriced), sum(_unpriced.values())