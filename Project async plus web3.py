
import asyncio
import aiohttp
from web3 import Web3

RPC_URL = "https://mainnet.era.zksync.io"
TOKEN_ADDRESS = "0x3355df6D4c9C3035724Fd0e3914dE96A5a83aaf4"

CONCURRENCY = 10


ERC20_ABI = [
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "name": "from", "type": "address"},
            {"indexed": True, "name": "to", "type": "address"},
            {"indexed": False, "name": "value", "type": "uint256"},
        ],
        "name": "Transfer",
        "type": "event",
    },
    {
        "inputs": [],
        "name": "decimals",
        "outputs": [{"name": "", "type": "uint8"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "symbol",
        "outputs": [{"name": "", "type": "string"}],
        "stateMutability": "view",
        "type": "function",
    },
]


async def rpc_call(session, method, params, request_id):
    body = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": params,
    }

    async with session.post(RPC_URL, json=body) as response:
        data = await response.json()
        if "error" in data:
            raise Exception(f"RPC Error: {data['error']}")
        return data["result"]


async def get_logs_for_range(session, semaphore, token_address, from_block, to_block, topics, range_id):
    async with semaphore:
        result = await rpc_call(
            session=session,
            method="eth_getLogs",
            params=[{
                "address": token_address,
                "fromBlock": hex(from_block),
                "toBlock": hex(to_block),
                "topics": topics,
            }],
            request_id=range_id,
        )

        return result


async def main():
    w3 = Web3(Web3.HTTPProvider(RPC_URL))
    token = w3.eth.contract(
        address=Web3.to_checksum_address(TOKEN_ADDRESS),
        abi=ERC20_ABI,
    )

    symbol = token.functions.symbol().call()
    decimals = token.functions.decimals().call()

    latest_block = w3.eth.block_number
    from_block = latest_block - 200
    to_block = latest_block

    print("Token:", symbol)
    print("Decimals:", decimals)
    print("Contract:", TOKEN_ADDRESS)
    print("Block range:", from_block, "to", to_block)

    transfer_topic = Web3.keccak(text="Transfer(address,address,uint256)").hex()
    topics = [transfer_topic]
    BLOCKS_PER_CHUNK = 50  # Jede Anfrage holt max. 50 Blöcke
    total_blocks = to_block - from_block
    num_chunks = max(1, total_blocks // BLOCKS_PER_CHUNK)

    chunks = []
    for i in range(num_chunks):
        start = from_block + i * BLOCKS_PER_CHUNK
        end = min(start + BLOCKS_PER_CHUNK - 1, to_block)
        if start <= end:
            chunks.append((start, end, i))

    print(f"\nDivide {total_blocks} Blocks into {len(chunks)} parallel requests")
    semaphore = asyncio.Semaphore(CONCURRENCY)

    async with aiohttp.ClientSession() as session:
        tasks = []
        for start_block, end_block, chunk_id in chunks:
            task = get_logs_for_range(
                session=session,
                semaphore=semaphore,
                token_address=TOKEN_ADDRESS,
                from_block=start_block,
                to_block=end_block,
                topics=topics,
                range_id=chunk_id
            )
            tasks.append(task)

        all_logs_chunks = await asyncio.gather(*tasks)

    logs = []
    for chunk_logs in all_logs_chunks:
        if chunk_logs:
            logs.extend(chunk_logs)

    print(f"\nRaw logs found: {len(logs)}")

    for raw_log in logs:

        formatted_log = {
            "address": Web3.to_checksum_address(raw_log["address"]),
            "blockHash": bytes.fromhex(raw_log["blockHash"][2:]),
            "blockNumber": int(raw_log["blockNumber"], 16),
            "data": raw_log["data"],
            "logIndex": int(raw_log["logIndex"], 16),
            "removed": raw_log.get("removed", False),
            "topics": [bytes.fromhex(t[2:]) for t in raw_log["topics"]],
            "transactionHash": bytes.fromhex(raw_log["transactionHash"][2:]),
            "transactionIndex": int(raw_log["transactionIndex"], 16),
        }

        event = token.events.Transfer().process_log(formatted_log)

        from_address = event["args"]["from"]
        to_address = event["args"]["to"]
        raw_value = event["args"]["value"]
        human_value = raw_value / (10 ** decimals)

        print("\nTransfer event")
        print("  block:", event["blockNumber"])
        print("  tx:", event["transactionHash"].hex())
        print("  from:", from_address)
        print("  to:", to_address)
        print("  raw value:", raw_value)
        print("  human value:", human_value, symbol)


if __name__ == "__main__":
    asyncio.run(main())