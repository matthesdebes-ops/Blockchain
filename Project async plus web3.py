
from web3 import Web3
import asyncio
import aiohttp


RPC_URL = "https://mainnet.era.zksync.io"
TOKEN_ADDRESS = "0x3355df6D4c9C3035724Fd0e3914dE96A5a83aaf4"
NUMBER_OF_BLOCKS = 20

CONCURRENCY = 5

ERC20_ABI = [
    {
        "anonymous": False,
        "inputs": [
            {
                "indexed": True,
                "name": "from",
                "type": "address",
            },
            {
                "indexed": True,
                "name": "to",
                "type": "address",
            },
            {
                "indexed": False,
                "name": "value",
                "type": "uint256",
            },
        ],
        "name": "Transfer",
        "type": "event",
    },
    {
        "inputs": [],
        "name": "decimals",
        "outputs": [
            {
                "name": "",
                "type": "uint8",
            }
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "symbol",
        "outputs": [
            {
                "name": "",
                "type": "string",
            }
        ],
        "stateMutability": "view",
        "type": "function",
    },
]


w3 = Web3(Web3.HTTPProvider(RPC_URL))

token = w3.eth.contract(
    address=Web3.to_checksum_address(TOKEN_ADDRESS),
    abi=ERC20_ABI,
)

latest_block = w3.eth.block_number

from_block = latest_block - 200
to_block = latest_block

symbol = token.functions.symbol().call()
decimals = token.functions.decimals().call()

print("Token:", symbol)
print("Decimals:", decimals)
print("Contract:", TOKEN_ADDRESS)
print("Block range:", from_block, "to", to_block)

transfer_topic = w3.keccak(text="Transfer(address,address,uint256)").hex()

logs = w3.eth.get_logs(
    {
        "address": Web3.to_checksum_address(TOKEN_ADDRESS),
        "fromBlock": from_block,
        "toBlock": to_block,
        "topics": [transfer_topic],
    }
)

print("\nRaw logs found:", len(logs))

for raw_log in logs[:20]:
    event = token.events.Transfer().process_log(raw_log)

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
async def rpc_call(session, method, params, request_id):
    """
    Send one JSON-RPC request.

    This is intentionally small and simple for teaching.
    """
    body = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": params,
    }

    async with session.post(RPC_URL, json=body) as response:
        data = await response.json()
        return data["result"]


async def download_block(session, semaphore, block_number):
    """
    Download one block by number.

    The semaphore keeps only CONCURRENCY requests active at the same time.
    """
    async with semaphore:
        block_number_hex = hex(block_number)

        block = await rpc_call(
            session=session,
            method="eth_getBlockByNumber",
            params=[block_number_hex, False],  # False = transaction hashes only
            request_id=block_number,
        )

        return block


async def main():
    semaphore = asyncio.Semaphore(CONCURRENCY)

    # One ClientSession for the whole program.
    # This allows aiohttp to reuse connections.
    async with aiohttp.ClientSession() as session:
        latest_block_hex = await rpc_call(
            session=session,
            method="eth_blockNumber",
            params=[],
            request_id=1,
        )

        latest_block = int(latest_block_hex, 16)

        print("Latest block:", latest_block)

        # Build a list of block numbers:
        # latest, latest - 1, latest - 2, ...
        block_numbers = []
        for n in range(NUMBER_OF_BLOCKS):
            block_numbers.append(latest_block - n)

        # Create one async task per block.
        tasks = []
        for block_number in block_numbers:
            task = download_block(session, semaphore, block_number)
            tasks.append(task)

        # Run all tasks concurrently.
        blocks = await asyncio.gather(*tasks)

    # Sort results because concurrent requests may finish in any order.
    blocks.sort(key=lambda block: int(block["number"], 16))

    print("\nDownloaded blocks:")
    for block in blocks:
        number = int(block["number"], 16)
        tx_count = len(block["transactions"])
        block_hash = block["hash"]

        print(f"  block={number} txs={tx_count} hash={block_hash}")


asyncio.run(main())