import time
import asyncio
import aiohttp
from statistics import mean

API_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "/model"

PROMPT = "Tell me a short story about a dragon and a wizard."
MAX_TOKENS = 256
CONCURRENCY = 8 
REQUESTS = 16

HEADERS = {"Content-Type": "application/json"}

async def benchmark_request(session, request_id):
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": PROMPT}
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
    }

    start_time = time.time()
    async with session.post(API_URL, headers=HEADERS, json=payload) as resp:
        ttft = time.time()
        data = await resp.json()
        end_time = time.time()

        gen_text = data["choices"][0]["message"]["content"]
        total_tokens = len(gen_text.split())
        return {
            "id": request_id,
            "latency": end_time - start_time,
            "ttft": ttft - start_time,
            "tokens": total_tokens,
            "output": gen_text,
        }


async def run_benchmark():
    connector = aiohttp.TCPConnector(limit=CONCURRENCY)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for i in range(REQUESTS):
            task = benchmark_request(session, i)
            tasks.append(task)

        results = []
        for i in range(0, len(tasks), CONCURRENCY):
            batch = tasks[i:i + CONCURRENCY]
            res = await asyncio.gather(*batch)
            results.extend(res)

    return results


def report(results):
    latencies = [r["latency"] for r in results]
    ttfts = [r["ttft"] for r in results]
    tokens = [r["tokens"] for r in results]
    total_time = sum(latencies)

    print("\nBenchmark Results:")
    print(f"Total Requests:\t\t{len(results)}")
    print(f"Concurrency Level:\t\t{CONCURRENCY}")
    print(f"Avg Latency:\t\t{mean(latencies):.2f} sec")
    print(f"Avg TTFT:\t\t{mean(ttfts):.2f} sec")
    print(f"Avg Tokens/Resp:\t\t{mean(tokens):.2f}")
    print(f"Throughput:\t\t{sum(tokens) / total_time:.2f} tokens/sec")


if __name__ == "__main__":
    results = asyncio.run(run_benchmark())
    report(results)
