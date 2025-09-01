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
GPUS = 1  # Assumed for per-GPU metrics; adjust as needed

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

    total_tokens = sum(tokens)
    total_time = sum(latencies)
    avg_latency = mean(latencies)
    avg_ttft = mean(ttfts)
    avg_tokens = mean(tokens)
    max_time = max(r["latency"] for r in results)  # Approximates total wall time

    # Derived metrics
    req_throughput = len(results) / max_time
    output_throughput = total_tokens / max_time
    total_token_throughput = output_throughput * 2  # assumes input = output tokens
    avg_tpot = (avg_latency - avg_ttft) / avg_tokens if avg_tokens > 0 else 0
    per_user_output_tps = output_throughput / CONCURRENCY
    per_user_speed = total_tokens / (CONCURRENCY * max_time)
    per_gpu_throughput = output_throughput / GPUS

    print("\nBenchmark Results:")
    print(f"Total Requests:\t\t\t\t{len(results)}")
    print(f"Concurrency Level:\t\t\t{CONCURRENCY}")
    print(f"Total Latency (ms):\t\t\t{total_time * 1000:.4f}")
    print(f"Avg Latency (ms):\t\t\t{avg_latency * 1000:.4f}")
    print(f"Avg TTFT (ms):\t\t\t\t{avg_ttft * 1000:.4f}")
    print(f"Avg Tokens/Response:\t\t\t{avg_tokens:.2f}")
    print(f"Avg TPOT (ms):\t\t\t\t{avg_tpot * 1000:.4f}")
    print(f"Request Throughput (req/sec):\t\t{req_throughput:.4f}")
    print(f"Total Output Throughput (tokens/sec):\t{output_throughput:.4f}")
    print(f"Total Token Throughput (tokens/sec):\t{total_token_throughput:.4f}")
    print(f"Per User Output Throughput (tps/user):\t{per_user_output_tps:.4f}")
    print(f"Per GPU Output Throughput (tps/gpu):\t{per_gpu_throughput:.4f}")
    print(f"Per User Output Speed (tps/user):\t{per_user_speed:.4f}")

if __name__ == "__main__":
    results = asyncio.run(run_benchmark())
    report(results)

