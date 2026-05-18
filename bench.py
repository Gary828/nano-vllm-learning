import os
import time
from random import randint

from nanovllm import LLM, SamplingParams


def benchmark_once():
    num_seqs = 256
    max_input_len = 1024
    max_ouput_len = 1024

    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    llm = LLM(path, enforce_eager=False, max_model_len=4096)

    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    sampling_params = [SamplingParams(temperature=0.6, max_tokens=randint(100, max_ouput_len)) for _ in range(num_seqs)]

    llm.generate(["Benchmark: "], SamplingParams())
    t = time.time()
    result = llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = time.time() - t

    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    ttft = result.get("ttft", 0)
    total_time = result.get("total_time", t)
    throughput = total_tokens / total_time if total_time > 0 else 0
    print(f"Total: {total_tokens}tok, Time: {total_time:.2f}s, TTFT: {ttft:.3f}s, Throughput: {throughput:.2f}tok/s")


def main():
    benchmark_once()


if __name__ == "__main__":
    main()
