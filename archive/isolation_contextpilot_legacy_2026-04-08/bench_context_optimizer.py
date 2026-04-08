"""
Benchmark script for ContextPilot integration in nano-vllm.

Tests the performance improvement from ContextPilot's context reordering
optimization for KV cache prefix sharing.

Usage:
    python bench_context_optimizer.py
"""

import os
import time
from copy import deepcopy
from random import randint, seed, shuffle

import numpy as np
import xxhash

from nanovllm import LLM, SamplingParams


def generate_shared_prefix_prompts(
    num_groups: int = 16,
    seqs_per_group: int = 16,
    prefix_len: int = 256,
    suffix_len: int = 128,
    seed_value: int = 42,
) -> list[list[int]]:
    """Generate prompts with shared prefixes."""
    seed(seed_value)
    shared_prefixes = [
        [randint(0, 100000) for _ in range(prefix_len)]
        for _ in range(num_groups)
    ]

    prompts = []
    for prefix in shared_prefixes:
        for _ in range(seqs_per_group):
            suffix = [randint(0, 100000) for _ in range(suffix_len)]
            prompts.append(prefix + suffix)

    return prompts


def generate_random_prompts(
    num_prompts: int = 256,
    min_len: int = 100,
    max_len: int = 512,
    seed_value: int = 42,
) -> list[list[int]]:
    """Generate random prompts without shared prefixes (baseline)."""
    seed(seed_value)
    return [
        [randint(0, 100000) for _ in range(randint(min_len, max_len))]
        for _ in range(num_prompts)
    ]


class CacheSimulator:
    """Simulate nano-vllm block-manager prefix caching."""

    def __init__(self, block_size: int = 256):
        self.block_size = block_size

    def compute_hash(self, token_ids: list[int], prefix_hash: int = -1) -> int:
        h = xxhash.xxh64()
        if prefix_hash != -1:
            h.update(prefix_hash.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def simulate(self, prompts: list[list[int]]) -> dict:
        hash_table = {}
        first_time_tokens = 0
        cached_tokens = 0
        total_tokens = 0

        for prompt in prompts:
            num_blocks = (len(prompt) + self.block_size - 1) // self.block_size
            prev_hash = -1

            for b in range(num_blocks):
                start = b * self.block_size
                end = min(start + self.block_size, len(prompt))
                block = prompt[start:end]

                if len(block) == self.block_size:
                    h = self.compute_hash(block, prev_hash)
                else:
                    h = -1

                if h in hash_table:
                    cached_tokens += len(block)
                else:
                    first_time_tokens += len(block)
                    hash_table[h] = True

                prev_hash = h

            total_tokens += len(prompt)

        return {
            "first_time_tokens": first_time_tokens,
            "cached_tokens": cached_tokens,
            "total_tokens": total_tokens,
            "cache_hit_rate": cached_tokens / total_tokens if total_tokens else 0.0,
            "effective_speedup": total_tokens / first_time_tokens if first_time_tokens else 1.0,
        }


def simulate_contextpilot_theory(prompts: list[list[int]]) -> dict:
    """Return cache metrics with and without ContextPilot reordering."""
    from nanovllm.context_optimizer import ContextOptimizer

    sim = CacheSimulator()

    # 1. Naive order (already generated in group order)
    naive = sim.simulate(prompts)

    # 2. Shuffle to simulate real dispatch, then ContextPilot reorder
    shuffled = list(prompts)
    shuffle(shuffled)
    shuf = sim.simulate(shuffled)

    t0 = time.time()
    optimizer = ContextOptimizer()
    reordered, _ = optimizer.reorder(shuffled)
    reorder_ms = (time.time() - t0) * 1000
    opt = sim.simulate(reordered)

    return {
        "naive_hit": naive["cache_hit_rate"],
        "shuffled_hit": shuf["cache_hit_rate"],
        "optimized_hit": opt["cache_hit_rate"],
        "naive_speedup": naive["effective_speedup"],
        "shuffled_speedup": shuf["effective_speedup"],
        "optimized_speedup": opt["effective_speedup"],
        "reorder_ms": reorder_ms,
    }


def flush_gpu_state(llm: LLM, num_prompts: int = 8, prompt_len: int = 512) -> float:
    """
    Flush GPU prefix-cache state between runs.

    nano-vllm retains block hashes even after deallocation.
    We overwrite these by processing fresh random prompts.
    """
    sp = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=32)
    prompts = [[randint(0, 100000) for _ in range(prompt_len)] for _ in range(num_prompts)]
    t0 = time.time()
    llm.generate(prompts, [sp] * num_prompts, use_tqdm=False, use_context_optimizer=False)
    return time.time() - t0


def benchmark_run(
    llm: LLM,
    prompts: list[list[int]],
    use_optimizer: bool,
    output_len: int = 64,
) -> dict:
    """Run a single benchmark trial, including warmup + measurement."""
    sampling_params = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=output_len)
        for _ in range(len(prompts))
    ]

    # Warmup
    llm.generate([prompts[0]], sampling_params[0], use_tqdm=False, use_context_optimizer=use_optimizer)

    # Measure
    t_start = time.time()
    result = llm.generate(
        prompts,
        sampling_params,
        use_tqdm=False,
        use_context_optimizer=use_optimizer,
    )
    elapsed = time.time() - t_start

    outputs = result["outputs"]
    total_input_tokens = sum(len(p) for p in prompts)
    total_output_tokens = sum(len(o["token_ids"]) for o in outputs)
    total_tokens = total_input_tokens + total_output_tokens
    throughput = total_tokens / elapsed

    return {
        "total_time": elapsed,
        "throughput": throughput,
        "total_tokens": total_tokens,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
    }


def benchmark_pair(llm: LLM, prompts: list[list[int]], output_len: int = 64) -> dict:
    """
    Run both conditions and attempt to remove GPU-warmup bias.

    We use an A-B-A-B interleaving pattern and average the differences.
    """
    # Make independent copies so reordering doesn't mutate the original
    p_without = deepcopy(prompts)
    p_with = deepcopy(prompts)

    print("    Run 1/4: WITHOUT...", end=" ", flush=True)
    flush_gpu_state(llm)
    r1 = benchmark_run(llm, p_without, use_optimizer=False, output_len=output_len)
    print(f"{r1['total_time']:.2f}s")

    print("    Run 2/4: WITH......", end=" ", flush=True)
    flush_gpu_state(llm)
    r2 = benchmark_run(llm, p_with, use_optimizer=True, output_len=output_len)
    print(f"{r2['total_time']:.2f}s")

    print("    Run 3/4: WITH......", end=" ", flush=True)
    flush_gpu_state(llm)
    r3 = benchmark_run(llm, p_with, use_optimizer=True, output_len=output_len)
    print(f"{r3['total_time']:.2f}s")

    print("    Run 4/4: WITHOUT...", end=" ", flush=True)
    flush_gpu_state(llm)
    r4 = benchmark_run(llm, p_without, use_optimizer=False, output_len=output_len)
    print(f"{r4['total_time']:.2f}s")

    # Average each condition
    t_without = (r1["total_time"] + r4["total_time"]) / 2.0
    t_with = (r2["total_time"] + r3["total_time"]) / 2.0
    speedup = t_without / t_with if t_with else 1.0

    return {
        "without_time": t_without,
        "with_time": t_with,
        "speedup": speedup,
        "without_throughput": (r1["throughput"] + r4["throughput"]) / 2.0,
        "with_throughput": (r2["throughput"] + r3["throughput"]) / 2.0,
    }


def print_results(label: str, result: dict):
    print(f"\n{label}:")
    print(f"  Total time:     {result['total_time']:.2f}s")
    print(f"  Throughput:     {result['throughput']:.2f} tok/s")
    print(f"  Input tokens:   {result['total_input_tokens']}")
    print(f"  Output tokens:  {result['total_output_tokens']}")


def main():
    model_path = os.path.expanduser("/root/study/lite_llama/my_weight/qwen3-0.6B")

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    print("=" * 60)
    print("ContextPilot Benchmark for nano-vllm (fixed)")
    print("=" * 60)

    print("\nInitializing LLM...", end=" ", flush=True)
    t_start = time.time()
    llm = LLM(model_path, enforce_eager=False, tensor_parallel_size=1)
    print(f"done ({time.time() - t_start:.2f}s)")

    scenarios = [
        (
            "Shared Prefixes (prefix=256, groups=8, seqs=32)",
            generate_shared_prefix_prompts(8, 32, 256, 128),
        ),
        (
            "Long Shared Prefixes (prefix=512, groups=4, seqs=64)",
            generate_shared_prefix_prompts(4, 64, 512, 128),
        ),
        (
            "Random Baseline (no shared prefixes)",
            generate_random_prompts(256, 100, 512),
        ),
    ]

    for title, prompts in scenarios:
        print("\n" + "=" * 60)
        print(f"Scenario: {title}")
        print("=" * 60)
        print(f"Prompts: {len(prompts)}, avg length: {sum(len(p) for p in prompts) // len(prompts)} tokens")

        # 1. Theory check via cache simulator
        print("\n  -- Cache Simulation (shuffled -> ContextPilot) --")
        theory = simulate_contextpilot_theory(prompts)
        print(f"    Naive order hit:    {theory['naive_hit']:.1%}")
        print(f"    Shuffled order hit: {theory['shuffled_hit']:.1%}")
        print(f"    ContextPilot hit:   {theory['optimized_hit']:.1%}")
        print(f"    Theoretical speedup: {theory['optimized_speedup']:.2f}x")
        print(f"    Reordering overhead: {theory['reorder_ms']:.2f} ms")

        # 2. Actual GPU inference (interleaved to reduce warmup bias)
        print("\n  -- GPU Inference (A-B-A-B interleaved) --")
        gpu = benchmark_pair(llm, prompts, output_len=64)
        print(f"    WITHOUT avg time: {gpu['without_time']:.2f}s")
        print(f"    WITH    avg time: {gpu['with_time']:.2f}s")
        print(f"    Speedup: {gpu['speedup']:.2f}x")

    print("\n" + "=" * 60)
    print("Benchmark complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
