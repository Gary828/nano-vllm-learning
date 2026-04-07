#!/usr/bin/env python3
"""
Profile nano-vllm + ContextPilot on realistic out-of-order shared-prefix workloads.

ContextPilot's value is maximized when prompts that share long prefixes arrive
out of order. This benchmark shuffles prompts to simulate real-world dispatch
patterns, then measures cache hit improvement after ContextPilot reordering.

Scenarios:
    1. Batch RAG: multiple queries retrieving chunks from the same document.
    2. Multi-turn chat: conversation turns from different sessions.
    3. Shared system prompt: many users sharing the same large system prompt.

Usage:
    python bench_profile.py --scenario rag
    python bench_profile.py --scenario chat
    python bench_profile.py --scenario system
    python bench_profile.py --all
    python bench_profile.py --run-inference  # Requires GPU + model weights
"""

import argparse
import os
import sys
import time
from collections import defaultdict
from random import randint, seed, shuffle

# Import ContextOptimizer directly to avoid pulling torch
import importlib.util
spec = importlib.util.spec_from_file_location(
    "context_optimizer",
    os.path.join(os.path.dirname(__file__), "nanovllm", "context_optimizer.py")
)
co = importlib.util.module_from_spec(spec)
spec.loader.exec_module(co)
ContextOptimizer = co.ContextOptimizer

try:
    import xxhash
    import numpy as np
    XXHASH_AVAILABLE = True
except ImportError:
    XXHASH_AVAILABLE = False


def compute_block_hash(token_ids: list[int], prefix_hash: int = -1) -> int:
    """xxhash64 with prefix chaining (matches nano-vllm BlockManager)."""
    h = xxhash.xxh64()
    if prefix_hash != -1:
        h.update(prefix_hash.to_bytes(8, "little"))
    h.update(np.array(token_ids).tobytes())
    return h.intdigest()


def simulate_cache_alloc(prompts: list[list[int]], block_size: int = 256):
    """
    Simulate nano-vllm KV cache allocation and return metrics.

    Returns:
        first_time_tokens: tokens that must be computed (cache miss)
        cached_tokens: tokens that can reuse cached KV
        total_tokens: total input tokens
    """
    hash_table = {}
    first_time_tokens = 0
    cached_tokens = 0
    total_tokens = 0

    for prompt in prompts:
        num_blocks = (len(prompt) + block_size - 1) // block_size
        prev_hash = -1

        for b in range(num_blocks):
            start = b * block_size
            end = min(start + block_size, len(prompt))
            block = prompt[start:end]

            if len(block) == block_size:
                h = compute_block_hash(block, prev_hash)
            else:
                h = -1

            if h in hash_table:
                cached_tokens += len(block)
            else:
                first_time_tokens += len(block)
                hash_table[h] = True

            prev_hash = h

        total_tokens += len(prompt)

    return first_time_tokens, cached_tokens, total_tokens


def generate_rag_prompts(
    num_queries: int = 64,
    num_docs: int = 4,
    chunks_per_doc: int = 4,
    chunk_size: int = 128,
    query_specific_len: int = 64,
) -> list[list[int]]:
    """
    Simulate RAG where multiple queries hit the same document chunks.

    Each prompt = system_prefix(64) + shared_doc_chunks + query_suffix(64)
    Queries arrive in random order, creating out-of-order shared prefixes.
    """
    seed(42)

    # Shared document chunks (same across queries hitting same doc)
    doc_chunks = {}
    for d in range(num_docs):
        doc_chunks[d] = [
            [d * 1000 + c] + [randint(10000, 20000) for _ in range(chunk_size - 1)]
            for c in range(chunks_per_doc)
        ]

    # System prefix (shared globally)
    system_prefix = [0] + [randint(1, 5000) for _ in range(63)]

    prompts = []
    for q in range(num_queries):
        doc_id = q % num_docs
        query_suffix = [5000 + q] + [randint(5001, 6000) for _ in range(query_specific_len - 1)]
        prompt = list(system_prefix)
        for chunk in doc_chunks[doc_id]:
            prompt.extend(chunk)
        prompt.extend(query_suffix)
        prompts.append(prompt)

    return prompts


def generate_multi_turn_chat(
    num_sessions: int = 16,
    max_turns: int = 8,
    system_len: int = 256,
    turn_len: int = 128,
) -> list[list[int]]:
    """
    Simulate parallel chat sessions. Each prompt is one turn,
    sharing system prompt + conversation history.

    ContextPilot reorders turns so sessions with shared system prompt
    are scheduled together.
    """
    seed(43)

    system_prompts = [
        [s + 100] + [randint(200, 5000) for _ in range(system_len - 1)]
        for s in range(num_sessions)
    ]

    prompts = []
    for s in range(num_sessions):
        history = list(system_prompts[s])
        for t in range(max_turns):
            turn = [t + 1000] + [randint(1001, 5000) for _ in range(turn_len - 1)]
            prompt = history + turn
            prompts.append(prompt)
            history.extend(turn)

    return prompts


def generate_shared_system_prompt(
    num_requests: int = 128,
    num_system_prompts: int = 4,
    system_len: int = 512,
    user_len: int = 128,
) -> list[list[int]]:
    """
    Simulate batch inference where many users share a few system prompts.
    """
    seed(44)

    system_prompts = [
        [s + 200] + [randint(300, 5000) for _ in range(system_len - 1)]
        for s in range(num_system_prompts)
    ]

    prompts = []
    for r in range(num_requests):
        sys_id = r % num_system_prompts
        user = [r + 6000] + [randint(6001, 8000) for _ in range(user_len - 1)]
        prompts.append(list(system_prompts[sys_id]) + user)

    return prompts


def profile_scenario(name: str, prompts: list[list[int]], run_inference: bool = False):
    """Profile a single scenario: naive, shuffled, ContextPilot-reordered."""

    print(f"\n{'-' * 60}")
    print(f"Scenario: {name}")
    print(f"  Prompts: {len(prompts)}, Avg length: {sum(len(p) for p in prompts) // len(prompts)} tokens")

    # 1. Naive order (already generated in an advantageous order for some scenarios)
    ft_naive, c_naive, tt = simulate_cache_alloc(prompts)
    hit_naive = c_naive / tt if tt else 0.0

    # 2. Simulate realistic out-of-order arrival
    shuffled = list(prompts)
    shuffle(shuffled)
    ft_shuf, c_shuf, _ = simulate_cache_alloc(shuffled)
    hit_shuf = c_shuf / tt if tt else 0.0

    # 3. Apply ContextPilot reordering to the shuffled batch
    t0 = time.time()
    optimizer = ContextOptimizer()
    reordered, order = optimizer.reorder(shuffled)
    reorder_ms = (time.time() - t0) * 1000

    ft_opt, c_opt, _ = simulate_cache_alloc(reordered)
    hit_opt = c_opt / tt if tt else 0.0

    # Compute speedups based on first-time tokens (fewer first-time = less prefill work)
    naive_speedup = tt / ft_naive if ft_naive else 1.0
    shuf_speedup = tt / ft_shuf if ft_shuf else 1.0
    opt_speedup = tt / ft_opt if ft_opt else 1.0
    gain_over_shuffled = opt_speedup / shuf_speedup if shuf_speedup else 1.0

    print(f"  Naive order:          {hit_naive:.1%} cache hit, {ft_naive}/{tt} first-time (speedup {naive_speedup:.2f}x)")
    print(f"  Shuffled order:       {hit_shuf:.1%} cache hit, {ft_shuf}/{tt} first-time (speedup {shuf_speedup:.2f}x)")
    print(f"  ContextPilot reorder: {hit_opt:.1%} cache hit, {ft_opt}/{tt} first-time (speedup {opt_speedup:.2f}x)")
    print(f"  Gain vs shuffled:     {gain_over_shuffled:.2f}x")
    print(f"  Reordering overhead:  {reorder_ms:.2f} ms")

    result = {
        "name": name,
        "num_prompts": len(prompts),
        "total_tokens": tt,
        "naive_hit": hit_naive,
        "shuffled_hit": hit_shuf,
        "optimized_hit": hit_opt,
        "naive_speedup": naive_speedup,
        "shuffled_speedup": shuf_speedup,
        "optimized_speedup": opt_speedup,
        "gain_vs_shuffled": gain_over_shuffled,
        "reorder_ms": reorder_ms,
    }

    if run_inference:
        from nanovllm import LLM, SamplingParams
        model_path = os.path.expanduser("/root/study/lite_llama/my_weight/qwen3-0.6B")
        print(f"  Running actual inference...")
        llm = LLM(model_path, enforce_eager=True, tensor_parallel_size=1)
        sp = [SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=32) for _ in prompts]

        # Warmup
        llm.generate([prompts[0]], sp[0], use_tqdm=False)

        t0 = time.time()
        r1 = llm.generate(prompts, sp, use_tqdm=False, use_context_optimizer=False)
        t_naive = time.time() - t0

        t0 = time.time()
        r2 = llm.generate(prompts, sp, use_tqdm=False, use_context_optimizer=True)
        t_opt = time.time() - t0

        actual_speedup = t_naive / t_opt if t_opt else 1.0
        print(f"    Naive time: {t_naive:.2f}s, Optimized time: {t_opt:.2f}s, Speedup: {actual_speedup:.2f}x")
        result["inference_naive_time"] = t_naive
        result["inference_opt_time"] = t_opt
        result["inference_speedup"] = actual_speedup

    return result


def main():
    parser = argparse.ArgumentParser(description="Profile nano-vllm + ContextPilot")
    parser.add_argument("--scenario", choices=["rag", "chat", "system", "all"], default="all")
    parser.add_argument("--run-inference", action="store_true", help="Run actual GPU inference (requires model)")
    args = parser.parse_args()

    if not XXHASH_AVAILABLE:
        print("Error: xxhash and numpy are required for cache simulation")
        sys.exit(1)

    print("=" * 60)
    print("  nano-vllm + ContextPilot Profile")
    print("=" * 60)

    scenarios = []
    if args.scenario in ("rag", "all"):
        scenarios.append(("RAG (shared doc chunks)", generate_rag_prompts(64, 4, 4, 128, 64)))
    if args.scenario in ("chat", "all"):
        scenarios.append(("Multi-turn Chat", generate_multi_turn_chat(16, 8, 256, 128)))
    if args.scenario in ("system", "all"):
        scenarios.append(("Shared System Prompt", generate_shared_system_prompt(128, 4, 512, 128)))

    results = []
    for name, prompts in scenarios:
        results.append(profile_scenario(name, prompts, run_inference=args.run_inference))

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Scenario':<30} {'Hit (shuf)':<12} {'Hit (opt)':<12} {'Gain':<8}")
    print("-" * 60)
    for r in results:
        print(f"{r['name']:<30} {r['shuffled_hit']:<12.1%} {r['optimized_hit']:<12.1%} {r['gain_vs_shuffled']:<8.2f}x")


if __name__ == "__main__":
    main()
