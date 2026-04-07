"""
Benchmark: ContextPilot Intra-Context Reordering + Cache-Aware Scheduler for nano-vllm.

Simulates a realistic multi-request long-context scenario:
  - A "knowledge base" of document chunks (pre-tokenized)
  - Each request selects a SUBSET of chunks in random order + unique query
  - Without ContextPilot: different chunk orderings → different token prefixes → no cache reuse
  - With ContextPilot: shared chunks reordered to form common prefixes → cache hits

This demonstrates the FULL ContextPilot value chain:
  1. Intra-context reordering  (ContextPilot: create shared prefixes)
  2. Inter-context scheduling   (ContextPilot: group similar requests)
  3. Cache-aware batch packing  (nano-vllm scheduler: exploit cache hits)

Usage:
    conda run -n nano python bench_contextpilot.py
"""

import os
import sys
import time
import logging
from copy import deepcopy
from random import randint, seed, shuffle, sample

os.environ["CONTEXTPILOT_LOG_LEVEL"] = "ERROR"
logging.disable(logging.WARNING)

from contextpilot.context_index.index_construction import ContextIndex
from contextpilot.context_ordering.inter_scheduler import InterContextScheduler

from nanovllm import LLM, SamplingParams


# ---------------------------------------------------------------------------
# Scenario generation
# ---------------------------------------------------------------------------

def make_knowledge_base(num_docs: int, tokens_per_doc: int, seed_val: int = 42):
    """Create a 'knowledge base' of pre-tokenized document chunks."""
    seed(seed_val)
    return {
        doc_id: [randint(100, 100000) for _ in range(tokens_per_doc)]
        for doc_id in range(num_docs)
    }


def make_requests(
    kb: dict,
    num_requests: int,
    docs_per_request: int,
    query_len: int,
    seed_val: int = 42,
):
    """
    Each request picks `docs_per_request` chunks from the knowledge base
    in random order, appends a unique query suffix.

    Returns list of (doc_ids, query_tokens) pairs.
    """
    seed(seed_val)
    doc_ids = list(kb.keys())
    requests = []
    for _ in range(num_requests):
        chosen = sample(doc_ids, docs_per_request)
        shuffle(chosen)  # random order — this is what ContextPilot fixes
        query = [randint(100, 100000) for _ in range(query_len)]
        requests.append((chosen, query))
    return requests


def assemble_prompt(kb, doc_ids, query_tokens):
    """Concatenate document chunks + query into a flat token sequence."""
    tokens = []
    for did in doc_ids:
        tokens.extend(kb[did])
    tokens.extend(query_tokens)
    return tokens


# ---------------------------------------------------------------------------
# ContextPilot integration (offline, no server needed)
# ---------------------------------------------------------------------------

def apply_contextpilot(requests):
    """
    Use ContextPilot to:
      1. Reorder docs within each request (intra-context) → create shared prefixes
      2. Schedule request execution order (inter-context) → group similar requests

    Returns (reordered_requests, execution_order).
    """
    # Extract doc_id lists for ContextPilot
    contexts = [req[0] for req in requests]

    # Step 1: Build index + intra-context reorder
    idx = ContextIndex(alpha=0.001)
    result = idx.fit_transform(contexts)

    # Step 2: Inter-context scheduling
    scheduler = InterContextScheduler()
    _, _, execution_order, _ = scheduler.schedule_contexts(result)

    # Reconstruct requests with reordered doc_ids + original queries
    reordered_requests = []
    for i, orig_idx in enumerate(execution_order):
        new_doc_ids = result.reordered_contexts[orig_idx]
        original_query = requests[orig_idx][1]
        reordered_requests.append((new_doc_ids, original_query))

    return reordered_requests, execution_order


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def flush_cache(llm):
    """Reset prefix cache so runs don't leak cache state to each other."""
    llm.scheduler.block_manager.reset_prefix_cache()


def run_once(llm, prompts, cache_aware, max_tokens=1):
    sp = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=max_tokens)
          for _ in range(len(prompts))]
    t0 = time.perf_counter()
    result = llm.generate(prompts, sp, use_tqdm=False, use_context_optimizer=cache_aware)
    elapsed = time.perf_counter() - t0
    total_in = sum(len(p) for p in prompts)
    total_out = sum(len(o["token_ids"]) for o in result["outputs"])
    return {
        "time": elapsed,
        "ttft": result["ttft"],
        "input_tokens": total_in,
        "output_tokens": total_out,
    }


def benchmark_scenario(llm, kb, requests, label, max_tokens=1):
    print(f"\n{'='*70}")
    print(f"Scenario: {label}")
    print(f"{'='*70}")
    print(f"  Knowledge base: {len(kb)} docs, {len(next(iter(kb.values())))} tokens/doc")
    print(f"  Requests: {len(requests)}, {len(requests[0][0])} docs/req")

    # --- Baseline: random doc order, no cache-aware scheduling ---
    baseline_prompts = [assemble_prompt(kb, r[0], r[1]) for r in requests]
    avg_len = sum(len(p) for p in baseline_prompts) // len(baseline_prompts)
    print(f"  Avg prompt length: {avg_len} tokens")

    # --- ContextPilot: reorder + schedule ---
    t_cp = time.perf_counter()
    cp_requests, exec_order = apply_contextpilot(requests)
    cp_time = time.perf_counter() - t_cp
    cp_prompts = [assemble_prompt(kb, r[0], r[1]) for r in cp_requests]
    print(f"  ContextPilot reorder time: {cp_time*1000:.1f}ms")

    # Show prefix sharing improvement
    def count_shared_prefix_tokens(prompts):
        """Count total tokens that share a prefix with the previous prompt."""
        total = 0
        for i in range(1, len(prompts)):
            shared = 0
            for a, b in zip(prompts[i-1], prompts[i]):
                if a == b:
                    shared += 1
                else:
                    break
            total += shared
        return total

    base_shared = count_shared_prefix_tokens(baseline_prompts)
    cp_shared = count_shared_prefix_tokens(cp_prompts)
    print(f"  Adjacent shared prefix tokens — baseline: {base_shared}, ContextPilot: {cp_shared}")
    if base_shared > 0:
        print(f"  Prefix sharing improvement: {cp_shared/base_shared:.1f}x")
    else:
        print(f"  Prefix sharing improvement: {cp_shared} vs 0 (∞x)")

    # A-B-B-A benchmark
    results = []
    for i, (name, prompts, ca) in enumerate([
        ("Baseline", baseline_prompts, False),
        ("ContextPilot", cp_prompts, True),
        ("ContextPilot", cp_prompts, True),
        ("Baseline", baseline_prompts, False),
    ]):
        print(f"  Run {i+1}/4: {name}...", end=" ", flush=True)
        flush_cache(llm)
        r = run_once(llm, deepcopy(prompts), cache_aware=ca, max_tokens=max_tokens)
        ttft_str = f", TTFT={r['ttft']:.3f}s" if r['ttft'] else ""
        print(f"{r['time']:.3f}s{ttft_str}")
        results.append(r)

    t_base = (results[0]["time"] + results[3]["time"]) / 2
    t_cp = (results[1]["time"] + results[2]["time"]) / 2

    print(f"\n  {'─'*50}")
    print(f"  Baseline        — {t_base:.3f}s")
    print(f"  ContextPilot    — {t_cp:.3f}s")
    print(f"  Speedup:          {t_base/t_cp:.2f}x")

    def avg_ttft(a, b):
        if a and b: return (a+b)/2
        return a or b
    ttft_b = avg_ttft(results[0]["ttft"], results[3]["ttft"])
    ttft_c = avg_ttft(results[1]["ttft"], results[2]["ttft"])
    if ttft_b and ttft_c:
        print(f"  TTFT speedup:     {ttft_b/ttft_c:.2f}x")


def main():
    model_path = "/root/study/lite_llama/my_weight/qwen3-0.6B"
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    print("="*70)
    print("ContextPilot x nano-vllm: Full Integration Benchmark")
    print("="*70)
    print()
    print("ContextPilot creates shared token prefixes by reordering document")
    print("chunks within each request. Combined with nano-vllm's cache-aware")
    print("scheduler, this maximizes KV cache prefix reuse.")

    print("\nLoading model...", end=" ", flush=True)
    t0 = time.time()
    llm = LLM(
        model_path,
        enforce_eager=False,
        tensor_parallel_size=1,
        max_num_batched_tokens=4096,
        max_num_seqs=256,
    )
    print(f"done ({time.time()-t0:.1f}s)")

    # Warmup — trigger CUDA graph capture and JIT compilation
    print("Warmup...", end=" ", flush=True)
    warmup_sp = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=8)
    warmup_prompts = [[randint(0, 100000) for _ in range(512)] for _ in range(16)]
    llm.generate(warmup_prompts, [warmup_sp]*16, use_tqdm=False, use_context_optimizer=False)
    flush_cache(llm)
    print("done")

    # ── Scenario 1: Dense overlap, many requests ──
    kb1 = make_knowledge_base(num_docs=20, tokens_per_doc=256, seed_val=42)
    reqs1 = make_requests(kb1, num_requests=128, docs_per_request=5, query_len=64, seed_val=42)
    benchmark_scenario(llm, kb1, reqs1,
        "20 docs x 256tok, 128 reqs picking 5 docs each", max_tokens=1)

    # ── Scenario 2: Longer docs, fewer reqs ──
    kb2 = make_knowledge_base(num_docs=15, tokens_per_doc=512, seed_val=123)
    reqs2 = make_requests(kb2, num_requests=64, docs_per_request=4, query_len=64, seed_val=123)
    benchmark_scenario(llm, kb2, reqs2,
        "15 docs x 512tok, 64 reqs picking 4 docs each", max_tokens=1)

    # ── Scenario 3: With generation to show end-to-end benefit ──
    kb3 = make_knowledge_base(num_docs=20, tokens_per_doc=256, seed_val=456)
    reqs3 = make_requests(kb3, num_requests=128, docs_per_request=5, query_len=64, seed_val=456)
    benchmark_scenario(llm, kb3, reqs3,
        "20x256tok, 128 reqs, max_tokens=32", max_tokens=32)

    # ── Scenario 4: High overlap (3 docs from pool of 8) ──
    kb4 = make_knowledge_base(num_docs=8, tokens_per_doc=512, seed_val=789)
    reqs4 = make_requests(kb4, num_requests=128, docs_per_request=3, query_len=64, seed_val=789)
    benchmark_scenario(llm, kb4, reqs4,
        "8 docs x 512tok, 128 reqs picking 3 (high overlap)", max_tokens=1)

    print(f"\n{'='*70}")
    print("Benchmark complete")
    print("="*70)


if __name__ == "__main__":
    main()
