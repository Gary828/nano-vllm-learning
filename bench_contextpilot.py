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
import time
import logging
import argparse
import io
from collections import deque
from contextlib import redirect_stdout
from copy import deepcopy
from random import randint, seed, shuffle, sample

os.environ["CONTEXTPILOT_LOG_LEVEL"] = "ERROR"
logging.disable(logging.WARNING)

from contextpilot.context_index.index_construction import ContextIndex
from contextpilot.context_ordering.inter_scheduler import InterContextScheduler

from nanovllm import LLM, SamplingParams
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence


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

def apply_contextpilot(requests, verbose=False):
    """
    Use ContextPilot to:
      1. Reorder docs within each request (intra-context) → create shared prefixes
      2. Schedule request execution order (inter-context) → group similar requests

    Returns (reordered_requests, execution_order).
    """
    # Extract doc_id lists for ContextPilot
    contexts = [req[0] for req in requests]

    # Step 1: Build index + intra-context reorder
    if verbose:
        idx = ContextIndex(alpha=0.001)
        result = idx.fit_transform(contexts)
    else:
        with redirect_stdout(io.StringIO()):
            idx = ContextIndex(alpha=0.001)
            result = idx.fit_transform(contexts)

    # Step 2: Inter-context scheduling
    scheduler = InterContextScheduler()
    _, _, execution_order, _ = scheduler.schedule_contexts(result)

    # Reconstruct requests with reordered doc_ids + original queries
    reordered_requests = []
    for orig_idx in execution_order:
        new_doc_ids = result.reordered_contexts[orig_idx]
        original_query = requests[orig_idx][1]
        reordered_requests.append((new_doc_ids, original_query))

    return reordered_requests, execution_order, result.stats


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def flush_cache(llm):
    """Reset prefix cache so runs don't leak cache state to each other."""
    llm.scheduler.block_manager.reset_prefix_cache()


def format_optional(value, suffix="s", precision=3):
    if value is None:
        return "n/a"
    return f"{value:.{precision}f}{suffix}"


def percentile(values, q):
    """Linear-interpolated percentile for a list of numeric values."""
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def summarize_distribution(values):
    """Return mean/p50/p95 summary, skipping missing values."""
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {
        "mean": sum(values) / len(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
    }


def merge_metric_lists(*lists):
    merged = []
    for values in lists:
        if values:
            merged.extend(v for v in values if v is not None)
    return merged


def avg_optional(a, b):
    if a is not None and b is not None:
        return (a + b) / 2
    return a or b


def count_shared_prefix_tokens(prompts):
    """Count total tokens that share a prefix with the previous prompt."""
    total = 0
    for i in range(1, len(prompts)):
        shared = 0
        for a, b in zip(prompts[i - 1], prompts[i]):
            if a == b:
                shared += 1
            else:
                break
        total += shared
    return total


def simulate_prompt_packing(
    prompts,
    *,
    cache_aware,
    max_num_batched_tokens,
    max_num_seqs,
    num_blocks,
    block_size,
):
    """
    Offline estimate of prompt-side scheduling only.

    It mirrors the scheduler's prefill packing logic, then deallocates each
    scheduled batch immediately to approximate the max_tokens=1 prompt path.
    """
    bm = BlockManager(num_blocks=num_blocks, block_size=block_size)
    waiting = deque(Sequence(prompt) for prompt in prompts)
    rounds = []

    def prefix_key(seq):
        if seq.num_blocks >= 1:
            tokens = seq.block(0)
            if len(tokens) == bm.block_size:
                return bm.compute_hash(tokens)
        return -1

    while waiting:
        if cache_aware and len(waiting) > 1:
            waiting = deque(sorted(waiting, key=prefix_key))

        batch = []
        predicted_new_total = 0
        actual_new_total = 0
        cached_total = 0

        while waiting and len(batch) < max_num_seqs:
            seq = waiting[0]
            if cache_aware:
                predicted_new = len(seq) - bm.count_cached_blocks(seq) * bm.block_size
                predicted_new = max(predicted_new, 1)
            else:
                predicted_new = len(seq)
            if predicted_new_total + predicted_new > max_num_batched_tokens or not bm.can_allocate(seq):
                break

            seq = waiting.popleft()
            bm.allocate(seq)
            actual_new = len(seq) - seq.num_cached_tokens
            batch.append(seq)
            predicted_new_total += predicted_new
            actual_new_total += actual_new
            cached_total += seq.num_cached_tokens

        if not batch:
            raise RuntimeError("Prompt packing simulation could not schedule any sequence.")

        rounds.append({
            "num_seqs": len(batch),
            "predicted_new_tokens": predicted_new_total,
            "actual_new_tokens": actual_new_total,
            "cached_tokens": cached_total,
        })

        for seq in batch:
            bm.deallocate(seq)

    total_prompt_tokens = sum(len(prompt) for prompt in prompts)
    total_new_tokens = sum(round_info["actual_new_tokens"] for round_info in rounds)
    total_cached_tokens = sum(round_info["cached_tokens"] for round_info in rounds)
    first_round = rounds[0]
    return {
        "prefill_rounds": len(rounds),
        "avg_batch_seqs": sum(round_info["num_seqs"] for round_info in rounds) / len(rounds),
        "first_batch_seqs": first_round["num_seqs"],
        "first_batch_new_tokens": first_round["actual_new_tokens"],
        "total_new_tokens": total_new_tokens,
        "total_cached_tokens": total_cached_tokens,
        "cache_hit_rate": total_cached_tokens / total_prompt_tokens if total_prompt_tokens else 0.0,
    }


def print_distribution_summary(title, baseline, contextpilot, note=None):
    if not baseline or not contextpilot:
        return
    print()
    print(f"  {title}")
    if note:
        print(f"  {note}")
    print("  Method         | Mean        | P50         | P95")
    print("  -------------- | ----------- | ----------- | -----------")
    print(
        f"  Baseline       | {baseline['mean']:>9.3f}s | "
        f"{baseline['p50']:>9.3f}s | {baseline['p95']:>9.3f}s"
    )
    print(
        f"  ContextPilot   | {contextpilot['mean']:>9.3f}s | "
        f"{contextpilot['p50']:>9.3f}s | {contextpilot['p95']:>9.3f}s"
    )
    if contextpilot["mean"] > 0 and contextpilot["p50"] > 0 and contextpilot["p95"] > 0:
        print(
            f"  {title} speedup: "
            f"mean {baseline['mean'] / contextpilot['mean']:.2f}x, "
            f"p50 {baseline['p50'] / contextpilot['p50']:.2f}x, "
            f"p95 {baseline['p95'] / contextpilot['p95']:.2f}x"
        )


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
        "batch_first_token_time": result.get(
            "batch_first_token_time",
            result.get("ttft_token", result.get("ttft")),
        ),
        "batch_first_decode_step_time": result.get("ttfd_decode_step"),
        "per_request_ttft": result.get("per_request_ttft", []),
        "per_request_completion_time": result.get("per_request_completion_time", []),
        "input_tokens": total_in,
        "output_tokens": total_out,
        "requests_per_second": len(prompts) / elapsed if elapsed > 0 else None,
        "total_tokens_per_second": (total_in + total_out) / elapsed if elapsed > 0 else None,
        "output_tokens_per_second": total_out / elapsed if elapsed > 0 else None,
    }


def benchmark_scenario(llm, kb, requests, label, max_tokens=1, verbose_contextpilot=False, focus=None):
    print(f"\n{'='*70}")
    print(f"Scenario: {label}")
    print(f"{'='*70}")
    print(f"  Knowledge base: {len(kb)} docs, {len(next(iter(kb.values())))} tokens/doc")
    print(f"  Requests: {len(requests)}, {len(requests[0][0])} docs/req")
    print(f"  max_tokens: {max_tokens}")
    if focus:
        print(f"  Focus: {focus}")

    # --- Baseline: random doc order, no cache-aware scheduling ---
    baseline_prompts = [assemble_prompt(kb, r[0], r[1]) for r in requests]
    avg_len = sum(len(p) for p in baseline_prompts) // len(baseline_prompts)
    print(f"  Avg prompt length: {avg_len} tokens")

    # --- ContextPilot: reorder + schedule ---
    t_cp = time.perf_counter()
    cp_requests, _, cp_stats = apply_contextpilot(requests, verbose=verbose_contextpilot)
    cp_time = time.perf_counter() - t_cp
    cp_prompts = [assemble_prompt(kb, r[0], r[1]) for r in cp_requests]
    print(f"  ContextPilot reorder time: {cp_time*1000:.1f}ms")
    print(f"  ContextPilot index: {cp_stats.get('total_nodes', 0)} nodes, {cp_stats.get('leaf_nodes', 0)} leaves")

    base_shared = count_shared_prefix_tokens(baseline_prompts)
    cp_shared = count_shared_prefix_tokens(cp_prompts)
    print(f"  Adjacent shared prefix tokens — baseline: {base_shared}, ContextPilot: {cp_shared}")
    if base_shared > 0:
        print(f"  Prefix sharing improvement: {cp_shared/base_shared:.1f}x")
    else:
        print(f"  Prefix sharing improvement: {cp_shared} vs 0 (∞x)")

    pack_base = simulate_prompt_packing(
        baseline_prompts,
        cache_aware=False,
        max_num_batched_tokens=llm.scheduler.max_num_batched_tokens,
        max_num_seqs=llm.scheduler.max_num_seqs,
        num_blocks=len(llm.scheduler.block_manager.blocks),
        block_size=llm.scheduler.block_manager.block_size,
    )
    pack_cp = simulate_prompt_packing(
        cp_prompts,
        cache_aware=True,
        max_num_batched_tokens=llm.scheduler.max_num_batched_tokens,
        max_num_seqs=llm.scheduler.max_num_seqs,
        num_blocks=len(llm.scheduler.block_manager.blocks),
        block_size=llm.scheduler.block_manager.block_size,
    )

    print()
    print("  Prompt-side scheduler estimate (offline prefill simulation)")
    print("  Method         | Prefill Rounds | Avg Batch Seqs | First Batch Seqs | Total New Tok | Cached Tok")
    print("  -------------- | -------------- | -------------- | ---------------- | ------------- | ----------")
    print(
        f"  Baseline       | {pack_base['prefill_rounds']:>14} | "
        f"{pack_base['avg_batch_seqs']:>14.1f} | "
        f"{pack_base['first_batch_seqs']:>16} | "
        f"{pack_base['total_new_tokens']:>13} | "
        f"{pack_base['total_cached_tokens']:>10}"
    )
    print(
        f"  ContextPilot   | {pack_cp['prefill_rounds']:>14} | "
        f"{pack_cp['avg_batch_seqs']:>14.1f} | "
        f"{pack_cp['first_batch_seqs']:>16} | "
        f"{pack_cp['total_new_tokens']:>13} | "
        f"{pack_cp['total_cached_tokens']:>10}"
    )
    print(
        "  Prompt cache hit rate — "
        f"baseline: {pack_base['cache_hit_rate'] * 100:.1f}%, "
        f"ContextPilot: {pack_cp['cache_hit_rate'] * 100:.1f}%"
    )
    if pack_cp["total_new_tokens"] > 0:
        print(f"  Prompt-side new-token reduction: {pack_base['total_new_tokens'] / pack_cp['total_new_tokens']:.2f}x")
    if pack_cp["prefill_rounds"] > 0:
        print(f"  Prefill-round reduction: {pack_base['prefill_rounds'] / pack_cp['prefill_rounds']:.2f}x")
    if pack_base["first_batch_seqs"] > 0:
        print(f"  First-batch size gain: {pack_cp['first_batch_seqs'] / pack_base['first_batch_seqs']:.2f}x vs baseline")

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
        metric_parts = []
        if r["batch_first_token_time"] is not None:
            metric_parts.append(f"batch-first-token={r['batch_first_token_time']:.3f}s")
        if r["batch_first_decode_step_time"] is not None:
            metric_parts.append(f"batch-first-decode={r['batch_first_decode_step_time']:.3f}s")
        metric_suffix = f", {', '.join(metric_parts)}" if metric_parts else ""
        print(f"{r['time']:.3f}s{metric_suffix}")
        results.append(r)

    t_base_generate = (results[0]["time"] + results[3]["time"]) / 2
    t_cp_generate = (results[1]["time"] + results[2]["time"]) / 2

    batch_first_token_b = avg_optional(
        results[0]["batch_first_token_time"],
        results[3]["batch_first_token_time"],
    )
    batch_first_token_c = avg_optional(
        results[1]["batch_first_token_time"],
        results[2]["batch_first_token_time"],
    )
    batch_first_decode_b = avg_optional(
        results[0]["batch_first_decode_step_time"],
        results[3]["batch_first_decode_step_time"],
    )
    batch_first_decode_c = avg_optional(
        results[1]["batch_first_decode_step_time"],
        results[2]["batch_first_decode_step_time"],
    )
    request_throughput_b = avg_optional(results[0]["requests_per_second"], results[3]["requests_per_second"])
    request_throughput_c = avg_optional(results[1]["requests_per_second"], results[2]["requests_per_second"])
    total_tok_s_b = avg_optional(results[0]["total_tokens_per_second"], results[3]["total_tokens_per_second"])
    total_tok_s_c = avg_optional(results[1]["total_tokens_per_second"], results[2]["total_tokens_per_second"])
    output_tok_s_b = avg_optional(results[0]["output_tokens_per_second"], results[3]["output_tokens_per_second"])
    output_tok_s_c = avg_optional(results[1]["output_tokens_per_second"], results[2]["output_tokens_per_second"])

    per_request_ttft_b = summarize_distribution(
        merge_metric_lists(results[0]["per_request_ttft"], results[3]["per_request_ttft"])
    )
    per_request_ttft_c = summarize_distribution(
        merge_metric_lists(results[1]["per_request_ttft"], results[2]["per_request_ttft"])
    )
    per_request_completion_b = summarize_distribution(
        merge_metric_lists(
            results[0]["per_request_completion_time"],
            results[3]["per_request_completion_time"],
        )
    )
    per_request_completion_c = summarize_distribution(
        merge_metric_lists(
            results[1]["per_request_completion_time"],
            results[2]["per_request_completion_time"],
        )
    )

    base_total = t_base_generate
    cp_total = cp_time + t_cp_generate

    print(f"\n  {'─'*50}")
    print("  Stage breakdown (avg of A-B-B-A runs)")
    print("  Method         | ContextPilot Overhead | LLM Generate (prefill+decode) | End-to-end | Batch First-Token | Batch First-Decode")
    print("  -------------- | --------------------- | ----------------------------- | ---------- | ----------------- | ------------------")
    print(
        f"  Baseline       | {0.0:>19.3f}s | {t_base_generate:>27.3f}s | {base_total:>8.3f}s | "
        f"{format_optional(batch_first_token_b):>17} | {format_optional(batch_first_decode_b):>18}"
    )
    print(
        f"  ContextPilot   | {cp_time:>19.3f}s | {t_cp_generate:>27.3f}s | {cp_total:>8.3f}s | "
        f"{format_optional(batch_first_token_c):>17} | {format_optional(batch_first_decode_c):>18}"
    )
    print(f"  LLM-generate speedup: {t_base_generate/t_cp_generate:.2f}x")
    print(f"  End-to-end speedup (incl. ContextPilot overhead): {base_total/cp_total:.2f}x")

    if batch_first_token_b is not None and batch_first_token_c is not None and batch_first_token_c > 0:
        print(f"  Batch first-token speedup: {batch_first_token_b/batch_first_token_c:.2f}x")
    if batch_first_decode_b is not None and batch_first_decode_c is not None and batch_first_decode_c > 0:
        print(f"  Batch first-decode speedup: {batch_first_decode_b/batch_first_decode_c:.2f}x")

    print()
    print("  Throughput (avg of A-B-B-A runs)")
    print("  Method         | Requests/s  | Total Tok/s | Output Tok/s")
    print("  -------------- | ----------- | ----------- | ------------")
    print(
        f"  Baseline       | {format_optional(request_throughput_b, suffix='', precision=2):>11} | "
        f"{format_optional(total_tok_s_b, suffix='', precision=1):>11} | "
        f"{format_optional(output_tok_s_b, suffix='', precision=1):>12}"
    )
    print(
        f"  ContextPilot   | {format_optional(request_throughput_c, suffix='', precision=2):>11} | "
        f"{format_optional(total_tok_s_c, suffix='', precision=1):>11} | "
        f"{format_optional(output_tok_s_c, suffix='', precision=1):>12}"
    )

    print_distribution_summary(
        "Per-request TTFT (merged across A-B-B-A runs)",
        per_request_ttft_b,
        per_request_ttft_c,
    )
    print_distribution_summary(
        "Per-request Completion Time (merged across A-B-B-A runs)",
        per_request_completion_b,
        per_request_completion_c,
        note="  Note: completion time equals TTFT when max_tokens=1.",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark ContextPilot + nano-vllm with clear stage breakdown output."
    )
    parser.add_argument(
        "--verbose-contextpilot",
        action="store_true",
        help="Show detailed ContextPilot distance/clustering logs.",
    )
    args = parser.parse_args()

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

    scenario_specs = [
        {
            "label": "20 docs x 256tok, 128 reqs picking 5 docs each",
            "focus": "Prefill-only / dense overlap / many requests",
            "num_docs": 20,
            "tokens_per_doc": 256,
            "num_requests": 128,
            "docs_per_request": 5,
            "query_len": 64,
            "seed_val": 42,
            "max_tokens": 1,
        },
        {
            "label": "15 docs x 512tok, 64 reqs picking 4 docs each",
            "focus": "Prefill-only / longer context / fewer requests",
            "num_docs": 15,
            "tokens_per_doc": 512,
            "num_requests": 64,
            "docs_per_request": 4,
            "query_len": 64,
            "seed_val": 123,
            "max_tokens": 1,
        },
        {
            "label": "20x256tok, 128 reqs, max_tokens=32",
            "focus": "Short decode / end-to-end impact after prompt optimization",
            "num_docs": 20,
            "tokens_per_doc": 256,
            "num_requests": 128,
            "docs_per_request": 5,
            "query_len": 64,
            "seed_val": 456,
            "max_tokens": 32,
        },
        {
            "label": "8 docs x 512tok, 128 reqs picking 3 (high overlap)",
            "focus": "Prefill-only / high-overlap upper bound",
            "num_docs": 8,
            "tokens_per_doc": 512,
            "num_requests": 128,
            "docs_per_request": 3,
            "query_len": 64,
            "seed_val": 789,
            "max_tokens": 1,
        },
        {
            "label": "64 docs x 256tok, 64 reqs picking 4 (low-overlap control)",
            "focus": "Prefill-only / low-overlap control to bound best-case claims",
            "num_docs": 64,
            "tokens_per_doc": 256,
            "num_requests": 64,
            "docs_per_request": 4,
            "query_len": 64,
            "seed_val": 321,
            "max_tokens": 1,
        },
        {
            "label": "8 docs x 512tok, 64 reqs picking 3, max_tokens=128",
            "focus": "High overlap + long decode tail / expose completion-latency behavior",
            "num_docs": 8,
            "tokens_per_doc": 512,
            "num_requests": 64,
            "docs_per_request": 3,
            "query_len": 64,
            "seed_val": 654,
            "max_tokens": 128,
        },
    ]

    for spec in scenario_specs:
        kb = make_knowledge_base(
            num_docs=spec["num_docs"],
            tokens_per_doc=spec["tokens_per_doc"],
            seed_val=spec["seed_val"],
        )
        reqs = make_requests(
            kb,
            num_requests=spec["num_requests"],
            docs_per_request=spec["docs_per_request"],
            query_len=spec["query_len"],
            seed_val=spec["seed_val"],
        )
        benchmark_scenario(
            llm,
            kb,
            reqs,
            spec["label"],
            max_tokens=spec["max_tokens"],
            verbose_contextpilot=args.verbose_contextpilot,
            focus=spec["focus"],
        )

    print(f"\n{'='*70}")
    print("Benchmark complete")
    print("="*70)


if __name__ == "__main__":
    main()
