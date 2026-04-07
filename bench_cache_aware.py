"""
Benchmark: Cache-Aware Scheduler (ContextPilot-powered) vs FIFO Baseline.

Demonstrates that prefix-hash sorting in the scheduler groups same-prefix
sequences together, allowing more sequences per prefill batch and fewer
total prefill steps.

Usage:
    python bench_cache_aware.py
"""

import os
import time
from copy import deepcopy
from random import randint, seed, shuffle

from nanovllm import LLM, SamplingParams


def generate_shared_prefix_prompts(
    num_groups: int = 8,
    seqs_per_group: int = 32,
    prefix_len: int = 512,
    suffix_len: int = 128,
    seed_value: int = 42,
) -> list[list[int]]:
    """Generate prompts with shared prefixes, then shuffle to simulate real dispatch."""
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

    # Shuffle to simulate realistic unordered request arrival
    shuffle(prompts)
    return prompts


def flush_gpu_cache(llm, num_prompts=4, prompt_len=512):
    """Overwrite prefix cache state with random prompts."""
    sp = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=1)
    prompts = [[randint(0, 100000) for _ in range(prompt_len)] for _ in range(num_prompts)]
    llm.generate(prompts, [sp] * num_prompts, use_tqdm=False, use_context_optimizer=False)


def run_once(llm, prompts, use_optimizer, max_tokens=1):
    """Single benchmark run. Returns dict with timing and stats."""
    sp = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=max_tokens)
          for _ in range(len(prompts))]

    t0 = time.perf_counter()
    result = llm.generate(prompts, sp, use_tqdm=False, use_context_optimizer=use_optimizer)
    elapsed = time.perf_counter() - t0

    total_input = sum(len(p) for p in prompts)
    total_output = sum(len(o["token_ids"]) for o in result["outputs"])

    return {
        "time": elapsed,
        "ttft": result["ttft"],
        "input_tokens": total_input,
        "output_tokens": total_output,
        "throughput": (total_input + total_output) / elapsed,
    }


def benchmark_scenario(llm, prompts, label, max_tokens=1):
    """A-B-A-B interleaved benchmark for one scenario."""
    print(f"\n{'='*60}")
    print(f"Scenario: {label}")
    print(f"{'='*60}")
    print(f"  Prompts: {len(prompts)}, avg len: {sum(len(p) for p in prompts)//len(prompts)} tokens")
    print(f"  max_tokens: {max_tokens}")

    # A-B-B-A pattern
    runs = []
    for i, (name, opt) in enumerate([
        ("FIFO", False), ("Cache-Aware", True),
        ("Cache-Aware", True), ("FIFO", False),
    ]):
        print(f"  Run {i+1}/4: {name}...", end=" ", flush=True)
        flush_gpu_cache(llm)
        p = deepcopy(prompts)
        r = run_once(llm, p, use_optimizer=opt, max_tokens=max_tokens)
        ttft_str = f"{r['ttft']:.3f}s" if r['ttft'] is not None else "N/A"
        print(f"{r['time']:.3f}s  (TTFT={ttft_str})")
        runs.append(r)

    t_fifo = (runs[0]["time"] + runs[3]["time"]) / 2
    t_aware = (runs[1]["time"] + runs[2]["time"]) / 2
    def avg_ttft(a, b):
        if a is not None and b is not None:
            return (a + b) / 2
        return a or b

    ttft_fifo = avg_ttft(runs[0]["ttft"], runs[3]["ttft"])
    ttft_aware = avg_ttft(runs[1]["ttft"], runs[2]["ttft"])

    print(f"\n  Results:")
    print(f"    FIFO        — time: {t_fifo:.3f}s, TTFT: {ttft_fifo:.3f}s" if ttft_fifo else f"    FIFO        — time: {t_fifo:.3f}s")
    print(f"    Cache-Aware — time: {t_aware:.3f}s, TTFT: {ttft_aware:.3f}s" if ttft_aware else f"    Cache-Aware — time: {t_aware:.3f}s")
    print(f"    Speedup:  {t_fifo/t_aware:.2f}x total", end="")
    if ttft_fifo and ttft_aware:
        print(f", {ttft_fifo/ttft_aware:.2f}x TTFT")
    else:
        print()


def main():
    model_path = os.path.expanduser("/root/study/lite_llama/my_weight/qwen3-0.6B")
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    print("="*60)
    print("Cache-Aware Scheduler Benchmark (ContextPilot x nano-vllm)")
    print("="*60)

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

    # Warmup
    print("Warmup...", end=" ", flush=True)
    flush_gpu_cache(llm)
    print("done")

    # Scenario 1: Many groups, moderate prefix — prefill-dominated
    prompts1 = generate_shared_prefix_prompts(
        num_groups=8, seqs_per_group=32, prefix_len=512, suffix_len=128, seed_value=42,
    )
    benchmark_scenario(llm, prompts1, "8 groups x 32 seqs, prefix=512, suffix=128", max_tokens=1)

    # Scenario 2: Fewer groups, longer prefix — higher cache hit potential
    prompts2 = generate_shared_prefix_prompts(
        num_groups=4, seqs_per_group=64, prefix_len=768, suffix_len=128, seed_value=123,
    )
    benchmark_scenario(llm, prompts2, "4 groups x 64 seqs, prefix=768, suffix=128", max_tokens=1)

    # Scenario 3: More groups, more seqs — stress test
    prompts3 = generate_shared_prefix_prompts(
        num_groups=16, seqs_per_group=16, prefix_len=512, suffix_len=128, seed_value=456,
    )
    benchmark_scenario(llm, prompts3, "16 groups x 16 seqs, prefix=512, suffix=128", max_tokens=1)

    # Scenario 4: Many groups, longer prefix — maximize scheduling benefit
    prompts4 = generate_shared_prefix_prompts(
        num_groups=32, seqs_per_group=16, prefix_len=512, suffix_len=128, seed_value=789,
    )
    benchmark_scenario(llm, prompts4, "32 groups x 16 seqs, prefix=512, suffix=128", max_tokens=1)

    # Scenario 5: With decode to show total-time + TTFT benefit
    prompts5 = generate_shared_prefix_prompts(
        num_groups=16, seqs_per_group=16, prefix_len=512, suffix_len=128, seed_value=101,
    )
    benchmark_scenario(llm, prompts5, "16x16, prefix=512, suffix=128, max_tokens=32", max_tokens=32)

    print(f"\n{'='*60}")
    print("Benchmark complete")
    print("="*60)


if __name__ == "__main__":
    main()
