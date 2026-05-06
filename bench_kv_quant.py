import argparse
import math
import os
import time
from contextlib import AbstractContextManager
from random import randint, seed

import torch

from nanovllm import LLM, SamplingParams
from nanovllm.layers.kv_quant import normalize_kv_cache_quant_dtype


def make_low_overlap_prompts(num_requests: int, prompt_len: int, seed_val: int) -> list[list[int]]:
    seed(seed_val)
    prompts = []
    for i in range(num_requests):
        prompt = [i + 1]
        prompt.extend(randint(100, 100000) for _ in range(prompt_len - 1))
        prompts.append(prompt)
    return prompts


def make_sampling_params(num_requests: int, max_tokens: int) -> list[SamplingParams]:
    return [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=max_tokens)
        for _ in range(num_requests)
    ]


class KvPathProfiler(AbstractContextManager):

    TARGETS = (
        "store_kvcache_int8",
        "_quantize_store_int8_pair_fused",
        "_gather_dequant_int8_fused",
        "store_kvcache_fp8",
        "_store_fp8_pair_fused",
        "_gather_cast_fp8_fused",
        "materialize_quantized_blocks",
        "build_local_block_tables",
    )

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self._module = None
        self._orig: dict[str, object] = {}
        self._stats: dict[str, dict[str, float | int]] = {}

    def __enter__(self):
        if not self.enabled:
            return self
        import nanovllm.layers.kv_quant as kvq

        self._module = kvq
        for name in self.TARGETS:
            if not hasattr(kvq, name):
                continue
            fn = getattr(kvq, name)
            if not callable(fn):
                continue
            self._orig[name] = fn

            def _wrap(fn_name, fn_impl):
                def _wrapped(*args, **kwargs):
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    out = fn_impl(*args, **kwargs)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    dt = time.perf_counter() - t0
                    row = self._stats.setdefault(fn_name, {"calls": 0, "total_s": 0.0})
                    row["calls"] = int(row["calls"]) + 1
                    row["total_s"] = float(row["total_s"]) + dt
                    return out

                return _wrapped

            setattr(kvq, name, _wrap(name, fn))
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._module is not None:
            for name, fn in self._orig.items():
                setattr(self._module, name, fn)
        return False

    def summary(self) -> list[dict]:
        rows = []
        for name, s in self._stats.items():
            calls = int(s["calls"])
            total_s = float(s["total_s"])
            rows.append(
                {
                    "name": name,
                    "calls": calls,
                    "total_s": total_s,
                    "avg_ms": (total_s / calls * 1000.0) if calls else 0.0,
                }
            )
        rows.sort(key=lambda x: x["total_s"], reverse=True)
        return rows


def run_case(
    model_path: str,
    prompts: list[list[int]],
    max_tokens: int,
    kv_cache_quant,
    max_model_len: int,
    max_num_batched_tokens: int,
    enforce_eager: bool,
    profile_kv_path: bool = False,
):
    quant_mode = normalize_kv_cache_quant_dtype(kv_cache_quant)
    torch.cuda.empty_cache()
    llm = LLM(
        model_path,
        enforce_eager=enforce_eager,
        kv_cache_quant=quant_mode,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=len(prompts),
    )
    sampling_params = make_sampling_params(len(prompts), max_tokens)
    init_allocated = torch.cuda.memory_allocated()
    init_reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()

    with KvPathProfiler(profile_kv_path) as kv_profiler:
        t0 = time.perf_counter()
        result = llm.generate(prompts, sampling_params, use_tqdm=False, use_context_optimizer=False)
        elapsed = time.perf_counter() - t0
    generate_peak_allocated = torch.cuda.max_memory_allocated()
    generate_peak_reserved = torch.cuda.max_memory_reserved()

    block_size = llm.model_runner.block_size
    num_blocks = llm.model_runner.config.num_kvcache_blocks
    seq_blocks = math.ceil((len(prompts[0]) + max_tokens) / block_size)
    est_max_live_seqs = num_blocks // max(seq_blocks, 1)
    total_output_tokens = sum(len(output["token_ids"]) for output in result["outputs"])
    llm.exit()
    metrics = {
        "mode": quant_mode or "baseline",
        "time": elapsed,
        "ttft": result.get("ttft_token"),
        "init_allocated_gb": init_allocated / 1024**3,
        "init_reserved_gb": init_reserved / 1024**3,
        "generate_peak_gb": generate_peak_allocated / 1024**3,
        "generate_peak_reserved_gb": generate_peak_reserved / 1024**3,
        "runtime_peak_delta_gb": max(generate_peak_allocated - init_allocated, 0) / 1024**3,
        "runtime_reserved_delta_gb": max(generate_peak_reserved - init_reserved, 0) / 1024**3,
        "num_kvcache_blocks": num_blocks,
        "est_max_live_seqs": est_max_live_seqs,
        "total_output_tokens": total_output_tokens,
        "kv_profile": kv_profiler.summary() if profile_kv_path else [],
    }
    del llm
    torch.cuda.empty_cache()
    return metrics


def print_summary(results: list[dict], prompt_len: int, max_tokens: int):
    print("\n" + "=" * 72)
    print("KV Cache Quant Benchmark")
    print("=" * 72)
    print(f"Prompt length: {prompt_len} tokens")
    print(f"Max output tokens: {max_tokens}")
    print()
    print("Mode       | Time(s) | TTFT(s) | Init GB | Gen Peak GB | Runtime +GB | KV Blocks | Est. Max Live Seqs")
    print("---------- | ------- | ------- | ------- | ----------- | ----------- | --------- | ------------------")
    for row in results:
        ttft = f"{row['ttft']:.3f}" if row["ttft"] is not None else "n/a"
        print(
            f"{row['mode']:<10} | "
            f"{row['time']:>7.3f} | "
            f"{ttft:>7} | "
            f"{row['init_allocated_gb']:>7.3f} | "
            f"{row['generate_peak_gb']:>11.3f} | "
            f"{row['runtime_peak_delta_gb']:>11.3f} | "
            f"{row['num_kvcache_blocks']:>9} | "
            f"{row['est_max_live_seqs']:>18}"
        )
        if row.get("kv_profile"):
            print("  kv-profile:")
            for p in row["kv_profile"]:
                print(
                    f"    {p['name']}: calls={p['calls']}, "
                    f"total={p['total_s']:.3f}s, avg={p['avg_ms']:.3f}ms"
                )

    base = results[0]
    print()
    for quant in results[1:]:
        print(f"{quant['mode']} KV block gain: {quant['num_kvcache_blocks'] / max(base['num_kvcache_blocks'], 1):.2f}x")
        print(f"{quant['mode']} estimated live-sequence gain: {quant['est_max_live_seqs'] / max(base['est_max_live_seqs'], 1):.2f}x")
        print(f"{quant['mode']} init allocated delta: {quant['init_allocated_gb'] - base['init_allocated_gb']:+.3f} GB")
        print(f"{quant['mode']} generate peak delta: {quant['generate_peak_gb'] - base['generate_peak_gb']:+.3f} GB")
        print(f"{quant['mode']} runtime temporary delta: {quant['runtime_peak_delta_gb'] - base['runtime_peak_delta_gb']:+.3f} GB")


def main():
    parser = argparse.ArgumentParser(description="Benchmark KV cache quantization capacity and runtime.")
    parser.add_argument("--model-path", default="/root/study/lite_llama/my_weight/qwen3-0.6B")
    parser.add_argument("--num-requests", type=int, default=64)
    parser.add_argument("--prompt-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run all modes with eager execution for stable cross-mode comparison.",
    )
    parser.add_argument(
        "--kv-cache-dtypes",
        default="baseline,int8,fp8_e4m3fn",
        help="Comma-separated modes to benchmark. Supported: baseline,int8,fp8_e4m3fn,fp8_e5m2",
    )
    parser.add_argument(
        "--profile-kv-path",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Collect lightweight per-function timing for kv quant path.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: model not found at {args.model_path}")
        return

    prompts = make_low_overlap_prompts(args.num_requests, args.prompt_len, args.seed)
    max_model_len = args.prompt_len + args.max_tokens + 16
    max_num_batched_tokens = max(args.num_requests * max(args.prompt_len, 1), max_model_len)
    modes = []
    for item in args.kv_cache_dtypes.split(","):
        item = item.strip().lower()
        if item == "baseline":
            modes.append(None)
        else:
            modes.append(item)

    results = [
        run_case(
            args.model_path,
            prompts,
            args.max_tokens,
            mode,
            max_model_len,
            max_num_batched_tokens,
            args.enforce_eager,
            args.profile_kv_path,
        )
        for mode in modes
    ]
    print_summary(results, args.prompt_len, args.max_tokens)


if __name__ == "__main__":
    main()
