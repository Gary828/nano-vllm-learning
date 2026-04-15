import argparse
import os
import random

import torch

from nanovllm import LLM, SamplingParams


def build_passkey_prompt(length: int, passkey: int) -> str:
    filler = " ".join(f"filler{i}" for i in range(length))
    return (
        f"{filler}\n"
        f"The secret passkey is {passkey}.\n"
        "Repeat the secret passkey only."
    )


def load_prompts() -> list[str]:
    prompts = [
        "Summarize what a KV cache does in transformer inference in one sentence.",
        "List three practical ways to reduce GPU memory usage during LLM serving.",
        "Explain the difference between prefill and decode in two sentences.",
    ]
    random.seed(0)
    for context_len in (256, 1024, 2048):
        passkey = random.randint(10000, 99999)
        prompts.append(build_passkey_prompt(context_len, passkey))
    return prompts


def run_generation(model_path: str, prompts: list[str], kv_cache_quant: bool, max_tokens: int):
    torch.manual_seed(0)
    torch.cuda.empty_cache()
    llm = LLM(
        model_path,
        kv_cache_quant=kv_cache_quant,
        enforce_eager=False,
        max_model_len=32768,
        max_num_batched_tokens=32768,
        max_num_seqs=len(prompts),
    )
    sampling_params = [
        SamplingParams(temperature=1e-4, max_tokens=max_tokens, ignore_eos=False)
        for _ in prompts
    ]
    result = llm.generate(prompts, sampling_params, use_tqdm=False, use_context_optimizer=False)
    outputs = [item["text"].strip() for item in result["outputs"]]
    llm.exit()
    del llm
    torch.cuda.empty_cache()
    return outputs


def compare_outputs(base_outputs: list[str], quant_outputs: list[str]):
    matches = sum(a == b for a, b in zip(base_outputs, quant_outputs))
    print("\n" + "=" * 72)
    print("KV Cache Quant Quality Check")
    print("=" * 72)
    print(f"Exact output match rate: {matches}/{len(base_outputs)} = {matches / max(len(base_outputs), 1):.2%}")
    print()
    for idx, (base, quant) in enumerate(zip(base_outputs, quant_outputs), start=1):
        status = "match" if base == quant else "diff"
        print(f"[{idx}] {status}")
        print(f"  baseline: {base}")
        print(f"  quant   : {quant}")


def main():
    parser = argparse.ArgumentParser(description="Compare generation quality with and without KV cache quantization.")
    parser.add_argument("--model-path", default="/root/study/lite_llama/my_weight/qwen3-0.6B")
    parser.add_argument("--max-tokens", type=int, default=24)
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: model not found at {args.model_path}")
        return

    prompts = load_prompts()
    base_outputs = run_generation(args.model_path, prompts, False, args.max_tokens)
    quant_outputs = run_generation(args.model_path, prompts, True, args.max_tokens)
    compare_outputs(base_outputs, quant_outputs)


if __name__ == "__main__":
    main()
