import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    prompts = [
        "who are you?",
        "介绍一下中国的国宝大熊猫",
        "Write a quick sorting algorithm in python",
        "write me a quick sort algorithm with c++",
        "what's LLM?",
        "Can you explain what is AI?",
    ]
    sampling_params = [
        SamplingParams(
            temperature=0.6,
            max_tokens=256,
        )
        for prompt in prompts
    ]
    result = llm.generate(prompts, sampling_params)
    outputs = result["outputs"]

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt}")
        print(f"Output: {output['text']}")
        print(f"Token ids: {output['token_ids']}")
        print(f"Token count: {len(output['token_ids'])}")
        print(f"Tokenizer count: {len(tokenizer.encode(output['text']))}")


if __name__ == "__main__":
    main()
