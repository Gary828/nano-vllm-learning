<p align="center">
<img width="300" src="assets/logo.png">
</p>

<p align="center">
<a href="https://trendshift.io/repositories/15323" target="_blank"><img src="https://trendshift.io/api/badge/repositories/15323" alt="GeeeekExplorer%2Fnano-vllm | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

# Nano-vLLM

# 基于Nano-vLLM改进

**参考自:** [git@github.com:GeeeekExplorer/nano-vllm.git](https://github.com/GeeeekExplorer/nano-vllm)

## 特性

* 🚀 **Fast offline inference** - Comparable inference speeds to vLLM
* 📖 **Readable codebase** - Clean implementation in ~ 1,200 lines of Python code
* ⚡ **Optimization Suite** - Prefix caching, Tensor Parallelism, Torch compilation, CUDA graph, CUDA graph etc.

- 当前支持：`Qwen3`、`MiniCPM`、`Llama`
- 已验证：
  - Qwen3-0.6B / 1.7B / 4B / 8B / 14B / 30B（含 FP8 KV cache）
  - MiniCPM4.1-0.5B / 8B
  - Llama

## Model Download

To download the model weights manually, use the following command:
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## 快速开始

完整示例见 [`example.py`](example.py)。当前 `LLM.generate()` 返回的是一个 dict：

- `outputs`: 生成结果列表（每项含 `text` 和 `token_ids`）
- `ttft` / `ttft_token`: batch 首 token 时间（秒）
- `per_request_ttft`: 各请求 TTFT（秒）
- `per_request_completion_time`: 各请求完成时间（秒）
- `ttfd_decode_step`: 首次 decode 调度时间（秒）
- `total_time`: 整个 batch 耗时（秒）

```python
from nanovllm import LLM, SamplingParams

llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
result = llm.generate(prompts, sampling_params)
text = result["outputs"][0]["text"]
```

可选开启 KV cache quant：

```python
llm = LLM("/YOUR/MODEL/PATH", kv_cache_quant="int8")
llm = LLM("/YOUR/MODEL/PATH", kv_cache_quant="fp8_e4m3fn")
llm = LLM("/YOUR/MODEL/PATH", kv_cache_quant="fp8_e5m2")
```

## Benchmark

- [`bench.py`](bench.py)：基础离线吞吐
- [`bench_cache_aware.py`](bench_cache_aware.py)：cache-aware 调度策略效果
- [`bench_contextpilot.py`](bench_contextpilot.py)：测试 ContextPilot + cache-aware scheduling
- [`bench_kv_quant.py`](bench_kv_quant.py)：测试 KV cache quant（`baseline / int8 / fp8`）

### 离线吞吐

**Test Configuration:**
- Hardware: RTX 4070 Laptop (8GB)
- Model: Qwen3-0.6B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens

- 测试配置：
- 硬件：RTX 4070 Laptop（8GB）
- 模型：Qwen3-0.6B
- 请求：256 条
- 输入长度：100-1024 随机
- 输出长度：100-1024 随机

| Framework      | Total Tokens | Total Time | Throughput (tokens/sec) |
|----------------|--------------|------------|--------------------------|
| vLLM           | 133,966      | 98.37      | 1361.84                  |
| Nano-vLLM      | 133,966      | 93.41      | 1434.13                  |

### ContextPilot + cache-aware scheduling

在 [`bench_contextpilot.py`](bench_contextpilot.py) 的长上下文合成场景中，当前集成结果大致为：

- baseline：prefill TTFT ~ 1.4s，decode 总耗时 ~ 66s
- contextpilot+cache-aware：prefill TTFT ~ 0.42s，decode 总耗时 ~ 40s

### KV cache quant

在 [`bench_kv_quant.py`](bench_kv_quant.py) 中，Qwen3-0.6B（A5000）典型结果：

- `baseline`: 约 13.6k tok/s
- `int8`: 约 13.6k tok/s（显存明显下降）
- `fp8_e4m3fn`: 约 13.9k tok/s

详细测试与原理说明见 `docs/kv_cache_quant_interview.md`。

## 相关文档

- [`docs/contextpilot_scheduler_interview.md`](docs/contextpilot_scheduler_interview.md)：ContextPilot + cache-aware 调度设计与效果
- [`docs/cache_aware_scheduler_refactor.md`](docs/cache_aware_scheduler_refactor.md)：调度器重构说明
- [`docs/kv_cache_quant_interview.md`](docs/kv_cache_quant_interview.md)：KV cache quant 设计与 benchmark 解读
- [`docs/nanovllm_minicpm41_interview.md`](docs/nanovllm_minicpm41_interview.md)：MiniCPM 支持说明
- [`docs/llama_main_qwen3_minicpm41_adaptation.md`](docs/pr88_llama_main_qwen3_minicpm41_adaptation.md)：Qwen3 / Llama / MiniCPM4.1 模型支持演进分析

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)
