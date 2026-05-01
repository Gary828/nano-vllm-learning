<p align="center">
<img width="300" src="assets/logo.png">
</p>

# 基于Nano-vLLM改进

**参考自:** [git@github.com:GeeeekExplorer/nano-vllm.git](https://github.com/GeeeekExplorer/nano-vllm)

## 特性

- 当前支持：`Qwen3`、`MiniCPM`、`Llama`
- 已验证：
  - `Qwen3-0.6B`
  - `MiniCPM4.1`
  - 本地 `Llama-3.2-1B-Instruct` 单请求端到端 smoke（2026-04-17）
- 已集成能力：
  - Prefix cache
  - Cache-aware scheduling
  - ContextPilot 基准链路
  - 实验性 KV cache quant（`int8` / `fp8`）
- 在仓库提供的 `bench.py` 配置上，离线吞吐与 vLLM 处于同一量级

## 安装

```bash
pip install git+git@github.com:Gary828/nano-vllm-learning.git
```

## 模型下载

```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## 快速开始

完整示例见 [`example.py`](example.py)。当前 `LLM.generate()` 返回的是一个 dict：

```python
from nanovllm import LLM, SamplingParams

llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]

result = llm.generate(prompts, sampling_params)
print(result["outputs"][0]["text"])
```

注意：

- 当前 `SamplingParams.temperature` 必须大于 `1e-10`，也就是暂不支持 `temperature=0` 的 greedy 模式。

可选的 KV cache quant 模式：

```python
llm = LLM("/YOUR/MODEL/PATH", kv_cache_quant="int8")
llm = LLM("/YOUR/MODEL/PATH", kv_cache_quant="fp8_e4m3fn")
llm = LLM("/YOUR/MODEL/PATH", kv_cache_quant="fp8_e5m2")
```

## Benchmark

仓库内包含 4 个 benchmark：

- [`bench.py`](bench.py)：通用离线吞吐 benchmark
- [`bench_cache_aware.py`](bench_cache_aware.py)：仅测试 cache-aware scheduler
- [`bench_contextpilot.py`](bench_contextpilot.py)：测试 ContextPilot + cache-aware scheduling
- [`bench_kv_quant.py`](bench_kv_quant.py)：测试 KV cache quant（`baseline / int8 / fp8`）

### 离线吞吐

- 测试配置：
- 硬件：RTX 4070 Laptop（8GB）
- 模型：Qwen3-0.6B
- 请求数：256
- 输入长度：100–1024 tokens 随机采样
- 输出长度：100–1024 tokens 随机采样

结果：
| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     | 98.37    | 1361.84               |
| Nano-vLLM      | 133,966     | 93.41    | 1434.13               |

### ContextPilot + cache-aware scheduling

在 [`bench_contextpilot.py`](bench_contextpilot.py) 的长上下文合成场景中，当前集成结果大致为：

- **1.39x ~ 1.85x** 端到端加速
- **最高 1.44x TTFT** 改善

收益主要来自两层：

- ContextPilot 通过重排文档块创造共享前缀
- Scheduler 通过 cache-aware batching 利用这些共享前缀

### KV cache quant

当前主干包含实验性的 KV cache quant：

- `int8`：相对稳定，静态 KV 容量约提升 **1.75x ~ 1.84x**
- `fp8_e4m3fn / fp8_e5m2`：功能可用，但当前 runtime 仍未优化

当前已知结论：

- KV quant benchmark 已明确证明 **静态显存收益**
- 但长上下文场景下，当前实现会先把量化 KV 重新 materialize 成浮点再做 attention，因此 runtime 可能退化

## 文档

分析文档见 [`docs/`](docs)：

- [`docs/TECHNICAL_REPORT.md`](docs/TECHNICAL_REPORT.md)：ContextPilot + cache-aware scheduler 技术报告
- [`docs/CONTEXTPILOT_INTEGRATION_REPORT.md`](docs/CONTEXTPILOT_INTEGRATION_REPORT.md)：早期集成失败与复盘
- [`docs/kv_cache_quant_interview.md`](docs/kv_cache_quant_interview.md)：KV cache quant 设计与 benchmark 解读
- [`docs/nanovllm_minicpm41_interview.md`](docs/nanovllm_minicpm41_interview.md)：MiniCPM 支持说明
- [`docs/llama_main_qwen3_minicpm41_adaptation.md`](docs/pr88_llama_main_qwen3_minicpm41_adaptation.md)：Qwen3 / Llama / MiniCPM4.1 模型支持演进分析
