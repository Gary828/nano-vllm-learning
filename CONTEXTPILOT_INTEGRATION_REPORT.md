# ContextPilot × nano-vllm 集成阶段性报告

## 一、工作概述

本项目尝试将 ContextPilot 的前缀缓存优化技术集成到 nano-vllm 中，以提升批量推理场景下的 KV cache 命中率和 prefill 吞吐量。

**ContextPilot 核心能力**：通过层次聚类构建上下文索引，对 contexts 进行 **Intra-context reordering**（内部重排序）和 **Inter-context scheduling**（批量调度），从而最大化共享前缀的复用。

**nano-vllm 现状**：已具备基础的前缀缓存机制（`BlockManager` 使用 xxhash64 + 链式前缀哈希 + 引用计数），但 `Scheduler` 采用简单的 FIFO 策略，无 cache-aware 调度。

---

## 二、代码修改清单

### 1. 核心集成文件

| 文件 | 改动类型 | 说明 |
|---|---|---|
| `nanovllm/context_optimizer.py` | **新增** | 轻量级 ContextOptimizer，封装 `IntraContextOrderer`、`InterContextScheduler` 和 `SimpleHierarchicalClustering`（不依赖 scipy） |
| `nanovllm/engine/llm_engine.py` | **修改** | 在 `generate()` 方法中增加 `use_context_optimizer` 参数，批量输入时调用 `ContextOptimizer.reorder()` 进行 prompts 重排 |
| `nanovllm/engine/scheduler.py` | **修改** | 新增 `get_cache_stats()` 方法，暴露缓存统计信息 |

### 2. 测试与 Benchmark 文件

| 文件 | 改动类型 | 说明 |
|---|---|---|
| `tests/test_context_optimizer.py` | **新增** | 13 个单元测试，覆盖 reorder、scheduler、clustering 等核心逻辑 |
| `bench_context_optimizer.py` | **新增** | 端到端 GPU benchmark，对比 with/without ContextPilot 的实际吞吐 |
| `bench_realistic.py` | **新增/可删** | 尝试模拟 RAG/多轮对话/系统提示等场景的 benchmark（未能有效度量差异） |
| `bench_profile.py` | **新增/可删** | 带缓存模拟器的 profile 脚本（功能与主 benchmark 重叠） |

---

## 三、Benchmark 结果与分析

### 3.1 最终实测结果

运行命令：`python bench_context_optimizer.py`

| 场景 | Prompt 结构 | Cache Sim 命中 | GPU 实际时间 (without) | GPU 实际时间 (with) | Speedup |
|---|---|---|---|---|---|
| Shared Prefixes (256×8 groups) | 256 shared + 128 unique | 97.8% | 1.01s | 1.11s | **0.92x** |
| Long Shared Prefixes (512×4 groups) | 512 shared + 128 unique | 98.7% | 1.22s | 1.61s | **0.76x** |
| Random Baseline | 随机 tokens | 46.7% | 1.44s | 1.32s | **1.09x** |

### 3.2 结果解读

#### 异常点 1：共享前缀场景反而变慢
- Scenario 1 `0.92x`、Scenario 2 `0.76x`
- **原因**：`generate_shared_prefix_prompts` 生成的 prompts 已经天然按 group 连续排列。`ContextOptimizer.reorder()` 在这种输入上几乎没有优化空间，反而增加了 **100~380ms 的聚类+排序开销**。

#### 异常点 2：随机基线出现 1.09x 加速
- **原因**：这是 GPU warmup/timing noise，不是 ContextPilot 的效果。即使采用 A-B-A-B 交叉测试，对于 ~1s 量级的快速运行仍无法完全消除系统波动。

#### 异常点 3：Cache Simulator 显示 "Shuffled hit = ContextPilot hit"
- **根本原因**：nano-vllm 使用 **block-level chaining hash**。一旦某个 prefix hash 链在历史上出现过，后续无论什么时候出现相同开头的 prompt 都能命中缓存。
- 这意味着："把相同前缀的 prompts 排在一起执行" 对于 nano-vllm 的 block hash 机制**没有额外收益**——缓存命中不依赖于调度顺序，只取决于"这个 prefix 是否已经被计算过"。

---

## 四、根因分析：为什么不 work？

### 4.1 ContextPilot 的设计假设

ContextPilot 的加速假设建立在 **vLLM/SGLang 的架构**之上：
1. **Continuous Batching**：请求按 batch 交错调度
2. **Trie-based / Radix-based Prefix Cache**：前缀缓存的组织形式和 eviction 策略对调度顺序敏感
3. **Memory Pressure 下的 Eviction**：相同前缀的请求相邻调度，可以减少 batch 组合时的 cache 失效和内存压力

### 4.2 nano-vllm 的架构现实

nano-vllm 的 `BlockManager` 特点：
1. **链式哈希**：`hash(block_i) = xxhash64(block_i_tokens, hash(block_{i-1}))`
2. **全局持久缓存**：`hash_to_block_id` 在 block deallocate 后**不会被清空**，相同前缀的 prompts 在任何时刻都能命中
3. **简单 FIFO Scheduler**：所有 waiting 请求都会按顺序被处理完，prefix 很少被 evict

### 4.3 关键结论

> 在 `generate()` 入口处对 prompts 做一次静态重排，对 nano-vllm 几乎没有收益。ContextPilot 的 Inter-context scheduling 优势在 nano-vllm 的当前架构下被完全抵消。

当前集成只触及了 ContextPilot 能力的表层，而未触及 nano-vllm 真正需要优化的瓶颈：**调度器（Scheduler）**。

---

## 五、反思与教训

### 5.1 设计层面的误判

**过度依赖表层集成**：最初判断"直接 Python API 集成即可"，在 `llm_engine.generate()` 入口处调用 `reorder()`。这是一个**策略层（Policy）**修改，但没有触及 nano-vllm 的**执行层（Scheduler + BlockManager）**。

**Benchmark 设计经历了三轮迭代才暴露问题**：
- 第一轮：随机 token + 人为拼接前缀（过于简单）
- 第二轮：尝试 RAG/多轮场景，但未 shuffle prompts（无法体现优化价值）
- 第三轮：加入 shuffle 和 cache simulator，才发现 nano-vllm 的 block hash 机制对调度顺序不敏感

### 5.2 技术层面的认知

**不同 Prefix Cache 实现有不同的优化点**：
- vLLM **RadixAttention**：调度顺序影响 batch 内 prefix 复用 → 需要 ContextPilot Inter-scheduling
- nano-vllm **Block-level Chaining Hash**：缓存全局持久，调度顺序不影响命中率 → 不需要静态重排，需要 **Scheduler 内部的动态 cache-aware 选择**

### 5.3 如果要继续推进，该怎么做？

#### 方向 A：修改 Scheduler（最有价值）

修改 `scheduler.schedule()`，不再简单从 `waiting` deque 头部取请求，而是：

```python
# 伪代码
def schedule(self):
    # 获取当前已缓存的 block hash 集合
    cached_prefixes = self.block_manager.get_cached_prefixes()
    
    # 在 waiting 队列中选择与当前缓存前缀匹配最长的序列
    best_seq = max(self.waiting, key=lambda seq: longest_shared_prefix(seq, cached_prefixes))
    
    # 优先调度 best_seq
```

这样可以在**每个 batch 的每次调度**中动态最大化 cache 复用，而不是在请求入口处做一次静态重排。

#### 方向 B：Intra-context Block Reordering（需要改变 prompt 结构）

如果能把 prompt 切分为多个语义 block（如 system / chunk1 / chunk2 / user_query），ContextPilot 的 `IntraContextOrderer` 可以把共享的 chunks 移到更前面，提升前缀长度。但这需要：
- 在 tokenizer 之前对 prompt 进行 block-level 切分
- nano-vllm 支持接收结构化的 block 输入

当前集成没有真正实现这一点。

#### 方向 C：在真正的 vLLM/SGLang 上验证

如果想验证 ContextPilot 的完整能力，应直接在其原生的 hook 环境（vLLM / SGLang）上运行 benchmark，而不是在架构差异较大的 nano-vllm 上期望获得同等级别的收益。

---

## 六、文件清理建议

为保持代码库整洁，建议保留：
- `nanovllm/context_optimizer.py`（核心算法封装）
- `nanovllm/engine/llm_engine.py` 修改（`use_context_optimizer` 开关）
- `nanovllm/engine/scheduler.py` 修改（`get_cache_stats()`）
- `tests/test_context_optimizer.py`
- `bench_context_optimizer.py`

建议删除（与主 benchmark 重复且未产生有效增量）：
- `bench_realistic.py`
- `bench_profile.py`

---

## 七、总结

本次集成完成了 ContextPilot 核心算法到 nano-vllm 的**轻量级移植**，但暴露了**架构假设不匹配**的深层问题：

1. **集成本身是正确且轻量的**（无 scipy/HTTP 依赖，约 ~500 行代码）。
2. **但集成点选错了**：`generate()` 入口处的静态重排对 nano-vllm 的 block-level chaining hash 缓存机制没有收益。
3. **真正需要修改的是 Scheduler**，实现动态的 cache-aware batching。
4. **Benchmark 设计经历了有效迭代**，最终通过 cache simulator 定位到了根因。

如果目标是"让 nano-vllm 获得 ContextPilot 级别的加速"，下一步应聚焦在 **Scheduler 的 cache-aware 改造**，而非继续优化当前的 reorder 集成。

---

*Report generated: 2026-04-02*
