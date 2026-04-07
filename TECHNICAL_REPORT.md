# ContextPilot x nano-vllm: Cache-Aware Scheduling 技术报告

## 一、项目背景与目标

### 1.1 两个项目

- **nano-vllm**: 轻量级 vLLM 实现（~1200 行核心代码），具备 prefix caching（block-level chaining hash）和 continuous batching，但调度器为简单 FIFO。
- **ContextPilot**: KV cache 优化系统，通过层次聚类和上下文重排序最大化 prefix cache 复用。已在 vLLM/SGLang 上验证 1.5-3.5x prefill 加速。

### 1.2 目标

将 ContextPilot 的优化能力集成到 nano-vllm 中，在多请求长上下文场景下获得可测量的性能提升。

### 1.3 前置尝试的失败（阶段性报告总结）

此前已有一次集成尝试：在 `generate()` 入口对 prompts 做静态重排。结果 **0.76-1.09x**（无提升甚至变慢）。

根因：nano-vllm 的 block hash 全局持久（`hash_to_block_id` 在 deallocate 后不清空），调度顺序对 cache 命中率无影响。静态重排增加了聚类开销却无收益。

---

## 二、设计方案

### 2.1 核心洞察

分析 nano-vllm 调度器源码后发现**两个可优化点**：

**问题 1：FIFO 调度不感知 cache**

```python
# 原始 scheduler.py
seq = self.waiting[0]  # 永远取队首，不考虑 cache 状态
```

Shuffled 输入下，不同前缀的序列交替排列，无法利用 batch 内的 prefix 共享。

**问题 2：batch budget 检查过于保守**

```python
# 原始检查：用序列全长 len(seq) 做 budget，不区分 cached/uncached
if num_batched_tokens + len(seq) > self.max_num_batched_tokens:
    break
```

一个 1344-token 序列，即使有 1280 tokens 被 cache（只需计算 64 tokens），也占 1344 的 budget。导致一个 batch 只能塞 3 个序列（而非 20+）。

### 2.2 两层优化架构

```
应用层 (ContextPilot)                    引擎层 (nano-vllm Scheduler)
┌─────────────────────────┐              ┌──────────────────────────────┐
│ 1. Intra-Context Reorder│              │ 3. Prefix-Hash Sorting       │
│    把共享 docs 移到开头   │  ─token─→   │    按首 block hash 分组排序   │
│ 2. Inter-Context Schedule│              │ 4. Cache-Aware Budget Check  │
│    相似请求相邻执行       │              │    用 predicted_new 做 budget │
└─────────────────────────┘              └──────────────────────────────┘
```

- **层 1-2（ContextPilot）**：解决 "token 前缀不同" 的根本问题——通过重排 document chunks 创造共享前缀。
- **层 3-4（Scheduler）**：解决 "调度器不利用 cache" 的效率问题——通过排序和精确 budget 估算最大化 batch 利用率。

### 2.3 为什么分两层？

单独做任一层效果有限：
- 只做 ContextPilot 重排（前置尝试）→ 调度器的保守 budget 检查浪费了重排创造的 cache 收益
- 只做 Scheduler 优化 → 如果 prompts 本身没有共享前缀，再好的调度也无 cache 可利用

---

## 三、代码修改详解

### 3.1 `block_manager.py` — 新增 cache 感知方法（+30 行）

**新增 `count_cached_blocks(seq)`**：预测序列能命中多少 cache blocks（含已释放但未被覆盖的）。

```python
def count_cached_blocks(self, seq: Sequence) -> int:
    h = -1
    hits = 0
    for i in range(seq.num_blocks):
        token_ids = seq.block(i)
        h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
        block_id = self.hash_to_block_id.get(h, -1)
        if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
            break  # 链式哈希：一旦 miss，后续全部 miss
        hits += 1
    return hits
```

**为什么需要它**：调度器在 allocate 之前需要预测 "这个序列实际需要计算多少新 token"，才能做精确的 batch budget 估算。

**新增 `count_reusable_blocks(seq)`**：只统计在 `used_block_ids` 中的 cache hits（不占 free block）。

**修改 `can_allocate(seq)`**：

```python
# 原始：需要 free_blocks >= num_blocks（不考虑 cache hit 共享 block 的情况）
# 修改后：
def can_allocate(self, seq):
    reusable = self.count_reusable_blocks(seq)
    return len(self.free_block_ids) >= (seq.num_blocks - reusable)
```

**为什么修改**：如果序列 A 的前缀 block 已被序列 B 占用（ref_count > 0），A 可以通过 ref_count++ 共享该 block，不消耗 free block。原始检查不考虑这一点，在内存紧张时误拒可调度的序列。

**新增 `reset_prefix_cache()`**：清空 hash 映射，用于 benchmark 间隔离。

### 3.2 `scheduler.py` — cache-aware 调度（+20 行）

**新增 `_sort_waiting_by_prefix()`**：

```python
def _sort_waiting_by_prefix(self):
    if len(self.waiting) <= 1:
        return
    bm = self.block_manager
    def prefix_key(seq):
        if seq.num_blocks >= 1:
            tokens = seq.block(0)
            if len(tokens) == bm.block_size:
                return bm.compute_hash(tokens)
        return -1
    self.waiting = deque(sorted(self.waiting, key=prefix_key))
```

**为什么按第一个 block 的 hash 排序**：nano-vllm 使用链式 hash（`hash(block_i) = xxhash64(tokens, hash(block_{i-1}))`），第一个 block 的 hash 决定了整条前缀链是否匹配。相同第一 block 的序列必然共享前缀。O(N log N) 排序，比聚类的 O(N^2) 快得多。

**修改 `schedule()` 的 prefill 循环**：

```python
# 原始：用 len(seq) 做 budget check
if num_batched_tokens + len(seq) > self.max_num_batched_tokens:
    break

# 修改后：用预测的新 token 数做 check
if self.cache_aware:
    predicted_new = len(seq) - self.block_manager.count_cached_blocks(seq) * self.block_manager.block_size
    predicted_new = max(predicted_new, 1)
else:
    predicted_new = len(seq)
if num_batched_tokens + predicted_new > self.max_num_batched_tokens:
    break
```

**为什么这是关键改动**：以 `max_num_batched_tokens=4096`，1344-token 序列为例：

| 模式 | 每序列 budget 占用 | 一个 batch 能塞 |
|------|-------------------|----------------|
| 原始 FIFO | 1344（全长） | 3 个序列 |
| Cache-Aware | 64（仅未缓存部分） | 20+ 个序列 |

### 3.3 `llm_engine.py` — 简化集成（改 2 行）

```python
# 移除旧的 ContextOptimizer 静态重排（15 行 try/except 块）
# 替换为一行：
self.scheduler.cache_aware = use_context_optimizer
```

调度逻辑下沉到 Scheduler，`generate()` 只控制开关。

### 3.4 `config.py` — 新增配置项（+1 行）

```python
cache_aware: bool = True
```

### 3.5 总结：改动量

| 文件 | 改动行数 | 性质 |
|------|---------|------|
| `block_manager.py` | +30 | 新增 3 个方法，修改 1 个方法 |
| `scheduler.py` | +20 | 新增 1 个方法，修改 schedule() |
| `llm_engine.py` | -15, +2 | 移除旧集成，简化为一行 |
| `config.py` | +1 | 新增配置字段 |
| **总计** | **~50 行净增** | |

---

## 四、遇到的问题与解决

### 4.1 问题：首次集成（静态重排）无效

**现象**：在 `generate()` 入口做 ContextOptimizer.reorder()，benchmark 显示 0.76-1.09x。

**根因分析**：
1. nano-vllm 的 `hash_to_block_id` 在 deallocate 后不清空 → 调度顺序不影响 cache 命中率
2. 聚类开销（100-380ms）吃掉了可能的收益
3. Benchmark 的 prompts 本身已按 group 排列（未 shuffle）

**解决**：放弃 "静态重排" 思路，转向 "动态调度优化"。将优化点从 generate() 入口移到 Scheduler 内部。

### 4.2 问题：Greedy 调度改进后仅 1.03-1.06x

**现象**：实现了 prefix-hash sorting，但加速只有 3-6%。

**根因分析**：排序确实把相同前缀的序列分组了，但调度器的 **batch budget check** 仍用 `len(seq)`（全长），即使 cache hit 也占满了 budget → 每个 batch 塞不进更多序列。

**解决**：修改 budget check 为 `predicted_new`（预测的新 token 数）。这是本项目最关键的改动，将加速从 1.06x 提升到 1.12-1.16x。

### 4.3 问题：Benchmark 中 Run 间 cache 泄漏

**现象**：A-B-B-A 测试中，Run 3（ContextPilot）比 Run 2 快 8 倍（0.1s vs 0.8s）。

**根因**：`flush_cache()` 只生成 4 个短 prompt（8 blocks），不足以覆盖前一轮的数百个 cached blocks。`hash_to_block_id` 映射持久存在。

**解决**：新增 `reset_prefix_cache()` 方法直接清空 hash 映射表，确保每轮 benchmark 从干净状态开始。

### 4.4 问题：首次 Run 异常慢（冷启动）

**现象**：Scenario 1 Run 1 = 16.8s，Run 4 = 1.3s。

**根因**：CUDA graph capture 和 JIT 编译在首次推理时触发。

**解决**：在 benchmark 前增加 warmup 阶段（16 个 prompt，max_tokens=8），预热 CUDA graph 和 kernel。

---

## 五、Benchmark 设计与性能结果

### 5.0 三个 Benchmark 的区别

本项目涉及三个 benchmark 脚本，各自测试的东西完全不同：

#### 原始 `bench.py` —— 纯引擎吞吐基准（与 ContextPilot 无关）

nano-vllm 自带的 benchmark。256 个**完全随机**的 prompt，互相没有任何关系，测的是引擎原始吞吐量。

```python
# 每个 prompt 都是独立的随机 token，长度也随机
prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, 1024))]
                    for _ in range(256)]
```

**特点**：没有共享前缀、没有文档结构、没有 cache 复用的可能性。任何 prefix cache 优化在这个场景下都不会有收益，因为根本没有可复用的前缀。这个 benchmark 用来验证我们的改动**不会导致性能退化**。

#### 新增 `bench_cache_aware.py` —— 验证调度器优化（共享前缀已存在）

专门构造的场景：多组 prompt **天然自带共享前缀**，但被 shuffle 打乱了顺序。

```python
# 8 个 group，每 group 32 个 prompt，共享 512 个 token 的前缀
# 每个 prompt = [共享前缀 512 tokens] + [各自不同 128 tokens]
# 生成后 shuffle，模拟请求无序到达
shared_prefixes = [[randint(0, 100000) for _ in range(512)] for _ in range(8)]
prompts = []
for prefix in shared_prefixes:
    for _ in range(32):
        suffix = [randint(0, 100000) for _ in range(128)]
        prompts.append(prefix + suffix)
shuffle(prompts)  # 打乱：A1,B1,C1,A2,B2,...
```

**测试的是**：共享前缀已经存在，调度器能否通过排序和更好的 batch packing 来利用它。这里**不涉及 ContextPilot 的文档重排**，只测引擎层的调度优化。

**类比**：桌上有一副已经按花色分好的扑克牌，被打乱了顺序。优化做的事是重新按花色排好。

#### 新增 `bench_contextpilot.py` —— 验证 ContextPilot 完整链路（共享前缀需要创造）

模拟真实使用场景：有一个"知识库"（20 篇文档），每个请求从中随机选 5 篇、随机排列，加一个 unique query。

```python
# 知识库：20 篇文档，每篇 256 tokens
kb = {doc_id: [randint(100, 100000) for _ in range(256)] for doc_id in range(20)}

# 每个请求：从 20 篇中随机选 5 篇，随机排列
for _ in range(128):
    chosen = sample(doc_ids, 5)
    shuffle(chosen)         # 关键：每个请求的文档顺序都不同
    query = [randint(...)]  # 加 unique query
    requests.append((chosen, query))

# 请求 1: [Doc3, Doc7, Doc12, Doc1, Doc15] + query
# 请求 2: [Doc7, Doc1, Doc9, Doc3, Doc18] + query  ← 共享 Doc1,3,7 但顺序不同
# → token 前缀完全不同 → prefix cache 无法命中
```

**测试的是**：ContextPilot 通过重排文档顺序**创造**共享前缀（把共享文档移到开头），再配合 cache-aware scheduler 利用这些前缀。

**类比**：每个人手里有不同组合的扑克牌，散乱地握着。ContextPilot 做的事是让每个人把共同持有的牌排到最前面，这样前几张牌一模一样，可以"复印"而不是重新"画"。

#### 三者关系总结

| | bench.py | bench_cache_aware.py | bench_contextpilot.py |
|---|---|---|---|
| **来源** | nano-vllm 原有 | 本项目新增 | 本项目新增 |
| **Prompt 结构** | 完全随机 | 人造共享前缀 + shuffle | 模拟知识库选文档 |
| **共享前缀** | 不存在 | 天然存在，被打乱 | 不存在，需要 ContextPilot 创造 |
| **测试目标** | 引擎原始吞吐 | 调度器排序 + budget 优化 | ContextPilot 重排 + 调度器全链路 |
| **用到 ContextPilot？** | 否 | 否（只用引擎层优化） | 是（应用层 + 引擎层） |
| **预期加速** | 无（基准线） | 1.12-1.16x | 1.4-1.85x |

---

### 5.1 Cache-Aware Scheduler 单独效果（bench_cache_aware.py）

场景：已有共享前缀的 prompts（shuffled），对比 FIFO vs Cache-Aware Scheduler。

| Scenario | Speedup |
|----------|---------|
| 8 groups x 32 seqs, prefix=512 | **1.12x** |
| 16 groups x 16 seqs, prefix=512 | **1.16x** |
| 16x16, max_tokens=32 | **1.08x total, 1.18x TTFT** |

### 5.2 ContextPilot 完整集成效果（bench_contextpilot.py）

场景：模拟多请求长上下文——知识库 + 每请求选子集 chunks（随机顺序）。

| Scenario | Prefix Sharing | Speedup | TTFT |
|----------|---------------|---------|------|
| 20 docs x 256tok, 128 reqs, 5 docs/req | 33x | **1.47x** | - |
| 15 docs x 512tok, 64 reqs, 4 docs/req | 12x | **1.60x** | - |
| 20 docs, 128 reqs, max_tokens=32 | 46x | **1.39x** | **1.44x** |
| 8 docs x 512tok, 128 reqs, 3 docs/req | 18x | **1.85x** | - |

### 5.3 结果分析

**为什么完整集成（1.39-1.85x）比单独 Scheduler（1.08-1.16x）效果好得多？**

因为两层优化解决的是**不同的问题**：

- **Scheduler 单独**：prompts 已有共享前缀，Scheduler 只是更好地利用它。改善幅度受限于 "原始 FIFO 已经有一定 cache 命中" 的事实。
- **ContextPilot + Scheduler**：ContextPilot 将 adjacent shared prefix tokens 从 2048 提升到 68096（33x），创造了大量原本不存在的 cache 命中机会。Scheduler 再把这些机会高效利用。

**两层各自的贡献（定性分析）**：
- ContextPilot Intra-Reorder：将 "无共享前缀" 变为 "大量共享前缀"（**创造 cache 机会**）
- Cache-Aware Budget Check：让调度器能在一个 batch 中塞入 20+ 个 cached 序列（**利用 cache 机会**）
- Prefix-Hash Sorting：确保同前缀序列在 waiting 队列中相邻（**组织 cache 机会**）

---

## 六、整体流程分析

### 6.1 数据流

```
用户请求（各自选了不同 document chunks，随机排序）
    │
    ▼
ContextPilot (应用层，offline，~80ms)
    ├─ 层次聚类：发现请求间共享哪些 chunks
    ├─ Intra-Context Reorder：把共享 chunks 移到每个 prompt 开头
    └─ Inter-Context Schedule：相似请求相邻执行
    │
    ▼
Tokenize + 送入 nano-vllm generate()
    │
    ▼
Scheduler.schedule() (引擎层，每个 prefill batch 执行一次)
    ├─ _sort_waiting_by_prefix()：按首 block hash 分组排序
    ├─ count_cached_blocks()：预测每个序列能 cache hit 多少 blocks
    ├─ predicted_new budget check：精确估算 batch 需要的新 token 数
    └─ can_allocate()：考虑共享 block 的内存需求
    │
    ▼
ModelRunner.run() (GPU)
    ├─ prepare_prefill()：只处理 seq[num_cached_tokens:] 的新 token
    ├─ Flash Attention：cached token 的 KV 从 block table 读取
    └─ Triton Kernel：只写入新 token 的 KV 到 cache
    │
    ▼
输出结果
```

### 6.2 关键路径上的每一步如何减少计算

| 步骤 | 原始 | 优化后 | 效果 |
|------|------|--------|------|
| Prompt 组装 | chunks 随机排列 | 共享 chunks 在前 | 创造共享 token 前缀 |
| 调度排序 | FIFO | prefix-hash sort | 相同前缀的序列在同一 batch |
| Budget check | `len(seq)` | `predicted_new` | 一个 batch 塞更多 cached 序列 |
| Prefill 计算 | 所有 token | 仅未缓存 token | 跳过 cached 前缀的计算 |

### 6.3 时间复杂度

| 组件 | 复杂度 | 说明 |
|------|--------|------|
| ContextPilot 聚类 | O(N^2) | N=请求数，距离矩阵计算 |
| Prefix-hash sorting | O(N log N) | 每次 prefill batch 前执行 |
| count_cached_blocks | O(B) | B=序列的 block 数，每序列调用一次 |
| 总 overhead | ~80ms | 128 请求时的 ContextPilot 耗时 |

---

## 七、与 vLLM 原生 ContextPilot 集成的对比

| 维度 | vLLM + ContextPilot | nano-vllm + ContextPilot (本项目) |
|------|--------------------|---------------------------------|
| 集成方式 | monkey-patch BlockPool + HTTP server | 直接修改 Scheduler (~50 行) |
| 缓存架构 | RadixAttention (Trie-based) | Block-level chaining hash |
| 调度顺序敏感性 | 高（eviction 策略依赖顺序） | 低（hash 全局持久） |
| 优化机制 | Intra-reorder + eviction-aware scheduling | Intra-reorder + cache-aware batch packing |
| 改动量 | 0 行（monkey-patch） | ~50 行 |
| 加速效果 | 1.5-3.5x prefill | 1.4-1.9x total |

---

## 八、面试要点总结

### 关键技术点

1. **Prefix Cache 的工作原理**：block-level chaining hash (`hash(block_i) = xxhash64(tokens, hash(block_{i-1}))`)，链式结构意味着一旦 miss 后续全部 miss。

2. **为什么重排 document order 能创造 cache 命中**：LLM prefix cache 是 token-level 的，只有 token 序列完全相同才能命中。不同的 document 排列 → 不同的 token 序列 → cache miss。ContextPilot 把共享 documents 统一移到开头 → 相同的 token 前缀 → cache hit。

3. **Scheduler 的 batch packing 优化**：原始 budget check 用全长 `len(seq)` 是因为注意力计算需要读完整 KV cache。但 prefill 的 GPU 计算量只与**新 token 数**成正比（cached token 的 KV 已在显存中），所以 budget 应该用 `predicted_new`。

4. **`can_allocate` 的 false negative**：当序列 A 和 B 共享前缀 block（ref_count > 1），B 不需要额外的 free block 来存放共享部分。原始实现要求 `free >= num_blocks` 会在内存紧张时误拒本可调度的序列。

5. **为什么要先做 Greedy 验证再做完整集成**：避免重蹈覆辙（前置尝试直接搬算法但架构假设不匹配）。Greedy 版（~20 行）用最小代价验证了 "cache-aware scheduling 对 nano-vllm 有收益" 的假设，然后才引入完整的 ContextPilot 集成。

### 能展示的能力

- **系统分析能力**：从失败的集成出发，通过源码分析定位到真正的瓶颈（不是 eviction，而是 batch packing）
- **渐进式验证**：Greedy 验证 → 发现 budget 瓶颈 → 完整集成，每一步都有数据支撑
- **跨层优化**：理解应用层（ContextPilot）和引擎层（Scheduler）各自的职责和协同方式
- **Benchmark 工程**：A-B-B-A 消除 warmup bias、cache reset 隔离、冷启动处理

---

# 附录：如何给 nano-vllm 修改/增添新功能（通用方法论）

> 以下方法论基于本次 ContextPilot 集成的实践总结，适用于任何 nano-vllm 级别的推理引擎优化。

## Step 1：理解架构分层

nano-vllm 的核心是一个 4 层架构：

```
LLMEngine.generate()           ← 用户 API 层
    │
Scheduler.schedule()           ← 调度层（决定"执行什么"）
    │
ModelRunner.run()              ← 执行层（决定"怎么执行"）
    │
Attention / KV Cache           ← 计算层（GPU kernel）
```

**任何新功能的第一步**：确定它属于哪一层。

| 功能类型 | 所属层 | 示例 |
|----------|--------|------|
| 请求预处理/后处理 | API 层 | prompt 重排、结果聚合 |
| 调度策略 | 调度层 | cache-aware scheduling、优先级调度 |
| 执行优化 | 执行层 | speculative decoding、chunked prefill |
| 算子优化 | 计算层 | 自定义 attention kernel、量化 |

## Step 2：阅读关键数据结构

nano-vllm 的状态集中在 3 个数据结构上：

**Sequence**（`sequence.py`）：
```
token_ids, num_cached_tokens, block_table, status (WAITING/RUNNING/FINISHED)
```
→ 理解一个请求的生命周期。

**BlockManager**（`block_manager.py`）：
```
blocks[], hash_to_block_id{}, free_block_ids, used_block_ids
```
→ 理解 KV cache 的物理布局和复用机制。

**Scheduler**（`scheduler.py`）：
```
waiting deque, running deque, schedule() → (seqs, is_prefill)
```
→ 理解 prefill/decode 两阶段调度。

## Step 3：用最小改动验证假设

**原则**：在写 100 行代码之前，先用 10 行验证核心假设是否成立。

本项目的例子：
```python
# 10 行代码验证 "prefix-hash 排序有没有用"
def _sort_waiting_by_prefix(self):
    if len(self.waiting) <= 1:
        return
    bm = self.block_manager
    def prefix_key(seq):
        tokens = seq.block(0)
        if len(tokens) == bm.block_size:
            return bm.compute_hash(tokens)
        return -1
    self.waiting = deque(sorted(self.waiting, key=prefix_key))
```

→ 跑 benchmark，发现只有 1.03-1.06x → 说明排序本身不够，需要更深的改动（budget check）。

如果这 10 行就给出了 2x 加速，那就不需要后续的复杂优化了。

## Step 4：定位真正的瓶颈

**方法**：在关键路径上加计数器/日志，而非猜测。

```python
# 在 schedule() 中加临时日志
print(f"batch: {len(scheduled_seqs)} seqs, {num_batched_tokens} new tokens, "
      f"budget_used={num_batched_tokens}/{self.max_num_batched_tokens}")
```

本项目中，这个日志直接暴露了瓶颈：`budget_used=3840/4096`（只塞了 3 个序列），而理论上应该能塞 20+。

## Step 5：改动后的验证清单

1. **单元测试**：新方法的正确性（`count_cached_blocks` 返回正确数值）
2. **集成测试**：不破坏原有功能（原始 `bench.py` 不退化）
3. **性能测试**：A-B-B-A 交叉测试消除系统噪声
4. **边界条件**：空队列、单序列、内存不足等

## Step 6：Benchmark 设计原则

| 原则 | 做法 | 反例 |
|------|------|------|
| **隔离变量** | A-B-B-A 交叉执行 | 先跑完所有 A 再跑 B（warmup bias） |
| **清理状态** | `reset_prefix_cache()` | 只生成几个随机 prompt flush（不够彻底） |
| **预热** | warmup 阶段触发 CUDA graph | 第一个 scenario 作为 warmup（不公平） |
| **Shuffle 输入** | 模拟真实无序到达 | 用已排好序的输入（高估 baseline） |
| **约束资源** | 设小 `max_num_batched_tokens` | 用默认值 16384（一个 batch 全塞完，看不出差异） |

## Step 7：常见优化方向参考

| 方向 | 改动位置 | 核心思路 |
|------|---------|---------|
| Cache-aware scheduling | `scheduler.py` | 优先调度 cache hit 高的序列 |
| Chunked prefill | `scheduler.py` + `model_runner.py` | 长 prompt 分多个 chunk 处理，与 decode 交叉 |
| Speculative decoding | `model_runner.py` + `scheduler.py` | draft model 预测 + verify |
| Prefix cache eviction | `block_manager.py` | LRU/LFU 替代当前的不淘汰策略 |
| Structured output | `sampler.py` | 约束 sampling 空间（JSON schema 等） |
| KV cache offload | `model_runner.py` | GPU ↔ CPU 异步搬运 |
| Continuous batching 优化 | `scheduler.py` | preemption 策略、fairness |

每个方向都遵循同样的流程：定位层级 → 阅读数据结构 → 最小验证 → 定位瓶颈 → 实现 → 测试。
