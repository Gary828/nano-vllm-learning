# ContextPilot × nano-vllm 对话回顾总结（2026-04-05）

> 目标：给“初学者复盘”用，一份文档回顾今天所有关键概念、误区、结论和后续动作。

---

## 1. 一句话总览

- 本次讨论的核心是：**为什么“静态重排”在 nano-vllm 上一开始效果差，以及后来为什么“ContextPilot + cache-aware Scheduler”会变快。**
- 最关键结论：  
  - **ContextPilot** 负责“创造共享前缀机会”（把能拼车的人凑一起）。  
  - **Scheduler** 负责“利用共享前缀机会”（把车装满，减少每轮新算 token）。

---

## 2. 你最关心的核心问题（直接回答版）

## Q1：`len(seq)` 做 batch budget 检查为什么有问题？

- `len(seq)` 是请求总长度（含已缓存前缀 + 新 token）。
- 但 prefill 真正计算的是“未缓存部分”。
- 如果预算按 `len(seq)` 扣费，就会把“本来不用算的 cached token”也算进预算，导致一批塞不下更多请求。

**改进：**
- 用 `predicted_new`（预计新算 token）替代 `len(seq)` 参与 budget 检查。  
- 结果：同样预算下，batch 能容纳更多请求，prefill 轮次下降。

---

## Q2：既然 `hash_to_block_id` 全局持久，为什么还需要 ContextPilot？

- 全局持久只表示“历史见过的前缀以后还能命中”。
- 但 nano-vllm 是**前缀链命中**：从第 0 个 block 开始检查，一旦 miss，后面就不再算命中。
- 所以“文档集合相同但顺序不同”时，通常仍然 miss 很多。

**ContextPilot 做的事：**
- 把共享文档尽量移动到前面，让原本“有重叠但不同序”的请求变成“真正同前缀”。
- 这样 `count_cached_blocks` 才会明显上升，`predicted_new` 才会明显下降。

---

## Q3：nano-vllm、ContextPilot、当前集成在 evict 上有什么区别？

### nano-vllm（当前仓库）
- 主要是 `deallocate`（请求结束释放 block），不是完整“请求级 eviction 同步系统”。
- `hash_to_block_id` 映射不会在 deallocate 时自动清空（测试时可手动 `reset_prefix_cache`）。

### ContextPilot 官方（配 vLLM/SGLang）
- 是 stateful 的“索引-引擎联动”体系。
- 引擎发生 cache eviction 时，通过回调 `POST /evict` 通知 ContextPilot。
- ContextPilot 从 live index 里移除对应 request_id，保持索引和真实缓存一致。

### 当前 ContextPilot + nano-vllm 基准（你现在主要跑的）
- 主要是离线重排 + nano 内部 cache-aware 调度。
- 不是官方那套完整在线 `/evict` 联动路径。
- 当前提速主要来自“重排 + 调度打包”，不是来自在线 eviction 回调。

---

## 3. 关键概念词典（复习速查）

- **Prefill**：处理 prompt 的阶段，计算量大。
- **Decode**：逐 token 生成阶段。
- **KV Cache**：缓存历史 token 的 key/value，避免重复计算。
- **Prefix Cache Hit**：新请求前缀和历史一致，可复用 KV。
- **block-level chaining hash**：每个 block hash 依赖前一个 block hash；前面 miss 会连锁影响后面。
- **`count_cached_blocks`**：从第 0 block 起，连续统计可命中的 block 数。
- **`predicted_new`**：该请求本轮预计新算 token 数（预算应按这个算）。
- **Intra-context reorder**：单请求内部把共享文档前置。
- **Inter-context scheduling**：请求间排序，让相似请求相邻执行。
- **A-B-B-A**：benchmark 交叉执行顺序，减少 warmup 偏差。

---

## 4. 当前 benchmark 是怎么设计的（`bench_contextpilot.py`）

### 设计思路
- 构造知识库（多个 doc chunk）。
- 每个请求随机挑若干 doc，并打乱顺序（模拟真实检索返回）。
- Baseline：乱序请求直接跑。
- CP 组：先用 ContextPilot 做 intra + inter，再跑 nano。

### 防偏差措施
- 先 warmup（减少首次 JIT/cudagraph 冷启动影响）。
- 每轮前 reset prefix cache（避免 run 之间 cache 污染）。
- 用 A-B-B-A 交叉执行。

### 当前局限
- 主要测性能（time/TTFT），不测真实语义质量。
- `cp_time` 当前打印了，但 speedup 默认是 infer-only 口径，需额外算 end-to-end 口径。

---

## 5. 为什么“完整方案”比“只改 Scheduler”更快

- 只改 Scheduler：只是更高效利用“本来就有的共享前缀”。
- ContextPilot + Scheduler：先把“原本没有的共享前缀”创造出来，再高效利用。
- 所以完整方案提升通常更明显（报告里达到更高 speedup 区间）。

---

## 6. 今天形成的最终认知模型（最重要）

把系统看成两层：

1. **机会层（ContextPilot）**  
   负责把请求组织成“可共享前缀”的形态。

2. **执行层（Scheduler + BlockManager）**  
   负责把这些机会转成真实吞吐：  
   - 相似请求相邻调度  
   - budget 按 `predicted_new` 计费  
   - 尽可能在同一 prefill batch 装入更多可复用请求

**两层缺一不可：**
- 只有机会层，没有执行层 → cache 机会浪费。
- 只有执行层，没有机会层 → 没有足够共享前缀可利用。

---

## 7. 常见误区（你今天已经跨过去了）

- 误区 1：有全局持久 hash 就等于调度顺序不重要  
  - 更准确：顺序对“是否见过前缀”影响小，但对“本轮 batch 装箱效率”仍有影响，特别是配合 `predicted_new`。

- 误区 2：共享了同一批文档就一定能命中  
  - 更准确：要看**前缀顺序是否一致**，不是只看集合重叠。

- 误区 3：benchmark 只看一次跑分就够  
  - 更准确：要考虑 warmup、cache 污染、执行顺序偏差。

---

## 8. 建议的下一步实验矩阵（更严谨）

建议跑 4 组，拆解贡献：

- **M0**: 原始请求 + FIFO（基线）
- **M1**: 原始请求 + cache-aware（仅调度器贡献）
- **M2**: CP 重排请求 + FIFO（仅“创造前缀”贡献）
- **M3**: CP 重排请求 + cache-aware（完整收益）

每组同时报告：
- **infer-only**（只算 generate）
- **end-to-end**（CP 重排时间 + generate）

这样就能清楚回答：  
“到底是重排贡献大，还是调度贡献大，还是两者协同贡献最大。”

---

## 9. 你现在可以怎么复习（最省力）

按这个顺序回看最有效：

1. 先看本文件第 2 节（3 个核心问答）  
2. 再看第 6 节（两层模型）  
3. 最后看第 8 节（实验矩阵）  

如果这三块都理解了，你对 ContextPilot × nano-vllm 的主线已经完整了。

