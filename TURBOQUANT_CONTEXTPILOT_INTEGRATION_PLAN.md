# 目前参考的 TurboQuant 实现有问题，所以main分支已清理有关的代码
下面的方案仅作参考

# TurboQuant × ContextPilot 集成总方案（nano-vllm）

> 目标：在不破坏当前 `nano-vllm` 主干（含 cache-aware 调度与 ContextPilot 基准链路）的前提下，增量引入 TurboQuant KV 压缩能力，并建立可复现实验体系，系统比较不同组合的效果。

---

## 0. 文档范围与结论先行

### 0.1 本文覆盖范围
- 当前仓库（`nano-vllm`）与 `nano-vllm-with-TurboQuant` 的整合路径
- TurboQuant 与 ContextPilot 的可结合性分析
- 代码改造分阶段计划（文件级、接口级、风险级）
- 对比实验设计（实验矩阵、指标、结果模板、判定标准）

### 0.2 核心结论（先回答你的问题）
- **TurboQuant 可以与 ContextPilot 结合**，且两者作用层级不同、理论上互补：
  - ContextPilot：主要优化 **请求组织/顺序与前缀复用**（偏 prefill 侧）
  - TurboQuant：主要优化 **KV 缓存表示与解码期内存/吞吐权衡**（偏 decode 侧）
- 组合效果不是“必然相加”，需要按场景验证：
  - 高并发、长上下文、显存紧张：`ContextPilot + TurboQuant` 往往最有价值
  - 低并发、短上下文：TurboQuant 可能因当前实现额外算子而降低 tok/s，ContextPilot 收益也有限
- 必须做“组合对比”而非单点 benchmark，建议至少比较 4 个主配置：
  - A: Baseline（无 CP、无 TQ）
  - B: CP only
  - C: TQ only
  - D: CP + TQ

---

## 1. 现状基线（当前仓库 vs TurboQuant 分支）

## 1.1 当前仓库（`nano-vllm`）关键能力
- 已有 prefix cache / cache-aware scheduler
- `LLM.generate` 返回结构为字典，包含 `outputs`、`ttft`、`ttft_token`、`ttfd_decode_step`、`total_time`
- 已有 ContextPilot 方向的 benchmark 代码与报告资产

## 1.2 TurboQuant 分支关键增量
- 新增 `kv_quant_bits` 配置（`None/3/4`）
- 新增 `nanovllm/turboquant/`（`compressor.py`、`lloyd_max.py`）
- Attention decode 路径引入 TQ 异步估计逻辑 + 持久化压缩 K 缓存
- `model_runner` 支持按模型架构路由（Qwen / MiniCPM）
- `scheduler` 增补多 EOS 兼容（`token_id in eos_set`）

## 1.3 合并冲突重点（从工程角度）
- `nanovllm/config.py`：当前分支有 `cache_aware`，TQ 分支有 `kv_quant_bits`，需要并存
- `nanovllm/engine/llm_engine.py`：
  - 需保留当前仓库返回契约
  - 同时引入 TQ 分支对多 EOS/`trust_remote_code` 的兼容
- `nanovllm/engine/scheduler.py`：
  - 必须保留 cache-aware 排序与预测逻辑
  - 同时将 EOS 判断升级为集合兼容
- benchmark 脚本接口期望不一致：
  - TQ 分支脚本多处假设 `generate` 直接返回 list
  - 当前仓库 `generate` 返回 dict，需要脚本统一适配

---

## 2. 目标架构：统一运行平面

## 2.1 配置维度
- `use_context_optimizer`（已有）
- `cache_aware`（已有，默认 True）
- `kv_quant_bits`（新增，默认 None）
- `model`（支持 Qwen / MiniCPM 自动识别）

## 2.2 运行模式矩阵

| 模式 | ContextPilot | Cache-Aware | TurboQuant | 适用场景 |
|---|---:|---:|---:|---|
| A Baseline | Off | Off/On | Off | 参考基线 |
| B CP Only | On | On | Off | 前缀高重合、追求 TTFT/吞吐 |
| C TQ Only | Off | On | On | 显存紧张、长上下文 |
| D CP+TQ | On | On | On | 高并发 + 显存受限 + 前缀可优化 |

## 2.3 关键约束
- 默认行为保持与当前主干一致（`kv_quant_bits=None`）
- 不破坏 `generate` 返回结构（避免已有脚本/工具链回归）
- 将 TQ 视为可选能力，不影响无 GPU 或无 scipy 场景的基础使用

---

## 3. 分阶段集成计划（建议 6 阶段）

## Phase 0：工程准备与风险隔离
### 目标
- 保证可回滚、可比较、可追踪

### 动作
- 新建分支：`feat/turboquant-contextpilot`
- 冻结当前 benchmark 基线输出（作为回归基准）
- 清理工作区脏状态，避免历史改动污染验证

### 产出
- `BASELINE_RESULTS.md`（记录当前版本指标）
- 干净分支 + 可复现实验命令清单

---

## Phase 1：最小可用 TurboQuant 接入（不开启默认）
### 目标
- 在不改变默认行为情况下，接入 `kv_quant_bits` 与基础 TQ 模块

### 改造点
- `nanovllm/config.py`
  - 增加 `kv_quant_bits: int | None = None`
  - 保留 `cache_aware`
  - `AutoConfig.from_pretrained(..., trust_remote_code=True)`（为 MiniCPM 等模型兼容）
- 新增 `nanovllm/turboquant/__init__.py`
- 新增 `nanovllm/turboquant/compressor.py`
- 新增 `nanovllm/turboquant/lloyd_max.py`

### DoD（完成定义）
- `kv_quant_bits=None` 时行为与当前主干一致
- `import nanovllm.turboquant` 可用，且不影响现有用法

---

## Phase 2：Attention 与 KV 缓存路径融合
### 目标
- 引入持久化压缩 K 缓存与 decode TQ 路径

### 改造点
- `nanovllm/engine/model_runner.py`
  - 分配常规 KV cache 的同时，按层分配 TQ K cache（`k_mse/signs/rnorm`）
  - 将 `tq_engine` 与 TQ cache 引用注入每层 attention module
- `nanovllm/layers/attention.py`
  - 增加 `gather_paged(...)`（向量化 gather）
  - 增加 `_tq_compress_store(...)`（增量压缩新 token key）
  - decode 分支支持 `_tq_decode(...)`
- `nanovllm/turboquant/compressor.py`
  - 使用 GQA-aware 计算路径，避免 `repeat_interleave` 产生大规模中间张量

### DoD
- `kv_quant_bits=3` 可以完成端到端生成
- `kv_quant_bits=None` 路径吞吐与当前版本近似（允许小幅波动）

---

## Phase 3：运行时兼容与模型路由
### 目标
- 保持当前框架 API 契约，同时吸收 TQ 分支模型兼容能力

### 改造点
- `nanovllm/engine/llm_engine.py`
  - tokenizer 加 `trust_remote_code=True`
  - `eos` 规范化为 set（兼容 `eos_token_id` 为 list）
  - `exit()` 增加幂等保护（避免重复释放异常）
  - **保留当前 `generate` 返回结构（dict）**
- `nanovllm/engine/scheduler.py`
  - EOS 判断由 `token_id == eos` 改为 `token_id in eos`
  - 保留当前 cache-aware 排序/估计逻辑
- `nanovllm/engine/model_runner.py`
  - 可选：引入 `get_model_class()` 自动识别模型架构（Qwen/MiniCPM）
  - 可选：`NCCL_PORT` 环境变量化，提升多进程测试稳定性

### DoD
- 现有 benchmark（非 TQ）无需改参数仍可运行
- 多 EOS 模型不出现提前终止/无法终止异常

---

## Phase 4：ContextPilot × TurboQuant 协同接入
### 目标
- 在统一开关体系下可组合运行 `use_context_optimizer=True + kv_quant_bits=3/4`

### 组合策略
- 保持 ContextPilot 作用在“请求重排/调度层”
- 保持 TurboQuant 作用在“KV 表示/attention 计算层”
- 二者在代码上解耦，仅在运行配置维度组合

### 关键检查
- CP 重排后，`num_cached_tokens` 变化是否与预期一致
- TQ 压缩写入使用的 `slot_mapping` 与 block table 是否对齐
- prefix cache 命中场景下，TQ 是否重复压缩历史 key（应避免）

### DoD
- 组合模式 D（CP+TQ）可稳定运行完整 benchmark 轮次
- 无 deadlock / 维度错配 / cache 写越界

---

## Phase 5：实验体系搭建（重点）
### 目标
- 系统比较四类模式 A/B/C/D，在不同负载下给出可解释结论

### 5.1 实验维度
- 功能开关：
  - `use_context_optimizer`: `False / True`
  - `kv_quant_bits`: `None / 3 / 4`
- 负载类型：
  - Prefix 高重合（CP 最易获益）
  - Prefix 低重合（CP 收益弱）
  - 短输入短输出
  - 长输入长输出
  - 高并发（batch 8/16/32）
  - 多轮对话（长时缓存压力）

### 5.2 主实验矩阵（建议首轮）

| 编号 | ContextPilot | TurboQuant | bits | 目标 |
|---|---:|---:|---:|---|
| M1 | Off | Off | - | 基线 |
| M2 | On | Off | - | 评估 CP 纯收益 |
| M3 | Off | On | 3 | 评估 TQ-3bit 纯收益 |
| M4 | On | On | 3 | 评估协同收益 |
| M5 | Off | On | 4 | 精度优先 TQ |
| M6 | On | On | 4 | CP + TQ-4bit |

### 5.3 观测指标
- **性能**
  - Prefill tok/s
  - Decode tok/s
  - End-to-end tok/s
  - TTFT（token 出现时间）
  - TTFD（首个 decode step）
- **资源**
  - Peak GPU memory
  - 可分配 `num_kvcache_blocks`
  - OOM 发生率
- **缓存**
  - Prefix cache hit rate
  - `num_cached_tokens / total_tokens`
- **质量**
  - 固定 prompt 的输出可用性（人工 + 自动规则）
  - 代码题/事实题正确性对比（通过率或 rubric 分数）

### 5.4 运行协议（保证可比）
- 每组实验至少重复 3 次，取中位数
- 每次先 warmup 再计时
- 固定随机种子
- 固定模型、温度、max_tokens
- 清理 cache 状态（确保组间独立）
- 单卡/同驱动/同 CUDA 版本

### 5.5 结果报告模板

#### 性能表
| Mode | Workload | Prefill tok/s | Decode tok/s | E2E tok/s | TTFT(s) | Peak GB |
|---|---|---:|---:|---:|---:|---:|
| M1 | high-overlap-long |  |  |  |  |  |
| M2 | high-overlap-long |  |  |  |  |  |
| M3 | high-overlap-long |  |  |  |  |  |
| M4 | high-overlap-long |  |  |  |  |  |

#### 收益分解表（相对 M1）
| Mode | ΔPrefill | ΔDecode | ΔE2E | ΔTTFT | ΔMem |
|---|---:|---:|---:|---:|---:|
| M2 |  |  |  |  |  |
| M3 |  |  |  |  |  |
| M4 |  |  |  |  |  |

---

## Phase 6：优化与产品化（可选）
### 优先级 P0（建议）
- TQ 脚本统一适配当前 `generate` 返回格式（dict）
- TQ 依赖收敛（`scipy` 可选化或预计算 centroids）
- 将 benchmark 脚本统一到一个 `benchmark_matrix.py`

### 优先级 P1
- TQ decode 路径 profiler（定位热点：gather/matmul/softmax）
- 探索 Triton fused kernel（长期项）
- bit-packed 存储替换 FP16 K cache（长期项）

---

## 4. TurboQuant 与 ContextPilot 的协同机理（细化）

## 4.1 为什么“能结合”
- ContextPilot 改变的是“请求进入引擎前的组织方式”
- TurboQuant 改变的是“引擎内部 KV 表示与 decode attention 计算方式”
- 两者边界清晰，不冲突，天然可叠加

## 4.2 可能出现的相互影响

### 正向影响
- CP 提升 prefix 复用后，新 token 数量下降，TQ 需要压缩的新增 key 也下降
- TQ 降低 KV 存储压力后，可承载更大并发，使 CP 调度空间更大

### 负向影响
- 当前 TQ decode 可能较慢（未 fused），在低内存压力场景拉低 tok/s
- CP 增强 prefill 命中后，系统瓶颈可能更偏 decode，TQ decode 算力开销会更显著

## 4.3 预期“谁赢”的场景判断
- **场景 S1：短上下文、低并发**
  - 推荐：M1/M2，TQ 收益不稳定
- **场景 S2：长上下文、显存紧张**
  - 推荐：M3/M4，重点看是否能提升可服务并发/避免 OOM
- **场景 S3：高重合知识块检索型请求**
  - 推荐：M2/M4，通常 CP 收益明显，M4 可能在内存上进一步放大收益

---

## 5. 文件级改造清单（执行版）

## 5.1 必改文件
- `nanovllm/config.py`
  - 合并字段：`cache_aware` + `kv_quant_bits`
  - `trust_remote_code=True`
- `nanovllm/engine/model_runner.py`
  - TQ cache 分配 + module 注入
  - 可选模型架构路由
- `nanovllm/layers/attention.py`
  - `gather_paged`
  - `_tq_compress_store`
  - `_tq_decode`
- `nanovllm/engine/llm_engine.py`
  - EOS set 化
  - return 契约保持为当前主干 dict 结构
- `nanovllm/engine/scheduler.py`
  - EOS 判断改为 `in`
  - 保留 cache-aware 特性

## 5.2 新增文件
- `nanovllm/turboquant/__init__.py`
- `nanovllm/turboquant/compressor.py`
- `nanovllm/turboquant/lloyd_max.py`
- （可选）`nanovllm/models/minicpm.py`
- （建议）`benchmark_matrix.py`（统一实验入口）

## 5.3 脚本兼容改造
- `benchmark_tq.py`、`benchmark_full.py`、`run_quality.py` 等：
  - 从 `outputs = llm.generate(...)` 改为读取 `result["outputs"]`
  - 或统一提供 `normalize_outputs(result)` 工具函数

---

## 6. 回归测试与验收标准

## 6.1 功能正确性
- `kv_quant_bits=None`：与当前主干结果一致（允许微小采样随机差异）
- `kv_quant_bits=3/4`：可稳定生成，且不崩溃
- CP on/off 开关可控，策略切换生效

## 6.2 性能与资源验收（建议阈值）
- 基础路径（无 TQ）性能回归不超过 3%
- TQ 模式下显存占用曲线可解释（至少给出峰值与 cache block 变化）
- 组合模式（CP+TQ）在至少一个目标场景上优于单项模式（如更高并发或更低 OOM）

## 6.3 质量验收
- 固定评测集（问答/代码/摘要）输出可读性和任务完成度不过度下降
- 对温度接近 0 的设定，回答核心事实一致率可接受

---

## 7. 风险清单与规避策略

## 7.1 最高风险项
- **R1：接口回归**（`generate` 返回结构变化导致脚本崩）
  - 策略：主干契约不变，脚本做兼容层
- **R2：调度能力回退**（误覆盖 cache-aware 逻辑）
  - 策略：合并时保留当前 scheduler 主体，仅注入 EOS 修复
- **R3：依赖问题**（`scipy` 缺失导致 TQ 不可用）
  - 策略：`extras_require[turboquant]` 或预计算 centroids
- **R4：性能误判**
  - 策略：严格执行实验协议（warmup/重复/中位数/固定种子）

## 7.2 中风险项
- NCCL 端口冲突（多进程测试）
  - 策略：支持 `NCCL_PORT` 环境变量
- TQ 压缩 cache 与 block 映射错位
  - 策略：增加 shape/索引断言与小样本可视化校验

---

## 8. 推荐执行节奏（两周版）

### Week 1
- D1-D2：Phase 0 + Phase 1
- D3-D4：Phase 2（核心融合）
- D5：Phase 3（接口与兼容）+ 基础回归

### Week 2
- D1-D2：Phase 4（CP+TQ 协同验证）
- D3-D4：Phase 5（矩阵 benchmark）
- D5：分析与报告（结论 + 下一步优化优先级）

---

## 9. 立即可执行的最小行动清单

- [ ] 清理并保存当前工作区改动（stash/备份分支）
- [ ] 创建 `feat/turboquant-contextpilot`
- [ ] 先合入 TQ 两个提交（优先手工解决 `config.py`）
- [ ] 恢复并校验 `LLM.generate` 当前返回契约
- [ ] 修复 benchmark 脚本的输出读取兼容
- [ ] 先跑 M1/M2/M3/M4 四组，拿到第一轮真实对比数据

---

## 10. 最终要回答的业务问题（报告模板）

1. 在你的目标场景下（并发/上下文长度），CP、TQ、CP+TQ 哪个最优？
2. 最优方案是“快”还是“能装更多请求”？收益主要来自哪一项？
3. 是否值得进入下一阶段（Triton fused / bit-packed）投入？

---

如果需要，我可以基于这个文档继续直接产出两个工程物件：
- `benchmark_matrix.py`（一键跑 M1~M6）
- `RESULT_ANALYSIS_TEMPLATE.md`（自动填表的结论模板）
