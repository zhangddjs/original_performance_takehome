# Anthropic 性能工程 Take-Home 挑战攻略指南

## 一、项目概述

**GitHub 仓库**: https://github.com/anthropics/original_performance_takehome

**目标**: 在一个**自定义 VLIW SIMD 模拟处理器**上优化一个 **batched tree traversal kernel**，将 baseline 的 **147,734 cycles** 尽可能降低。

**申请门槛**: 优化到 **< 1487 cycles**（Opus 4.5 发布时的最佳成绩），发邮件至 `performance-recruiting@anthropic.com`。

**社区排行榜**: https://www.kerneloptimization.fun/

---

## 二、项目文件结构

```
original_performance_takehome/
├── perf_takehome.py      # 主入口 - 你要修改的 KernelBuilder.build_kernel()
├── problem.py            # 模拟器核心 - Machine, Core, 指令集定义 (只读)
├── watch_trace.py        # Perfetto trace 热重载服务器
├── watch_trace.html      # trace 可视化页面
└── tests/
    └── submission_tests.py  # 提交验证测试 (不可修改!)
```

**关键**: 只修改 `perf_takehome.py` 中的 `KernelBuilder.build_kernel()` 方法。**绝对不要修改 `tests/` 目录和 `problem.py`**。

---

## 三、处理器架构详解

### 3.1 VLIW (Very Long Instruction Word) 架构

与传统 CPU 不同，VLIW 架构中**指令并行由程序员/编译器负责调度**，而非硬件。你需要将多个操作打包进一个"指令束"(instruction bundle)，每个 cycle 执行一个 bundle。

### 3.2 执行单元（每 cycle 可用）

| 单元 | 数量 | 操作 |
|------|------|------|
| **ALU** | 12 | 标量运算: +, -, *, //, ^, &, \|, <<, >>, %, <, == |
| **VALU** | 6 | 向量运算 (8 元素 SIMD): 同 ALU，加 vbroadcast, multiply_add |
| **LOAD** | 2 | load, vload (8 word), const |
| **STORE** | 2 | store, vstore (8 word) |
| **FLOW** | 1 | select, vselect, cond_jump, jump, halt |
| **DEBUG** | 64 | compare (不计 cycle) |

**理论峰值**: 每 cycle 可做 12 + 6×8 = **60 次算术运算**。

### 3.3 内存模型

- **主内存 (Main Memory)**: 存放问题数据（树、索引、值）。ALU/VALU 不能直接访问。
- **Scratch Space**: 1536 words 的快速存储，类似寄存器文件。所有计算在这里进行。
- **瓶颈**: 每 cycle 最多 **2 loads + 2 stores**。

### 3.4 SIMD 向量宽度

`VLEN = 8`，每条 VALU 指令同时处理 8 个元素。

---

## 四、问题：Batched Tree Traversal

### 4.1 算法逻辑

```python
# 参考实现 (problem.py 中的 reference_kernel)
for round in range(16):           # 16 轮
    for i in range(256):          # 256 个 batch item
        idx = indices[i]
        val = values[i]
        val = myhash(val ^ tree[idx])        # XOR + 6步哈希
        idx = 2*idx + (1 if val%2==0 else 2) # 树遍历
        idx = 0 if idx >= n_nodes else idx    # 环绕
        values[i] = val
        indices[i] = idx
```

- **树**: 高度 10 的完美二叉树 (2047 个节点)
- **Batch size**: 256 个 item
- **Rounds**: 16 轮
- **总计**: 256 × 16 = 4096 次遍历步骤

### 4.2 哈希函数 (6 个阶段)

每个阶段: `a = (a op1 const) op2 (a op3 shift)`

| 阶段 | 公式 |
|------|------|
| 0 | `a = (a + 0x7ED55D16) + (a << 12)` |
| 1 | `a = (a ^ 0xC761C23C) ^ (a >> 19)` |
| 2 | `a = (a + 0x165667B1) + (a << 5)` |
| 3 | `a = (a + 0xD3A2646C) ^ (a << 9)` |
| 4 | `a = (a + 0xFD7046C5) + (a << 3)` |
| 5 | `a = (a ^ 0xB55A4F09) ^ (a >> 16)` |

每阶段 = 3 ALU 操作，总计每次哈希 **18 次 ALU 操作**。

---

## 五、优化路线图（从 147K → <1487 cycles）

### 阶段 1: SIMD 向量化 (~147K → ~25K cycles, ~6x)

**核心思路**: 用 SIMD 一次处理 8 个 batch item，而非逐个处理。

- 将 256 个 item 分成 32 组，每组 8 个
- 用 `vload/vstore` 批量加载/存储
- 用 `valu` 做向量哈希运算
- 树查找变为 vector gather（但因为索引不连续，只能用标量 load）

### 阶段 2: VLIW 指令打包 (~25K → ~13K cycles, ~2x)

**核心思路**: 把独立操作打包进同一个 cycle。

Baseline 的问题：每个 bundle 只放一个操作，12 个 ALU 空置 11 个。

```python
# Baseline: 4 cycles
{"alu": [op1]}     # Cycle 1
{"alu": [op2]}     # Cycle 2  
{"load": [ld1]}    # Cycle 3
{"valu": [vop1]}   # Cycle 4

# 优化后: 1 cycle
{"alu": [op1, op2], "load": [ld1], "valu": [vop1]}
```

**实现**: 构建一个 instruction scheduler，追踪数据依赖和 slot 限制，自动打包。

### 阶段 3: Branchless 编程 (消除分支)

用 `select`/`vselect` 替代条件跳转：

```python
# 替代 if val%2==0: idx=2*idx+1 else idx=2*idx+2
offset = vselect(val_mod2_is_zero, broadcast_1, broadcast_2)
idx = 2*idx + offset
```

### 阶段 4: 软件流水线 (Pipelining) (~13K → ~2700 → ~1500 cycles, 关键突破!)

**这是从中等分数跳到高分的核心技术。**

**核心思路**: 不同 batch 组处于不同的计算阶段，同时执行，填满所有执行单元。

```
Cycle N:   [Batch A - Stage 3] [Batch B - Stage 2] [Batch C - Stage 1]
Cycle N+1: [Batch A - Stage 4] [Batch B - Stage 3] [Batch C - Stage 2] [Batch D - Stage 1]
```

将每一轮的计算分解为多个阶段（stages），让不同 batch 的不同 stage 的指令共享同一个 cycle 的执行槽位。

**实现要点**:
- 将内循环分成 pipeline stages
- 为每个 stage 分配独立的 scratch 空间区域
- 建立依赖追踪的 instruction scheduler
- 将 VALU 指令拆成多个标量 ALU 指令以更好地填充空闲 slot

### 阶段 5: 消除内存访问瓶颈 (~1500 → <1487 cycles, 突破门槛!)

**瓶颈分析**: 16 轮 × 32 batch × 每 batch 需要 8 个标量 load（树节点值）= 4096 loads，每 cycle 只能做 2 个 load → 理论下限 2048 cycles（如果只看 load 瓶颈）。

**关键优化**: 避免某些树层级的内存访问。

- **Level 0 (根节点)**: 值固定，用 `const` 加载到 scratch，无需 load
- **Level 1**: 只有 2 个可能值（左/右子节点），可以预加载
- **Level 2**: 只有 4 个可能值，也可以预加载
- **树高度 10 会环绕回来**，所以 level 0/1/2 会被遍历两次
- 不需要 load 的轮次可以和其他轮次通过 pipelining 重叠执行

### 阶段 6: 极致优化 (进入 <1363 乃至更低)

- 常量预计算和广播优化
- 更激进的 pipeline 深度
- 利用 `multiply_add` fused 指令减少 cycle
- 精细调优 batch 分组大小
- 利用所有 12 个 ALU slot（将 VALU 拆成标量运算填充）

---

## 六、调试工具使用

### 6.1 运行测试

```bash
# 运行基本测试，查看 cycle 数
python perf_takehome.py Tests.test_kernel_cycles

# 生成 Perfetto trace
python perf_takehome.py Tests.test_kernel_trace

# 启动热重载 trace 查看器 (Chrome)
python watch_trace.py

# 验证提交
python tests/submission_tests.py
```

### 6.2 Perfetto Trace

打开 `watch_trace.html` 或将 `trace.json` 拖到 https://ui.perfetto.dev/，可以看到每个 cycle 中每个执行单元的使用情况。

### 6.3 Debug 指令

在 kernel 中插入 `("debug", ("compare", scratch_addr, expected_key))` 来验证中间值是否正确。Debug 指令不计 cycle。

### 6.4 提交前验证

```bash
# 确保 tests/ 未被修改
git diff origin/main tests/

# 运行提交测试
python tests/submission_tests.py
```

---

## 七、参考资料汇总

### 必读资料

| 资源 | 链接 | 说明 |
|------|------|------|
| **原文博客** | https://www.anthropic.com/engineering/AI-resistant-technical-evaluations | Tristan Hume 的设计思路 |
| **GitHub 仓库** | https://github.com/anthropics/original_performance_takehome | 源代码 |
| **架构详解** | https://trirpi.github.io/posts/anthropic-performance-takehome/ | 对模拟器架构和指令集的深度拆解 |
| **1474 cycles 攻略** | https://matthewtejo.substack.com/p/taking-on-anthropics-public-performance | 实战经验，含 AI 辅助工作流 |
| **1338 cycles 攻略 (Medium)** | https://medium.com/@indosambhav/my-journey-through-the-anthropic-performance-optimization-challenge-7a5dc46dd6e0 | 19 步优化过程 |

### 背景知识

| 主题 | 推荐资源 |
|------|---------|
| **VLIW 架构** | Wikipedia: Very Long Instruction Word |
| **SIMD 编程** | Intel Intrinsics Guide, GPU shared memory programming |
| **软件流水线** | "Software Pipelining" - Wikipedia; 编译器教材中的 modular scheduling |
| **Branchless 编程** | Algorithmica HPC: https://en.algorithmica.org/hpc/pipelining/branchless/ |
| **指令调度** | List scheduling, dependency graph analysis |
| **Perfetto** | https://perfetto.dev/ - trace 可视化工具 |
| **Zachtronics 游戏** | Shenzhen I/O, TIS-100 - 类似的极限优化编程思维训练 |

### 社区讨论

| 来源 | 链接 |
|------|------|
| **HN 首发帖** | https://news.ycombinator.com/item?id=46700594 |
| **HN 攻略讨论** | https://news.ycombinator.com/item?id=46871285 |
| **社区排行榜** | https://www.kerneloptimization.fun/ |

---

## 八、建议的工作流

### 第 1 天: 理解问题 (4-6 小时)
1. 克隆仓库，运行 baseline，观察 ~147K cycles
2. **仔细阅读** `problem.py` 理解模拟器架构（执行引擎、slot 限制、内存模型）
3. 阅读 `perf_takehome.py` 理解 baseline kernel 如何生成指令
4. 运行 trace，在 Perfetto 中观察每个 cycle 的执行单元利用率
5. 阅读上面列出的架构详解博客

### 第 2 天: 基础优化 (4-6 小时)
1. 实现 SIMD 向量化 → 目标 ~25K cycles
2. 实现基础 VLIW 指令打包 → 目标 ~13K cycles
3. 实现 branchless select

### 第 3 天: 进阶优化 (6-8 小时)
1. 构建 instruction scheduler（依赖追踪 + 自动打包）
2. 实现软件流水线 → 目标 ~2700 cycles
3. 迭代优化 pipeline 结构

### 第 4 天: 突破门槛 (6-8 小时)
1. 优化树节点加载（预加载 level 0/1/2）
2. 调优 pipeline 深度和 batch 分组
3. VALU 拆分填充 ALU slot
4. 目标 < 1487 cycles

### 第 5 天+: 极致优化
1. 分析 trace 找剩余瓶颈
2. 尝试更激进的调度策略
3. 目标 < 1363 cycles（超越 Opus 4.5 改进版 harness）

---

## 九、注意事项

1. **不要让 AI 修改 tests/ 目录** — 这是 LLM 常见的"作弊"行为，会被发现
2. **理解你提交的代码** — Anthropic 可能会跟进面试讨论你的方案
3. **README 提到门槛可能随新模型更新** — 尽量做到尽可能低
4. **可以用 AI 辅助学习和编码**，但关键洞察需要自己理解
5. **发邮件时说明你运行了验证命令** (`git diff` + `submission_tests.py`)
