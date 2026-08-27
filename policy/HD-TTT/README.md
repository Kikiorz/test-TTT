# CreditTTT V3

**Hindsight-to-Local Query-conditioned Fast-Weight Learning for Long-Horizon
Robot Imitation**

这是本目录唯一的算法说明和复现实验记录。CreditTTT V3 建立在
SmolVLA 的 action expert 上，把跨物理时刻的历史压缩为持久的 TTT
fast weights，并用成功示范中的完整未来动作在训练阶段提供长程信用，再把
这种信用蒸馏为部署时可计算的局部更新目标。

> 当前状态（2026-08-27）：代码和训练/评测入口已整理完成；V3 的正式
> MIKASA 成功率实验尚未完成。文末的数字是已保存的早期诊断或历史控制，
> 不能当作论文结论。

`lib/` 和 `policy/Method1_lerobot-pi0-ttt` 不属于本方法，未被修改。
本仓库保留 LeRobot 的 `src` 运行时依赖（包括普通 SmolVLA）；论文方法只
指 `src/lerobot/policies/smolvla_ttt` 和 `examples/mikasa` 中标为 V3 的
路径。

## 1. 方法主旨

CreditTTT 只解决一个问题：当当前视觉观测不能恢复较早交互中的信息时，
策略怎样在未来动作中继续使用这段历史。它不增加 progress head、belief
decoder 或 recovery module，也不把成功标签误当成历史重要性。

三个相互闭合的组成部分是：

1. **Hindsight Control Attribution (HCA)**：训练一个严格因果的
   full-history action teacher。只删除某个历史事件写入 fast weights 的
   表示，保持真实观测、真实状态、已经执行的动作和未来专家目标不变；未来
   动作误差的增加就是该事件的控制信用。
2. **Hindsight-to-Local Query-conditioned Distillation (QH2L)**：对同一
   事件的写入前后状态，在后来的真实查询上重放部署流程，并让学生匹配 teacher
   的最终执行动作差异。未来查询只用于离线构造训练目标，不进入部署输入。
3. **Causal Memory Deployment (CMD)**：用正确、错误、reset 和 null memory
   的干预训练/审计 reader，使正确写入必须被 action head 使用。部署时仍是同一
   个“先更新、再读取”的因果状态机，而不是另一个推理模块。

HCA 提供远期监督，QH2L 把监督接回 writer 的元梯度，CMD 检查 reader 是否
真正利用了状态；三者缺一时，论文中的机制闭环不成立。

## 2. 因果定义与部署计算

对一个 episode 的物理时刻 `t`，记：

| 符号 | 含义 |
| --- | --- |
| `o_t, s_t` | 当前 RGB observation 和 proprioceptive state |
| `ell` | 任务语言指令 |
| `p_t = E_theta(o_t, s_t, ell)` | SmolVLA 的语言/视觉/状态 prefix |
| `b_{t-1}` | 上一个物理决策实际执行的归一化 slot-0 action |
| `x_{t,r}` | 第 `r` 个 flow-matching denoising step 的 50-slot noisy chunk |
| `W_t` | 处理当前 observation 前的持久 fast-weight state |
| `a_t` | 最终归一化 action chunk；只把 `a_{t,0}` 发给环境 |

### 2.1 Writer 与 reader

生产配置使用 observation-causal `prefix_only` writer：

\[
z_t=[R,\,p_t,\,b_{t-1}],\qquad
W_t^+=W_t+g_t\bigl(U_\phi(W_t;z_t)-W_t\bigr).
\]

其中 `R` 是 16 个可学习 register anchors，`g_t=1` 是 canonical
recipe 的写入门控。当前 noisy action、denoising time 和未来 observation
不参与这次写入，因此不会以 teacher-forcing 的方式泄漏未来。

在选定的 action-expert 层，动作 query 读取更新后的状态：

\[
r_{t,r}=f_{W_t^+}(q_{t,r}),\qquad
v_{t,r}=A_\theta(h_{t,r},r_{t,r}),
\]

其中 `A_theta` 是原 SmolVLA 的 action output projection。TTT 位于该层
attention residual 之后、FFN 之前；因此 fast weights 改变的是已有 action
expert 的隐表示，而不是增加一个独立动作头。

### 2.2 Register mask

每个 action-expert suffix 的结构为

```text
[16 register tokens] + [50 noisy action/time tokens]
```

mask 是有意不对称的：register query 可以读取有效 register 和当前 action
槽位；action query 不能读取 register 的 key 列，并保留原来的下三角
action-to-action mask。这样 registers 丰富了单步 TTT workspace，但不成为
绕过 action 表示的直接捷径。`ttt_num_register_tokens=0` 是明确的结构消融。

### 2.3 Denoising 顺序

对每次新的 observation：

1. 采样第一步的纯高斯 action chunk `x_{t,1} ~ N(0,I)`。
2. 第一个（`t=1`）denoising step 调用 `update=True`：先写出
   `W_t^+`，再立即用它读取 query。
3. 其余 9 个 flow steps 调用 `update=False`，只读取同一个 `W_t^+`，不
   重复写入。
4. 对最终 chunk 积分，执行 slot 0，并把实际执行的归一化动作保存为下一时刻
   的 `b_t`。

`policy.reset()` 只在 episode 边界清空 fast state 和 action carry。于是跨
物理决策的历史是 recurrent state，而不是把多个 denoising step 错当成多
个环境事件。

## 3. 网络配置

下面是论文 launcher 的默认结构；配置类中较大的默认 `ttt_hidden_dim` 是
可复用的底层默认，论文 recipe 会显式覆盖它。

| 部件 | canonical 设置 |
| --- | --- |
| VLM | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` |
| VLM/action-expert 深度 | 16 个对齐层 |
| expert 宽度 | VLM hidden 的 0.75 倍（当前约 720）；FFN 当前约 2048 |
| attention | SmolVLA 原有 alternating cross-attention / causal suffix self-attention |
| action horizon | 50 slots；MIKASA 有效 action 维度为 7，内部 padding 到 32 |
| 物理执行频率 | `n_action_steps=1`（每个 observation 只执行 slot 0） |
| flow denoising | 10 steps |
| TTT layers | `[12, 13, 14, 15]`，必须包括最后一层 |
| fast MLP width | launcher 为 1024（config 的通用默认值为 4096） |
| registers | 16 个 expert-width learned vectors |
| writer | `prefix_only`，并包含上一步执行的 slot-0 action |
| inner loop | `ttt_second_order=true`；launcher 打开 bounded RMS numerical guard |

当前 launcher 日志中的参考模型约为 4.63e8 个总参数，新增 TTT/register
部分约 1.3e7；具体数量会随 checkpoint、dtype 和配置覆盖而变化，不能用这
两个近似值替代 checkpoint 的实际统计。

普通 `smolvla` baseline 不含 fast weights。`smolvla_ttt` 在
`hd_ttt_enabled=false` 时是干净的 SmolVLA-TTT/TTT control；CreditTTT 是
在同一 action-expert 上打开 V3 objective 后的模型。这样可以分别测量
“有没有 fast state”和“hindsight objective 是否有效”。

## 4. 训练阶段的三个目标

### 4.1 HCA：full-history causal teacher

`train_full_history_teacher.py` 先从冻结的 SmolVLA prefix 提取每个物理帧的
event token，再训练一个 causal recurrent action teacher。teacher 的输入只
有当前 event token 和 `b_{t-1}`，预测当前归一化执行 slot-0 action；它不看
当前专家 action、未来 observation、denoising noise 或未来 query。

对事件 `i` 和严格更晚的查询 `j`，在同一 episode 中重放两条分支：

- full：正常写入事件 `i`；
- counterfactual：只删除事件 `i` 的 fast-weight write。

两条分支都保留真实的 `o_j,s_j`、固定的 `b_{t-1}` 序列和同一个专家目标
`a_j^star`。令

\[
\ell_j^{\rm full}=\|\hat a_j^{\rm full}-a_j^\star\|_2^2,\qquad
\ell_{ij}^{\rm cf}=\|\hat a_j^{\rm cf(i)}-a_j^\star\|_2^2,
\]

\[
u_{ij}=\left[
\frac{\ell_{ij}^{\rm cf}-\ell_j^{\rm full}}
 {\tfrac12(|\ell_{ij}^{\rm cf}|+|\ell_j^{\rm full}|)+\varepsilon}
\right]_+,qquad
\Delta a^T_{ij}=\hat a_j^{\rm full}-\hat a_j^{\rm cf(i)}.
\]

`u_ij` 是“当前 observation 无法恢复、但未来动作仍需要”的控制信用，
不是 RL action value，也不是成功标签。实际物理轨迹从未被删除或重写。
低信用 pair 构成 null/invariance stratum。默认固定 delay bins 为
`1-16, 17-64, 65-256, 257-1024, 1025+`；短 Color episode
只报告其有数据的短 bin。

当前直接 action-teacher adapter 的 provenance 是：

```text
target_mode           = normalized_executed_slot0_action
flow_target_available = false
antithetic_noise      = false
intervention          = event_write_deletion
```

这不是 flow velocity teacher，也不是能看未来的 oracle transformer。若以后
实现 flow-integrated teacher，必须作为单独 protocol 记录。

### 4.2 QH2L：query-conditioned local effect

对事件 `i` 保存学生 trace 的 `W_i^-` 和 `W_i^+`。对每个后续 query
`q_j`，在相同的未来 observation、previous action、noise 和 timestep schedule
下重放完整的 10-step deployed flow，只替换最后选定 TTT 层的 state：

\[
\Delta a^S_{ij}=
A_\theta(h_j,f_{W_i^+}(q_j))-
A_\theta(h_j,f_{W_i^-}(q_j)).
\]

teacher target 对学生梯度 `detach`。以 detached robust scale `s_ij` 归一化，
正/空 pair 分别计算 Huber：

\[
\mathcal L_{\rm QH2L}=
\operatorname{mean}_{u_{ij}>\tau}
 \operatorname{Huber}\!\left(
 \frac{\Delta a^S_{ij}-\operatorname{sg}(\Delta a^T_{ij})}{s_{ij}}\right)
+\lambda_0\operatorname{mean}_{u_{ij}\le\tau}
 \operatorname{Huber}\!\left(\frac{\Delta a^S_{ij}}{s_{ij}}\right).
\]

正、null 两个分母独立 all-reduce（DDP 时），所以改变 pair 数量或 delay
分布不会隐式改变 loss 尺度。`ttt_second_order=true` 是必要条件：QH2L 的
梯度必须穿过学生的局部 fast-weight update 回到 writer 参数。

pair replay 的 micro-chunk（默认 4）只限制峰值显存；所有采样 pair 都会被
评估，不改变数学目标。event/future index 始终是 episode-local，禁止跨
episode pair。

### 4.3 CMD：reader/action 使用约束

CMD 使用同样的事件 snapshot，但在 reader replay 前 detach snapshot。因此它
训练 action reader 和共享 action tail，而不会偷偷添加第二个 writer objective。
四项可审计约束为：

1. 正确 memory 的动作向 full-history teacher action 蒸馏；
2. 正确减错误 memory 的动作差异匹配正 pair 的 teacher effect；
3. 对专家目标施加固定 margin，使正确 memory 优于错误 memory；
4. 对 null/irrelevant pair 施加近似不变性。

部署没有 teacher、未来帧、pair label 或 counterfactual branch；只保留同一个
因果 writer、fast state 和 action reader。

### 4.4 总损失

V3 配置 `hd_attribution_protocol=credit_ttt_v3_query_effect` 时：

\[
\boxed{\mathcal L=
\mathcal L_{\rm flow}
+\lambda_{\rm local}\mathcal L_{\rm QH2L}
+\lambda_{\rm CMD}\mathcal L_{\rm CMD}
+0.01\mathcal L_{\rm anchor}}
\]

论文 launcher 固定使用 `lambda_local=1`、`lambda_CMD=1`、
`hd_v3_null_weight=0.25`、`hd_v3_cmd_margin=0.05`。`L_anchor` 是很小的
固定 K/V binding anchor，只用于数值锚定，不是新的 hindsight target。旧版
`hd_effect_weight` 在 V3 强制为 0，避免把 V2 loss 混入主实验。

## 5. 数据、序列和优化协议

### 5.1 数据划分

正式 recipe 对每个任务单独训练一个模型，并使用官方全部 250 个示范：

| 用途 | episode |
| --- | --- |
| prefix cache、teacher、pair label、student | `[0, 250)` 全部示范 |
| validation | 默认不从官方示范中扣除；可做不参与选择的诊断 |
| simulator test | 固定 seed，绝不用于选超参数 |

teacher、normalization、label artifact 和 checkpoint 都是 task-local；不同任务
不能共享一个 student checkpoint。

### 5.2 Full-history sequence contract

student 的 `sequence_length` 在 preflight 时解析为不小于所选训练 episode 的
最长长度，`sequence_stride == sequence_length`，
`max_windows_per_episode=1`，`ttt_history_warmup_length=null`。一个训练
window 就是一个完整 episode，不跨边界。TBPTT 只是计算图截断，不能重置
episode 内 fast state；每个 window 开始时才 reset。

`_lerobot_sequence_offset` 是 episode-local offset。segment 从局部位置 (s)
开始时传入 `sequence_offset=window_offset+s`，新 episode 再归零；不能使用
拼接数据集的全局行号，也不能静默借用另一个 episode 的 previous action。

### 5.3 batch、DDP 和梯度累积

默认每卡 `batch_size=1`，因为不同长度的 episode 不能用时间 padding 混合。
若显存允许，可打开 `EQUAL_LENGTH_BATCHING=1` 并把 batch 调大：只把完全相同
的 `(physical_length, episode_local_offset)` trajectory 放在一起，每个样本有
独立 fast state，不插入 padding update。短 bucket 的重复仅用于补齐 batch/DDP，
不会丢弃官方示范，重复数量会写入 provenance sidecar。

DDP 的 flow loss 使用全局有效 action-slot 数加权；QH2L/CMD 的正/null 分母
使用 all-rank pair denominator，再抵消 Accelerate 的显式 gradient mean。因此
单卡和多卡的目标定义一致。`gradient_accumulation_steps` 只平均独立 window，
不会把 pair 分母跨 window 拼接；fast state 在 window 间清空。

参考 launcher 设定：

```text
NUM_PROCESSES=4                 # 四卡时
BATCH_SIZE=1                    # 可配合 EQUAL_LENGTH_BATCHING=1 增大
GRADIENT_ACCUMULATION_STEPS=1
TBPTT_SEGMENT_LENGTH=32
EPOCHS=150                      # 150 个完整 sequence epochs
PAIR_K=5
TTT_HIDDEN_DIM=1024
TTT_LAYERS=[12,13,14,15]
REGISTER_TOKENS=16
```

这里的 epoch 是一次完整 episode-window pass，不是把原始帧任意切成短
window 的“frame epoch”。硬件相关的 pair chunk、CPU offload、batch fill 是
执行参数，不能被报告为方法创新，也不能用 test SR 选择。

## 6. 代码入口

```text
policy/HD-TTT/
├── README.md                                  # 本文：唯一论文算法说明
├── examples/mikasa/
│   ├── train_credit_ttt.sh                     # teacher → labels → student
│   ├── train_full_history_teacher.py           # HCA causal action teacher
│   ├── mikasa_data.py                          # 与目标无关的 episode 预处理
│   ├── build_credit_labels.py                  # V3 event/future pair artifact
│   ├── evaluate_smolvla_ttt.py                 # CreditTTT/Clean-TTT evaluator
│   ├── evaluate_smolvla_baseline.py            # 原生 SmolVLA evaluator
│   ├── benchmark_credit_ttt_v3.py              # manifest、aggregate、audit
│   └── train_native_smolvla.sh                 # baseline training launcher
└── src/lerobot/policies/smolvla_ttt/
    ├── configuration_smolvla_ttt.py            # 配置与 protocol guard
    ├── modeling_smolvla_ttt.py                 # SmolVLA flow、TTT、V3 loss
    ├── smolvlm_with_expert_ttt.py              # VLM/action-expert backbone
    ├── ttt.py                                  # fast MLP、update/read/state
    ├── history_teacher.py                      # HCA teacher/replay
    ├── credit_ttt_v3.py                        # pair schema、采样、目标
    ├── hd_dataset.py                           # label/provenance loader
    ├── sequence.py                              # full episode/TBPTT semantics
    └── processor_smolvla_ttt.py                # 输入输出处理
```

`hd_ttt.py` 仍被 `modeling_smolvla_ttt.py` 用作底层 fast-state/数值辅助，
不是另一个论文算法入口。LeRobot 其他 `src/lerobot/policies/*` 也保留是为了
factory、baseline 和 checkpoint 兼容；它们不属于 CreditTTT 的贡献。

## 7. 安装与训练

先按 [MIKASA-Robo-VLA 官方安装说明](https://mikasarobo.github.io/installation.html)
配置 simulator，再在本仓库安装 LeRobot 依赖。版本范围以 `pyproject.toml` 为准；
MIKASA 的 Python 环境和本地开发环境可以分开。

```bash
cd policy/HD-TTT
python -m venv .venv
source .venv/bin/activate
pip install -e ".[smolvla,training]"
export PYTHONPATH="$PWD/src:${MIKASA_ROOT:-$PYTHONPATH}"
```

所有路径以下都是占位符。`plan` 不读取 GPU、不改数据、不启动训练；确认
输出后才设置 `EXECUTE=1`。

```bash
cd policy/HD-TTT

# 0. 查看完整协议和解析后的结构
TASK_ID=shuffle_long ./examples/mikasa/train_credit_ttt.sh plan

# 1. HCA teacher + frozen prefix feature cache
EXECUTE=1 TASK_ID=shuffle_long \
  DATASET_ROOT=/workspace/data_mikasa_robo/data_lerobot/shell_game_shuffle_color_lamp_touch_long_vla_v0 \
  BASE_CHECKPOINT=/workspace/checkpoints/clean_ttt \
  ./examples/mikasa/train_credit_ttt.sh teacher

# 2. event-write-deletion pair labels
EXECUTE=1 TASK_ID=shuffle_long INTERVENTION=delete \
  ./examples/mikasa/train_credit_ttt.sh labels

# 3. QH2L + CMD student（每个任务、每个 seed 独立输出）
EXECUTE=1 TASK_ID=shuffle_long SEED=1000 \
  NUM_PROCESSES=4 BATCH_SIZE=1 \
  ./examples/mikasa/train_credit_ttt.sh student
```

也可以运行 `EXECUTE=1 ... train_credit_ttt.sh all` 按顺序执行三个阶段；
它只训练和写 artifact，不会伪造 simulator 结果。训练输出目录必须包含
`training_metadata.json`，其中记录 sequence 长度、DDP、重复样本、配置和
execution bounds。

## 8. MIKASA benchmark

### 8.1 比较矩阵

正式结果用官方 runner、相同 episode seeds 和相同 task-local 数据统计：

| 方法 | 含义 | 默认执行 cadence |
| --- | --- | ---: |
| Native-SmolVLA | 原始 `lerobot/smolvla_base`，无 fast state | K=50 |
| Native-SmolVLA-K1 | 同一 native checkpoint，每次只执行 slot 0 的 cadence control | K=1 |
| Clean-TTT | 相同 TTT backbone，关闭 hindsight objective | K=1 |
| CreditTTT | 本文 V3（QH2L+CMD） | K=1 |
| Utility-KVB | 可选机制对照 | K=1 |

Native K=50 与 TTT K=1 的动作 cadence 不同，不能把两者直接差异全部归因
于 memory；因此论文主表必须包含 Native-K1 control。Clean-TTT 与 CreditTTT
使用相同 backbone、optimizer budget、action cadence、reset protocol 和
episode seeds，只改变 hindsight objective。

推荐的公开四任务 profile：

| ID | 环境 | 数据集 |
| --- | --- | --- |
| `shell_touch` | `ShellGameTouch-VLA-v0` | `shell_game_touch_vla_v0` |
| `intercept_medium` | `InterceptMedium-VLA-v0` | `intercept_medium_vla_v0` |
| `remember_color3` | `RememberColor3-VLA-v0` | `remember_color_3_vla_v0` |
| `remember_color9` | `RememberColor9-VLA-v0` | `remember_color_9_vla_v0` |

早期开发还支持 `color`（Short/Spatial）和 `shuffle_long`
（Medium/Tracking/MP）；它们通过 `--task-set legacy_two` 作为单独的历史
profile，不能和四任务主表混合。

默认每个方法在 50 个固定 simulator episodes 上评测，起始环境 seed 为
`4242424242`，torch seed 为 `7000+i`。报告 per-episode `success_once` 的
SR、mean return、95% hierarchical paired-bootstrap CI（10,000 replicates）
以及 common-seed exact McNemar test。teacher extraction/fitting 是训练成本，
不计入部署 inference cost。

### 8.2 评测命令

CreditTTT/clean evaluator：

```bash
PYTHONPATH="$PWD/src:$MIKASA_ROOT" \
python examples/mikasa/evaluate_smolvla_ttt.py \
  --checkpoint /workspace/checkpoints/credit_color \
  --dataset-root /workspace/data_mikasa_robo/data_lerobot/shell_game_color_lamp_touch_vla_v0 \
  --task ShellGameColorLampTouch-VLA-v0 \
  --num-episodes 50 --sim-backend gpu --device cuda \
  --output /workspace/eval/credit_color_50.json
```

原生 baseline（K=50）：

```bash
PYTHONPATH="$PWD/src:$MIKASA_ROOT" \
python examples/mikasa/evaluate_smolvla_baseline.py \
  --checkpoint lerobot/smolvla_base \
  --dataset-root /workspace/data_mikasa_robo/data_lerobot/shell_game_color_lamp_touch_vla_v0 \
  --task ShellGameColorLampTouch-VLA-v0 \
  --num-episodes 50 --execution-action-steps 50 \
  --sim-backend gpu --device cuda \
  --output /workspace/eval/native_color_50.json
```

加 `--execution-action-steps 1` 得到 Native-K1 cadence control。先用
`benchmark_credit_ttt_v3.py manifest` 固定 task-local checkpoint map，再用
`aggregate` 汇总；评测脚本不会自动选择最好 checkpoint，也不会伪造缺失结果。

## 9. 已有实验记录（不等于论文结果）

下表只记录当前确实存在的 artifact。V3 正式训练在远端收到停止指令时为
`220/1650` optimizer steps，故 `0/10` 是 early diagnostic，不是完整训练
失败结论。

| artifact / protocol | task | episodes | memory | SR | mean return |
| --- | --- | ---: | --- | ---: | ---: |
| Native SmolVLA zero-shot (`color_native_50.json`) | Color | 50 | 无，K=50 | 0/50 = 0% | 1.2109 |
| CreditTTT V3 checkpoint 000220 | Color | 10 | persistent, K=1 | 0/10 = 0% | 0.5184 |
| 同一 V3 checkpoint | Color | 10 | reset every step, K=1 | 0/10 = 0% | 0.5794 |
| 历史 Clean-TTT（旧协议） | Color | 50 | persistent | 7/50 = 14% | 3.2794 |
| 历史 HD-TTT/V2（旧协议） | Color | 50 | persistent | 6/50 = 12% | 3.1717 |

历史 Clean/HD 数字使用不同 checkpoint、训练协议和时间点，只能作为工程
sanity reference；不能与 V3 early-220 或 Native zero-shot 做显著性比较。
Shuffle-Long 的 V3 SR、完整 150-epoch V3 SR、正式训练 Native baseline、
Native-K1 control 和四任务 aggregate 当前均为 **not measured**。论文表格只
能填入官方 runner 生成并写入 frozen manifest 的结果。

## 10. 必做机制审计与消融

在声称“方法有效”前，至少保存以下独立证据：

- full-history teacher 相对短历史的 action loss 改善；
- history write deletion 在预期方向改变 teacher；
- QH2L teacher/student effect cosine 的 95% CI 下界为正；
- local effect 与短时端到端 gradient credit 一致；
- 长 delay bin 中 writer gradient 非零；
- top-attributed events 优于 random，且 recall@8 高于 random；
- correct/wrong/reset memory drift 大于 irrelevant-memory drift；
- 所有 loss、fast state 和 action 都 finite，RMS ratio 有界。

结构消融应预注册并固定数据、seed、optimizer update 和 cadence：

1. Clean-TTT vs CreditTTT；
2. 0 registers vs 16 registers；
3. `suffix` writer vs causal `prefix_only` writer；
4. QH2L-only、CMD-only、QH2L+CMD；
5. event-write deletion vs 单独命名的 content replacement；
6. persistent state vs reset-every-step；
7. full-history vs 明确命名且匹配 seen frames 的 bounded window；
8. Native K=50 vs Native K=1。

`V3_ABLATION=full|qh2l_only|cmd_only` 会把 objective family 写入 checkpoint
provenance。inner learning rate、residual gate、null weight、pair budget 等
是实现参数，可以影响性能，但必须由训练 recipe 预先固定，不能用 simulator
test SR 调参。若做 ±20% 邻域稳健性分析，应作为额外审计而非挑选最好数字。

## 11. 论文边界与可复现性

- 当前 teacher 目标是 final normalized executed slot-0 action，不是 flow velocity。
- 当前 writer 使用 prefix + previous executed action；不是声称每次写入都看见
  VLM 最后一层完整 contextualized hidden state。
- 当前 QH2L effect 在最后选定 TTT layer、最终执行 slot 0 定义，未宣称独立
  归因全部 50 个 action slots 或每个中间 velocity。
- full-flow pair replay 只在训练时执行，second-order 可能昂贵；chunking 只
  是显存界限。部署仍是每个 observation 一次 write、九次 read-only denoise。
- Color episode 没有长 delay 数据；长程结论必须来自 Shuffle-Long 或其他足够
  长的任务。
- tensor shape/finite smoke 不能替代 simulator success。缺少官方 aggregate
  时，结果必须写 `not measured`。

每次冻结实验请同时保存：Git commit、manifest SHA256、checkpoint/config hash、
dataset revision、episode list、Python/PyTorch/Transformers 版本、GPU、seed、
所有 launcher 环境覆盖和 `training_metadata.json`。

## 12. 本次迁移整理

- 删除了仓库内开发测试与测试 artifact；后续测试由项目维护者重新编写。
- 删除重复的旧 README，当前只保留本文件。
- 删除旧 MIKASA HD/V1/V2 的 `build_hd_labels.py`、
  `build_hd_window_labels.py`、`train_hd_ttt.sh` 入口；V3 预处理已独立到
  `mikasa_data.py`，所以 canonical teacher 不再依赖旧算法脚本。
- 仓库内 `.venv`、build、ruff/pytest、Python bytecode 等生成物已移出项目目录；
  它们不应提交到迁移后的 Git 仓库。
- 所有改动范围仅为 `policy/HD-TTT`；`lib` 与
  `policy/Method1_lerobot-pi0-ttt` 保持不变。

本目录按 Apache-2.0 LeRobot 代码许可分发；论文中请同时注明原始 SmolVLA、
RoboTTT/TTT 以及 MIKASA-Robo-VLA 的对应来源和许可证。
