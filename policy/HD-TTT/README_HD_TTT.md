# HD-TTT：面向长程控制记忆的 Hindsight-Distilled Test-Time Training

本文档描述 `policy/HD-TTT` 中当前可运行的算法实现、训练协议和 MIKASA-Robo-VLA 实验方法。它是本项目的算法说明，不是对上游 LeRobot 或任何论文实现的逐字复现。

## 先看结论

HD-TTT 在 SmolVLA 的 action expert 中加入可跨物理时刻保存的 TTT fast weights，并用成功示范的完整历史离线计算“某段过去是否影响未来控制”的信用，再把这个信用蒸馏为部署时可计算的局部写入目标和因果写入 gate。

部署时只需要当前 observation、language、proprioception、fast weights，以及模型内部采样的 Gaussian action noise：没有 hindsight teacher、未来 observation、专家 action 或离线标签。

当前实现的三个方法组件是：

1. **Hindsight Control Attribution（HCA）**：对完整成功轨迹做事件级 zero-write counterfactual，得到历史事件对未来 flow-action 预测的控制信用。
2. **Hindsight-to-Local TTT Distillation（H2L）**：用 HCA 信用加权本地 K/V fast-weight 重构目标，并训练一个只看当前 causal prefix 的写入 gate。
3. **Causal Memory Deployment**：用 true-memory / wrong-memory 两条 detached replay 分支约束 reader 真正使用记忆，并在 episode 边界显式 reset。

> `hd_ttt_enabled=false` 只关闭 HD 辅助目标和 HD gate，保留 SmolVLA-TTT 的 fast-weight 路径。它不是原生无 TTT 的 SmolVLA。原生 SmolVLA 应使用单独的 baseline policy/evaluator。

### 当前研究边界（不要过度解读）

当前版本的 clean teacher 是“从 episode 开始做完整因果 fast-weight replay”的冻结 SmolVLA-TTT teacher，不是额外训练的、显式读取全部历史的 oracle Transformer/SSM。因此 HCA 严格测量的是**某段 fast-weight write path 被移除后对未来预测的影响**；论文中不应把它表述成已经解决了任意历史信息的 oracle credit assignment。

同样，当前 H2L 是 hindsight-credit-weighted local K/V objective 加 causal write-gate distillation，尚未提供独立的 content/address target 来直接指定 fast weights 应存储什么。counterfactual grounding 约束 reader 使用 true/wrong memory，但不等同于完整的 content distillation。

实现里 gate 的输入是 prefix-only；TTT 的 K/V writer 则作用于 expert suffix hidden states，所以当前部署 writer 仍可能受 action/time/noise 表示影响。这是需要单独报告的 noise-sensitivity 实验问题，而不是可由固定随机种子自动排除的事实。正式结果还应报告：50-slot 与实际执行 slot-0 的 attribution 口径、`max-events=8` 相对 exhaustive 的召回率、以及相同总训练预算的 clean-continued 对照。

## 1. 代码范围和保护边界

本算法的 base code 全部位于：

```text
/home/zeno-rp/2026test/test-TTT/policy/HD-TTT
```

以下目录是外部/对照代码，算法开发时不修改：

```text
/home/zeno-rp/2026test/test-TTT/lib
/home/zeno-rp/2026test/test-TTT/policy/Method1_lerobot-pi0-ttt
```

`policy/HD-TTT` 是独立的 LeRobot policy tree；它不依赖上述受保护目录中的实现。

## 2. 符号和问题定义

| 符号 | 含义 | 当前实现 |
| --- | --- | --- |
| $o_t$ | 当前视觉 observation | MIKASA 的 top/wrist RGB |
| $l_t$ | language instruction | 当前 task instruction |
| $s_t$ | proprioceptive state | MIKASA 7 维 state，内部 pad 到 32 维 |
| $x_t$ | action expert 的 noisy action chunk | 50 个槽位，内部 pad 到 32 维 |
| $v_t$ | flow-matching velocity prediction | `action_out_proj` 输出，取前 7 维作为 MIKASA action |
| $W_t^\ell$ | 第 $\ell$ 个 TTT 层的 fast state | 每个 episode/trajectory 独立保存 |
| $P_t$ | 当前 causal prefix | image tokens + language tokens + state token |
| $Q,K,V$ | TTT 的 query/key/value projection | 这里是投影名称，不是额外的机器人状态变量 |

当前设计不引入额外的 belief/progress/recovery head；`W_t` 应该成为完整历史的固定大小 latent state。瞬时的 prefix pooled context 只用于预测本步 gate，不是另一套长期记忆。

## 3. 网络结构

### 3.1 SmolVLA backbone 和 action expert

默认配置从 `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` 构造 SmolVLA：

- 使用 VLM 的前 16 个 text layers；
- 构造 16 层、宽度为 VLM `0.75×` 的 action expert（当前 hidden size 为 720，FFN intermediate size 为 2048）；
- `attention_mode="cross_attn"`，`self_attn_every_n_layers=2`：action expert 在交替层读取缓存的 VLM prefix K/V，其余层保留 suffix self-attention；
- TTT hook 位于 action expert 每个选中层的 attention residual 之后、expert MLP 之前；
- VLM 和 vision encoder 默认冻结，`ttt_training_stage="ttt_only"` 时只训练新增 TTT/register 参数。

对 MIKASA，两个 128×128 RGB 相机和 7 维 proprioception 经过标准 SmolVLA processor；action expert 仍使用固定 50-slot chunk，环境每个物理决策只执行 1 个 action。

### 3.2 Register tokens

默认 `ttt_num_register_tokens=16`。register 是 action expert suffix 中的**可学习参数**，不是额外的环境输入，也不是第二个 action head。

suffix 的顺序为：

```text
[16 register tokens] + [50 noisy action/time tokens]
```

attention mask 是有意不对称的：

- register query 可以读取当前 prefix 和完整的 50 个 action token，也可以读取其他 register；
- action query 保留原来的 causal action-to-action triangle；
- action query 不能直接读取 register key 列。

因此 register 是“前置、参与单步 attention、后置读出”的工作区：它能汇总当前 action block 并影响后续层的隐藏状态，但不会成为 action token 的直接可见 shortcut。将 `REGISTER_TOKENS=0` 或 `--policy.ttt_num_register_tokens=0` 可做无 register 对照。

### 3.3 参数规模（当前 recipe 的实际 checkpoint）

下面的数字由当前 SmolVLA-TTT checkpoint 的 `model.safetensors` state dict 统计，具体数字会随 backbone/config 改变：

| 部分 | 参数量（约） | 说明 |
| --- | ---: | --- |
| VLM + action expert | 448.41M | 主要冻结的 backbone/expert |
| 4 个 TTT 层 | 12.129M | 每层含 Q/K/V、fast MLP 初始权重、inner-lr 和 residual gate |
| 16 registers | 11.5K | 16×720 |
| state/action/time projections | 1.635M | `ttt_only` 时冻结；action-head stage 时可解冻 |
| 整个 checkpoint | 462.19M | 当前实现的典型总量 |

`ttt_only` 下实际优化约 12.14M 个新增参数（约占总参数 2.6%）；HD learned gate 只额外增加一个很小的 prefix-context linear head。不要把“HD 关闭”误写成“没有 TTT 参数”。

## 4. TTT 的单步机制

### 4.1 局部 K/V inner update

在选中的层，对该层 attention residual 的 token 表示 $h_{t,n}$ 做归一化并投影：

\[
k_{t,n}=K(\operatorname{LN}(h_{t,n})),\qquad
v_{t,n}=V(\operatorname{LN}(h_{t,n})),\qquad
q_{t,n}=Q(\operatorname{LN}(h_{t,n})).
\]

fast MLP $f_W$ 的本地目标为：

\[
\ell_{\mathrm{KV}}(W;t)=\frac12\left\|f_W(k_t)-\operatorname{sg}(v_t)\right\|_2^2.
\]

先做 inner gradient update，再用更新后的 state 读取 query：

\[
\widehat W_t=W_t-\eta_t\nabla_W\ell_{\mathrm{KV}}(W_t;t),
\qquad
W_{t+1}=W_t+g_t(\widehat W_t-W_t),
\]

\[
\operatorname{TTT}(h_t)=h_t+\tanh(\gamma)\,f_{W_{t+1}}(q_t).
\]

这就是本项目的 **update-then-apply**：当前 interaction 先写入 fast weights，当前 token 的 residual read 使用更新后的 $W_{t+1}$。`g_t=0` 是严格的 zero-write intervention；$g_t\in(0,1)$ 是可微插值。

### 4.2 Denoising 时序

默认 `num_steps=10`，$dt=-1/10$：

```text
x_1 ~ N(0, I)
step 0: t=1.0   -> update fast state, then read with updated state
step 1: t=0.9   -> read only
...
step 9: t=0.1   -> read only
x_{k+1} = x_k + dt * v_k
```

每个物理 observation 的 callback 只在第一个 denoising step 传 `update=True`；后续 9 步读取同一个已更新的 fast state。下一个物理 observation 到来时才进行下一次写入。episode 结束时调用 `policy.reset()`，不得跨 episode 复用 state。

训练的普通 `hd_phase_mode="random"` 使用随机 flow interpolation；正式 HD recipe 使用 `hd_phase_mode="deployment"`，让写入 interaction 与部署一致：$t=1$、输入是纯高斯 action noise，而不是 teacher-forced future action chunk。外层 flow loss 仍然用专家 action 作为 target；writer 仍读取当前 noisy-action tokens，但不会看到未来/专家 action，gate context 则完全不读取 action。

## 5. 三个 HD-TTT 组件

### 5.1 Hindsight Control Attribution（HCA）

HCA 只在训练/离线 label 阶段运行。对一条完整成功 episode：

1. 用 clean SmolVLA-TTT teacher 从 episode 开头做 full-history causal replay，得到每个未来 frame 的 flow velocity 和 action loss $L_j$。
2. 将过去划分为长度 `event_block_size=B` 的事件 $E_i=[b_i,b_i+B)$。对每个事件做 zero-write replay：该 block 的写入 gate 设为 0，其余事件正常写入。
3. 只在事件结束之后统计未来，并忽略 episode 边界后的 frame/padded action slot：

\[
C_{i,j}=\left[L^{-i}_j-L_j\right]_+,\qquad j\ge b_i+B.
\]

这里 $C_{i,j}$ 是“删除事件 $i$ 后，未来动作预测变差了多少”，不是 RL action value，也不修改真实轨迹。当前 builder 输出：

- `hd_attribution`：所有事件对每个 future 的最大正信用（HCA 权重）；
- `hd_write_gate`：事件级 $u_i=\max_j C_{i,j}$ 的 episode 内 max-normalized frame 映射（builder 的 frame-label 实现采用 max；通用 `compute_hindsight_attribution` primitive 默认保留 row sum）；
- `hd_rho`：与保存的单一 wrong-memory branch 对齐的 selected-event future dependency；
- `hd_C`、事件区间、eligible counts、total credits 等审计信息（保存在 episode metadata/detail 中）。

默认 `max-events=0` 会 replay 所有 causal blocks；正式远程实验为控制开销显式使用 `max-events=8`，未采样 block 的 `hd_write_gate` 安全默认值为 1.0，但它们的 `hd_write_gate_observed=0`，不会被当成观测到的 gate target。

grounding 的 selected event 采用：优先选择 eligible future frame 数不少于 `grounding_min_future_frames=64` 的正信用事件，并按 mean credit 选最大；短 episode 没有满足阈值的事件时，退化为 total credit 最大的正事件。该策略写入 metadata：

```text
grounding_event_policy=min_future_horizon_mean_else_total_credit
```

Color 的短 episode 常触发 `total_credit_fallback`；Shuffle-Long 通常走 `min_future_horizon_mean`。两种 selection mode 都应在实验日志中报告。

### 5.2 Hindsight-to-Local（H2L）

部署时不能计算 $C_{i,j}$，所以不能直接把未来 action loss 当作 online objective。H2L 做的是：

\[
L_{\mathrm{H2L}}
=\operatorname{WeightedMean}_t
\left[u_t\,\ell_{\mathrm{KV}}(W_t;t)\right].
\]

实现直接从每个 TTT layer 的当前 inner K/V prediction loss 计算该项，不要求预先存储 `hd_local_*` 张量；旧格式的 projected local K/V 字段只作为兼容 fallback。

同时，在最早的选中 TTT 层预测一个共享 gate：

\[
g_t=\sigma\bigl(h_\phi(\operatorname{Pool}(P_t))\bigr),
\qquad
L_{\mathrm{gate}}=\operatorname{SmoothL1}(g_t,\operatorname{sg}(u_t)).
\]

`Pool(P_t)` 只包含当前 image/language/state prefix；它在 action suffix 嵌入前计算，因此不包含 noisy action、flow noise 或 denoising timestep。离线 `u_t` 只是训练 target，部署时由 gate 自己预测。H2L 的准确表述是“hindsight-credit-weighted local K/V objective”，不是直接 future-action distillation。

### 5.3 Causal Memory Deployment / counterfactual grounding

为防止模型学会“写了但不读”，训练时在同一 batch 维护三条数值 state：

- `main`：正常训练路径，可反向传播；
- `true`：all-write 的 detached replay；
- `wrong`：对 selected event 使用 zero-write 的 detached replay。

学生的 true/wrong velocity 差异应匹配 teacher 的差异：

\[
d_s=v^{\mathrm{true}}_s-v^{\mathrm{wrong}}_s,\qquad
d_T=\operatorname{sg}(v^{\mathrm{true}}_T-v^{\mathrm{wrong}}_T).
\]

高 dependency 的 future 用 direction matching，低 dependency 的 future 用 invariance：

\[
L_{\mathrm{ground}}
=\rho\,\|d_s-d_T\|^2
 +\lambda_{\mathrm{inv}}(1-\rho)\,\|d_s\|^2.
\]

teacher velocity、wrong branch 和 intervention gate 都在 reader-only replay 中 detach；因此 grounding 主要训练 query/readout/action pathway，不会绕过部署时的 writer 规则。true/wrong state 在 TBPTT segment 边界各自继续携带，只有 episode/window 生命周期结束才丢弃。

### 5.4 总训练目标

当前实现对 flow/HCA/grounding 使用有效 action slot mask；H2L local-writer 则使用 `hd_writer_valid`（因此可以在 history warm-up frame 上训练），总目标可写为：

\[
L = L_{\mathrm{flow}}
 +\lambda_{\mathrm{HCA}}L_{\mathrm{HCA}}
 +\lambda_{\mathrm{H2L}}L_{\mathrm{H2L}}
 +\lambda_{\mathrm{G}}L_{\mathrm{ground}}
 +\lambda_{\mathrm{gate}}L_{\mathrm{gate}}.
\]

默认值为：

```text
hd_hca_weight=1.0
hd_h2l_weight=1.0
hd_grounding_weight=1.0
hd_invariance_weight=0.25
hd_write_gate_weight=1.0
hd_counterfactual_margin=0.0
```

历史 warm-up frame 会推进 recurrent state，但 action/HCA/grounding target 被 mask；它们仍可通过 `hd_writer_valid` 参与 local writer objective。terminal repeated/padded action slots 使用原始 slot-valid mask，避免 episode 尾部重复动作放大损失。

## 6. 训练和部署的严格闭环

```text
成功示范 episode
        │
        ├─ clean teacher full-history replay
        │       └─ event zero-write replay -> C, u, rho, true/wrong velocities
        │
        └─ frame-level HD label artifact（训练时使用）
                         │
当前 prefix + noisy action ──> SmolVLA expert ──> TTT update/read
                         │                         │
                         └─ flow + HCA + H2L + gate + grounding
                                                   │
                                           HD-TTT checkpoint
                                                   │
部署：当前 prefix + x1~N(0,I) ──> 第一步写入/读取，后九步只读取
                                                   │
                                           action，保存 W，下一帧继续
```

部署端不加载 label artifact；label loader、teacher replay 和 counterfactual branch 都不会被调用。

## 7. Full-history 数据协议

### 7.1 为什么必须 full history

HCA 的问题是“删除早期 interaction 后，未来动作是否变差”。如果训练窗口在 episode 中间开始，早期信息本来就不存在，信用会被错误地归零。因此正式 recipe 要求：

- 每个 episode 只取一个覆盖全段的 window：`max_windows_per_episode=1`；
- `sequence_length` 不小于选中 episode 的最大长度；
- `sequence_stride=sequence_length`；
- `ttt_history_warmup_length=full`（CLI 内部传 JSON `null`）；
- episode 边界不能串 state；
- TBPTT 只能截断 outer gradient，不能清空数值 fast state。

当前两个任务的完整 episode 长度分别不超过 29 和 513，因此正式配置为 Color `sequence_length=64`、Shuffle-Long `sequence_length=513`。训练脚本启动前会计算真实 window 数和 `steps_per_epoch`；对 250 个 episode、4 个进程，每个任务都是 250 windows、63 distributed steps/epoch、150 epoch = 9450 steps。日志中的 `epch` 是按样本计数的旧 tracker 字段，不能替代脚本打印的 window epoch。

### 7.2 HD label provenance contract

label artifact 必须由 clean `smolvla_ttt` teacher 生成，不能用普通 SmolVLA 或已启用 HD gate 的 checkpoint。loader 严格校验：

- dataset repo id、fps、episode/frame/global index；
- teacher config SHA256；
- TTT layer indices、register 数；
- `event_block_size`、`max_events`、`grounding_min_future_frames`、threshold、phase mode；
- `history_mode="full_episode_replay"`；
- fixed `hd_noise` 和 `hd_time`（deployment phase 时 `hd_time=1`）。

不匹配的旧 artifact 会直接拒绝，而不是静默解释。推荐保留每个 label shard 的 metadata 和 merge audit trail。

### 7.3 当前正式 recipe 的超参数

除任务专属的 `sequence_length` 外，Color 和 Shuffle-Long 使用同一套模型/优化设置：

| 项目 | 值 |
| --- | --- |
| TTT layers | `[12, 13, 14, 15]` |
| TTT fast hidden dim | `1024` |
| fast inner learning rate | `0.1`（每层可学习 multiplier） |
| residual effective gate init | `0.05` |
| second-order TTT | `false`（first-order，控制显存） |
| register tokens | `16` |
| action chunk / executed steps | `50` / `1` |
| denoising steps | `10` |
| training stage | `ttt_only`（VLM、expert 和 projection head 冻结） |
| optimizer | AdamW，lr `1e-4`，betas `(0.9, 0.95)`，weight decay `1e-10` |
| scheduler | warmup `1000`，decay `30000`，final lr `2.5e-6` |
| gradient clip | `10` |
| distributed precision | 4 GPU、`bf16`、batch size `1` |
| image resize | `[224, 224]`（padding 保持比例） |
| train / label / eval seed | `1000` / `1729` / `7000`（eval seed 可按实验登记） |

HD 阶段在 clean checkpoint 上开启 `hd_ttt_enabled=true`、`hd_learned_write_gate=true`，并使用 `hd_phase_mode=deployment`、`event_block_size=4`、`max_events=8`、`grounding_min_future_frames=64`、`attribution_threshold=0`。这些值构成当前两任务的可复现实验 protocol；做 ablation 时必须把改动写入 checkpoint config 和 label metadata。

## 8. MIKASA-Robo-VLA 实验

MIKASA 的官方安装、任务定义和评估协议见：

- [官方安装说明](https://mikasarobo.github.io/installation.html)
- [官方 benchmarking protocol](https://mikasarobo.github.io/benchmarking.html)
- [官方 evaluation protocol](https://mikasarobo.github.io/evaluation_protocol.html)

当前项目使用两个单任务实验数据集（不是完整 90-task benchmark score）：

| task env id | LeRobot dataset repo id | 本地数据规模 | 官方类别 |
| --- | --- | ---: | --- |
| `ShellGameColorLampTouch-VLA-v0` | `shell_game_color_lamp_touch_vla_v0` | 250 episodes / 3,857 frames，长度 10–29 | Short / Spatial / PPO，max 30 |
| `ShellGameShuffleColorLampTouch-Long-VLA-v0` | `shell_game_shuffle_color_lamp_touch_long_vla_v0` | 250 episodes / 83,335 frames，长度 145–513 | Medium / Tracking / MP，max 600 |

两个任务必须分别加载各自的 dataset statistics；评估脚本故意限制一次只评估一个 task，避免把一个任务的 normalization 套到另一个任务上。正式官方评估使用 canonical start seed `4242424242`，建议固定 `--torch-seed` 以便 persistent/reset-memory 成对比较。

### 8.1 环境准备

下面示例假设已经按照 MIKASA 官方说明安装 simulator，并使用 benchmark 的 Python 3.11 环境；路径可按机器修改：

```bash
cd /workspace/test-TTT/policy/HD-TTT
source /workspace/MIKASA-Robo/.venv/bin/activate
export HF_HOME=/workspace/hf_cache
export HF_LEROBOT_HOME=/workspace/data_mikasa_robo
export PYTHONPATH="$PWD/src:/workspace/MIKASA-Robo"
```

先做 simulator/runner smoke test：

```bash
python /workspace/MIKASA-Robo/examples/eval_demo.py \
  --task ShellGameColorLampTouch-VLA-v0 \
  --num-episodes 1 --start-seed 4242424242 \
  --sim-backend cpu --output-dir /tmp/mikasa_color_dummy
```

### 8.2 第一阶段：clean SmolVLA-TTT

先从 `lerobot/smolvla_base` 初始化新增 TTT/register 参数，训练 clean TTT teacher。以 Color 为例：

```bash
REPO_ROOT=/workspace/test-TTT/policy/HD-TTT \
PYTHON_BIN=/workspace/MIKASA-Robo/.venv/bin/python \
ACCELERATE_BIN=/workspace/MIKASA-Robo/.venv/bin/accelerate \
DATASET_REPO_ID=shell_game_color_lamp_touch_vla_v0 \
DATASET_ROOT=/workspace/data_mikasa_robo/data_lerobot/shell_game_color_lamp_touch_vla_v0 \
OUTPUT_DIR=/workspace/experiments/mikasa_color_clean150_full \
PRETRAINED_PATH=lerobot/smolvla_base \
EPOCHS=150 NUM_PROCESSES=4 \
SEQUENCE_LENGTH=64 SEQUENCE_STRIDE=64 MAX_WINDOWS_PER_EPISODE=1 \
TBPTT_SEGMENT_LENGTH=16 HISTORY_WARMUP_LENGTH=full RESIZE='[224,224]' \
  HD_ENABLED=false HD_LEARNED_GATE=false HD_PHASE_MODE=random \
SAVE_FREQ=500 LOG_FREQ=50 \
bash examples/mikasa/train_hd_ttt.sh
```

Shuffle-Long 只需替换 dataset/replay 参数：

```bash
DATASET_REPO_ID=shell_game_shuffle_color_lamp_touch_long_vla_v0 \
DATASET_ROOT=/workspace/data_mikasa_robo/data_lerobot/shell_game_shuffle_color_lamp_touch_long_vla_v0 \
OUTPUT_DIR=/workspace/experiments/mikasa_shuffle_clean150_full \
SEQUENCE_LENGTH=513 SEQUENCE_STRIDE=513 MAX_WINDOWS_PER_EPISODE=1 \
TBPTT_SEGMENT_LENGTH=16 HISTORY_WARMUP_LENGTH=full \
bash examples/mikasa/train_hd_ttt.sh
```

实际运行时保留上一段中的 `REPO_ROOT`、Python、accelerate、pretrained 和其它环境变量。clean teacher 的最终路径应为：

clean 阶段的 `HD_PHASE_MODE=random` 是普通 flow-matching 训练；只有 HD student 和 label replay 的 `deployment` phase 才强制使用 $t=1$ 的 Gaussian writer input。

```text
<clean_output>/checkpoints/009450/pretrained_model
```

（若 `SAVE_FREQ` 不同，以实际最后一步目录为准。）

### 8.3 第二阶段：生成 HCA labels

label pass 必须使用 clean teacher，并建议使用 deployment phase。一个 shard 的命令如下：

```bash
python examples/mikasa/build_hd_labels.py \
  --dataset-repo-id shell_game_color_lamp_touch_vla_v0 \
  --dataset-root /workspace/data_mikasa_robo/data_lerobot/shell_game_color_lamp_touch_vla_v0 \
  --checkpoint /workspace/experiments/mikasa_color_clean150_full/checkpoints/009450/pretrained_model \
  --output /workspace/labels/color_full_shard0.pt \
  --episode-start 0 --episode-end 63 \
  --event-block-size 4 --max-events 8 \
  --grounding-min-future-frames 64 \
  --attribution-threshold 0.0 --frame-batch-size 4 \
  --phase-mode deployment --device cuda --seed 1729
```

把 episode 划分为不重叠 shard（例如 `[0,63)`, `[63,126)`, `[126,189)`, `[189,250)`），可在不同 GPU 并行生成，然后显式 merge：

```bash
python examples/mikasa/build_hd_labels.py \
  --merge \
    /workspace/labels/color_full_shard0.pt \
    /workspace/labels/color_full_shard1.pt \
    /workspace/labels/color_full_shard2.pt \
    /workspace/labels/color_full_shard3.pt \
  --output /workspace/labels/color_full_all.pt
```

Shuffle-Long 使用相同 label 参数，只替换 dataset、teacher 和输出文件。`--max-events 8` 是明确的 compute-budget 设置；若要做 exhaustive ablation，改为 `--max-events 0`，并同时在训练配置中设置 `HD_MAX_EVENTS=0`。

### 8.4 第三阶段：HD-TTT training

从 clean teacher checkpoint 开启新 run，不要 resume clean run 的 optimizer state（learned gate 是新参数，没有对应 optimizer slots）：

```bash
REPO_ROOT=/workspace/test-TTT/policy/HD-TTT \
PYTHON_BIN=/workspace/MIKASA-Robo/.venv/bin/python \
ACCELERATE_BIN=/workspace/MIKASA-Robo/.venv/bin/accelerate \
DATASET_REPO_ID=shell_game_color_lamp_touch_vla_v0 \
DATASET_ROOT=/workspace/data_mikasa_robo/data_lerobot/shell_game_color_lamp_touch_vla_v0 \
OUTPUT_DIR=/workspace/experiments/mikasa_color_hd150_full \
PRETRAINED_PATH=/workspace/experiments/mikasa_color_clean150_full/checkpoints/009450/pretrained_model \
LABEL_PATH=/workspace/labels/color_full_all.pt \
EPOCHS=150 NUM_PROCESSES=4 \
SEQUENCE_LENGTH=64 SEQUENCE_STRIDE=64 MAX_WINDOWS_PER_EPISODE=1 \
TBPTT_SEGMENT_LENGTH=16 HISTORY_WARMUP_LENGTH=full RESIZE='[224,224]' \
HD_ENABLED=true HD_LEARNED_GATE=true HD_PHASE_MODE=deployment \
HD_EVENT_BLOCK_SIZE=4 HD_MAX_EVENTS=8 \
HD_GROUNDING_MIN_FUTURE_FRAMES=64 HD_ATTRIBUTION_THRESHOLD=0.0 \
bash examples/mikasa/train_hd_ttt.sh
```

### 8.5 官方 runner 评估和 memory ablation

主结果使用 episode-persistent memory：

```bash
python examples/mikasa/evaluate_smolvla_ttt.py \
  --checkpoint /workspace/experiments/mikasa_color_hd150_full/checkpoints/009450/pretrained_model \
  --dataset-repo-id shell_game_color_lamp_touch_vla_v0 \
  --dataset-root /workspace/data_mikasa_robo/data_lerobot/shell_game_color_lamp_touch_vla_v0 \
  --task ShellGameColorLampTouch-VLA-v0 \
  --num-episodes 50 --start-seed 4242424242 --torch-seed 7000 \
  --sim-backend gpu \
  --output /workspace/eval/color_hd_persistent.json \
  --official-output-dir /workspace/eval/color_hd_persistent_official
```

只清除跨物理步 memory 的 paired diagnostic：

```bash
python examples/mikasa/evaluate_smolvla_ttt.py \
  --checkpoint /workspace/experiments/mikasa_color_hd150_full/checkpoints/009450/pretrained_model \
  --dataset-repo-id shell_game_color_lamp_touch_vla_v0 \
  --dataset-root /workspace/data_mikasa_robo/data_lerobot/shell_game_color_lamp_touch_vla_v0 \
  --task ShellGameColorLampTouch-VLA-v0 \
  --num-episodes 50 --start-seed 4242424242 --torch-seed 7000 \
  --sim-backend gpu --reset-memory-every-step \
  --output /workspace/eval/color_hd_reset.json \
  --official-output-dir /workspace/eval/color_hd_reset_official
```

Shuffle-Long 必须使用自己的 dataset root 和 task id，不能与 Color 混在一次调用。clean checkpoint 的评估可直接自动识别 HD=false；对 HD checkpoint 做 clean ablation 时显式加 `--no-hd-ttt-enabled --no-hd-learned-write-gate`。

## 9. 消融矩阵

| 实验 | 关键设置 | 目的 |
| --- | --- | --- |
| Native SmolVLA | `evaluate_smolvla_baseline.py` | 无 TTT 的真正 baseline |
| Clean SmolVLA-TTT | `HD_ENABLED=false` | 测试 fast weights/register 本身 |
| No-register | `REGISTER_TOKENS=0` | 测试显式 register 的贡献 |
| HD-TTT | `HD_ENABLED=true`, `HD_LEARNED_GATE=true` | 完整方法 |
| Direct label-gated training | `HD_LEARNED_GATE=false` | 训练期直接用 label gate 的旧/对照路径，不是部署方案 |
| Reset memory | `--reset-memory-every-step` | 验证长程 memory 是否被真正使用 |
| Exhaustive attribution | `HD_MAX_EVENTS=0` | 去除事件采样 compute budget |
| No grounding | `HD_GROUNDING_WEIGHT=0` | 测试 reader grounding 的作用 |

所有 paired memory 比较应固定相同 checkpoint、episode seeds 和 `--torch-seed`；否则 flow noise 差异会混入 memory 差异。

## 10. 当前验证状态

代码层面当前测试集为：

```text
67 passed
```

覆盖 TTT inner update、state detach/carry、RoPE position、register mask、full-history/window contract、label provenance、grounding branch 和 MIKASA adapter。已完成的本地 Color HD one-step smoke 能得到有限 loss、H2L、gate 和 grounding diagnostics；它只证明数值路径可运行，不是 benchmark success rate。

正式的 Color/Shuffle 150-epoch 训练和官方 50-episode 分数应从远端实验目录的最终 checkpoint 与 JSON 读取。README 不预填未经最终 protocol 验证的旧分数；报告时至少同时给出：

- persistent-memory SR/return；
- reset-memory SR/return；
- clean TTT、no-register、native SmolVLA 对照；
- label selection mode、event budget、teacher SHA 和完整训练配置。

## 11. 文件地图

```text
src/lerobot/policies/smolvla_ttt/
├── configuration_smolvla_ttt.py   # policy/config、TTT/HD 开关和严格校验
├── modeling_smolvla_ttt.py        # SmolVLA flow、register、TTT hook、HD losses
├── smolvlm_with_expert_ttt.py     # VLM/action-expert 交替 self/cross attention
├── ttt.py                         # fast MLP、update-then-apply、RoPE、state
├── hd_ttt.py                      # HCA/H2L/grounding tensor primitives
├── hd_dataset.py                  # frame/window labels、metadata/provenance 校验
├── sequence.py                    # episode-contiguous windows、TBPTT state carry
└── processor_smolvla_ttt.py       # policy/data processors

examples/mikasa/
├── train_hd_ttt.sh                # clean/HD 两阶段训练 recipe
├── build_hd_labels.py             # clean teacher full-history + event interventions
├── evaluate_smolvla_ttt.py        # 官方 MIKASA runner adapter（persistent/reset）
└── evaluate_smolvla_baseline.py   # 原生 SmolVLA baseline adapter
```

## 12. 可复现性和常见错误

- 记录 git commit、Python/PyTorch/Transformers/LeRobot 版本、GPU、dataset root、teacher config SHA 和所有 `HD_*` 环境变量。
- label 生成和 HD training 的 `event_block_size`、`max_events`、`grounding_min_future_frames`、phase mode 必须完全一致。
- 不要用普通 `lerobot/smolvla_base` 直接生成 HCA labels；它没有训练好的 TTT/register fast weights，builder 会拒绝。
- 不要把 Color normalization 用到 Shuffle-Long，反之亦然。
- 不要把 `max_windows_per_episode=4` 的 bounded-window artifact 冒充 full-history artifact；正式 full-history 必须一 episode 一个完整 window。
- 不要在 eval episode 之间忘记 `policy.reset()`；否则前一条轨迹的 fast state 会污染后一条轨迹。
- `max-events>0` 时未采样事件的 gate=1 是安全 fallback，不是观测到的高信用；gate loss 必须使用 `hd_write_gate_observed`。
- 当前实现不支持 `torch.compile` 和启用 RTC；两者都会与 inner autograd/state update 冲突。
- 远端 `/workspace` 可能是非持久卷；训练完成后立即把 checkpoint、labels、logs 和 eval JSON 复制到持久存储。

## 13. 参考

- MIKASA-Robo-VLA: <https://mikasarobo.github.io/>
- MIKASA benchmarking: <https://mikasarobo.github.io/benchmarking.html>
- MIKASA evaluation protocol: <https://mikasarobo.github.io/evaluation_protocol.html>
- SmolVLA/LeRobot 上游实现：<https://github.com/huggingface/lerobot>
- RoboTTT（fast weights / recurrent test-time state 的实现启发）及 Chronos（历史 latent state 的概念启发）是本实现的出发点；本文档中的 HCA、H2L 和 causal grounding 是本项目自己的组合与训练协议。
