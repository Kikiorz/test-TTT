# HD-TTT：面向长程控制记忆的 Hindsight-Distilled Test-Time Training

本文档描述 `policy/HD-TTT` 中当前可运行的算法实现、训练协议和 MIKASA-Robo-VLA 实验方法。它是本项目的算法说明，不是对上游 LeRobot 或任何论文实现的逐字复现。

> **当前实验状态（2026-08-26）**：V2 的 clean-prefix、V2 label、writer-connected
> action-effect/second-order smoke 已经跑通；此前 4 卡报错已定位为 variable-length
> TBPTT 在 DDP 中的 collective 顺序问题，并在 `58db6d4` 修复。正式的 Color/Shuffle
> V2 长训练和官方 SR 仍在进行，本文档不会把旧版 v1 分数当作 V2 结果。

如果只想快速了解 V2 的执行顺序和当前风险，可先看本文档末尾的
[“V2 当前问题与验证状态”](#v2-当前问题与验证状态)；下面的定义和公式仍是完整的
实现参考。

## 先看结论

HD-TTT 在 SmolVLA 的 action expert 中加入可跨物理时刻保存的 TTT fast weights，并用成功示范的完整历史离线计算“某段过去是否影响未来控制”的信用，再把这个信用蒸馏为部署时可计算的局部写入/读取目标。论文主路径（v2）用与动作效果对齐的 true/wrong writer replay 训练 fast-weight 的写入内容；learned write gate 只保留为 legacy/no-effect 消融，不是 v2 的必要模块。这不是把未来信息带到部署端。

部署时只需要当前 observation、language、proprioception、fast weights，以及模型内部采样的 Gaussian action noise：没有 hindsight teacher、未来 observation、专家 action 或离线标签。

当前实现的三个方法组件是：

1. **Hindsight Control Attribution（HCA）**：对完整成功轨迹做事件级 zero-write counterfactual，得到历史事件对未来 flow-action 预测的控制信用。
2. **Hindsight-to-Local TTT Distillation（H2L）**：用 HCA 信用加权本地 K/V fast-weight
   写入目标；v2 进一步用 memory 对已执行 action slot 的影响（action-effect
   distillation）约束写入内容。只看当前 causal prefix 的 learned write gate 是
   legacy/no-effect 的可选消融，不是 v2 主路径。
3. **Causal Memory Deployment**：用 true-memory / wrong-memory 两条因果 replay 分支
   约束 fast-weight 对动作的真实影响，并在 episode 边界显式 reset。

> `hd_ttt_enabled=false` 只关闭 HD 辅助目标和 HD gate，保留 SmolVLA-TTT 的 fast-weight 路径。它不是原生无 TTT 的 SmolVLA。原生 SmolVLA 应使用单独的 baseline policy/evaluator。

### 当前研究边界（不要过度解读）

当前默认的 clean teacher 仍是“从 episode 开始做完整因果 fast-weight replay”的冻结 SmolVLA-TTT teacher，不是额外训练的、显式读取全部历史的 oracle Transformer/SSM。因此 HCA 严格测量的是**某段 fast-weight write path 被移除后对未来预测的影响**；论文中不应把它表述成已经解决了任意历史信息的 oracle credit assignment。

v2 已实现 action-effect distillation：teacher 提供 slot-0 的 true-minus-wrong velocity effect，student 用带 writer 梯度的两条 replay 分支匹配高依赖 effect、抑制低依赖变化。它是“控制效果”监督，不是把 latent fast weights 逐元素回归到一个可解释的 content/address target；论文应明确这一区别。`history_teacher.py` 中的 `CausalHistoryTeacher` 是独立的、可选的因果历史编码器/实验工具，当前默认 label builder **不会自动调用它**，也不应把它写成现有主结果的 oracle。

v2 论文 recipe 使用 `ttt_writer_mode=prefix_only`：K/V writer 的输入是当前 observation/language/state prefix，经共享投影进入 action-expert 宽度，并附带固定的 learned register anchors；action/noise/time suffix 只保留为 query/read path。`suffix` 仍作为兼容模式保留，旧 checkpoint/消融可能继续有 denoising-noise 依赖，必须在表格中单列。正式结果还应报告：50-slot 与实际执行 slot-0 的 attribution 口径、`max-events` 相对 exhaustive 的召回率、以及相同总训练预算的 clean-continued 对照。

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

TTT writer 有两个明确模式：`suffix` 是原始兼容路径，使用 action-expert
suffix hidden states 写 K/V；`prefix_only` 是 v2 论文路径，使用当前 causal
observation/language/state prefix 写 K/V，而 suffix 仍负责 query/read。prefix
与 expert 宽度不同之处由一个共享的无 bias learned projection 适配（MIKASA
默认 `960 -> 720`）；padding mask 在 inner update 前生效。两种模式的
fast-state 形状和 update-then-apply 时序相同，因此可以做公平的 writer-mode
ablation。

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
| prefix writer adapter（仅 v2） | 0.691M | `960 × 720`、无 bias；suffix 模式不实例化 |
| 整个 checkpoint（suffix / prefix） | 462.19M / 462.88M | 由当前 `model.safetensors` state dict 统计的约数 |

`ttt_only` 下 suffix 模式实际优化约 12.14M 个新增参数；v2 prefix 模式再加约
0.691M writer adapter（learned gate 仅在单独 ablation 打开），仍约占整个
checkpoint 的 2.8% 以下。参数量随 backbone/config 改变，论文表格应以保存的
`config.json` 和 state dict 重新统计；不要把“HD 关闭”误写成“没有 TTT 参数”。

## 4. TTT 的单步机制

### 4.0 数值稳健的内循环（v2 实现约束）

v2 保留 `ttt_base_inner_lr` 和每层可学习 multiplier 作为底层参数，但不让
它们把结果变成单点调参：`ttt_stable_inner_update=true` 时，inner gradient
先按每个 trajectory/fast-tensor 的 RMS 做无量纲相对化，learned multiplier
在固定对称范围内生效，并对非有限 candidate 做安全的 no-op 回退。该层还在
float32 中完成 inner arithmetic；外层仍可使用 bf16。这样参数改变的是记忆
写入强度，而不是数值尺度/动作单位，且同一机制可用于 clean teacher、HD student
和部署。这个稳定器是实现层的数值不变性，不是额外的 memory head，也不构成
论文的独立贡献；v2 recipe 强制打开它，legacy `false` 保留旧 checkpoint 的
逐位兼容路径。训练日志同时审计实际 inner-lr、gate、state/gradient 的有限性。

### 4.1 局部 K/V inner update

在选中的层，对该层 attention residual 的 token 表示 $h_{t,n}$ 做归一化并投影：

\[
k_{t,n}=K(\operatorname{LN}(h_{t,n})),\qquad
v_{t,n}=V(\operatorname{LN}(h_{t,n})),\qquad
q_{t,n}=Q(\operatorname{LN}(h_{t,n})).
\]

在兼容的 `suffix` 模式，$k,v,q$ 都来自 action-expert suffix；在论文的
`prefix_only` 模式，写入侧改为
$k,v=K/V(\operatorname{LN}(\operatorname{Proj}([R,P_t])))$，其中 $P_t$ 是当前
image/language/state prefix，$R$ 是不依赖当前噪声动作的 learned register
anchors，$q$ 仍来自 expert suffix 的 attention residual。因此 prefix writer
不读取当前 noisy action、flow noise 或 timestep；这是一项输入路径约束，不是把
action token 从模型中删除。

fast MLP $f_W$ 的本地目标为：

\[
\ell_{\mathrm{KV}}(W;t)=\frac12\left\|f_W(k_t)-\operatorname{sg}(v_t)\right\|_2^2.
\]

先做 inner gradient update，再用更新后的 state 读取 query：

\[
\widehat W_t=W_t-\eta_t\,s_t\,
\frac{\nabla_W\ell_{\mathrm{KV}}(W_t;t)}
{\max(\operatorname{RMS}(\nabla_W\ell_{\mathrm{KV}}),r_0)+\epsilon},
\qquad
W_{t+1}=W_t+g_t(\widehat W_t-W_t),
\]

其中 $r_0=\operatorname{RMS}(W_0)$，$s_t$ 是当前 state RMS 相对 $r_0$ 的
$[0.25,4]$ trust-region 截断；这是 `ttt_stable_inner_update=true` 时的实际
更新。上式中的 RMS、截断边界和有限值回退都是统一的数值实现细节，不是新的
任务相关超参数。关闭 stable mode 时保留原始的 $W_t-\eta_t\nabla_W\ell$ 路径，
便于 legacy/clean ablation 与旧 checkpoint 逐位兼容。

\[
\operatorname{TTT}(h_t)=h_t+\tanh(\gamma)\,f_{W_{t+1}}(q_t).
\]

这就是本项目的 **update-then-apply**：当前 interaction 先写入 fast weights，当前 token 的 residual read 使用更新后的 $W_{t+1}$。`g_t=0` 是严格的 zero-write intervention；$g_t\in(0,1)$ 是可微插值。

实现中有意区分两个表面上相同的 K/V loss：真正用于 inner update 的目标保留
当前可学习的 $v_t$，这样 v2 action-effect loss 可以沿着
$v\rightarrow\nabla_W\ell_{KV}\rightarrow W_{t+1}$ 形成二阶 meta-gradient；
对外暴露、作为 H2L 辅助项的 local loss 则使用
$\operatorname{sg}(v_t)$，避免 value projection 通过移动自己的 target 来
co-adapt。前者训练 writer 的 effect 路径，后者提供稳定的独立 local-writer
监督。

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

训练的普通 `hd_phase_mode="random"` 使用随机 flow interpolation；正式 HD-v2
recipe 使用 `hd_phase_mode="deployment"`，让写入 interaction 与部署一致：
$t=1$、输入是纯高斯 action noise，而不是 teacher-forced future action chunk。
外层 flow loss 仍然用专家 action 作为 target。`prefix_only` writer 的写入侧
只读取当前 prefix；`suffix` 兼容模式仍读取当前 noisy-action suffix，不能与
prefix-only 的噪声独立性结论混用。无论 writer 模式如何，gate context 都完全
不读取 action/noise/time。

## 5. 三个 HD-TTT 组件

### 5.1 Hindsight Control Attribution（HCA）

HCA 只在训练/离线 label 阶段运行。对一条完整成功 episode：

1. 用 clean SmolVLA-TTT teacher 从 episode 开头做 full-history causal replay，得到每个未来 frame 的 flow velocity 和 action loss $L_j$。
2. 将过去划分为长度 `event_block_size=B` 的事件 $E_i=[b_i,b_i+B)$。对每个事件做 zero-write replay：该 block 的写入 gate 设为 0，其余事件正常写入。
3. 只在事件结束之后统计未来，并忽略 episode 边界后的 frame/padded action slot。
   legacy 兼容协议使用

\[
C^{\mathrm{legacy}}_{i,j}=\left[L^{-i}_j-L_j\right]_+,
\qquad j\ge b_i+B.
\]

   论文 v2 对同一事件用 common-random-number 的 $z,-z$ 两次 replay，并先计算
   有符号、无量纲的相对退化：

\[
d_{i,j}=\frac{L^{-i}_j-L_j}
 {\tfrac12(|L^{-i}_j|+|L_j|)+\epsilon},\qquad
C_{i,j}=\max(d_{i,j},0),\quad H_{i,j}=\max(-d_{i,j},0).
\]

   两次 replay 的 $d$ 取平均；$C$ 和 harmful 部分 $H$ 分开聚合。事件/未来
   聚合使用 $\lceil\sqrt{n_{\rm valid}}\rceil$ 个 strongest entries 的均值，
   再用 episode 内 90th-percentile（而非单个 maximum）做稳健归一化。

这里的信用是“删除事件 $i$ 后，未来动作预测变差了多少”，不是 RL action value，也不修改真实轨迹。当前 builder 输出：

- `hd_attribution`：v2 的稳健正信用（legacy 时保留原始 max 语义）；
- `hd_write_gate`：事件级稳健 $u_i$ 映射到 frame 的 gate target；
- `hd_rho`：与保存的单一 selected wrong-memory branch 对齐的 future dependency；
- `hd_signed_attribution` / `hd_harm_attribution`：有符号/有害诊断，便于审计而不
  把有害写入误当成正记忆；
- `hd_C`、事件区间、eligible counts、total credits 等审计信息（保存在 episode
  metadata/detail 中）。

默认 `max-events=0` 会 replay 所有 causal blocks；论文 recipe 可把
`max-events=8` 作为预注册的 compute budget，但必须把该值写入 label metadata
并报告相对 exhaustive（`0`）的事件召回率。未采样 block 的 `hd_write_gate` 安全
默认值为 1.0，但它们的 `hd_write_gate_observed=0`，不会被当成观测到的 gate target。

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

旧的 learned-gate/no-effect 消融会在最早的选中 TTT 层预测一个共享 gate；正式
v2 关闭该预测头（`hd_learned_write_gate=false`），由 all-write 的局部目标和
action-effect distillation 约束写入内容：

\[
g_t=\sigma\bigl(h_\phi(\operatorname{Pool}(P_t))\bigr),
\qquad
L_{\mathrm{gate}}=\operatorname{SmoothL1}(g_t,\operatorname{sg}(u_t)).
\]

`Pool(P_t)` 只包含当前 image/language/state prefix；它在 action suffix 嵌入前计算，因此不包含 noisy action、flow noise 或 denoising timestep。离线 `u_t` 只是训练 target，部署时由 gate 自己预测。H2L 的准确表述是“hindsight-credit-weighted local K/V objective”，不是直接 future-action distillation。

### 5.3 Action-effect distillation（仅 v2）

仅当 label metadata 的 `attribution_protocol` 为
`v2_relative_antithetic_robust` 时，builder 额外保存 selected event 的
slot-0 effect `hd_teacher_effect`。在线训练为每个物理 frame 运行两条带 writer
梯度的 replay（true write / selected-event zero-write），并令

\[
d_s=v^{\rm true}_{s,0}-v^{\rm wrong}_{s,0},\qquad
d_T=\operatorname{sg}(\texttt{hd\_teacher\_effect}_{0}).
\]

高依赖 frame 用 robust-scaled unit-beta Huber 匹配 $d_s$ 与 $d_T$，低依赖 frame
约束 $d_s\approx0$。teacher effect 的 median non-zero RMS 只用于无量纲缩放，
不会引入 task/action-unit 超参数；`hd_effect_weight` 仅控制该辅助项在总 loss
中的优化权重。为避免 TBPTT 分段改变尺度，训练器会在切 segment **之前**，从完整
physical window 的 slot-0、active action dimensions 计算一个 detached robust
median floor，并把同一个 floor 传给该 window 的所有 segment；只有直接调用
`action_effect_distillation_loss` 且未显式传 floor 时，才按当前调用 batch 估计（兼容
旧 API）。当前 v2 builder 只生成一个 branch，student 明确消费 event axis
的第 0 项。读取旧的 K>1 artifact 仍然兼容，但额外 branch 不参与主路径；若要研究
多事件，必须另行实现并报告独立 ablation。
该项直接让 writer 受到最终控制效果约束，但仍不是 latent fast-weight 的逐元素
可解释内容监督。

### 5.4 Causal Memory Deployment / counterfactual grounding

为防止模型学会“写了但不读”，训练时在同一 batch 维护 main 与 true/wrong
两条反事实数值 state：

- `main`：正常训练路径，可反向传播；
- `true` / `wrong`：selected event 的 all-write 与 zero-write replay。

v2 主路径直接在这两条 replay 上保留 writer-connected graph，匹配 slot-0 的
action effect；因此不再额外叠加旧的 detached reader-grounding pair（后者只保留
给 legacy/no-effect ablation）。所有 replay state 的数值部分都在 TBPTT segment
间携带，episode/window 结束才丢弃；跨 segment 的 outer meta-gradient 仍按标准
TBPTT 截断，H2L 的事件级局部 writer loss 负责在写入发生的 segment 提供信用。

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

legacy grounding 的 teacher velocity、wrong branch 和 intervention gate 都在
reader-only replay 中 detach；因此它主要训练 query/readout/action pathway。v2
effect replay 对 true/wrong writer 保留梯度，并在 segment 边界 detach 数值 state，
以控制图规模；这与 H2L 的局部写入信用共同构成部署闭环。

### 5.5 总训练目标

当前实现对 flow/HCA/effect 使用有效 action slot mask；H2L local-writer 则使用
`hd_writer_valid`（因此可以在 history warm-up frame 上训练）。v2 的 HD 项按整段
物理帧归一化，再与各 TBPTT segment 的 flow numerator 相加。总目标可写为：

\[
L = L_{\mathrm{flow}}
 +\lambda_{\mathrm{HCA}}L_{\mathrm{HCA}}
 +\lambda_{\mathrm{H2L}}L_{\mathrm{H2L}}
 +\lambda_{\mathrm{E}}L_{\mathrm{effect}}
 +\lambda_{\mathrm{G}}L_{\mathrm{ground}}
 +\lambda_{\mathrm{gate}}L_{\mathrm{gate}}.
\]

默认值为：

```text
hd_hca_weight=1.0
hd_h2l_weight=1.0
hd_effect_weight=0.0          # compatibility default; v2 paper path = 1.0
hd_grounding_weight=1.0
hd_invariance_weight=0.25
hd_write_gate_weight=1.0
hd_counterfactual_margin=0.0
```

上面的数值是兼容性默认值，不代表逐任务调参结果。论文 v2 recipe 在两个任务
共享同一组值，并显式设置 `hd_effect_weight=1.0`、`hd_grounding_weight=0`、
`hd_learned_write_gate=false`、`hd_attribution_protocol=v2_relative_antithetic_robust`
与 `ttt_writer_mode=prefix_only`；legacy/no-effect、learned-gate 和 detached
grounding 仅作为结构/训练路径对照。

历史 warm-up frame 会推进 recurrent state，但 action/HCA/grounding target 被 mask；它们仍可通过 `hd_writer_valid` 参与 local writer objective。terminal repeated/padded action slots 使用原始 slot-valid mask，避免 episode 尾部重复动作放大损失。

## 6. 训练和部署的严格闭环

```text
成功示范 episode
        │
        ├─ clean teacher full-history replay
        │       └─ event zero-write replay -> C, u, rho, true/wrong/effect labels
        │
        └─ frame-level HD label artifact（训练时使用）
                         │
当前 prefix + noisy action ──> SmolVLA expert ──> TTT update/read
                         │                         │
                         └─ flow + HCA + H2L + effect + gate + grounding
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
- teacher config SHA256、`teacher_ttt_writer_mode`；
- TTT layer indices、register 数；
- `event_block_size`、`max_events`、`grounding_min_future_frames`、threshold、phase mode；
- `attribution_protocol`（legacy 或 `v2_relative_antithetic_robust`）、
  `attribution_slot_mode`、`attribution_replays` 和 `effect_branches`；
- `history_mode="full_episode_replay"`；
- fixed `hd_noise` 和 `hd_time`（deployment phase 时 `hd_time=1`）。

v2 训练时 policy 的 `hd_attribution_protocol` 必须与 artifact 完全一致；
`ttt_writer_mode` 也必须与 clean teacher 一致。旧 artifact 若没有协议字段只能
按 legacy 解释，不能与 v2 shard 混 merge。推荐保留每个 label shard 的 metadata
和 merge audit trail，并在论文附录列出完整 JSON contract。

关键字段的对应关系如下（短名 `v2` 只用于 builder CLI，落盘时使用完整字符串）：

| metadata 字段 | v2 论文值 | 训练侧对应项 | 作用 |
| --- | --- | --- | --- |
| `format` | `hd_ttt_labels_v2` | — | label schema 版本 |
| `attribution_protocol` | `v2_relative_antithetic_robust` | `policy.hd_attribution_protocol` | 防止 legacy/v2 混用 |
| `attribution_slot_mode` | `slot0` | action-effect branch | 与实际执行的第一 slot 对齐 |
| `attribution_replays` | `2` | 固定算法常数 | common-random-number $z,-z$ |
| `effect_branches` | `1` | `hd_effect_weight`（主路径消费 branch 0） | selected event；旧 K>1 artifact 兼容读取但忽略额外 branch |
| `teacher_ttt_writer_mode` | `prefix_only` | `policy.ttt_writer_mode` | 保证 teacher/student 输入路径一致 |
| `history_mode` | `full_episode_replay`（frame）或 `bounded_window_replay`（window） | `ttt_history_warmup_length` | 防止截断历史被冒充 full history |

`hd_effect_weight=0` 是合法的 no-effect 对照，但仍应保留 v2 metadata；它表示
“同一 v2 attribution 下去掉 effect loss”，而不是把 artifact 当作 legacy。

### 7.3 当前正式 recipe 的超参数

除任务专属的 `sequence_length` 外，Color 和 Shuffle-Long 使用同一套模型/优化设置：

| 项目 | 值 |
| --- | --- |
| TTT layers | `[12, 13, 14, 15]` |
| TTT fast hidden dim | `1024` |
| fast inner learning rate | `0.1`（每层可学习 multiplier） |
| stable inner update（v2） | `true`（RMS-relative、bounded multiplier、finite fallback） |
| residual effective gate init | `0.05` |
| second-order TTT | `true`（v2 effect writer meta-gradient；clean/legacy 可用 `false`） |
| register tokens | `16` |
| writer mode（论文主路径） | `prefix_only` |
| action chunk / executed steps | `50` / `1` |
| denoising steps | `10` |
| training stage | `ttt_only`（VLM、expert 和 projection head 冻结） |
| optimizer | AdamW，lr `1e-4`，betas `(0.9, 0.95)`，weight decay `1e-10` |
| scheduler | warmup `1000`，decay `30000`，final lr `2.5e-6` |
| gradient clip | `10` |
| distributed precision | 4 GPU、`bf16`、batch size `1` |
| image resize | `[224, 224]`（padding 保持比例） |
| train / label / eval seed | `1000` / `1729` / `7000`（eval seed 可按实验登记） |

HD-v2 阶段在 clean prefix-writer checkpoint 上开启
`hd_ttt_enabled=true`、`hd_learned_write_gate=false`、
`hd_effect_weight=1.0`，并使用 `hd_attribution_protocol=v2_relative_antithetic_robust`、
`hd_phase_mode=deployment`、`event_block_size=4`、`max_events=8`、
`grounding_min_future_frames=64`、`attribution_threshold=0`。这些值构成当前两
任务共享的预注册 protocol；做 ablation 时必须把改动写入 checkpoint config 和
label metadata，不能只在命令行临时覆盖而不留 provenance。

`examples/mikasa/train_hd_ttt.sh` 也把这一条作为入口契约：当
`HD_ENABLED=true` 且调用者没有显式设置 `TTT_WRITER_MODE` 时，自动选择
`prefix_only`；clean/legacy 调用仍默认 `suffix`。显式设置始终优先，因此
`suffix` 只能作为登记过的兼容或结构消融，而不会因为漏写一个环境变量而被误报成
论文主路径。

### 7.4 参数使用与跨任务复现（辅助审计，不是算法贡献）

HD-TTT 并不是“去掉参数”的方法。`η`、residual gate、各辅助项的
`λ`、事件块长度和 replay budget 都是可解释的底层参数，当然可以影响 SR；算法层面的
贡献是因果事件删除、无量纲稳健信用和 writer/reader effect 对齐，而不是某个精确
小数值。下面的规则只是帮助复现和排除逐任务过拟合，不把“参数不敏感”本身宣称为
新的算法目标：

1. Color 与 Shuffle-Long 共享同一份 v2 config；不按 task 选择不同的
   `λ`、gate 初始化或事件阈值。只允许 `sequence_length` 服从 episode 长度
   的数据约束，不能用它调 SR。
2. 在主配置邻域对每个连续权重做一次独立的
   `{0.8×, 1.0×, 1.2×}` 扰动（至少覆盖 `ttt_base_inner_lr`、
   `ttt_effective_gate_init`、`hd_hca_weight`、`hd_h2l_weight`、
   `hd_effect_weight`、`hd_grounding_weight`、`hd_write_gate_weight`）。
   固定 dataset split、teacher、训练/评估 seed 和总 update budget；只有同时
   扰动 `event_block_size`、`max-events` 或其它 label-generating knob 时才
   重新生成 labels，并更新 metadata。不得从邻域结果挑选最佳值作为主结果。
3. 对每个扰动报告两任务的 mean±std SR/return、最差 SR 和相对 spread
   `(max-min)/|baseline|`。同时报告 `max-events=8` 相对 exhaustive 的事件
   召回率；这能区分 compute budget 变化与模型超参数敏感性。
4. `prefix_only`↔`suffix`、v2↔legacy、effect weight=0、register=0 是
   **结构性消融**，不应混在连续参数审计表里。所有结果使用同一个 checkpoint
   provenance/label protocol，并给出失败或不稳定的设置，而不是只展示成功项。

训练日志还记录 `hd_aux_to_flow_ratio` 与 `hd_aux_fraction`，以及选中 TTT 层的
`inner_lr`/`effective_gate` 最小值和最大值。这些字段是 detached stability
diagnostics，只用于审计不同设置下的损失尺度和内循环控制范围，不会自动重标定
loss，也不会引入新的训练旋钮；v2 的 TBPTT 汇总在整段序列的分子相加后再计算
ratio，因此不会随 segment length 改变定义。

若邻域内 SR 有平滑的小幅变化，说明结果依赖算法机制而非单点调参；若某一
参数出现陡峭跃迁，应在正文中如实标注并增加更细的局部扫描，不能用默认值
“写死”来隐藏敏感性。

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
  TTT_WRITER_MODE=prefix_only \
  HD_ENABLED=false HD_LEARNED_GATE=false HD_PHASE_MODE=random \
  HD_ATTRIBUTION_PROTOCOL=legacy HD_EFFECT_WEIGHT=0.0 \
SAVE_FREQ=500 LOG_FREQ=50 \
bash examples/mikasa/train_hd_ttt.sh
```

Shuffle-Long 只需替换 dataset/replay 参数：

```bash
DATASET_REPO_ID=shell_game_shuffle_color_lamp_touch_long_vla_v0 \
DATASET_ROOT=/workspace/data_mikasa_robo/data_lerobot/shell_game_shuffle_color_lamp_touch_long_vla_v0 \
OUTPUT_DIR=/workspace/experiments/mikasa_shuffle_clean150_full \
SEQUENCE_LENGTH=513 SEQUENCE_STRIDE=513 MAX_WINDOWS_PER_EPISODE=1 \
TBPTT_SEGMENT_LENGTH=16 HISTORY_WARMUP_LENGTH=full TTT_WRITER_MODE=prefix_only \
HD_ATTRIBUTION_PROTOCOL=legacy HD_EFFECT_WEIGHT=0.0 \
HD_ENABLED=false HD_LEARNED_GATE=false HD_PHASE_MODE=random \
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
  --attribution-protocol v2 \
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

从 clean teacher checkpoint 开启新 run，不要 resume clean run 的 optimizer state（HD
effect 路径会新增训练状态）：

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
TTT_WRITER_MODE=prefix_only HD_ATTRIBUTION_PROTOCOL=v2 HD_EFFECT_WEIGHT=1.0 \
HD_ENABLED=true HD_LEARNED_GATE=false HD_GROUNDING_WEIGHT=0 \
HD_PHASE_MODE=deployment TTT_SECOND_ORDER=true \
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
| HD-TTT (v2, paper) | `HD_ENABLED=true`, `HD_LEARNED_GATE=false`, `HD_GROUNDING_WEIGHT=0`, `HD_ATTRIBUTION_PROTOCOL=v2`, `HD_EFFECT_WEIGHT=1`, `TTT_WRITER_MODE=prefix_only`, `TTT_SECOND_ORDER=true` | 完整方法：robust HCA + local writer + writer-connected action effect |
| No action-effect | 同上但 `HD_EFFECT_WEIGHT=0` | 去除 writer/content-effect 对齐，保留其它 HD 项 |
| Legacy HD | `HD_ATTRIBUTION_PROTOCOL=legacy`, `HD_EFFECT_WEIGHT=0` | 复现旧 raw-hinge/max 标签协议 |
| Suffix writer | v2 配置但 `TTT_WRITER_MODE=suffix` | 测试 prefix-only 输入路径的结构贡献 |
| Direct label-gated training | `HD_LEARNED_GATE=false` | 训练期直接用 label gate 的旧/对照路径，不是部署方案 |
| Reset memory | `--reset-memory-every-step` | 验证长程 memory 是否被真正使用 |
| Exhaustive attribution | `HD_MAX_EVENTS=0`（labels 与 training 同步） | 去除事件采样 compute budget |
| No grounding | `HD_GROUNDING_WEIGHT=0` | 测试 reader grounding 的作用 |

所有 paired memory 比较应固定相同 checkpoint、episode seeds 和 `--torch-seed`；否则 flow noise 差异会混入 memory 差异。结构消融与 7.4 的连续参数邻域实验分开汇报，并保持相同总 update/compute budget。

## 10. 当前验证状态

代码层面测试覆盖 TTT inner update、state detach/carry、RoPE position、register
mask、prefix/suffix writer、full-history/window contract、v2 attribution/effect
labels、label provenance、grounding branch 和 MIKASA adapter。提交前应在当前
环境运行 `uv run pytest tests -q`（或项目指定 Python 环境）并把实际的
`N passed` 写入实验记录；README 不固定一个会随 v2 测试增长而过期的数字。已
完成的本地 Color one-step smoke 只证明数值路径可运行，不是 benchmark success
rate。

正式的 Color/Shuffle 150-epoch 训练和官方 50-episode 分数应从远端实验目录的最终 checkpoint 与 JSON 读取。README 不预填未经最终 protocol 验证的旧分数；报告时至少同时给出：

- persistent-memory SR/return；
- reset-memory SR/return；
- clean TTT、no-register、native SmolVLA 对照；
- label selection mode、event budget、teacher SHA 和完整训练配置。

## V2 当前问题与验证状态

这一节专门区分“算法尚未证明的边界”和“已经修复的工程问题”，避免把一次运行
失败误判成 V2 数学本身失效。

### 已实现的 V2 闭环

```text
clean prefix-only teacher
  → full/bounded causal replay
  → robust z/-z event attribution (HCA)
  → credit-weighted local K/V objective (H2L)
  → writer-connected true/wrong slot-0 effect loss
  → deployment-time causal fast-weight memory
```

单个物理时刻的实际顺序是：当前 observation prefix 写入一次 fast state；第一个
denoising step 先 update 再 apply；后九个 denoising step 只读取同一个 state；执行
slot 0 后进入下一物理时刻。部署侧没有 teacher、未来帧或专家动作。V2 主路径使用
`prefix_only` writer、16 个 register、最后四个 action-expert 层和 `effect_branches=1`。

### 当前已知问题

| 状态 | 问题 | 解释和处理 |
| --- | --- | --- |
| 已修复（`58db6d4`） | DDP 各 rank 的 episode 长度不同，TBPTT segment 数不同 | 原先 rank 在不同 collective 位置调用 reduce，主进程会误报 “another distributed rank”。现在先同步最大 segment 数，短 rank 用 differentiable zero loss 参与 collective，且不推进 fast state。 |
| 已验证 | clean prefix one-step、V2 label smoke、writer-connected effect/second-order one-step | 这些只证明张量/梯度路径 finite，不代表 benchmark success rate。 |
| 已验证 | 修复后的 3-GPU fp32 与 4-GPU bf16 variable-length DDP 1-step | 不同 Color episode 长度可安全混合；修复后两种精度均通过，state RMS ratio 保持在约 `0.97–1.02`。 |
| 进行中 | 4-GPU bf16 长训练的数值和吞吐 | `MIXED_PRECISION=bf16` 是默认；长跑仍需观察，不应把 1-step 通过外推为 150-epoch 结论。若失败先保留 rank/local-batch 日志，再与 `MIXED_PRECISION=no` 对照，不能静默吞掉 finite marker。 |
| 协议边界 | Shuffle-Long 的 label teacher | Color teacher 不能用于 Shuffle 的科学标签；必须先在 Shuffle 自己的数据上训练 clean prefix teacher。现有单 episode Shuffle smoke 仅用于格式/计时检查。 |
| 计算边界 | Shuffle episode 长度 145–513，full replay 很慢 | 采用预注册的 bounded `L=64, stride=64, context=128, K=4`，并明确 `history_mode=bounded_window_replay`，不能冒充 full-history。 |
| 研究边界 | HCA 的监督对象 | 它是“删除 learned fast-weight event 后未来动作预测的退化”，不是任意历史 oracle 的真实 action value；论文应写成 control attribution under the clean teacher。 |
| 研究边界 | effect 的覆盖范围 | 当前只消费 selected event 的 slot 0 / branch 0；全 50-slot 或多事件 effect 尚未实现，不能在论文中暗示已覆盖。 |
| 研究边界 | effect 与完整 rollout 的差异 | label/effect 只在 deployment-matched 的首个 denoising phase（$t=1$、Gaussian input）监督已执行 slot 0；部署随后还有 9 个 read-only denoising reads。它是首步 velocity effect，不等于 10-step 积分后 action chunk 的完整 credit，需作为明确限制和 ablation。 |
| 研究边界 | harmful credit 的使用 | v2 会计算并保存 signed/harmful attribution 供审计，但主 all-write writer 不用它抑制写入；不能宣称已实现 harmful-write rejection 或 selective suppression。 |
| 研究边界 | prefix writer 的表征 | `prefix_only` writer 使用当前原始 VLM prefix embedding 经共享 adapter（加 static registers），不是经过完整 VLM 深层 contextualization 的 hidden state；这是当前计算/输入独立性折中，应在消融中与 suffix writer 区分。 |
| 复现限制 | teacher/data provenance | 当前 contract 记录 config SHA、dataset id/fps/index 和协议字段，但未对 teacher 权重内容或 dataset 原始内容做完整 cryptographic hash；正式实验需额外保存 checkpoint manifest 与数据版本摘要。 |
| 需监控 | 长 episode 的 state drift | stable update 约束每一步而非整个 episode；记录 `ttt_state_rms_ratio_*`。如果 Shuffle 越界，应先报告并做独立机制实验，不在主结果中临时加入 decay。 |
| 尚无结论 | 正式 Color/Shuffle V2 150-epoch 和官方 SR | 旧 v1 的 `.08/.12` 只属于旧协议，不能作为 V2 提升或失败的证据。 |

### 为什么 DDP 报错不是 V2 NaN

Color 的一个 DDP batch 可能同时包含 29-frame 和 15-frame window。旧 trainer 让前者
执行 4 个 TBPTT segment，而后者只执行 2 个；短 rank 随后进入参数梯度 reduce，长 rank
仍在执行 segment finite-guard reduce，collective 顺序因此错位。单卡各 episode 都是
finite，且修复后 3 卡 fp32 日志中所有 rank 的 segment guard 均为 `local_bad=False`。
这说明故障来自分布式控制流，而不是 HCA/H2L/effect 的损失公式。

### 参数如何使用而不变成逐任务调参

`inner_lr`、residual gate、各 loss weight、event block 和 replay budget 都是可解释
的底层旋钮；它们可以影响 SR，但不是论文贡献。主实验对 Color 与 Shuffle 使用同一
份 V2 配置和相同 update budget，只允许序列长度服从数据的容量约束。参数邻域实验
只作为审计（例如独立 ±20% 扰动），不能从邻域结果挑一个最优值再回填主结果。真正
需要单独成表的是结构性消融：no-register、suffix writer、no-effect、legacy HCA、
reset-memory 和 exhaustive attribution。

## 11. 文件地图

```text
src/lerobot/policies/smolvla_ttt/
├── configuration_smolvla_ttt.py   # policy/config、TTT/HD 开关和严格校验
├── modeling_smolvla_ttt.py        # SmolVLA flow、register、TTT hook、HD losses
├── smolvlm_with_expert_ttt.py     # VLM/action-expert 交替 self/cross attention
├── ttt.py                         # fast MLP、update-then-apply、RoPE、state
├── hd_ttt.py                      # HCA/H2L/grounding tensor primitives
├── history_teacher.py             # 可选的独立 causal history teacher 工具（非默认 builder）
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
- label 生成和 HD training 的 `event_block_size`、`max_events`、`grounding_min_future_frames`、phase mode、`attribution_protocol`、writer mode 必须完全一致；v2 还要核对 `attribution_slot_mode=slot0`、`attribution_replays=2` 和 `effect_branches=1`。旧的 K>1 effect artifact 可读取，但主路径始终只消费 branch 0。
- `build_hd_labels.py` / `build_hd_window_labels.py` 的 `--attribution-protocol v2` 要与训练时 `HD_ATTRIBUTION_PROTOCOL=v2`（配置中的完整字符串 `v2_relative_antithetic_robust`）对应；不要把 builder 的 v2 默认误认为 policy config 已自动切换。
- 不要用普通 `lerobot/smolvla_base` 直接生成 HCA labels；它没有训练好的 TTT/register fast weights，builder 会拒绝。
- v2 labels 应由与 student 相同 `ttt_writer_mode=prefix_only` 的 clean teacher 生成；suffix teacher 只能用于明确的 suffix ablation。
- 不要把 Color normalization 用到 Shuffle-Long，反之亦然。
- 不要把 `max_windows_per_episode=4` 的 bounded-window artifact 冒充 full-history artifact；正式 full-history 必须一 episode 一个完整 window。
- 不要在 eval episode 之间忘记 `policy.reset()`；否则前一条轨迹的 fast state 会污染后一条轨迹。
- `max-events>0` 时未采样事件的 gate=1 是安全 fallback，不是观测到的高信用；gate loss 必须使用 `hd_write_gate_observed`。
- `history_teacher.py` 是可选工具，不会自动替换 clean replay teacher；若实验启用它，必须额外记录 teacher format/state hash，并将其作为独立 teacher ablation。
- 当前实现不支持 `torch.compile` 和启用 RTC；两者都会与 inner autograd/state update 冲突。
- 远端 `/workspace` 可能是非持久卷；训练完成后立即把 checkpoint、labels、logs 和 eval JSON 复制到持久存储。

## 13. 参考

- MIKASA-Robo-VLA: <https://mikasarobo.github.io/>
- MIKASA benchmarking: <https://mikasarobo.github.io/benchmarking.html>
- MIKASA evaluation protocol: <https://mikasarobo.github.io/evaluation_protocol.html>
- SmolVLA/LeRobot 上游实现：<https://github.com/huggingface/lerobot>
- RoboTTT（fast weights / recurrent test-time state 的实现启发）及 Chronos（历史 latent state 的概念启发）是本实现的出发点；本文档中的 HCA、H2L 和 causal grounding 是本项目自己的组合与训练协议。
