# Method Declaration / 方法说明

UniMatch-V2 Surgical — semi-supervised surgical semantic segmentation with
HPTA/HPTA-MoE (visual adapter) + LC-PAM (text/affinity prior) +
MGER-guided edge routing (edge refiner as detached routing prior)
+ TCR (temporal consistency) + TS-MDR (text-morphology router) + an
instance-aware multi-class loss. All on by default; the earlier separate
image-level classification head has been cut (see the note above Section 8).
An optional **Rein in-backbone PEFT** (§2b) can replace or complement HPTA,
adapting the frozen DINOv2 *between* its layers instead of only post-hoc.

本文件给出各模块的数学定义、张量维度与损失聚合方式。公式与维度为中英共用，文字说明分中英两段。

---

## 0 · Notation / 符号表

| Symbol | Dimension | Meaning (EN) | 含义 (中) |
|---|---|---|---|
| $B$ | scalar | batch size | 批大小 |
| $H,\,W$ | scalar | image height / width (multiple of $p$) | 图像高/宽（$p$ 的整数倍）|
| $p$ | scalar | backbone patch size ($p=14$ for DINOv2) | 主干 patch 尺寸 |
| $H_p,W_p$ | scalar | patch grid, $H_p=H/p,\ W_p=W/p$ | patch 网格尺寸 |
| $N$ | scalar | number of patch tokens, $N=H_pW_p$ | patch token 数 |
| $C$ | scalar | number of segmentation classes (`nclass`) | 分割类别数 |
| $C'$ | scalar | number of non-background classes, $C'\le C$ | 非背景类别数 |
| $D$ | scalar | backbone embed dim (S/B/L/G = 384/768/1024/1536) | 主干特征维度 |
| $d$ | scalar | text-affinity latent dim (`d_latent`, def. 128) | 文本-亲和潜在维度 |
| $X$ | $\mathbb{R}^{B\times3\times H\times W}$ | input image (ImageNet-normalized) | 输入图像 |
| $Y$ | $\{0,\dots,C{-}1,255\}^{B\times H\times W}$ | GT label (255 = ignore) | 真值标签（255 忽略）|
| $P$ | $\mathbb{R}^{B\times N\times D}$ | last-layer DINOv2 patch tokens | 主干末层 patch token |
| $L_{\mathrm{dpt}}$ | $\mathbb{R}^{B\times C\times H\times W}$ | DPT decoder logits | DPT 解码 logits |
| $\tau_c$ | scalar | confidence threshold `conf_thresh` | 伪标签置信阈值 |
| $m$ | scalar | EMA decay | EMA 衰减系数 |

Subscripts: $\,_x$ labeled stream, $\,_u$ unlabeled stream, $\,_w$ weak view, $\,_{s1},\,_{s2}$ two strong views.
下标：$x$ 有标注流，$u$ 无标注流，$w$ 弱增广，$s1/s2$ 两条强增广视图。

---

## 1 · Backbone + DPT / 主干与解码头

**EN.** A frozen-or-tunable DINOv2 ViT encodes $X$ into 4 multi-scale features
$\{F_l\}_{l=1}^{4}$ and last-layer patch tokens $P$. The DPT head $g_{\mathrm{dpt}}$
fuses them into dense logits.

**中.** DINOv2 主干（可冻结/可微调）将 $X$ 编码为 4 个多尺度特征 $\{F_l\}$ 与末层
patch token $P$，DPT 头 $g_{\mathrm{dpt}}$ 融合得到稠密 logits。

$$
\{F_l\}_{l=1}^4,\,P=\mathrm{Enc}(X),\qquad
L_{\mathrm{dpt}}=g_{\mathrm{dpt}}(\{F_l\})\in\mathbb{R}^{B\times C\times H\times W}.
$$

---

## 2 · HPTA — HOM-lite Visual Adapter / 视觉适配器

**EN.** A lightweight adapter inserted on each of the 4 features (kept *outside*
the frozen backbone). For a token tensor $x\in\mathbb{R}^{B\times N\times D}$ with
bottleneck $h=\max(8,\lfloor D/r\rfloor)$ ($r$=`reduction`), it mixes a local
(depthwise-conv) branch and a global (MLP) branch with an input-aware gate, via a
zero-initialized residual ($\gamma{=}0$ at start → identity).

**中.** 在 4 个特征上各插入一个轻量适配器（位于冻结主干**之外**）。瓶颈维
$h=\max(8,\lfloor D/r\rfloor)$，融合“局部（深度卷积）”与“全局（MLP）”两支，
由输入相关门控加权，并用零初始化残差（$\gamma{=}0$ 起始为恒等）。

$$
z=W_\downarrow\,\mathrm{LN}(x)\in\mathbb{R}^{B\times N\times h},\quad
\hat z=\mathrm{reshape}(z)\in\mathbb{R}^{B\times h\times H_p\times W_p}
$$
$$
u_{\mathrm{loc}}=\mathrm{Conv}_{1\times1}\!\big(\mathrm{GELU}(\mathrm{DWConv}_{3\times3}(\hat z))\big),\qquad
\bar z=\tfrac1N\textstyle\sum_n z_n\in\mathbb{R}^{B\times h}
$$
$$
\alpha=\mathrm{softmax}(W_g\bar z)\in\mathbb{R}^{B\times2},\quad
u_{\mathrm{glob}}=\mathrm{Broadcast}_N(\mathrm{MLP}(\bar z))\in\mathbb{R}^{B\times N\times h}
$$
$$
u=\alpha_0\,u_{\mathrm{loc}}+\alpha_1\,u_{\mathrm{glob}}
$$
$$
x'=x+\gamma\,\mathrm{Drop}\big(W_\uparrow\,u\big)\in\mathbb{R}^{B\times N\times D}.
$$

---

## 2a · HPTA-MoE — Mixture-of-Experts Token Routing / 专家混合 token 加权

**EN.** An optional upgrade of §2: the 2-branch gate becomes a **4-expert
mixture with per-token soft routing**. Experts on the bottleneck $z$ are
$E_1$ local (DWConv), $E_2$ global (pooled MLP), $E_3$ low-rank ($h\!\to\!h/2\!\to\!h$),
$E_4$ identity. The router weights are computed **per token**, conditioned on the
edge and text modules when present: $g_{\mathrm{edge}}$ = per-patch MGER-refined
boundary prior when MGER is enabled (raw Sobel fallback otherwise),
$g_{\mathrm{txt}}$ = per-token LC-PAM latent from
`AffinitySideBranch.moe_text_condition` injected through `moe_text_cond_fn`
inside `DPT.forward`; both default to $0$ when the corresponding module or
condition dimension is off. Zero-init $\gamma$ ⇒ identity start.

**中.** §2 的可选升级：2 支门控扩展为**4 专家、逐 token soft 路由**。瓶颈 $z$ 上的
专家为 $E_1$ 局部(深度卷积)、$E_2$ 全局(池化 MLP)、$E_3$ 低秩($h\!\to\!h/2\!\to\!h$)、
$E_4$ 恒等。路由权重**逐 token** 计算，并在边缘/文本模块存在时与之配合：
$g_{\mathrm{edge}}$=逐 patch MGER 细化边界先验（MGER 关闭时退化为 raw Sobel），
$g_{\mathrm{txt}}$=逐 token LC-PAM 潜在信号（由 `AffinitySideBranch.moe_text_condition` 生成，并通过
`DPT.forward` 内的 `moe_text_cond_fn` 注入）；对应模块或条件维度关闭时取 $0$。
零初始化 $\gamma$ 起始为恒等。

$$
w=\mathrm{softmax}\!\big(W_r\,[\,z\,;\,g_{\mathrm{txt}}\,;\,g_{\mathrm{edge}}\,]\big)\in\mathbb{R}^{B\times N\times4},\qquad
u=\textstyle\sum_{k=1}^{4} w_{:,:,k}\,E_k(z)
$$
$$
x'=x+\gamma\,\mathrm{Drop}\big(W_\uparrow\,u\big)\in\mathbb{R}^{B\times N\times D}.
$$

---

## 2b · Rein — In-Backbone Token-Refinement PEFT / Rein 层间细化适配器

**EN.** HPTA (§2) refines only the four *extracted* features, i.e. it sits
*outside* the frozen backbone, so the ViT's internal attention/MLP never adapt
to the target domain. **Rein** (following Wei *et al.*, CVPR 2024, "Stronger,
Fewer, Superior") instead injects a lightweight token refinement **between**
frozen DINOv2 blocks, so the correction propagates through the subsequent frozen
layers while DINOv2 stays frozen — true in-backbone parameter-efficient
adaptation. For each adapted layer $l$, the block-output tokens
$x^{(l)}\in\mathbb{R}^{B\times T\times D}$ ($T=1+N$, CLS + patch tokens) attend to
a small learnable token dictionary $T_l\in\mathbb{R}^{M\times d_r}$, which is
generated **low-rank** from a basis $E\in\mathbb{R}^{r\times d_r}$ **shared across
all layers** (the "fewer" parameters). A zero-initialized scale $s_l$ makes the
module an identity map at the start (stable on a frozen VFM).

**中.** HPTA（§2）只细化 4 个**提取后**的特征，即位于冻结主干**之外**，ViT 内部
的注意力/MLP 始终不对目标域适应。**Rein**（参照 Wei 等，CVPR 2024《Stronger,
Fewer, Superior》）改为在冻结 DINOv2 的**层与层之间**注入轻量 token 细化，使修正
量沿后续冻结层传播，而 DINOv2 仍保持冻结——即真正的“主干内”参数高效适配。对每个
被适配层 $l$，块输出 token $x^{(l)}\in\mathbb{R}^{B\times T\times D}$（$T=1+N$，
CLS+patch）对一个小的可学习 token 字典 $T_l\in\mathbb{R}^{M\times d_r}$ 做注意力；
$T_l$ 由**跨层共享**的基底 $E\in\mathbb{R}^{r\times d_r}$ 低秩生成（即“更少参数”）。
零初始化尺度 $s_l$ 使模块起始为恒等（在冻结 VFM 上稳定）。

$$
T_l=A_l\,E\in\mathbb{R}^{M\times d_r},\qquad
A_l\in\mathbb{R}^{M\times r},\ \ E\in\mathbb{R}^{r\times d_r}\ \text{(shared)}
$$
$$
z=W_\downarrow\,\mathrm{LN}\big(x^{(l)}\big)\in\mathbb{R}^{B\times T\times d_r},\qquad
S=\mathrm{softmax}\!\Big(\tfrac{z\,T_l^\top}{\sqrt{d_r}}\Big)\in\mathbb{R}^{B\times T\times M}
$$
$$
\Delta^{(l)}=W_\uparrow\,\mathrm{GELU}\big(S\,T_l\big)\in\mathbb{R}^{B\times T\times D}
$$
$$
\boxed{\ x'^{(l)}=x^{(l)}+s_l\,\mathrm{Drop}\big(\Delta^{(l)}\big)\ },\qquad
s_l\!\leftarrow\!0\ \text{(identity start)} .
$$

The refined $x'^{(l)}$ is fed to the next (frozen) block. The DINOv2 weights stay
frozen; only $\{A_l,E,W_\downarrow,W_\uparrow,s_l,\mathrm{LN}\}$ train, and — being
named outside `backbone.*` — they join the $\eta\cdot\eta_{\mathrm{mult}}$
learning-rate group. For ViT-B ($D{=}768$) with $M{=}100,\ r{=}16,\ d_r{=}64$ and
all 12 layers adapted, Rein adds $\approx\!1.23\text{M}$ trainable parameters
($\ll$ the 86.6M frozen backbone). Rein may **replace** HPTA (in-backbone vs.
post-hoc adaptation) or **complement** it; it is enabled with `--rein`
(`REIN=1`), tokens/rank/dim via `--rein-tokens/--rein-rank/--rein-dim`.

**中（补充）.** 细化后的 $x'^{(l)}$ 送入下一冻结块。DINOv2 权重保持冻结，仅
$\{A_l,E,W_\downarrow,W_\uparrow,s_l,\mathrm{LN}\}$ 可训练；因命名不在 `backbone.*`
之下，它们进入 $\eta\cdot\eta_{\mathrm{mult}}$ 学习率组。ViT-B（$D{=}768$，
$M{=}100,r{=}16,d_r{=}64$，12 层全适配）下 Rein 仅增加约 **1.23M** 可训练参数
（远小于 86.6M 冻结主干）。Rein 可**替代** HPTA（主干内 vs. 事后适配），也可与其
**并用**；用 `--rein`（`REIN=1`）开启，token 数/秩/维度由
`--rein-tokens/--rein-rank/--rein-dim` 控制。

---

## 3 · LC-PAM — Language-Conditioned Patch-Affinity Module / 文本-亲和先验支路

**EN.** A text prior over patches. Patch tokens are projected to a latent space
and compared (cosine, temperature $\tau$) against per-class text anchors
$T_{\mathrm{anchor}}$ (SigLIP-2 prompts). The dense affinity is scattered into the
DPT class order and fused **per pixel** with a learned gate $G$. Background
channels are masked out by $m$. The teacher's weak-view logits are intentionally
**not** fused (clean pseudo-labels).

**中.** 基于文本的 patch 先验。patch token 投影到潜在空间后，与逐类文本锚点
$T_{\mathrm{anchor}}$（SigLIP-2 提示）做余弦相似（温度 $\tau$）。稠密亲和散射回
DPT 类序，并用可学习门控 $G$ 做**逐像素**融合，背景通道由 $m$ 屏蔽。教师弱视图
logits **不**参与融合（保持伪标签干净）。

$$
p_{\mathrm{lat}}=\mathrm{normalize}\big(\phi_v(P)\big)\in\mathbb{R}^{B\times N\times d},\quad
\phi_v:\ \mathrm{LN}\!\to\!\mathrm{Lin}_{D\to256}\!\to\!\mathrm{GELU}\!\to\!\mathrm{Drop}\!\to\!\mathrm{Lin}_{256\to d}
$$
$$
T=\mathrm{normalize}(T_{\mathrm{anchor}})\in\mathbb{R}^{C'\times d},\qquad
\tau=\mathrm{clip}(e^{\log\tau},\,10^{-3},\,1)
$$
$$
A=\frac{p_{\mathrm{lat}}\,T^\top}{\tau}\in\mathbb{R}^{B\times N\times C'},\qquad
R=\mathrm{reshape}(A)+b_{\mathrm{cls}}\in\mathbb{R}^{B\times C'\times H_p\times W_p}
$$
$$
\tilde R=\mathrm{Up}(R)\in\mathbb{R}^{B\times C'\times H\times W},\qquad
\Pi_{:,c}=\tilde R_{:,\pi(c)}\cdot m_c,\ \ m_c=\mathbb{1}[\pi(c)\ge0]
$$
$$
G=\sigma\big(\mathrm{Conv}_{1\times1}(\mathrm{reshape}(p_{\mathrm{lat}}))\big)\in\mathbb{R}^{B\times1\times H\times W}
$$
$$
\boxed{\ \tilde L=(1-G\odot m)\odot L_{\mathrm{dpt}}+(G\odot m)\odot\Pi\ }\in\mathbb{R}^{B\times C\times H\times W}
$$

Image-level multi-label logits (for the auxiliary BCE):
图像级多标签 logits（用于辅助 BCE）：
$$
s=\mathrm{Pool}(A)+b_{\mathrm{cls}}\in\mathbb{R}^{B\times C'}.
$$

> Note / 注：$\pi:\{0,\dots,C{-}1\}\to\{-1,0,\dots,C'{-}1\}$ maps DPT class id → affinity class id ($-1$ = background). The raw prior lives at $A\!\sim\!\pm1/\tau\!\approx\!\pm14$ vs DPT logits $\sim\!\pm5$–$10$; before the gated mix, $\Pi$ is **affinely rescaled per pixel over the non-bg channels to the DPT logits' mean/std** (`AffinitySideBranch.match_prior_scale`, parameter-free, persisted in the checkpoint so train/eval match). A single per-pixel scale+shift preserves the prior's argmax/ranking while matching its magnitude, so the convex combination is no longer dominated by the prior once the gate opens. / 融合前对先验做逐像素矩匹配到 DPT 量级（保序、无参、存入 ckpt）。

**Metric variants / 度量变体.** `cosine` (above), `hyperbolic` (Poincaré-ball
distance affinity), `hyperbolic_pathway` and `cma` (Cross-Modal Adapter) — the
latter two additionally emit a text-aware patch tensor $X_{\mathrm{aware}}=P+\gamma\,E_{\mathrm{fused}}$
($\gamma$ ReZero, init 0) that **replaces** $P$ and re-decodes the DPT head.

---

## 4 · MGER-guided Edge Routing / MGER 引导的边缘路由

**EN.** A Sobel RGB prior $E$ is purified by a small CNN (MGER) into a clean
semantic-boundary map $\tilde E$. In the main method, $\tilde E$ is converted to
patch tokens and used as a **detached routing prior** for HPTA-MoE, so boundary
patches can prefer local adaptation experts without adding a second edge-feature
gating branch. The older edge residual adapters for DINO/text features are kept
only as an ablation option.

**中.** Sobel RGB 边缘先验 $E$ 经轻量 CNN（MGER）提纯为干净的语义边界图 $\tilde E$，
主方法中将 $\tilde E$ 转成 patch token，作为 **detach 的路由先验** 输入 HPTA-MoE，
让边界 patch 更倾向局部适配专家；不再额外使用旧的 DINO/text edge residual gating
支路。旧 edge residual adapters 仅作为消融选项保留。

$$
E=\mathrm{clip}\!\Big(\frac{\lVert\nabla\,\mathrm{gray}(X)\rVert-q_{\mathrm{lo}}}{q_{\mathrm{hi}}-q_{\mathrm{lo}}},0,1\Big)\in[0,1]^{B\times1\times H\times W}
\quad(\text{Sobel}+\text{per-image quantiles})
$$
$$
\tilde E=\sigma\big(\mathrm{CNN}([X;E])\big)\in[0,1]^{B\times1\times H\times W},\qquad
e_{\mathrm{tok}}=\mathrm{flatten}\big(\mathrm{resize}(\tilde E,H_p{\times}W_p)\big)\in\mathbb{R}^{B\times N\times1}
$$

MGER edge routing signal:
MGER 边缘路由信号：
$$
g_{\mathrm{edge}}=\mathrm{sg}[e_{\mathrm{tok}}]\in\mathbb{R}^{B\times N\times1},
\qquad
w=\mathrm{softmax}\!\big(W_r[\,z;\,g_{\mathrm{txt}};\,g_{\mathrm{edge}}\,]\big).
$$

---

## 5 · TCR — Temporal/Two-stream Consistency / 一致性正则 (DA-VSN style)

**EN.** Reference $p_{kf}$ = EMA teacher's weak-view probability, CutMix-aligned to
the student's strong view; current $p_{cf}$ = student strong-view probability.
A per-pixel gate enables the L1 pull **only** where the teacher is more confident
(lower normalized entropy). No global threshold, no extra hyperparameter.

**中.** 参考 $p_{kf}$ 为 EMA 教师弱视图概率（按相同 CutMix 框与强视图对齐）；当前
$p_{cf}$ 为学生强视图概率。逐像素门控**仅**在教师更自信（归一化熵更低）处启用
L1 牵引，无需全局阈值与额外超参。

$$
\mathcal H(p)=-\frac{1}{\log_2 C}\sum_{c=1}^{C}p_c\log_2 p_c\in[0,1],\qquad
g=\mathbb{1}\!\big[\mathcal H(p_{kf})<\mathcal H(p_{cf})\big]
$$
$$
L_{\mathrm{tcr}}=\frac{\sum g\,\lVert p_{cf}-p_{kf}\rVert_1}{\sum g},\qquad
p_{kf}=\mathrm{sg}\big[\mathrm{CutMix}(p^{w}_{\mathrm{ema}})\big]
$$
where $\mathrm{sg}[\cdot]$ is stop-gradient. $p_{cf},p_{kf}\in\mathbb{R}^{B\times C\times H\times W}$, $g\in\{0,1\}^{B\times H\times W}$.

---

## 6 · TS-MDR — Text-Semantic Morphology Dynamic Router / 文本-语义形态动态路由

**Part of the proposed method (default ON, semi-supervised trainer only; `--no-tsmdr` for ablation).**
/ **方法的一部分（半监督训练器默认开启；消融时用 `--no-tsmdr`）。**

**EN.** A text/shape morphology prior $s_c$ and a per-class morphology descriptor
$q_t$ drive an MLP router that emits per-class temporal/edge weights
$\lambda_{\mathrm{temp}},\lambda_{\mathrm{edge}}\in\mathbb{R}^{C}$, used to weight a
local directional consistency between $p_{cf}$ and the CutMix-aligned teacher prob.
An entropy regularizer $R_{\mathrm{ent}}$ keeps the routing distribution from collapsing.

**中.** 文本/形状形态先验 $s_c$ 与逐类形态描述 $q_t$ 驱动一个 MLP 路由器，输出逐类
时序/边缘权重 $\lambda_{\mathrm{temp}},\lambda_{\mathrm{edge}}\in\mathbb{R}^{C}$，用于加权
$p_{cf}$ 与对齐后教师概率之间的局部方向一致性；熵正则 $R_{\mathrm{ent}}$ 防止路由分布坍塌。

> Implementation note / 实现备注: `util/temporal_tsmdr.py` also ships a
> parameter-free `ShapeMorphologyRouter` that derives the same soft routing
> $\pi_t^c$ from mask geometry alone (soft sigmoid indicators on area/aspect,
> no text encoder), as a drop-in alternative to `TextSemanticMorphologyRouter`
> for ablating away the text-prior contribution. Not used in the default
> (text-driven) configuration. / 同文件另提供纯几何版路由器
> `ShapeMorphologyRouter`（无文本编码器，仅用面积/长宽比软指示函数），可作为
> 消融替代，默认配置不启用。

---

## 6b · Instance-Aware Multi-Class Loss / 实例感知多类损失

**Part of the proposed method (default ON, labeled branch + unlabeled branch;
`--no-instance-loss` / `--no-instance-loss-unlabeled` for ablation).**
/ **方法的一部分（有标注与无标注分支均默认开启；消融时用
`--no-instance-loss` / `--no-instance-loss-unlabeled`）。**

**EN.** An additive auxiliary segmentation loss, adapted from Kundu et al.
2026 (*Instance Awareness of Multi-class Semantic Segmentation Loss
Functions*) to our 2D setting. Motivation: `CombinedCETversky` (Section 8)
already reweights *classes* via per-class Tversky, but is still a single
pixel-level average over the whole frame — one large instance (e.g. the
gallbladder) and five tiny instances (e.g. clips) of a rare class contribute
gradient in proportion to their pixel count, not their instance count. The
paper's fix — one-vs-rest class decomposition + uniform averaging over
(class, instance) — repurposes an instance-imbalance loss into also fixing
class imbalance, since every instance of every class ends up with equal
weight regardless of size or class frequency.

**中.** 一个加性辅助分割损失（方法的一部分，默认开启），改编自 Kundu 等 2026 年的工作（*Instance
Awareness of Multi-class Semantic Segmentation Loss Functions*），适配到本项目
的 2D 场景。动机：`CombinedCETversky`（第 8 节）已经通过逐类 Tversky 重新加权
不同**类别**，但仍是整幅图像上的单一像素级平均——同一稀有类别里，一个大实例
（如胆囊）和五个微小实例（如钛夹）按像素数量而非实例数量贡献梯度。该论文的
解法——one-vs-rest 类别分解 + 对 (类别, 实例) 做均匀平均——把一个原本解决
"实例不平衡"的损失顺带也解决了"类别不平衡"，因为每个类别的每个实例最终获得
相同权重，与其大小或类别频率无关。

Implementation (`util/instance_aware_loss.py`): for each foreground class
$c$ (background excluded via the same non-background class list used
elsewhere, e.g. LC-PAM's `non_bg_mask` / the dataset's `--exclude-classes`),
extract the binary map $y_c$ and 2D-connected-component it
(`scipy.ndimage.label`, 8-connectivity) into instances
$\{S_1^c,\dots,S_{K_c}^c\}$. Each instance gets a **Blob-loss-style domain**
(the full frame minus *other* instances of the *same* class — background and
other classes stay visible, so false positives elsewhere are still
penalized), and a binary CE+Tversky loss restricted to that domain:
$$
\Omega_k^c=\Omega\setminus\bigcup_{j\neq k}S_j^c,\qquad
\ell_k^c=\mathrm{BCE}\!+\!\mathrm{Tversky}\big(\hat y_c|_{\Omega_k^c},\,y_c|_{\Omega_k^c}\big)
$$
$$
L_{\mathrm{inst}}=\frac1{|\mathcal C|}\sum_{c\in\mathcal C}\frac1{K_c}\sum_{k=1}^{K_c}\ell_k^c,
\qquad \mathcal C=\{c:K_c>0\}.
$$
This differs from the paper's CC loss (3D Voronoi tessellation) by using the
cheaper Blob-loss domain in 2D, and is combined **additively** with the
existing per-pixel loss rather than replacing it (Eq. 6 of the paper is
literally $L=\alpha L_{\mathrm{global}}+\beta L_{\mathrm{instance}}$; here
$L_{\mathrm{global}}$ stays `CombinedCETversky`/CE unchanged and
$w_{\mathrm{inst}}$ plays the role of $\beta$).

**Labeled vs. unlabeled / 有标注与无标注.** On the labeled branch this runs on
clean GT, active from epoch 0. On unlabeled strong views (semi-supervised
trainer only, on by default: `--no-instance-loss-unlabeled` restricts it to
the labeled branch for ablation), it runs on the
**same confidence-gated hard pseudo-label** already used for $L_u$
($\rho\ge\tau_c$ pixels only — low-confidence pixels are treated as ignore
*before* connected-component labeling, so unreliable regions cannot spawn
spurious "instances"), plus a minimum-component-size filter
(`--instance-loss-min-size`, default 16 px, drops noise blobs) and a warmup
delay (`--instance-loss-warmup`, default 5 epochs) before it switches on,
since pseudo-labels are noisiest early in training. / 有标注分支使用干净真值，
从第 0 轮起生效；无标注强视图（仅半监督训练器，默认开启，消融时用
`--no-instance-loss-unlabeled` 关闭）复用 $L_u$ 已有的置信度门控硬伪标签
（仅 $\rho\ge\tau_c$ 像素参与连通域标注，其余视为忽略），并额外做最小连通域
尺寸过滤与 warmup 延迟，避免伪标签噪声被当作"实例"放大。

Weight `--instance-loss-weight` (default $w_{\mathrm{inst}}=0.2$) is added to
Section 8's total loss as $w_{\mathrm{inst}}L_{\mathrm{inst}}$. On by default
alongside HPTA/LC-PAM/MGER-guided routing/TCR (and TS-MDR, Section 6); `--no-instance-loss`
lets it be individually ablated the same way `--no-visual-adapter` /
`--no-lcpam` / edge-routing flags isolate the other modules. / 与 HPTA/LC-PAM/MGER 路由/TCR
（以及第 6 节 TS-MDR）同为默认开启项；可用 `--no-instance-loss` 单独消融。

---

> **Cut from the method / 已从方法中移除**: an earlier draft carried a
> separate image-level classification head
> ($\hat y=\mathrm{ClsHead}(P\ \text{or}\ \mathrm{cls\_token})$,
> $L_{\mathrm{cls}}=\mathrm{ASL/BCE}(\hat y,y_{\mathrm{multi}})$). It was
> redundant with LC-PAM's own image-level aux BCE (Section 3, $s=\mathrm{Pool}(A)+b_{\mathrm{cls}}$,
> same $y_{\mathrm{multi}}$ target) and is **removed from the proposed
> method**. The underlying code (`util/cls_head.py`, `--cls-head`) is
> untouched for backward-compat with existing checkpoints/tools but is
> `enabled: false` in every shipped config and not part of `L` below. /
> 早期草稿中存在一个独立的图像级分类头，与 LC-PAM 自带的图像级辅助 BCE
> （第 3 节）功能重复，**已从提出的方法中移除**。底层代码
> （`util/cls_head.py`、`--cls-head`）为兼容旧 checkpoint/工具而保留，但所有
> 配置文件均为 `enabled: false`，不参与下方总损失。

---

## 8 · Loss aggregation / 总损失

**EN.** Supervised CE on the fused labeled logits; confidence-gated CE on the two
strong views; plus the weighted auxiliary terms.

**中.** 融合后有标注 logits 上的监督 CE；两条强视图上的置信度门控 CE；再加各加权辅助项。

$$
L_x=\mathrm{CE}(\tilde L_x,\,Y_x)
$$
$$
L_u=\frac12\sum_{v\in\{s1,s2\}}\frac{\sum\big(\mathbb{1}[\max_c p^{w}_{c}\ge\tau_c]\wedge\mathbb{1}[\text{valid}]\big)\,\mathrm{CE}(\tilde L_u^{v},\hat Y^{v})}{\sum\mathbb{1}[\text{valid}]}
$$
$$
L_{\mathrm{base}}=\tfrac12\,(L_x+L_u)
$$
$$
\boxed{\,L=L_{\mathrm{base}}
+\tfrac12\lambda_{\mathrm{tcr}}L_{\mathrm{tcr}}
+\tfrac12\lambda_{\mathrm{tsmdr}}L_{\mathrm{tsmdr}}
+\lambda_{\mathrm{route}}R_{\mathrm{ent}}
+w_{\mathrm{aff}}L_{\mathrm{aff}}
+w_rL_{\mathrm{mger}}
+w_{\mathrm{inst}}L_{\mathrm{inst}}\,}
$$

Boundary / edge terms / 边缘项: in the main method, the edge loss is only
$L_{\mathrm{mger}}$, a BCE($+$Dice) supervision between the MGER refined edge
map and a GT-mask-derived semantic boundary. The refined edge map is then used
as a detached HPTA-MoE routing prior. The older seg-side auxiliary boundary head
$L_{\mathrm{bnd}}$ and RGB-edge consistency $L_{\mathrm{econs}}$ remain available
only when the legacy `--edge-residual-adapters` ablation is explicitly enabled.
$L_{\mathrm{inst}}$ is the Section 6b instance-aware loss — **part of the
proposed method, on by default**. The image-level cls-head term
($\alpha L_{\mathrm{cls}}$) is cut (see note above Section 8) and no longer
appears in $L$. / 主方法的边缘损失仅保留 $L_{\mathrm{mger}}$：用 mask 生成的语义边界
监督 MGER refined edge，并将其作为 detach 的 HPTA-MoE 路由先验。旧的分割侧辅助
边界头 $L_{\mathrm{bnd}}$ 与 RGB-edge consistency $L_{\mathrm{econs}}$ 仅在显式开启
`--edge-residual-adapters` 的 legacy 消融中使用。

Default weights / 默认权重 (argparse defaults): $\tfrac12$ on $L_x,L_u$;
$\lambda_{\mathrm{tcr}}{=}0.1$ (`--temporal-weight`, requires
`--temporal-consistency`), $\lambda_{\mathrm{tsmdr}}{=}0.1$ (`--tsmdr-weight`;
**TS-MDR is on by default**, `--no-tsmdr` for ablation),
$w_{\mathrm{aff}}{=}0.5$ (`--affinity-aux-weight`, cosine warmup over
`--affinity-aux-warmup`), $w_r{=}0.1$ (`--edge-refiner-weight`),
$w_{\mathrm{inst}}{=}0.2$ (`--instance-loss-weight`; **on by default for both
the labeled and unlabeled branches**, `--no-instance-loss` /
`--no-instance-loss-unlabeled` for ablation; unlabeled branch still gated behind
`--instance-loss-warmup`=5 epochs). Each term has its own warm-up epoch. /
旧 edge residual adapters 的 $w_b,w_c$ 只作为消融超参保留，不属于主方法。

---

## 9 · EMA teacher / 指数滑动平均教师

$$
m_t=\min\!\Big(1-\frac{1}{t+1},\,m_{\max}\Big)\ (m_{\max}=0.996),\qquad
\theta_{\mathrm{ema}}\leftarrow m_t\,\theta_{\mathrm{ema}}+(1-m_t)\,\theta .
$$
Buffers EMA'd if floating-point, else copied. LC-PAM keeps a parallel EMA copy
for fair `--use-ema` evaluation. / 浮点 buffer 做 EMA，整型直接拷贝；LC-PAM 维护
并行 EMA 副本以保证 `--use-ema` 测试一致。

---

## 9b · Datasets / 数据集

Four surgical segmentation datasets. **EndoVis2018** uses the **official
train/val/test split**; **EndoVis2017** uses **TernausNet 4-fold cross-validation**
by whole video sequence (val folds `{1,3}{2,5}{4,8}{6,7}`, per-class IoU averaged
across folds, no public test set). / 共 4 个数据集；2018 官方划分，2017 用 4 折交叉验证。

| Dataset | `nclass` | Split / 划分 |
|---|---|---|
| `endoscapes_seg50` | 7  | processed train/val/test |
| `cholecseg8k`      | 13 | no official test → `test.txt` = `val.txt` |
| `endovis2018`      | 12 | **official** train/val/test |
| `endovis2017_type` | 8  | **4-fold CV** (`make_seq_splits.py`), report val |

---

## 10 · Evaluation metrics / 评测指标

All derived from the confusion matrix $\mathrm{CM}\in\mathbb{N}^{C\times C}$ over valid
pixels ($Y\neq255$). For class $c$: $\mathrm{TP}_c=\mathrm{CM}_{cc}$,
$\mathrm{FP}_c=\sum_i\mathrm{CM}_{ic}-\mathrm{TP}_c$,
$\mathrm{FN}_c=\sum_j\mathrm{CM}_{cj}-\mathrm{TP}_c$,
$\mathrm{GT}_c=\sum_j\mathrm{CM}_{cj}$.

均由有效像素（$Y\neq255$）的混淆矩阵导出。

$$
\mathrm{IoU}_c=\frac{\mathrm{TP}_c}{\mathrm{TP}_c+\mathrm{FP}_c+\mathrm{FN}_c},\qquad
\mathrm{Dice}_c=\frac{2\mathrm{TP}_c}{2\mathrm{TP}_c+\mathrm{FP}_c+\mathrm{FN}_c}
$$
$$
\textbf{PixAcc}_c=\frac{\mathrm{TP}_c}{\mathrm{GT}_c}\ (\text{= per-class pixel accuracy = recall}),\qquad
\textbf{PixAcc}=\frac{\sum_c\mathrm{TP}_c}{\sum_c\mathrm{GT}_c}\ (\text{global})
$$
$$
\mathrm{mIoU}=\frac1{|\mathcal K|}\sum_{c\in\mathcal K}\mathrm{IoU}_c,\quad
\mathrm{mAcc}=\frac1{|\mathcal K|}\sum_{c\in\mathcal K}\mathrm{PixAcc}_c,\quad
\mathrm{fwIoU}=\sum_c\frac{\mathrm{GT}_c}{\sum_{c'}\mathrm{GT}_{c'}}\,\mathrm{IoU}_c
$$
where $\mathcal K$ = kept (non-excluded) classes; "present-only" variants further
restrict to $\mathrm{GT}_c>0$. / $\mathcal K$ 为保留类别；"present-only" 进一步限制为 $\mathrm{GT}_c>0$。

**Progress bar / 训练进度条.** Each training epoch shows a rank-0 `tqdm` bar
(`Epoch e/E`, iteration count, it/s, ETA) with a live postfix of running averages:
`loss`, `x` ($L_x$), `s` ($L_u$), `mask` (pseudo-label keep-ratio), `lr`. Shown on
rank 0 only (no duplicate bars under DDP), throttled to ~1 Hz so `nohup`/redirected
logs stay readable, and it falls back to the plain iterator if `tqdm` is absent.
训练阶段在 rank 0 显示 `tqdm` 进度条（`Epoch e/E`、迭代数、it/s、ETA），实时刷新运行均值
`loss / x / s / mask / lr`；仅在 rank 0 显示（DDP 下不重复），约 1Hz 刷新以保证 `nohup`/重定向
日志可读；未安装 `tqdm` 时回退为普通迭代。

**Per-epoch reporting / 每轮上报.** Each evaluated epoch reports, for both the raw
and EMA models, the key metrics $\mathrm{mIoU}/\mathrm{mDice}/\mathrm{PixAcc}$ (plus
$\mathrm{mAcc}$) and a per-class $\mathrm{IoU}/\mathrm{Dice}/\mathrm{PixAcc}$ table on
**val** and a held-out **test** split (auto-detected `test.txt`; reused from val when
absent). Cost control: `EVAL_BS` batches evaluation (uniform-resolution sets);
the expensive test split is evaluated only on a new best-val epoch / the final epoch
/ every `TEST_INTERVAL` epochs; `EVAL_INTERVAL` sets the val cadence.
训练每轮对原始与 EMA 模型上报关键指标 $\mathrm{mIoU}/\mathrm{mDice}/\mathrm{PixAcc}$
（及 $\mathrm{mAcc}$）与逐类 $\mathrm{IoU}/\mathrm{Dice}/\mathrm{PixAcc}$，覆盖 **val** 与
留出 **test**（自动识别 `test.txt`，缺失则复用 val）。

**All-class vs present-only / 全类 vs present-only.** Metrics are computed under
`no_grad` and **never enter the loss**. Every trainer (semi-sup `unimatch_v2.py`
and the fully-supervised `train_supervised_*`) now prints **present-only** means
(absent classes — GT support 0 — dropped, marked `[absent]`/`[excl]`) alongside
all-class, and selects `best.pth` (raw and EMA) by **present-only val mIoU**, matching
the `test.py` headline (`miou_present`). `--exclude-classes` drops background
(auto `[0]` for EndoVis2017 → 7-instrument number, none elsewhere). Only the per-class
CSV/`mIoU` columns stay all-class for backward compatibility (present-only added as
extra `*_present` columns). Dropping absent classes is a constant offset (does not
change best-epoch); excluding background can. / 半监督与全监督训练期均上报 present-only
并据此选 best；指标不参与损失。

---

## 11 · Training efficiency / 训练效率

**EN.** Speed-relevant implementation choices and runtime knobs. None change the
model definition; the eval-resolution cap only affects the *monitoring* metrics
during training (the final `test.py` evaluation is always full-resolution).

**中.** 与速度相关的实现选择与运行开关。均不改变模型定义；评测分辨率上限仅影响
训练过程中的*监控*指标（最终 `test.py` 评测始终为全分辨率）。

| Knob / 开关 | Effect (EN) | 作用 (中) |
|---|---|---|
| frozen backbone (HPTA/PEFT) | only adapter+head+side-branches get gradients ($\sim$12M trainable) | 仅适配器+头+支路回传梯度 |
| fused attention (`F.scaled_dot_product_attention`) | replaces manual $O(N^2)$ softmax in DINOv2; lower mem, faster, no xFormers | 替代手写 $O(N^2)$ softmax，省显存更快 |
| TF32 (`allow_tf32`) | Ampere matmul/conv speedup; no-op on Volta | Ampere 上加速，Volta 无效 |
| SyncBN skipped when 1 GPU | avoids an `all_reduce` per BN forward (DPT head has BN) | 单卡时省去每个 BN 的 all_reduce |
| `persistent_workers`, `prefetch_factor`, `NUM_WORKERS` | hide CutMix/aug IO latency | 隐藏数据增强 IO 延迟 |
| `EVAL_SIZE` $S$ | cap eval long-side to $S$ px → $N'\!\approx(S/p)^2$ tokens, attention $\propto N'^2$ | 限制评测长边，注意力 $\propto N'^2$ |
| `EVAL_EMA_ONLY` | evaluate only the EMA model; select $\mathrm{best.pth}$ by EMA mIoU; halves val cost | 仅评测 EMA 模型，按 EMA mIoU 选最优，val 减半 |
| `EVAL_INTERVAL` $K$ | run val every $K$ epochs (final always) | 每 $K$ 轮评测一次（末轮必评）|
| `TEST_INTERVAL` $T$ | $-1$ never · $0$ final-only (default) · $>0$ every $T$ epochs + final | $-1$ 不测 · $0$ 仅末轮(默认) · $>0$ 每 $T$ 轮+末轮 |

Eval-resolution cost model / 评测成本模型: with patch size $p$ and cap $S$, the per-frame
attention cost scales as $\big((S/p)^2\big)^2=(S/p)^4$. E.g. endovis $1280{\times}1024$
uncapped ($N\!\approx\!6800$) vs $S{=}686$ ($N'\!\approx\!1900$) ⇒ attention $\sim$13× cheaper.

**Mixed precision (optional).** `torch.autocast` around the model forward with
$\mathrm{bf16}$ (Ampere, no `GradScaler`) or $\mathrm{fp16}+\text{GradScaler}$ (Volta),
keeping the entropy/Dice/BCE losses in $\mathrm{fp32}$, gives a further $\sim$1.5–2.5×.
混合精度（可选）：前向用 $\mathrm{bf16}$（Ampere，无需 GradScaler）或 $\mathrm{fp16}+$GradScaler（Volta），
损失项保持 $\mathrm{fp32}$，可再加速约 1.5–2.5×。
