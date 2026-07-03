# 多类别分割任务代码诊断报告

检查日期：2026-06-04  
检查范围：`docs/METHOD.md`、`docs/method_ieee.tex`、训练入口 `unimatch_v2.py`、DPT/adapter、LC-PAM、MGER-guided routing、TCR/TS-MDR、评估与测试路径。  
结论摘要：当前代码主线更接近 `METHOD.md` 和 `RUN.html` 描述的单阶段联合训练体系，而不是 `method_ieee.tex` 中“baseline / affinity_min / boundary / debias 四类独立两阶段”的旧体系。实现总体具备 HPTA/HPTA-MoE、LC-PAM、MGER-guided routing、TCR、可选 TS-MDR 和实例感知损失。MGER 当前作为 detached edge prior 注入 HPTA-MoE router；旧 edge residual adapter、RGB-edge consistency 和辅助 boundary head 仅作为显式开启的 legacy 消融。

## 1. 与方法说明的一致性概览

### 1.1 当前代码实际主线

- `unimatch_v2.py` 中显式关闭旧 standalone boundary：`cfg['boundary']['enabled'] = False`，当前边缘功能由 `--edge-enhance`、`--edge-refiner` 控制。
- `--joint-text-stage` 支持无 Stage-I checkpoint 时从类别名初始化文本锚点，并在主分割训练中联合训练 LC-PAM。
- `--visual-adapter` 会自动锁定 DINOv2 backbone，符合 HPTA/PEFT 设定。
- `test.py` 有独立的 affinity-side 推理融合逻辑，能加载 `affinity_side` 和 `affinity_side_ema`。

### 1.2 文档冲突

- `docs/METHOD.md` 描述 HPTA/HPTA-MoE + LC-PAM + MGER-guided routing + TCR + 可选 TS-MDR/cls-aux 的当前体系。
- `docs/method_ieee.tex` 描述旧的四类方法拆分，且称 `util/debias.py` 实现 debias，但当前仓库状态里 `util/debias.py` 已删除，训练入口也没有完整 debias 逻辑。
- `docs/RUN.md` 已声明过时，`docs/RUN.html` 才是当前运行说明。

建议：后续论文/实验记录应以 `METHOD.md`/`RUN.html` 为准，并更新或归档 `method_ieee.tex`，否则“方法描述-代码实现”会被审稿或复现实验反向质疑。

## 2. 确定性问题

### 2.1 TS-MDR 可单独触发未定义变量错误

位置：`unimatch_v2.py` 约 908-918、945-979。  
现象：`prob_u_w_cm1` 和 `prob_u_w_cm2` 只在 TCR v2 分支内部创建；TS-MDR 分支随后直接复用它们。若启用 `--tsmdr-enabled` 但未启用 `--temporal-consistency`，或 TCR warmup 条件尚未满足，而 TS-MDR 已满足 warmup，则会在 `q_t1 = _ext(... prev_prob=prob_u_w_cm1.detach())` 处触发 `UnboundLocalError`。

影响：TS-MDR 不是独立可选模块，实际依赖 TCR 分支的局部变量副作用。该行为与 `METHOD.md` 中“TS-MDR optional”的描述不一致。

建议：把 CutMix 对齐后的 teacher soft probability 构造提前到 TCR/TS-MDR 共同可见的位置，只要任一分支需要就计算；或让 TS-MDR 内部独立构造。

### 2.2 训练期验证没有接入 LC-PAM 融合

位置：`supervised.py` 的 `evaluate()` 约 98 行直接 `pred = model(img)`；`unimatch_v2.py` 约 1188-1200 调用该函数时只传 `model` / `model_ema`，没有传 `affinity_side` 或 `affinity_side_ema`。  
对比：`test.py` 约 153-181 在离线测试时会显式运行 `affinity_side(...)` 并调用 `affinity_side.fuse(...)`。

影响：

- 训练日志中的 val/test mIoU 对 LC-PAM 最终融合效果不敏感。
- `best.pth` 选择基于未融合 DPT 路径，可能不是最终推理路径的最佳 checkpoint。
- plateau-unfreeze 逻辑约 1213-1223 使用未融合 mIoU 判断 affinity projection 是否解冻，这与“LC-PAM 融合后训练/评估”的方法描述不一致。

建议：训练期评估复用 `test.py` 的融合推理逻辑，或实现一个轻量 wrapper model，把 `DPT + affinity_side(_ema)` 封装成统一 forward。若担心速度，可只在 `eval_interval` 或 final epoch 做 fused eval，但 best selection 需明确使用哪个指标。

### 2.3 `method_ieee.tex` 中 debias 模块描述已失效

位置：`method_ieee.tex` 的 “Class-Prior Bias Correction” 与 “Implementation Alignment” 仍指向 `util/debias.py`。当前 `git status` 显示该文件已删除，`unimatch_v2.py` 中仅保留 `bias_metrics` 的日志统计，没有实际 teacher logit debias 调整。

影响：若实验表格仍报告 `debias` 变体，当前代码不能按文档复现该方法。

建议：要么恢复并接入 debias 逻辑；要么在文档中明确 debias 是历史实验，不属于当前主线。

## 3. 线性/非线性变量处理检查

### 3.1 HPTA visual adapter 基本合理

位置：`model/visual_adapter.py`。  
实现：`LayerNorm -> Linear down -> local depthwise conv / global MLP -> softmax gate -> Linear up -> Dropout -> gamma residual`，`gamma` 初始化为 0。该实现与 `METHOD.md` 中“局部 + 全局 + 输入相关门控 + ReZero”的描述一致。

注意点：adapter 的 dropout 默认 0，正则主要来自残差零初始化和主训练增强；若 backbone 锁定且数据量小，可考虑给 `VISUAL_ADAPTER_DROPOUT` 小值，但这属于超参，不是代码错误。

### 3.2 DPT complementary dropout 实现与文档关系不清

位置：`model/semseg/dpt.py` 约 184-196。  
实现：对拼接后的两条 strong view 生成互补通道 mask，且随机保留一半样本不 dropout。该机制来自 UniMatch-V2 的 complementary dropout，但 `METHOD.md` 的 loss/模块说明没有显式描述。

风险：这不是错误，但在多模块消融中会影响“dropout/正则”解释。如果论文方法部分强调正则项，应补充说明 strong-view DPT head 使用 complementary feature dropout。

### 3.3 LC-PAM cosine 分支的尺度风险已在文档提到，代码未做额外校准

位置：`util/affinity_side.py` 约 405-447。  
实现：`A = normalized_patch @ normalized_text / tau`，再加 `cls_bias`、上采样、按非背景 mask scatter，最后用 `gate` 和 DPT logits 线性融合。`tau` clamp 到 `[1e-3, 1]`，当 `tau` 很小时 affinity logits 可远大于 DPT logits。

影响：gate 增大后，prior logit 尺度可能主导分割结果；这与 `METHOD.md` 的注意事项一致。代码没有温度外的 logit calibration 或 per-class normalization。

建议：如果训练中出现 gate 均值上升后 mIoU 抖动，可先记录 DPT logits 与 prior logits 的均值/标准差，再考虑只做日志诊断或温度/门控正则；不建议在未做实验前修改方法。

### 3.4 hyperbolic/CMA 路径存在“双重 bias”可能

位置：`model/cross_modal_adapter.py` 约 290-291 已在 `A` 中加入 `cls_bias`；`util/affinity_side.py` 约 376 对 CMA+edge 分支也加入 `cma_block.cls_bias`；随后约 417-418 对 dense prior 再统一加 `cls_bias`。Hyperbolic pathway 的 `A` 也在 `model/hyperbolic_fusion.py` 中加入 `cls_bias`，再经过 side branch dense 时又加 `cls_bias`。

影响：对 cosine 分支，`A` 不含 bias，side branch 统一加一次 bias 是合理的；但 CMA/hyperbolic_pathway 的 `A` 已经包含 block 内部 bias，再加 side branch 的 `cls_bias` 可能导致偏置重复。若两个 bias 都可训练，先验 dense logits 会发生不可预期平移，影响 gate 融合后的类别偏好。

建议：明确约定 `AffinitySideBranch.forward()` 接收的 `A` 是否包含 class bias。更稳妥的做法是让各 metric 输出 raw affinity，由 side branch 统一加一次 bias；或在 metric 已含 bias 时跳过 dense 的二次加 bias。

## 4. Loss 计算合理性

### 4.1 SSL 主损失与文档一致

位置：`unimatch_v2.py` 约 859-873。  
实现：有标注 CE；两条 strong view 的 CE 用 confidence gate 和 ignore mask 筛选，分母为有效像素数而不是高置信像素数；`loss = (loss_x + loss_u_s) / 2`。这与 `METHOD.md` 中公式一致。

注意：用有效像素作分母会让低置信比例高时无监督损失自然变小，这是 UniMatch 风格设计，不是 bug。

### 4.2 TCR v2 与文档一致，但 v1 权重不一致

位置：`unimatch_v2.py` 约 892-936；`util/temporal.py`。  
v2：teacher weak probability CutMix 对齐后作为 `p_kf`，student strong softmax 作为 `p_cf`，`entropy_gated_consistency` 内部 detach teacher，并按 gate 平均，且总损失加 `0.5 * temporal_weight`，与 `METHOD.md` 一致。  
v1：若启用 `--temporal-original`，代码使用 `loss += temporal_weight * loss_tcr`，没有额外 `0.5`。这是 legacy 分支，若用于消融，应在表格中说明权重不可直接与 v2 比。

### 4.3 MGER refined-edge loss 与 legacy boundary head

位置：`unimatch_v2.py` 约 1032-1046。  
实现：主方法只在 labeled branch 上监督 MGER refined edge，并通过 `edge_refiner_pos_weight` 使用稀疏边界正样本加权；MGER 输出 detach 后作为 HPTA-MoE 的 edge routing condition。  
文档：`METHOD.md` 与 `RUN.html` 已将主方法改为 MGER-guided routing。旧 edge boundary head、`L_bnd` 和 RGB-edge consistency 仅在显式开启 `--edge-residual-adapters` 时作为 legacy 消融存在。

影响：主方法不再依赖旧 boundary head，因此原先关于 boundary-head `pos_weight=1.0` 的风险不再适用于默认配置。若开启 legacy 消融，仍应在实验表格中单独标注其损失设定。

建议：论文主叙述只描述 MGER refined-edge supervision 与 detached routing prior；legacy boundary head 不应作为默认方法贡献点。

### 4.4 Affinity auxiliary BCE 只用有标注图像，合理但目标阈值写死

位置：`unimatch_v2.py` 约 997-1024。  
实现：只对 labeled branch 的 `side_x['cls_logits']` 做 BCE，类别存在条件为 mask 中像素数 `>=16`。`METHOD.md` 中 cls-aux 写的是 `min_pixels`，但 affinity aux 的 16 是硬编码。

影响：不同 crop size 和数据集类别尺度差异较大时，固定 16 可能不合适，尤其小目标/细结构类别容易被错误判为不存在或噪声存在。

建议：把 affinity aux 的 `min_pixels` 做成参数，或复用 `cls_min_pixels` 的配置逻辑。

## 5. 正则化与 dropout 检查

- HPTA adapter：`gamma=0`，dropout 可配，结构稳定。
- LC-PAM gate：`gate_conv.weight=0`、bias 默认 `-3`，初始 `sigmoid≈0.047`，符合“保守融合”。
- Edge adapters：`gamma=0`，`gate_dropout=0.05` 默认存在，但 CLI 只控制 adapter MLP 内 dropout，不控制 gate dropout。
- MGER：最后一层 weight=0、bias=-1.5，输出初始约 0.18，比较稳。
- Hyperbolic pathway：支持 `proj_dropout`、`fuse_dropout`、`fuse_layernorm`、`tau/distance clamp`；但 Stage-II 加载时 `grad_checkpoint=False`，即使 Stage-I checkpoint 含 checkpoint 配置也不会沿用。
- CMA：adapter dropout 在 Stage-II 构造时被强制为 0.0，fuse dropout 从 checkpoint 读取。若文档声称 CMA adapter dropout 参与 Stage-II 训练，需要修正文档或代码。

## 6. 训练速度和显存优化建议

以下建议不降低图像分辨率、不降低 batch size，也不改变方法本身。

### 6.1 优先修复训练期 fused evaluation

这既是正确性问题，也是速度问题。当前训练期评估快，是因为没跑 LC-PAM；一旦修复会变慢。建议只在 `--eval-interval` 到达时做 fused eval，并默认 `--eval-ema-only`，避免 raw/EMA 双跑。

### 6.2 减少 hyperbolic_pathway/CMA 的重复 backbone encode

位置：`unimatch_v2.py` 约 817 和 840。  
当前为了用 `X_aware` 重新 decode，代码重新 `encode(img_x)` / `encode(img_u_cat)` 获取浅层 features。注释称“reuse intermediate features”，但实际第一次 `model(...)` 没有返回 features[0..2]，所以又跑了一次 backbone。DINOv2 backbone 是主要时间/显存来源，这对 hyperbolic/CMA 训练很贵。

建议：给 `DPT.forward()` 增加可选返回 adapted features 的接口，例如 `return_features=True`，第一次 forward 时保留浅层 features，用 `x_aware` 替换最后一层后直接 decode。这样不改变模型方法，只减少重复计算。注意要控制返回对象生命周期，避免 baseline 路径额外持有大激活。

### 6.3 对 side branch 的 labeled/u1/u2 合并 forward

位置：`unimatch_v2.py` 约 798、826、828。  
当前 LC-PAM 对 x、u1、u2 三次调用，参数 clone 和 pooling/gate 上采样也做三次。可以在空间尺寸一致的训练 crop 下把 patches 拼接成一个 batch，一次 side forward 后拆分，减少 Python/DDP overhead 和小 kernel 调度开销。对 edge_prior 也可拼接。

注意：这不改变方法，但要小心 `last_x_aware` 这种模块状态变量。建议让 `AffinitySideBranch.forward()` 直接在返回 dict 中带 `x_aware`，不要依赖 `self.last_x_aware`。

### 6.4 避免 `last_edge_boundary_logits` 这种模块全局状态

位置：`model/semseg/dpt.py` 约 240。  
当前 forward 把 edge boundary logits 写到 `self.last_edge_boundary_logits`，训练入口再读取。该方式在 DDP、checkpoint、重入 forward、未来 mixed eval wrapper 中容易踩状态覆盖问题。

建议：让 DPT forward 在 `return_aux=True` 时返回 dict：`{'logits', 'cls_tok', 'patches', 'edge_boundary_logits', 'features'}`。这会让训练和测试更清晰，也能配合 6.2 减少重复 encode。

### 6.5 RGB Sobel quantile 可能是 CPU/GPU 同步热点

位置：`util/edge_enhance.py` 约 70-72。  
`flat.quantile(..., dim=1)` 在每个 batch、多个视图上调用，可能比简单 reduce 慢，且对高分辨率图像开销明显。

建议：可用 `torch.kthvalue`/近似分位或固定归一化统计替换精确 quantile；如果必须保持完全一致，至少只对 downsample 后的 edge magnitude 估计分位数，再上采样归一化参数。这不会降低输入图像分辨率，只降低边缘先验统计的计算量。

### 6.6 开启已有的无损速度选项

- `EVAL_EMA_ONLY=1`：训练期只评估 EMA。
- `TEST_INTERVAL=0` 或 `-1`：避免每轮跑 held-out test。
- `EDGE_REFINER_CHECKPOINT=1`：已有代码支持 MGER labeled branch checkpoint，显存紧张时开启。
- `NUM_WORKERS` 与 `prefetch_factor` 已支持，实际机器上应根据 IO 调整，避免 worker 过多反抢 CPU。
- 当前已开启 TF32 和 persistent workers，这是合理的。

## 7. 建议优先级

P0：

1. 修复 TS-MDR 对 `prob_u_w_cm1/cm2` 的隐式依赖。
2. 明确并修复训练期验证是否需要 LC-PAM fused logits；若当前结果表使用 `test.py`，要在文档中说明训练日志/best 选择与最终测试路径不同。

P1：

1. 统一 class bias 加法规则，避免 hyperbolic_pathway/CMA dense prior 双重 bias。
2. 对齐 edge boundary `pos_weight` 与方法说明。
3. 将 affinity aux 的 `min_pixels=16` 参数化。

P2：

1. 重构 DPT forward 返回 features/aux，减少 hyperbolic/CMA 重复 backbone encode。
2. 合并 side branch 三路 forward，去掉 `last_x_aware` 状态依赖。
3. 优化 Sobel quantile 计算。

总体判断：当前代码不是“完全错误”，主损失和主要模块设计大体成立；真正需要优先处理的是评估路径、TS-MDR 变量作用域、以及若干文档声称和实现细节不一致的问题。若这些不修，实验复现和论文表述会比模型本身更容易出问题。
