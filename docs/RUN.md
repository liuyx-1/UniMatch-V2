# Run Manual — UniMatch-V2 Surgical

本文件对应本地目录 `C:\u2v_fresh\UniMatch-V2`，服务器建议解压到 `/data/code/UniMatch-V2`。

当前支持：

```text
Datasets : endoscapes_seg50 | cholecseg8k | endovis2018 | endovis2017_parts | endovis2017_type
Rates    : 0.10 | 0.20 | 0.25 | 0.30 | 0.50
Variants : base | debias | boundary | tangent | full | cls | debias_cls | boundary_cls | full_cls | affinity_min | affinity
```

### 数据集概览

| 数据集 | 类别数 | 任务说明 | data_root 期望布局 |
|---|---|---|---|
| `endoscapes_seg50`     | 7  | 腹腔镜胆囊切除关键解剖+器械分割（CVS） | `<root>/{images,masks}/{train,val,test}/*.jpg` / `*.png` |
| `cholecseg8k`          | 13 | 腹腔镜胆囊切除全场景语义分割          | `<root>/{images,masks}/{train,val,test}/*.png` |
| `endovis2018`          | 12 | 机器人肾脏手术多类器械+组织            | `<root>/{train,test}/{image,label}/seq_*_frame*.png` |
| `endovis2017_parts`    | 4  | 机器人器械三段部件（shaft/wrist/clasper） | `<root>/{train,test}/{image,label}/seq_*_frame*.png` |
| `endovis2017_type`     | 8  | 机器人器械类型（7 种 + bg）            | `<root>/{train,test}/{image,label}/seq_*_frame*.png` |

均为高度长尾，背景与器械主导，稀有解剖/部件 recall 是关键指标。`splits/<dataset>/<rate>/{labeled,unlabeled,val}.txt` 由 `tools/prepare_*.py` 自动生成。

### 数据准备脚本

```bash
# Endoscapes-Seg50（已就位，无需再跑）
python tools/prepare_endoscapes.py --src /data/raw/endoscapes --dst /data/test/endoscapes_seg50_processed

# CholecSeg8k（视频帧 + COCO-style mask 重映射）
python tools/prepare_cholecseg8k.py \
  --src /data/raw/cholecseg8k \
  --dst /data/test/cholecseg8k_coco_semantic
# 或当数据已就位（/data/test/cholecseg8k/ + /data/test/cholecseg8k_coco_semantic/），
# 仅生成 splits 即可：
python tools/build_cholecseg8k_splits.py --root /data/test --rates 0.10 0.20 0.25 0.30 0.50 --seed 42

# EndoVis2018 (15 train + 4 test seq, release_1-4 + test_data)
python tools/prepare_endovis2018.py --root /data/raw/endovis2018 --out /data/test/endovis2018_processed --apply

# EndoVis2017 — 同源数据，两种标签 task
python tools/prepare_endovis2017.py --task parts --root /data/test/endovis2017/EndoVis2017/data --out /data/test/endovis2017_parts_processed --apply
python tools/prepare_endovis2017.py --task type  --root /data/test/endovis2017/EndoVis2017/data --out /data/test/endovis2017_type_processed  --apply
```

完成后在对应 `configs/<dataset>.yaml` 把 `data_root` 指到 `*_processed` 目录。

其中：

- `debias`: 伪标签 logit prior debias。
- `boundary`: MA-Canny 边界辅助损失。
- `tangent`: `boundary + tangent-space geometry alignment`，必须和 boundary 一起开，脚本中用 variant `tangent`。
- `cls`: 图像级多标签分类辅助头。
- `full`: `debias + boundary`。
- `affinity_min`: 仅加 Stage 1 affinity 侧支（gate 融合），不开 debias/boundary/cls。
- `affinity`: `debias + boundary + cls + Stage 1 affinity 侧支`，即论文里的完整方法。详见 §9e。

建议先跑 `base` 与 `debias` 得到类别不平衡主结论，再跑 `boundary` 与 `tangent` 做边界几何对照，最后跑 `affinity_min` 与 `affinity` 评估文本-视觉对齐先验的增益。

## 1. 上传与解压

本地打包后上传：

```powershell
scp -P 42118 "C:\u2v_fresh\_pack\UniMatch-V2_<stamp>.zip" root@xj-member.bitahub.com:/data/code/
```

服务器解压：

```bash
cd /data/code
rm -rf UniMatch-V2.old
[ -d UniMatch-V2 ] && mv UniMatch-V2 UniMatch-V2.old

# 先快速验 zip 自带顶层 UniMatch-V2/ 目录（防止散到根级）
unzip -l UniMatch-V2_<stamp>.zip | head -5 | grep -q "UniMatch-V2/" \
    || { echo "WARN: zip 内容无顶层目录，会散到 /data/code/。建议重新打包。"; }

unzip -q UniMatch-V2_<stamp>.zip -d /data/code
cd /data/code/UniMatch-V2
```

如果 zip 缺顶层目录、文件已经散到 `/data/code/`（症状：根目录出现 `unimatch_v2.py`、
`configs/`、`model/`、`util/` 等本应在 `UniMatch-V2/` 内的项），先收拢：

```bash
cd /data/code
mkdir -p UniMatch-V2
mv unimatch_v2.py supervised.py fixmatch.py test.py README.md \
   LICENSE requirements.txt \
   configs dataset model tools util scripts docs remote-sensing \
   UniMatch-V2/ 2>/dev/null
cd UniMatch-V2 && ls scripts/train.sh util/manifold_tangent.py   # 验证完整
```

## 2. 每个 shell 的初始化

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate unimatchv2

cd /data/code/UniMatch-V2
chmod +x scripts/*.sh

unset SPLITS
export DATA_ROOT=/data/test
export SPLITS=/data/splits
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 推荐写到 NVMe scratch，训练完再同步到 /data/code/exp
export EXP_ROOT=/localdisk-tmp/exp
export EXP_ARCHIVE=/data/code/exp
mkdir -p "$EXP_ROOT" "$EXP_ARCHIVE"
```

训练结束后归档：

```bash
rsync -a --info=progress2 "$EXP_ROOT/" "$EXP_ARCHIVE/"
```

## 3. 检查 data_root

`configs/endoscapes_seg50.yaml` 的 `data_root` 必须指向包含 `images/` 与 `masks/` 的 processed 数据目录。

```bash
grep '^data_root:' configs/endoscapes_seg50.yaml
find /data/test -maxdepth 4 -type d -name images
find /data/test -maxdepth 4 -type d -name masks
```

若训练报：

```text
FileNotFoundError: /data/test/images/train/...
```

说明 `data_root` 错了，可自动修正：

```bash
IMG=$(find /data/test -path "*/images/train/57_26900.jpg" -print -quit)
ROOT=${IMG%/images/train/57_26900.jpg}
echo "$ROOT"
sed -i "s|^data_root:.*|data_root: $ROOT|" configs/endoscapes_seg50.yaml
grep '^data_root:' configs/endoscapes_seg50.yaml
```

## 4. Tangent 单元测试

第一次跑 `tangent` 前建议执行：

```bash
python - <<'PY'
import torch
from util.manifold_tangent import tangent_alignment_loss

torch.manual_seed(0)
F = torch.randn(2, 5, 24, 24, device='cuda', requires_grad=True)
G = torch.randn(2, 7, 24, 24, device='cuda', requires_grad=True)

for name, A, B, tol in [
    ('same', F, F, 1e-7),
    ('scale', F, 5 * F, 1e-6),
    ('sign', F, -F, 1e-6),
]:
    loss = tangent_alignment_loss(A, B)
    print(name, float(loss))
    assert float(loss) <= tol, (name, float(loss), tol)

loss = tangent_alignment_loss(F, G)
print('random', float(loss))
assert float(loss) > 0
loss.backward()
assert torch.isfinite(F.grad).all()
assert torch.isfinite(G.grad).all()
print('tangent test ok')
PY
```

## 5. 快速试跑

先用 1 个 epoch 确认路径、DDP、loss 都没问题：

```bash
RATE=0.25 BS=2 LR=5e-6 EPOCHS=1 sh scripts/train.sh 1 29500 base endoscapes_seg50
RATE=0.25 BS=2 LR=5e-6 EPOCHS=1 sh scripts/train.sh 1 29501 debias endoscapes_seg50
RATE=0.25 BS=2 LR=5e-6 EPOCHS=1 sh scripts/train.sh 1 29504 tangent endoscapes_seg50
```

输出目录格式：

```text
$EXP_ROOT/<dataset>/unimatch_v2_<variant>_r<RATE>[_bs<BS>_lr<LR>_ep<EPOCHS>]/
```

## 6. 正式训练 Endoscapes-Seg50

先跑 v1/v2：

```bash
RATE=0.25 BS=2 LR=5e-6 sh scripts/train.sh 1 29500 base   endoscapes_seg50
RATE=0.25 BS=2 LR=5e-6 sh scripts/train.sh 1 29501 debias endoscapes_seg50
```

再跑边界/几何对照：

```bash
RATE=0.25 BS=2 LR=5e-6 sh scripts/train.sh 1 29502 boundary endoscapes_seg50
RATE=0.25 BS=2 LR=5e-6 sh scripts/train.sh 1 29504 tangent  endoscapes_seg50
```

最后可跑 full：

```bash
RATE=0.25 BS=2 LR=5e-6 sh scripts/train.sh 1 29503 full endoscapes_seg50
```

如果要跑 5 个标签比例：

```bash
for RATE in 0.10 0.20 0.25 0.30 0.50; do
  for V in base debias boundary tangent full; do
    RATE=$RATE BS=2 LR=5e-6 sh scripts/train.sh 1 29500 $V endoscapes_seg50
  done
done
```

## 7. 测试与类别偏差诊断

测试脚本会自动追加：

- `--train-id-path`
- `--train-freq-cache`
- `--ece`
- `--ap`

单个测试：

```bash
RATE=0.25 BS=2 LR=5e-6 sh scripts/test.sh base   endoscapes_seg50
RATE=0.25 BS=2 LR=5e-6 sh scripts/test.sh debias endoscapes_seg50
```

批量测试：

```bash
for V in base debias boundary tangent full; do
  RATE=0.25 BS=2 LR=5e-6 sh scripts/test.sh $V endoscapes_seg50
done
```

重点看每个目录下：

```text
test_metrics.csv
```

其中 bias 指标含义：

- `gap_HT`: head mIoU - tail mIoU，越小表示头尾差距越小。
- `TV_pred_vs_gt` / `KL_pred_vs_gt`: 预测类别分布与 GT 分布距离。
- `recall_CV`: 类别 recall 离散程度。
- `ECE_head/body/tail`: 不同频率组的校准误差。

## 8. 绘制训练曲线

```bash
python tools/plot_train_log.py \
  --run "base:$EXP_ROOT/endoscapes_seg50/unimatch_v2_base_r0.25_bs2_lr5e-6/train_log.csv" \
  --run "debias:$EXP_ROOT/endoscapes_seg50/unimatch_v2_debias_r0.25_bs2_lr5e-6/train_log.csv" \
  --run "boundary:$EXP_ROOT/endoscapes_seg50/unimatch_v2_boundary_r0.25_bs2_lr5e-6/train_log.csv" \
  --run "tangent:$EXP_ROOT/endoscapes_seg50/unimatch_v2_tangent_r0.25_bs2_lr5e-6/train_log.csv" \
  --run "full:$EXP_ROOT/endoscapes_seg50/unimatch_v2_full_r0.25_bs2_lr5e-6/train_log.csv" \
  --out "$EXP_ROOT/endoscapes_seg50/endoscapes_r0.25_curves.png"
```

图中包含：

- loss 曲线
- mIoU / EMA mIoU
- head/body/tail pseudo-label keep ratio
- tail keep ratio 跨 run 对比

`debias` 若有效，通常应看到 tail keep ratio 上升，且测试中的 `gap_HT` 缩小。

## 9. CholecSeg8k

命令同 Endoscapes，只替换 dataset：

```bash
RATE=0.25 BS=2 LR=5e-6 sh scripts/train.sh 1 29510 base   cholecseg8k
RATE=0.25 BS=2 LR=5e-6 sh scripts/train.sh 1 29511 debias cholecseg8k
RATE=0.25 BS=2 LR=5e-6 sh scripts/test.sh base   cholecseg8k
RATE=0.25 BS=2 LR=5e-6 sh scripts/test.sh debias cholecseg8k
```

## 9b. EndoVis2018 / EndoVis2017

```bash
# EndoVis2018 (12 类)
RATE=0.25 BS=2 LR=5e-6 sh scripts/train.sh 1 29520 base    endovis2018
RATE=0.25 BS=2 LR=5e-6 sh scripts/train.sh 1 29521 debias  endovis2018
RATE=0.25 BS=2 LR=5e-6 sh scripts/test.sh  base    endovis2018

# EndoVis2017-parts (4 类)
RATE=0.25 BS=2 LR=5e-6 sh scripts/train.sh 1 29530 base    endovis2017_parts
RATE=0.25 BS=2 LR=5e-6 sh scripts/train.sh 1 29531 debias  endovis2017_parts
RATE=0.25 BS=2 LR=5e-6 sh scripts/test.sh  base    endovis2017_parts

# EndoVis2017-type (8 类)
RATE=0.25 BS=2 LR=5e-6 sh scripts/train.sh 1 29540 base    endovis2017_type
RATE=0.25 BS=2 LR=5e-6 sh scripts/train.sh 1 29541 debias  endovis2017_type
RATE=0.25 BS=2 LR=5e-6 sh scripts/test.sh  base    endovis2017_type
```

EndoVis2017 两个 task 共享原始帧但 mask 不同，准备时已用 `_to_2d()` 把 RGB 标签转 2D index。`endovis2018` split 文件由 `prepare_endovis2018.py` 写入，含 `train/test_data` 嵌套目录。

## 9c. (历史 / 已废弃) — CoOp + Dense Alignment 探索

原 §9c (SigLIP-2 CoOp 离线训练) 与 §9d (Dense Alignment 联合训练 + 4 组对照) 是探索阶段产物，
最终方案放弃 CoOp + 软提示 + mask 监督路径，统一为下面 §9e 的 **Patch-Text Affinity** 弱监督先验。
若需复现历史实验，参见 git history 或 `tools/{soft_prompt_finetune,run_mmproto_pipeline,train_dense_align_merged}.py`。

## 9e. 两阶段完整方法：DINOv2 + SigLIP 文本对齐先验 + UniMatch-V2 分割

### 整体方法

我们最终的方法把"长尾多标签语义"与"半监督密集分割"解耦成**两个共享同一视觉骨干的阶段**——整套方法只有一个视觉编码器（UniMatch-V2 原生的 **DINOv2-B/14**）和一个文本编码器（**SigLIP-2 text encoder**），两阶段都不引入额外的视觉主干。第一阶段用 DINOv2 patch + SigLIP 文本对齐训出一个轻量的"图像级多标签 + 粗定位"先验生成器；第二阶段把先验作为**冻结侧支**接入 UniMatch-V2 分割主流水线，与原有 EMA debias、boundary 辅助、tangent 流形正则联合作用，专门补偿长尾稀有类与小目标在纯像素监督下学不动的问题。

**Stage 1 — DINOv2 视觉 + SigLIP 文本对齐先验**（[model/affinity_cls.py](C:/u2v_fresh/UniMatch-V2/model/affinity_cls.py) + [tools/train_affinity_per_ds.py](C:/u2v_fresh/UniMatch-V2/tools/train_affinity_per_ds.py)）：输入 518×518 图像经 DINOv2-B/14 得 37×37=1369 个 patch token，DINOv2 主干**默认可训**（小学习率 `--lr-vision` 默认 `1e-5`）以适配外科领域；类锚点来自 SigLIP-2 text encoder + [tools/class_templates.py](C:/u2v_fresh/UniMatch-V2/tools/class_templates.py) 中每类 4 条人写医学模板做 ensemble 平均（先单位化、再求均值、再单位化），text encoder 同样**默认可训**（`--lr-text` 默认 `1e-5`）。patch 与 text 各经一个轻量 MLP（768→256→128）投到共享 128 维潜空间并 L2 归一，构造亲和矩阵 `A[i,c] = p_i · T_c / tau`；沿空间维度做 top-k=5 池化加每类可学偏置得图像级 logits `s_c = topk(A[:,c]).mean + b_c`，A 按类切片重排即得粗粒度定位图 `L_c`（37×37）。每个数据集独立训一份 `affinity_<ds>.pt`，可选用 `--freeze-vision` / `--freeze-text` 退化为完全冻结模式；训练损失为类频权重补偿的 sigmoid-BCE，每 5 epoch 在该数据集 val split 上输出按**原类名**展开的 P/R/F1/AP 摘要。可训参数：投影头 ~46 万 + 可选 DINOv2 ~86M + 可选 SigLIP-text ~60M；checkpoint 额外保存 `dinov2_state_dict` 与 `siglip_text_state`（仅当对应编码器解冻），Stage 2 启动时若 `--affinity-warmstart` 指向的 ckpt 含 `dinov2_state_dict` 则自动覆盖 DPT 主干默认 pretrained，确保两阶段视觉空间衔接一致。

**Stage 2 — UniMatch-V2 半监督分割 + SigLIP 语义校正**（[unimatch_v2.py](C:/u2v_fresh/UniMatch-V2/unimatch_v2.py) + [scripts/train.sh](C:/u2v_fresh/UniMatch-V2/scripts/train.sh)）：与 Stage 1 **共享同一个 DINOv2 主干**，避免特征空间错配；DPT 解码器为分割主路径，沿用双强视图 + EMA 教师生成伪标签，按需开启 EMA pixel prior debias、boundary 辅助头（MA-Canny + BCE+Dice）、tangent 结构张量一致性、ML-Decoder + ASL 多标签 cls 头。Stage 1 产物经 `--affinity-prior` 加载作为**冻结侧支**：同一组 DINOv2 patch features 经载入的 visual_proj 与 T_anchor 计算亲和矩阵，按当前 batch 数据集的 `remap` 选出该数据集有效类索引，上采样到 crop 尺寸得 `[B, C_d, H, W]` 密集先验，与 DPT logits 通过新增 1x1 gate conv 学 per-pixel 融合权重 `g = sigmoid(conv(DPT_feat))` 做软融合 `L_final = (1-g)*L_DPT + g*L_prior`；最终 logits 与 GT/伪标签一起算 CE + boundary + tangent 全套 loss 反传，visual_proj、T_anchor、cls_bias、log_tau **全部冻结不更新**，仅 DPT 解码器、boundary head、tangent head、gate conv 与可选的小 lr DINOv2 微调进入优化器。

**端到端推理**：图像送入 DINOv2 主干后只前向一次，patch features 同时进入 DPT 主路径与 Stage 1 亲和侧支，两路 logits 经 gate 融合后 argmax 得最终分割 mask；测试脚本 [scripts/test.sh](C:/u2v_fresh/UniMatch-V2/scripts/test.sh) 自动追加 bias 诊断指标（`gap_HT / TV_pred_vs_gt / KL / recall_CV / ECE_head_body_tail`）写入 `test_metrics.csv`。

### Stage 1 运行步骤（每个数据集独立训一个模型）

每个数据集的类目语义、像素分布、稀有类构成差异很大，统一全局空间联合训会让大量类（如 endovis2017_parts 的 3 类）触及 top-k 评估天花板，无法体现方法对长尾稀有类的实际帮助。因此 Stage 1 改为**每个数据集独立训一个 affinity 模型**：每个模型只在该数据集的原始类空间上工作，类锚点由 [tools/class_templates.py](C:/u2v_fresh/UniMatch-V2/tools/class_templates.py) 中对应医学模板生成（缺失模板的类自动回退到 `"a photo of <class>"`），训完输出 `affinity_<ds>.pt`，下游 Stage 2 各数据集分割时各自加载自己的产物。

**Step 1 — manifest 与 val 构建**

```bash
# 1a) 各数据集 manifest（已生成；如需重建：）
python tools/build_siglip_manifest.py \
  --splits-root /data/splits \
  --data-root   /data/test \
  --out-dir     /data/pretrained/siglip_train

# 1b) 链接缺失的 val.txt 到 /data/splits/<ds>/val.txt
mkdir -p /data/splits/endoscapes_seg50 /data/splits/endovis2018 /data/splits/cholecseg8k
ln -sfn /data/test/endoscapes_seg50_processed/splits/val.txt /data/splits/endoscapes_seg50/val.txt
ln -sfn /data/test/endovis2018_processed/splits/val.txt      /data/splits/endovis2018/val.txt
ln -sfn /data/splits/unimatch_splits_cholecseg8k_0.25_seed42/val.txt /data/splits/cholecseg8k/val.txt

# 1c) 构建每数据集的 val manifest（产出 manifest_<ds>_val.jsonl）
python tools/build_merged_manifest.py \
  --manifest-dir /data/pretrained/siglip_train \
  --out-dir      /data/pretrained/siglip_train/merged
python tools/build_val_manifest.py \
  --splits-root /data/splits \
  --data-root   /data/test \
  --global-info /data/pretrained/siglip_train/merged/global_classes.json \
  --out-dir     /data/pretrained/siglip_train/merged_val
```

**Step 2 — 训练 per-dataset 亲和先验（DINOv2 视觉 + SigLIP 文本）**

```bash
mkdir -p /data/pretrained/siglip_train/affinity_per_ds
python tools/train_affinity_per_ds.py \
  --manifest-dir /data/pretrained/siglip_train \
  --val-dir      /data/pretrained/siglip_train/merged_val \
  --out-dir      /data/pretrained/siglip_train/affinity_per_ds \
  --model        google/siglip2-base-patch16-256 \
  --vision-backbone dinov2 --dinov2-input-size 518 \
  --epochs 40 --bs 16 --lr 1e-3 \
  --lr-vision 1e-5 --lr-text 1e-5 \
  --lambda-patch 1.0 --patch-grid 37 \
  --topk 5 --eval-every 5 \
  2>&1 | tee /data/pretrained/siglip_train/affinity_per_ds/run.log
```

可通过 `--datasets endovis2018 cholecseg8k` 只跑指定子集。如显存吃紧或数据极少（如 endoscapes 只 34 训）容易让主干过拟合，可改用 `--freeze-vision --freeze-text` 退化为完全冻结模式（约 46 万参数）。
默认配置下额外训练 ~150M 参数（DINOv2 + SigLIP text），bs=16 在 V100 上约占 14GB；若 OOM，先降到 bs=8 或加 `--freeze-text` 只放开视觉。

输出结构：

```text
affinity_per_ds/
├── endoscapes_seg50/affinity_endoscapes_seg50.pt + per_class_metrics.csv
├── cholecseg8k/affinity_cholecseg8k.pt + per_class_metrics.csv
├── endovis2018/...
├── endovis2017_parts/...
├── endovis2017_type/...
└── run.log
```

每个 `affinity_<ds>.pt` 字段：

```text
visual_proj      : MLP 768->256->128
text_proj        : MLP 768->256->128
T_anchor         : [C_d, 128] L2-norm 类锚点
log_tau, cls_bias
class_names      : 该数据集去除 background 后的原类名列表
orig_class_names : 该数据集完整原类名列表（含 background）
orig_to_new      : {orig_id: new_id} 映射
vision_backbone  : 'dinov2'
dataset          : '<ds>'
```

训练中每 5 epoch 输出一行 `[eval @ep5]  N=447  cls=10/11  all F1=0.64  mAP=0.65`；全部跑完最后打印 5 行 per-dataset summary。

**Step 3 — 产物可视化（每数据集独立做；以 endovis2018 为例）**

```bash
DS=endovis2018
OUT=/data/pretrained/siglip_train/affinity_per_ds/$DS
mkdir -p $OUT/vis_loc

# 3a) 每张图前 4 个 GT 类的 affinity 热图叠加
python tools/vis_siglip_locmap.py \
  --ckpt $OUT/affinity_$DS.pt \
  --manifest /data/pretrained/siglip_train/merged_val/manifest_${DS}_val.jsonl \
  --out-dir $OUT/vis_loc \
  --num-images 24 --top-classes 4 --classes-mode gt

# 3b) DINOv2 原生 patch 特征 t-SNE（不经投影，作对照基线）
python tools/vis_tsne_features.py --backbone dinov2 \
  --manifest /data/pretrained/siglip_train/manifest_${DS}.jsonl \
  --global-info /data/pretrained/siglip_train/merged/global_classes.json \
  --out $OUT/tsne_dinov2.png \
  --num-images 200 --patches-per-class 400

# 3c) 训过的 affinity-128D 投影特征 t-SNE
python tools/vis_tsne_features.py --backbone siglip-affinity \
  --ckpt $OUT/affinity_$DS.pt \
  --manifest /data/pretrained/siglip_train/manifest_${DS}.jsonl \
  --global-info /data/pretrained/siglip_train/merged/global_classes.json \
  --out $OUT/tsne_affinity.png \
  --num-images 200 --patches-per-class 400
```

### 输出与下游对接

每数据集的 `affinity_<ds>.pt` 字段：

```text
visual_proj      : MLP 768->256->128 state_dict
text_proj        : MLP 768->256->128 state_dict
T_anchor         : [C_d, 128] L2-norm 类锚点（已经过 text_proj）
log_tau          : scalar
cls_bias         : [C_d]
class_names      : 该数据集去除 background 后的原类名列表
orig_class_names : 完整原类名（含 background）
orig_to_new      : {orig_class_id: new_id}
vision_backbone  : 'dinov2'
dinov2_input_size: 518
dataset          : '<ds>'
```

Stage 2 分割训练加载方式：DINOv2 主干 frozen 共享，`visual_proj + T_anchor + log_tau + cls_bias` 作为该数据集专属的冻结侧支，主可训分支保持 UniMatch-V2 原有 DPT + boundary aux + debias 不变，新增一个 1x1 gate conv 学习 per-pixel 融合权重，最终 `L_final = (1-g)*L_DPT + g*upsample(prior)`。

### Stage 2 运行步骤（UniMatch-V2 分割训练 + Stage 1 侧支 + 自适应解冻）

Stage 2 已落地为 `affinity` 与 `affinity_min` 两个 variant，分别叠加 (debias + boundary + cls_aux + affinity 侧支) 与仅 (affinity 侧支)。代码改动覆盖 [unimatch_v2.py](C:/u2v_fresh/UniMatch-V2/unimatch_v2.py)（CLI / 侧支构造 / 融合 / 损失 / 自适应解冻 / 断点保存）、[util/affinity_side.py](C:/u2v_fresh/UniMatch-V2/util/affinity_side.py)（AffinitySideBranch 模块，含独立 frozen DINOv2 副本与 1×1 gate conv）、[scripts/train.sh](C:/u2v_fresh/UniMatch-V2/scripts/train.sh)、[scripts/test.sh](C:/u2v_fresh/UniMatch-V2/scripts/test.sh)。

**关键设计：单一 DINOv2、共享 patches、单方向融合、自适应解冻**

- **整个 Stage 2 只有一个 DINOv2** —— DPT 主分支的 backbone。AffinitySideBranch 不持有任何视觉编码器，仅在前向中接收**预先抽出的 patch tokens** `[B, N, 768]`（DPT 的 `forward(..., return_cls=True)` 直接返回 last-layer patches；comp_drop 路径也补丁为返回 *未* dropout 的 clean patches）。Stage 1 的 DINOv2 与 Stage 2 共享同一份权重并保持同步。
- DPT 输出 logits + 侧支输出 prior 通过**逐像素 gate** `g = σ(Conv1x1(visual_proj_latent))` 软融合：`L_final = (1-g·mask) · L_DPT + g·mask · L_prior`，其中 `mask` 为非背景类指示（背景通道不受 prior 影响）
- gate_conv bias 初始化为 -3，使 `σ(g) ≈ 0.05` 起步，前几个 epoch 主分支几乎独立训练
- **教师 EMA 不接 prior**：伪标签生成路径 `pred_u_w = model_ema(img_u_w)` 完全走 DPT，避免侧支噪声污染长尾伪标签
- 侧支的 visual_proj / T_anchor / log_tau / cls_bias **默认冻结**（text_proj 在 Stage 2 不再使用，因为 T_anchor 已在 Stage 1 离线 ensemble 完毕直接存于 ckpt）；当 val mIoU 在最近 `--affinity-plateau-window`(默认 5) 个 epoch 内变化幅度 < `--affinity-plateau-eps`(默认 0.001 即 0.1%) 且 `epoch >= --affinity-freeze-warmup`(默认 15) 时，**自动解冻**并加入 optimizer，新 param group lr = `cfg.lr × --affinity-unfreeze-lr-mult`(默认 1.0)。解冻后的视觉投影随 DPT 主干一起跟训，因为它们看到的就是同一组（漂移中的）DINOv2 patches，**特征空间始终对齐**
- 图像级 cls BCE aux loss 用 `--affinity-aux-weight`(默认 0.5) cosine 从 0 warmup 到目标值（持续 `--affinity-aux-warmup`(默认 5) ep）

**Step 4 — 主分割训练（5 个数据集 × 5 标注率 × 多 variant）**

```bash
# 单数据集示例（endoscapes_seg50，标注比例 0.25）
DS=endoscapes_seg50
RATE=0.25 BS=2 LR=5e-6 \
AFFINITY_WARMSTART=/data/pretrained/siglip_train/affinity_per_ds/$DS/affinity_$DS.pt \
sh scripts/train.sh 1 29505 affinity $DS

# 测试
RATE=0.25 BS=2 LR=5e-6 \
AFFINITY_WARMSTART=/data/pretrained/siglip_train/affinity_per_ds/$DS/affinity_$DS.pt \
sh scripts/test.sh affinity $DS

# 五数据集 × 五标注率 × 六 variant 全矩阵（base/debias/boundary/full/affinity_min/affinity）
for DS in endoscapes_seg50 cholecseg8k endovis2018 endovis2017_parts endovis2017_type; do
  for RATE in 0.10 0.20 0.25 0.30 0.50; do
    for V in base debias boundary full affinity_min affinity; do
      RATE=$RATE BS=2 LR=5e-6 \
      AFFINITY_WARMSTART=/data/pretrained/siglip_train/affinity_per_ds/$DS/affinity_$DS.pt \
      sh scripts/train.sh 1 $((29500+RANDOM%100)) $V $DS
    done
  done
done
```

`AFFINITY_WARMSTART` 是 Stage 1 产物路径。可选的 env vars 还有 `AFFINITY_AUX_WEIGHT`、`AFFINITY_FREEZE_WARMUP`、`AFFINITY_PLATEAU_EPS`、`AFFINITY_UNFREEZE_LR_MULT`，均透传到 `unimatch_v2.py` 对应 CLI。两个 variant 差异：

| variant | flags 等价于 | 用途 |
|---|---|---|
| `affinity_min` | (无 aux) + affinity 侧支 | 仅评估 prior 融合的纯增益 |
| `affinity`     | `--debias --boundary --cls-head` + affinity 侧支 | 与 full 拉齐做完整方法对比 |

**Step 5 — 训练监控**

输出目录 `$EXP_ROOT/<dataset>/unimatch_v2_<variant>_r<RATE>_bs<BS>_lr<LR>/`，含：

- `train_log.csv` — 每 epoch 一行，affinity 相关新列：`loss_aff`（aux BCE）、`gate_mean`（侧支 gate 全图平均）、`affinity_proj_unfrozen`（0/1 标志位）
- `out.log` — 实时日志，含每 1/8 epoch 的训练摘要，affinity variant 下额外打印 `Loss aff: X.XXX, gate: X.XXX`；侧支解冻时单行日志 `[affinity] mIoU plateaued at ep N → unfreezing proj with lr=X.XXe-X`
- `latest.pth` / `best.pth` — 含 `affinity_side` 子字典（visual_proj/text_proj/T_anchor/log_tau/cls_bias/gate_conv 全部 state_dict + `proj_unfrozen` 标志），断点恢复时自动重建侧支与对应 optimizer 状态

诊断信号：

- `gate_mean` 前 ~15 epoch 应稳定在 0.04~0.10；解冻后允许逐步上升到 0.15~0.30（说明主分支学到了在小目标/稀有类位置信任 prior）
- `loss_aff` 应单调下降；如果停滞在 0.6+ 说明侧支信号被 seg 主 loss 压制，可调大 `--affinity-aux-weight`
- `affinity_proj_unfrozen` 由 0 转 1 的 epoch 应在 15 与 30 之间；若到训练结束仍为 0，说明主分支 mIoU 还在持续提升（不需要侧支微调），是合理结果

**Step 6 — 训练曲线对比**

```bash
DS=endoscapes_seg50
python tools/plot_train_log.py \
  --run "base:$EXP_ROOT/$DS/unimatch_v2_base_r0.25_bs2_lr5e-6/train_log.csv" \
  --run "debias:$EXP_ROOT/$DS/unimatch_v2_debias_r0.25_bs2_lr5e-6/train_log.csv" \
  --run "full:$EXP_ROOT/$DS/unimatch_v2_full_r0.25_bs2_lr5e-6/train_log.csv" \
  --run "affinity:$EXP_ROOT/$DS/unimatch_v2_affinity_r0.25_bs2_lr5e-6/train_log.csv" \
  --out "$EXP_ROOT/$DS/curves_r0.25.png"
```

观察重点：`affinity` vs `full` 上 **tail keep_ratio**、**rare class IoU**、**gap_HT** 三件套；`gate_mean` 与 mIoU 的同期变化可判断 prior 何时开始发挥作用。

**Step 7 — 端到端推理（单图）**

```bash
DS=endoscapes_seg50
python tools/inference_single.py \
  --weights $EXP_ROOT/$DS/unimatch_v2_affinity_r0.25_bs2_lr5e-6/best.pth \
  --affinity-warmstart /data/pretrained/siglip_train/affinity_per_ds/$DS/affinity_$DS.pt \
  --image /path/to/test.jpg \
  --out  /path/to/pred.png
```

推理时只前向 DINOv2 一次：DPT 主分支与侧支共享同一组 patch tokens，侧支额外开销仅一次 patch×T_anchor 矩阵乘 + 一次 1×1 conv，约 +2% 计算开销，可忽略。输出 argmax mask + 可选 `--save-prior-heatmap` 单独导出每类 prior 热图供可视化。

## 10. 常见问题

### `--tangent` 单独使用报错

这是预期行为。tangent 依赖 boundary head。使用脚本 variant：

```bash
sh scripts/train.sh 1 29504 tangent endoscapes_seg50
```

或手写参数：

```bash
--boundary --tangent --tangent-weight 0.1
```

### checkpoint 找不到

确认测试时的 `RATE/BS/LR/EPOCHS/CROP` 与训练一致，因为这些变量会进入目录名。

### 想看实时日志

```bash
tail -F "$EXP_ROOT/endoscapes_seg50/unimatch_v2_debias_r0.25_bs2_lr5e-6/out.log"
```

### 解压后 `/data/code/` 散落 `unimatch_v2.py` 等文件

`scripts/package_and_upload.ps1` 旧版本（2026-05-19 21:39 之前）打的 zip 没有顶层
目录。用 `2026-05-19 21:39+` 重新打包即可。已发生散落时，按 §1 末尾的 `mv` 命令
把散到 `/data/code/` 根的文件收回 `UniMatch-V2/`。

