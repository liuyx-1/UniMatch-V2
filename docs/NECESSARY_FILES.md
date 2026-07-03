# 上传清单（必要文件 / 私有文件 / 排除项）

本文件汇总向 GitHub 仓库上传时**需要**哪些文件、哪些**保持私有**、哪些**直接排除**。
原则：**只发布推理与使用，不暴露训练/改进源码**；权重尽量以 TorchScript/TensorRT 引擎**封装**形式发布。

> 路径以服务器 `/root/autodl-tmp/code/` 为根；`UV2 =` `UniMatch-V2_local/`，`S2 =` `SAM2-Plus/`。

---

## ✅ Tier A — 需要上传（推理 + 使用 + 文档）

### A1. 文档与工程入口
- `UV2/docs/README.md`（上传后置为仓库根 `README.md`）
- `UV2/docs/OPERATION_MANUAL.md`
- `UV2/docs/NEEDLE_POSE_REGISTRATION.md`
- `UV2/docs/NECESSARY_FILES.md`（本文件）
- `UV2/docs/requirements-inference.txt`

### A2. 推理脚本（三版本 + 加速 + 导出）
- `UV2/realtime_stereo_keypoints.py`、`UV2/eval_pose_val.py`              （v1）
- `UV2/realtime_stereo_keypoints_v2.py`、`UV2/eval_pose_val_v2.py`        （v2）
- `UV2/realtime_stereo_keypoints_v3_accel.py`、`UV2/eval_pose_val_v3_accel.py`、`UV2/infer_accel.py`  （v3）
- `UV2/export_seg_engine.py`
- `UV2/infer_engine_only.py`（**引擎自包含推理入口**：不依赖 `test.py`/`model/`，封装发布首选）
- `S2/tools/needle_keypoints.py`（v1）、`S2/tools/needle_keypoints_v2.py`（v2）
- `S2/tools/calibrate_needle_radius.py`

### A3. 运行所需的最小模型/工具源码（推理依赖）
- `UV2/test.py`（`build_inference_model` / `infer_pred`）
- `UV2/model/`（**仅推理用到的 DINOv2 骨干 + DPT 头**；见下方“封装建议”——若发布引擎可不带本目录）
- `UV2/util/`（`utils.py`, `classes.py` 等推理依赖）
- `UV2/dataset/`（val 读取/transform 所需的最小子集）
- `UV2/configs/surgical_combined_base.yaml`

### A4. 标定与模型参数（小文件，必带）
- `S2/tools/needle_calib.json`（双目标定）
- `S2/tools/needle_model.json`（针规范半径，v2/v3 用）

### A5. 权重（见“封装建议”，二选一）
- 封装发布（推荐）：`exp/combined_r100_base/seg_trt_s640.ts` / `seg_trt_s512.ts`（TorchScript/TensorRT 引擎）
- 或原始权重：`exp/combined_r100_base/best.pth`（≈390 MB，需 Git LFS）

---

## 🔒 Tier B — 保持私有（不要上传：训练 / 改进源码 / 大权重）
- 训练脚本：`UV2/train_supervised_basic.py`、`UV2/train_supervised_smc.py`、`UV2/unimatch_v2.py`、`UV2/supervised.py`、`UV2/fixmatch.py`、`UV2/train_keypoints.py`
- 改进模块（自定义头：affinity/text、hyperbolic、CMA、edge/MGER、TS-MDR、TCR、cls aux 等）及其在 `UV2/model/`、`UV2/util/` 中的实现文件
- 预训练骨干：`UV2/pretrained/dinov2_vitb14_pretrain.pth`（上游权重，体积大，按需自行下载）
- 原始 `best.pth`（若已发布封装引擎，则不传原始权重，避免暴露可再训练的完整模型）
- 标注器训练相关：`S2/training/`、`S2/tools/train_first_frame_yolo.py` 等
- **`UV2/docs/` 内描述改进方法/论文的旧文档（务必排除）**：`METHOD.md`、`method_ieee.tex`、`method_smc.html`、`epla_lcpam_conflict_analysis.html`、`multiclass_segmentation_code_diagnostic.md`、`RUN.md`/`RUN.html`、`COMBINED_MULTICLASS.md`、`FULLSUP_INFER_STREAM_KEYPOINTS.html`、`speech_surgical_seg.html`、`*.csv`。
  > 上传时 `docs/` **只带本项目新增的 5 个**：`README.md`、`OPERATION_MANUAL.md`、`NEEDLE_POSE_REGISTRATION.md`、`NECESSARY_FILES.md`、`requirements-inference.txt`。

> **封装建议（已落地）**：用 `export_seg_engine.py` 把 `best.pth` 导成 TorchScript/TensorRT 引擎，再用
> **`infer_engine_only.py`** 推理——它**不导入** `test.py`/`model/`，所以发布时 **A3 的 `model/`、`test.py`、
> `dataset/` 全部可不上传**，使用者也拿不到网络结构与原始权重。最小可发布集即：
> `infer_engine_only.py` + `infer_accel.py` + `needle_keypoints_v2.py` + `<engine>.ts` + `needle_calib.json` + `needle_model.json` + 文档。
> （`realtime/eval_v*` 与 `export_seg_engine.py` 仍依赖 `model/`，若要发布它们则需带最小推理源码 A3，或仅作内部使用。）

---

## ❌ Tier C — 排除（与本项目无关）
- 其它项目目录：`PS-MT/`、`Surg-SegFormer/`、`EndoGSLAM/`、`Med3DVLM/`、`mean-teacher/`、`mmdetection/`、`co-tracker/`、`autonomous_surgery/`、`Endo-SemiS/`、`Min_Max_Similarity/`、`Semi-supervised-Segmentation/` 等
- 其它数据集与准备脚本：`UV2/tools/prepare_*.py`、`endovis*/cholecseg8k/endoscapes/coco/ade20k/...` 相关 config 与代码
- 论文/消融/绘图工具：`UV2/tools/make_*.py`、`plot_*.py`、`*ablation*`、`vis_*`、`build_*manifest*`（非缝针流程必需者）
- 虚拟环境、缓存、日志、原始视频与数据：`*/.venv/`、`__pycache__/`、`sam2_logs/`、`*.mp4`(原始)、`ROOT/` 下的图像/掩码数据本体
- 临时/中间产物：`exp/.../*.jsonl`、`*.csv`、`run_*.log`、`right_pred/`、`*_annot/`

---

## 上传后的目标仓库结构（建议）
```
<repo>/
  README.md                         ← 来自 docs/README.md
  requirements-inference.txt        ← 来自 docs/requirements-inference.txt
  docs/  OPERATION_MANUAL.md  NEEDLE_POSE_REGISTRATION.md  NECESSARY_FILES.md
  inference/                        ← A2 推理脚本（含 infer_accel.py, export_seg_engine.py）
  tools/   needle_keypoints.py  needle_keypoints_v2.py  calibrate_needle_radius.py
  calib/   needle_calib.json  needle_model.json
  configs/ surgical_combined_base.yaml
  model/   （可选；发布引擎则可省）
  weights/ seg_trt_s640.ts  seg_trt_s512.ts   （或 best.pth + Git LFS）
```
> 实际目录命名可调整；脚本内的相对 import（`from test import ...`、`--sam2-tools`、`load_nk`）需对应仓库布局做最小适配。

---

## 一句话总结
**上传 A 区（推理脚本 + 文档 + 标定/半径 json + 配置 + 封装引擎）**；
**封装权重而非裸 `best.pth`**；
**保留 B 区训练与改进源码私有**；**排除 C 区无关内容**。
