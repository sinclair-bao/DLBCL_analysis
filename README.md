# DLBCL 影像分析项目

弥漫大 B 细胞淋巴瘤（DLBCL）PET/CT 影像分析流程——从 PACS 原始 DICOM 到
AutoPET III 冠军模型病灶分割的全自动批处理管线。

---

## 目录

1. [技术路线概览](#1-技术路线概览)
2. [目录结构](#2-目录结构)
3. [环境配置](#3-环境配置)
4. [完整流程使用方法](#4-完整流程使用方法)
5. [各阶段详细说明与注意要点](#5-各阶段详细说明与注意要点)
   - [5.1 convert — DICOM 转 NIfTI + SUV](#51-convert--dicom-转-nifti--suv)
   - [5.2 preprocess — CT/PET 重采样与对齐](#52-preprocess--ctpet-重采样与对齐)
   - [5.3 export — nnU-Net 推理格式导出](#53-export--nnu-net-推理格式导出)
   - [5.4 segment — 病灶分割](#54-segment--病灶分割)
   - [5.5 qc — 分割结果质控可视化](#55-qc--分割结果质控可视化)
   - [5.6 analyze — 特征统计与绘图（占位）](#56-analyze--特征统计与绘图占位)
6. [脚本功能一览](#6-脚本功能一览)
7. [数据说明](#7-数据说明)
8. [Git 远程仓库](#8-git-远程仓库)

---

## 1. 技术路线概览

```
PACS 归档 DICOM
    │
    ▼  [convert]  pacs_dicom_to_nifti_suv.py
data/interim/<PatientID>/<StudyDate>/{CT,PET}/*.nii.gz
    │  · CT：原始 HU 值 NIfTI
    │  · PET：活度浓度图 _ACT.nii.gz + SUVbw 图 _SUVbw.nii.gz
    │
    ▼  [preprocess]  preprocess.py
data/interim/…/preprocessed/{CT,PET}/*.nii.gz
    │  · CT：各向同性重采样（默认 2mm）
    │  · PET：对齐到 CT 网格（同 shape/affine，nnU-Net 双通道硬性要求）
    │
    ▼  [export]  export_nnunet.py
data/nnunet_export/{PatientID}_{StudyDate}_0000.nii.gz   # CT
                   {PatientID}_{StudyDate}_0001.nii.gz   # PET(SUV)
    │  · nnU-Net 标准扁平命名（通道后缀 _0000/_0001）
    │  · 导出前校验 CT/PET 几何一致，不一致时报 error
    │
    ▼  [segment]  segmentation.py / infer_nnunet.py
data/processed/<PatientID>/<StudyDate>/masks/
    │  · {case}_lesion.nii.gz  —— AutoPET III nnU-Net 病灶掩码（uint8）
    │  · {case}.nii.gz          —— nnU-Net 原始多标签输出（可选保留）
    │  · {PET原名}_mask.nii.gz  —— SUV 阈值基线掩码（--method threshold 时）
    │
    ▼  [qc]  scripts/visualization/qc_segmentation.py
data/qc/<PatientID>/<PatientID>_<StudyDate>_qc.png
    │  · 2×3 面板（冠状面+矢状面 × PET MIP/Mask/Overlay）
    │  · 放射学方向（患者右侧在图像左侧）
    │
    ▼  [analyze]  plot_results.py（待实现）
results/tables/features.csv
results/figures/
```

**模型信息（segment/nnunet）：**

| 项目 | 说明 |
|------|------|
| 竞赛 | MICCAI 2024 AutoPET III Challenge — Model-Centric 类别**冠军** |
| 团队 | Team LesionTracer（DKFZ，Heidelberg） |
| 架构 | nnU-Net ResEncL + 双头输出（病灶 + 器官） |
| 训练数据 | AutoPET III 数据集，1611 例，多中心多示踪剂 |
| 权重路径 | `autoPET/Dataset222_AutoPETIII_2024/autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres_bs3/` |
| Fold 数 | 5-fold 集成推理 |
| 输入通道 | 0 = CT (HU)，1 = PET (SUV) |
| 输出标签 | 0 = background，1 = tumor |

---

## 2. 目录结构

```text
DLBCL/
├── main.py                          # 统一流程入口
├── README.md                        # 本文档
├── environment.yml                  # data-analysis conda 环境依赖
│
├── data/
│   ├── raw/                         # 原始 DICOM（Git 忽略）
│   ├── interim/                     # 转换 + 预处理中间产物（Git 忽略）
│   │   └── <PatientID>/<StudyDate>/
│   │       ├── CT/                  # DICOM→NIfTI，HU 值
│   │       ├── PET/                 # _ACT.nii.gz + _SUVbw.nii.gz
│   │       └── preprocessed/        # 重采样后，PET 已对齐到 CT 网格
│   │           ├── CT/
│   │           └── PET/
│   ├── nnunet_export/               # nnU-Net 推理输入（Git 忽略）
│   │   └── {PatientID}_{Date}_0000/0001.nii.gz
│   ├── processed/                   # 最终病灶掩码（Git 忽略）
│   │   └── <PatientID>/<StudyDate>/masks/
│   │       ├── {case}_lesion.nii.gz     # nnU-Net 病灶掩码
│   │       ├── {case}.nii.gz            # nnU-Net 原始输出
│   │       └── {PET原名}_mask.nii.gz    # SUV 阈值基线掩码（可选）
│   └── qc/                          # 分割质控图（Git 忽略，本地生成）
│       └── <PatientID>/<PatientID>_<StudyDate>_qc.png
│
├── autoPET/                         # 第三方模型（Git 忽略）
│   ├── autopet-3-submission-master/ # 模型代码（editable install 到 autopet 环境）
│   └── Dataset222_AutoPETIII_2024/  # 5-fold 预训练权重（~3.9GB）
│       └── autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres_bs3/
│           ├── fold_{0..4}/checkpoint_final.pth
│           ├── plans.json
│           └── dataset.json
│
├── scripts/
│   ├── common/
│   │   └── pipeline_utils.py        # 共用工具：StageResult / 目录发现 / 日志
│   ├── tools/
│   │   ├── pacs_dicom_to_nifti_suv.py  # DICOM → NIfTI + SUVbw（可复用）
│   │   └── dicom2suvmaps.py            # 早期原型（已被上者取代，保留备查）
│   ├── processing/
│   │   ├── preprocess.py            # CT/PET 重采样 + PET 对齐到 CT 网格
│   │   ├── export_nnunet.py         # 导出 nnU-Net 推理命名格式
│   │   ├── infer_nnunet.py          # AutoPET nnU-Net 推理（GPU）
│   │   └── segmentation.py          # 分割阶段统一入口（nnunet/threshold/both）
│   ├── visualization/
│   │   └── qc_segmentation.py       # 分割质控 MIP 图生成
│   └── analysis/
│       └── plot_results.py          # 特征绘图（占位，待实现）
│
├── results/
│   ├── figures/                     # 论文用图（Git 忽略大文件）
│   ├── tables/                      # 统计表（Git 忽略大文件）
│   └── models/                      # 自训练模型权重（Git 忽略）
│
├── logs/                            # 各阶段 CSV 日志
└── docs/                            # 实验记录、协议、数据字典
```

---

## 3. 环境配置

本项目使用 **两个 conda 环境**：

| 环境 | 用途 | 关键包 |
|------|------|--------|
| `data-analysis` | convert / preprocess / export / analyze | nibabel, SimpleITK, scipy, numpy, torch 2.13（可选） |
| `autopet` | segment（nnU-Net GPU 推理） | torch 2.5.1+cu124, nnunetv2 2.5.1（editable）, nibabel, SimpleITK |

### 3.1 data-analysis 环境

```bash
conda create -n data-analysis python=3.11 -y
conda activate data-analysis
conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia -y
pip install nibabel SimpleITK scipy scikit-image pandas matplotlib seaborn TotalSegmentator
```

### 3.2 autopet 环境

```bash
conda create -n autopet python=3.11 -y
conda activate autopet
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
cd autoPET/autopet-3-submission-master
pip install -e .
```

验证安装：

```bash
/home/sun/miniconda3/envs/autopet/bin/python -c "
import torch, nnunetv2
from nnunetv2.training.nnUNetTrainer.autoPET3_Trainer import autoPET3_Trainer
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
print('autoPET3_Trainer OK')
"
```

---

## 4. 完整流程使用方法

### 一键运行全流程

```bash
# 需在 autopet 环境（包含 GPU nnU-Net 推理）
/home/sun/miniconda3/envs/autopet/bin/python main.py --stage all \
    --dcm2niix-bin /home/sun/fsl/bin/dcm2niix

# 干跑，只查看计划不执行
/home/sun/miniconda3/envs/autopet/bin/python main.py --stage all --dry-run -v
```

### 分阶段运行

```bash
PYTHON=/home/sun/miniconda3/envs/autopet/bin/python

# 1. DICOM 转换
$PYTHON main.py --stage convert --dcm2niix-bin /home/sun/fsl/bin/dcm2niix

# 2. 预处理（CT 重采样 + PET 对齐）
$PYTHON main.py --stage preprocess --voxel-size 2.0

# 3. 导出 nnU-Net 格式
$PYTHON main.py --stage export

# 4. 分割（默认 nnU-Net，需 GPU）
$PYTHON main.py --stage segment

# 4b. 仅 SUV 阈值基线（无需 GPU，任意环境可用）
python main.py --stage segment --segment-method threshold

# 4c. 两种方法并行（对比评估）
$PYTHON main.py --stage segment --segment-method both

# 5. 生成分割质控图（可选，data-analysis 环境）
conda run -n data-analysis python scripts/visualization/qc_segmentation.py
```

### 调试单个病例

```bash
PYTHON=/home/sun/miniconda3/envs/autopet/bin/python

# 对单个病例跑全部阶段
$PYTHON scripts/processing/preprocess.py --patient-id 00857723 --study-date 20180905 -v
$PYTHON scripts/processing/export_nnunet.py --patient-id 00857723 --study-date 20180905 -v
$PYTHON scripts/processing/infer_nnunet.py --patient-id 00857723 --study-date 20180905 -v
$PYTHON scripts/processing/segmentation.py --method nnunet --patient-id 00857723 -v
```

### 常用参数

| 参数 | 作用 | 默认值 |
|------|------|--------|
| `--stage` | 运行阶段（convert/preprocess/export/segment/analyze/all） | all |
| `--voxel-size` | 预处理目标体素间距（mm） | 2.0 |
| `--segment-method` | 分割方法（nnunet/threshold/both） | nnunet |
| `--threshold-mode` | 阈值模式（absolute/relative） | absolute |
| `--threshold` | 阈值数值（absolute: SUV g/mL；relative: 占 SUVmax 比例） | 2.5 |
| `--folds` | nnU-Net 使用的 fold 编号，多个用空格分隔 | 0 1 2 3 4 |
| `--device` | 推理设备（cuda/cpu） | cuda |
| `--overwrite` | 强制重新处理已有输出 | False |
| `--dry-run` | 只打印计划，不实际执行 | False |
| `-v` / `--verbose` | 输出 DEBUG 日志 | False |

---

## 5. 各阶段详细说明与注意要点

### 5.1 convert — DICOM 转 NIfTI + SUV

**脚本：** `scripts/tools/pacs_dicom_to_nifti_suv.py`

**功能：** 批量将 PACS 自动归档的 DICOM 序列转换为 NIfTI，PET 序列额外换算为 SUVbw。

**输出结构：**
```
data/interim/<PatientID>/<StudyDate>/
    CT/   s3_WB_Standard.nii.gz
    PET/  s12_WB_3D_MAC_ACT.nii.gz       # 活度浓度原图（Bq/mL）
          s12_WB_3D_MAC_SUVbw.nii.gz     # SUVbw 图（优先用于后续步骤）
```

**注意要点：**
- 依赖 `dcm2niix`（FSL 自带：`/home/sun/fsl/bin/dcm2niix`）
- DICOM 目录须符合 PACS 三级结构 `Patient/Study/Series/Image`；IM 文件无扩展名但为标准 DICOM Part 10
- SUV 换算需要 DICOM 头包含完整的注射剂量/体重/扫描时间信息；前提不满足时仅输出 `_ACT.nii.gz` 并记录 warning
- 默认源目录 `data/raw/DICOM*`（glob 展开），可通过 `--source` 指定多个路径

```bash
python scripts/tools/pacs_dicom_to_nifti_suv.py \
    --source data/raw/DICOM \
    --output-root data/interim \
    --dcm2niix-bin /home/sun/fsl/bin/dcm2niix -v
```

---

### 5.2 preprocess — CT/PET 重采样与对齐

**脚本：** `scripts/processing/preprocess.py`

**功能：**
1. CT → `resample_to_output`（各向同性，默认 2mm，三次样条插值 order=3）
2. PET → `resample_from_to`（对齐到该 Study 参考 CT 的 shape/affine，线性插值 order=1）

**输出结构：**
```
data/interim/<PatientID>/<StudyDate>/preprocessed/
    CT/   s3_WB_Standard.nii.gz       # 2mm 各向同性
    PET/  s12_WB_3D_MAC_SUVbw.nii.gz  # 与 CT 同 shape/affine
```

**注意要点：**
- **PET 必须对齐到 CT 网格**，nnU-Net 双通道输入要求两通道 shape/affine 完全一致；使用 `resample_from_to` 而非各自独立重采样，这是与旧版代码的关键区别
- 无 CT 的 Study（仅 PET）会退化为独立各向同性重采样并记录 warning，后续 nnU-Net 推理不可用
- 建议 `--voxel-size 2.0`（原始体素间距约 3mm×2mm×2mm，2mm 是合理折衷）
- 重跑时加 `--overwrite`，否则已对齐的旧文件会被跳过

```bash
# 全量预处理
python scripts/processing/preprocess.py --voxel-size 2.0

# 重跑单个病例（含覆盖旧结果）
python scripts/processing/preprocess.py --patient-id 00857723 --study-date 20180905 --overwrite -v
```

---

### 5.3 export — nnU-Net 推理格式导出

**脚本：** `scripts/processing/export_nnunet.py`

**功能：** 将 `preprocessed/{CT,PET}` 以 nnU-Net 标准命名导出到扁平目录。

**输出：**
```
data/nnunet_export/
    00857723_20180905_0000.nii.gz   # CT  → channel 0
    00857723_20180905_0001.nii.gz   # PET → channel 1
```

**注意要点：**
- 导出前自动校验 CT/PET shape 和 affine 是否一致；不一致时报 error 并提示重跑 preprocess
- 优先使用 hardlink（节省磁盘），失败时 fallback 到 copy
- 若 study 缺少 preprocessed CT 或 PET 任意一方，记录 warning 并跳过
- `data/nnunet_export/` 已加入 `.gitignore`，不会被纳入版本控制

```bash
python scripts/processing/export_nnunet.py
python scripts/processing/export_nnunet.py --patient-id 00857723 --overwrite -v
```

---

### 5.4 segment — 病灶分割

**脚本：** `scripts/processing/segmentation.py`（统一入口）  
**底层脚本：** `scripts/processing/infer_nnunet.py`（nnU-Net 推理）

#### 方法 A：nnU-Net（默认，推荐）

调用 AutoPET III 冠军模型 5-fold 集成推理。

**输出：**
```
data/processed/<PatientID>/<StudyDate>/masks/
    {PatientID}_{StudyDate}.nii.gz          # nnU-Net 原始输出（labels: 0=bg, 1=tumor）
    {PatientID}_{StudyDate}_lesion.nii.gz   # 二值病灶掩码（uint8）
```

**注意要点：**
- **必须使用 `autopet` conda 环境**（含 torch+CUDA + nnunetv2 editable）
- 推理通过子进程调用 `/home/sun/miniconda3/envs/autopet/bin/nnUNetv2_predict_from_modelfolder`，不依赖当前 Python 环境
- RTX 4090 D 24GB 显存，单病例 5-fold 集成约 **1.5 分钟**（patch 192³，滑动窗口）
- 推理时 CUDA 显存峰值约 8–12GB，其他程序占用过多时可先释放
- 无需手动重采样到训练集 spacing，nnU-Net 推理内部按 `plans.json` 自动重采样
- 若只想用部分 fold 加速（降低精度），使用 `--folds 0 1 2`

```bash
PYTHON=/home/sun/miniconda3/envs/autopet/bin/python

# 推理所有已导出的 case
$PYTHON scripts/processing/segmentation.py --method nnunet

# 推理单个 case（--overwrite 强制重跑）
$PYTHON scripts/processing/segmentation.py --method nnunet \
    --patient-id 00857723 --study-date 20180905 --overwrite -v

# 仅用 fold 0（最快，测试用）
$PYTHON scripts/processing/segmentation.py --method nnunet --folds 0 -v
```

#### 方法 B：SUV 阈值基线

无需 GPU，适合快速验证或无 GPU 场景。

**输出：**
```
data/processed/<PatientID>/<StudyDate>/masks/
    {PET原始名}_mask.nii.gz    # 二值掩码（uint8，1=病灶，0=背景）
```

**注意要点：**
- `--threshold-mode absolute`（默认 2.5 g/mL）：PET 肿瘤学文献最常用固定阈值
- `--threshold-mode relative`（建议 0.41）：41% SUVmax，适合摄取较低的病灶
- 基线精度低于 nnU-Net，主要用于对比评估或调试

```bash
python scripts/processing/segmentation.py --method threshold
python scripts/processing/segmentation.py --method threshold \
    --threshold-mode relative --threshold 0.41

# 两种方法并行
$PYTHON scripts/processing/segmentation.py --method both
```

---

### 5.5 qc — 分割结果质控可视化

**脚本：** `scripts/visualization/qc_segmentation.py`

**功能：** 为每个完成 nnU-Net 分割的病例生成全身 PET MIP 质控图，便于快速审查分割结果的合理性。

**输出：** `data/qc/<PatientID>/<PatientID>_<StudyDate>_qc.png`

每张图为 **2×3 面板**（约 1.2MB PNG，120 dpi）：

|  | 列 1：PET MIP | 列 2：Lesion Mask MIP | 列 3：叠加图 |
|--|---|---|---|
| **行 1（冠状面）** | 临床伪彩 PET 全身 MIP（AP 方向投影） | 白色病灶轮廓 | 红色半透明病灶叠加在 PET MIP |
| **行 2（矢状面）** | 同上（RL 方向投影） | 同上 | 同上 |

- **显示方向**：放射学惯例，患者右侧在图像左侧，头在上
- **SUV 显示范围**：0–6 SUVbw（可用 `--suv-max` 调整）
- 顶部标注：患者 ID、检查日期、病灶体素数、SUV 原始最大值
- **增量执行**：已有图跳过，`--overwrite` 强制重新生成

```bash
# data-analysis 环境运行
# 生成全部病例
conda run -n data-analysis python scripts/visualization/qc_segmentation.py

# 指定病例（快速测试）
conda run -n data-analysis python scripts/visualization/qc_segmentation.py \
    --patient-id 00136597 --study-date 20220425

# 覆盖重新生成 + 自定义 SUV 上限
conda run -n data-analysis python scripts/visualization/qc_segmentation.py \
    --overwrite --suv-max 8.0
```

> **注意：** `data/qc/` 已加入 `.gitignore`，QC 图不入版本库。

---

### 5.6 analyze — 特征统计与绘图（占位）

**脚本：** `scripts/analysis/plot_results.py`

**当前状态：** 占位实现，调用 `plot_feature_distributions()` 会抛出 `NotImplementedError`，主流程会记录 warning 而非报错，不影响前序阶段。

**待实现内容：**
- 从 `data/processed/` 的病灶掩码 + 影像汇总出 PET/CT 定量特征（SUVmax、SUVmean、MTV、TLG 等）
- 写出 `results/tables/features.csv`
- 调用 `plot_feature_distributions()` 生成箱线图/小提琴图等

---

## 6. 脚本功能一览

### `main.py` — 流程统一入口

编排 `convert → preprocess → export → segment → analyze` 五个阶段，支持按阶段单独运行。所有阶段均为增量式（输出已存在则跳过），单个病例失败不影响其他病例。

```bash
/home/sun/miniconda3/envs/autopet/bin/python main.py --help
```

---

### `scripts/common/pipeline_utils.py` — 共用工具

| 功能 | 说明 |
|------|------|
| `StageResult` | 统一结果数据结构（stage/patient_id/study_date/status/output_path/message） |
| `discover_subject_studies()` | 遍历 `data/interim/<PatientID>/<StudyDate>/` 两级目录，返回三元组列表 |
| `write_stage_log_csv()` | 将阶段结果写入 CSV 日志 |
| `summarize()` | 统计 ok/skipped/warning/error 数量 |
| `setup_logging()` | 初始化日志（支持 verbose DEBUG 模式） |

---

### `scripts/tools/pacs_dicom_to_nifti_suv.py` — PACS DICOM 转换

| 项目 | 说明 |
|------|------|
| 输入 | PACS 三级 DICOM 归档目录（或 glob 模式） |
| 输出 | `data/interim/<PatientID>/<StudyDate>/<Modality>/` 下的 NIfTI（CT: HU，PET: _ACT + _SUVbw） |
| 核心类 | `PacsDicomToNiftiSuvConverter` |
| 依赖 | dcm2niix（`/home/sun/fsl/bin/dcm2niix`） |
| 可复用性 | 核心逻辑封装为类，可 import 到其他项目使用 |

SUVbw 换算公式：

```
SUVbw = ActivityConcentration(Bq/mL) × BodyWeight(g) / InjectedDose(Bq)
      × decay_correction_factor
```

---

### `scripts/tools/dicom2suvmaps.py` — 早期原型（已废弃）

早期的 DICOM → NIfTI 原型脚本，已被 `pacs_dicom_to_nifti_suv.py` 取代，保留备查。

---

### `scripts/processing/preprocess.py` — CT/PET 重采样

| 项目 | 说明 |
|------|------|
| 核心类 | `ImagePreprocessor` |
| CT 插值 | 三次样条（order=3），更平滑，HU 值允许小幅过冲 |
| PET 插值 | 线性（order=1），避免 SUV 热点周围振铃 |
| **关键：PET 对齐** | `resample_from_to(pet, ct_shape_affine)` 而非各自独立重采样 |
| 筛选功能 | `--patient-id` / `--study-date` 参数可只处理指定病例 |

---

### `scripts/processing/export_nnunet.py` — nnU-Net 格式导出

| 项目 | 说明 |
|------|------|
| 核心类 | `NnuNetExporter` |
| 命名规则 | `{PatientID}_{StudyDate}_0000.nii.gz`（CT），`_0001.nii.gz`（PET） |
| 几何校验 | 导出前用 nibabel 比对 shape 和 affine（atol=1e-5），不一致则 error |
| 磁盘优化 | 优先 `os.link`（hardlink），失败时 `shutil.copy2` |

---

### `scripts/processing/infer_nnunet.py` — AutoPET nnU-Net 推理

| 项目 | 说明 |
|------|------|
| 核心类 | `NnuNetInferrer` |
| 推理方式 | 子进程调用 autopet 环境 CLI，隔离 Python 环境依赖 |
| 模型 | AutoPET III LesionTracer，5-fold 集成 |
| 输入 | `data/nnunet_export/` 中的 `_0000/_0001.nii.gz` 对 |
| 输出 | `data/processed/.../masks/{case}.nii.gz`（原始）+ `{case}_lesion.nii.gz`（二值） |
| `extract_lesion_mask()` | 从多标签输出提取 label=1，写成 uint8 二值 NIfTI |
| 临时目录 | 每次推理使用 `tempfile.TemporaryDirectory`，推理完自动清理 |

---

### `scripts/processing/segmentation.py` — 分割阶段统一入口

| 类 | 说明 |
|----|------|
| `LesionSegmenter` | 统一入口，通过 `method` 参数路由到不同后端 |
| `NnunetSegmenter` | 委托 `infer_nnunet.NnuNetInferrer`，nnU-Net GPU 推理 |
| `ThresholdSegmenter` | SUV 阈值基线，直接对 PET numpy 数组操作，无 GPU 要求 |
| `threshold_suv_mask()` | 独立函数，可直接 import 用于自定义分割流程 |

三种模式：

| `--method` | 说明 |
|------------|------|
| `nnunet`（默认）| 使用 AutoPET III 模型，需 GPU + autopet 环境 |
| `threshold` | SUV 阈值基线，任意环境可用 |
| `both` | 两者并行，输出不同文件名，便于对比 |

---

### `scripts/analysis/plot_results.py` — 特征绘图（占位）

`plot_feature_distributions(table_csv, figure_out)` 当前抛出 `NotImplementedError`，
待特征提取脚本实现后替换为真实逻辑。

---

### `scripts/visualization/qc_segmentation.py` — 分割质控可视化

| 项目 | 说明 |
|------|------|
| 核心函数 | `make_qc_figure()` 生成单病例 2×3 面板 PNG |
| 批量入口 | `run()` 扫描全部含 `_lesion.nii.gz` 的病例，增量生成 |
| 输入 | preprocessed PET（`data/interim/.../preprocessed/PET/`）+ lesion mask |
| 输出 | `data/qc/<PatientID>/<PatientID>_<StudyDate>_qc.png` |
| 面板布局 | 2 行（冠状/矢状）× 3 列（PET MIP / Mask MIP / Overlay） |
| 显示方向 | 放射学惯例：患者右→图像左，头朝上 |
| 颜色方案 | 临床 PET 伪彩色（黑→紫→蓝→绿→黄→红→白），病灶红色半透明叠加 |
| 主要参数 | `--suv-max`（显示上限，默认 6.0）/ `--overwrite` / `--patient-id` |

---

## 7. 数据说明

- 原始 DICOM 批次位于 `data/raw/`（`DICOM`、`DICOMDIS`–`DICOMDIY` 等）
- 所有数据目录（`data/raw/`、`data/interim/`、`data/processed/`、`data/nnunet_export/`、`data/qc/`）均已加入 `.gitignore`，请勿强制添加上传
- `autoPET/` 第三方模型目录（~4GB）同样已被 `.gitignore` 忽略
- 中间结果可安全删除后重新生成（所有阶段均为增量幂等）

**本地数据集运行结果（截至 2026-08-14）：**

| 阶段 | 结果 |
|------|------|
| preprocess | 318 studies 完成，CT/PET shape 全部对齐（2mm 各向同性） |
| export | 310 cases 导出（8 studies 因缺 CT 或 PET 跳过） |
| segment (nnunet) | 310/310 lesion mask 生成完毕，零失败 |
| qc | 310 张质控 PNG 生成完毕（`data/qc/`） |

---

## 8. Git 远程仓库

| 远程 | 地址 |
|------|------|
| GitHub（origin） | `git@github.com:sinclair-bao/DLBCL_analysis.git` |
| Gitea（内网） | `ssh://git@100.100.211.88:2222/sinclair/DLBCL_analysis.git` |

推送到两个远程：

```bash
git push origin main && git push gitea main
```
