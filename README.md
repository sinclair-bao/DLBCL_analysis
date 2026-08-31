# DLBCL 影像分析项目

弥漫大 B 细胞淋巴瘤（DLBCL）PET/CT 影像分析流程——从 PACS 原始 DICOM 到
AutoPET III 冠军模型病灶分割，再到纵向随访浏览（基线病灶床映射、SUVmax / MTV / TLG、
影像组学）的批处理管线与桌面软件。

---

## 目录

1. [技术路线概览](#1-技术路线概览)
2. [目录结构](#2-目录结构)
3. [环境配置](#3-环境配置)
4. [完整流程使用方法](#4-完整流程使用方法)
5. [各阶段详细说明与注意要点](#5-各阶段详细说明与注意要点)
   - [5.1 convert — DICOM 转 NIfTI + SUV](#51-convert--dicom-转-nifti--suv)
   - [5.2 preprocess — CT/PET 重采样与对齐](#52-preprocess--ctpet-重采样与对齐)
   - [5.3 register — PET→CT 刚体配准（ANTs）](#53-register--petct-刚体配准ants)
   - [5.4 organ_seg — 器官分割（TotalSegmentator）](#54-organ_seg--器官分割totalsegmentator)
   - [5.5 export — nnU-Net 推理格式导出](#55-export--nnu-net-推理格式导出)
   - [5.6 segment — 病灶分割](#56-segment--病灶分割)
   - [5.7 qc — 分割结果质控可视化](#57-qc--分割结果质控可视化)
   - [5.8 analyze — 特征统计与绘图（占位）](#58-analyze--特征统计与绘图占位)
   - [5.9 longitudinal — 纵向分析与桌面软件](#59-longitudinal--纵向分析与桌面软件)
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
    ▼  [register]  register_pet_ct.py
data/processed/<PatientID>/<StudyDate>/registration/
    │  · pet_to_ct_0GenericAffine.mat  刚体变换矩阵（原始分辨率估计）
    │  · pet_orig_warped.nii.gz         原始 SUV 配准到 CT 空间（质控用）
    │  · pet_iso_aligned.nii.gz         2 mm SUV 配准后（影像组学工作图）
    │  · ct_iso_reference.nii.gz        2 mm CT 副本（影像组学固定图像）
    │
    ▼  [organ_seg]  organ_extraction/organ_segmentation.py
data/processed/<PatientID>/<StudyDate>/organs/organs.nii.gz
    │  · 11 类器官掩码（0=BG, 1=脾, 2=肾, 3=肝, 4=膀胱, 5=肺,
    │    6=脑, 7=心脏, 8=胃, 9=前列腺, 10=头颈腺体）
    │  · 输入 CT 为 data/interim 原始分辨率（无需 preprocess）
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
    ▼  [longitudinal]  scripts/longitudinal/ + scripts/gui/app.py
data/processed/<PatientID>/longitudinal_session.json
data/processed/<PatientID>/<FollowupDate>/longitudinal/
    │  · GUI 手动指定 baseline / interim / end
    │  · CT→CT 刚体+仿射，把基线病灶床映射到随访网格
    │  · native_lesion vs baseline_mapped：SUVmax / MTV / TLG + 组学
    │
    ▼  [analyze]  plot_results.py（队列分布图仍占位）
results/tables/longitudinal_features.csv
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
│   ├── processed/                   # 各阶段输出结果（Git 忽略）
│   │   └── <PatientID>/<StudyDate>/
│   │       ├── registration/        # ANTs PET-CT 配准产物
│   │       │   ├── pet_to_ct_0GenericAffine.mat   # 刚体变换矩阵
│   │       │   ├── pet_to_ct_affine.txt            # 人可读变换参数
│   │       │   ├── pet_orig_warped.nii.gz          # 原始空间 QC 图
│   │       │   ├── pet_iso_aligned.nii.gz          # 2mm 对齐后 PET
│   │       │   └── ct_iso_reference.nii.gz         # 2mm CT 参考图像
│   │       ├── organs/              # TotalSegmentator 器官分割
│   │       │   └── organs.nii.gz   # 11 类器官标签
│   │       ├── masks/               # 病灶分割掩码
│   │       │   ├── {case}_lesion.nii.gz     # nnU-Net 二值病灶掩码
│   │       │   ├── {case}.nii.gz            # nnU-Net 原始输出
│   │       │   └── {PET原名}_mask.nii.gz    # SUV 阈值基线掩码（可选）
│   │       └── longitudinal/        # 跨检查 CT–CT 映射（随访目录）
│   │           ├── baseline_<BL>_to_this_0GenericAffine.mat
│   │           └── baseline_lesion_warped.nii.gz
│   │   └── <PatientID>/
│   │       ├── longitudinal_session.json    # GUI 指定的 baseline/interim/end
│   │       └── longitudinal_features.csv    # 该患者代谢 + 组学表
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
│   │   ├── register_pet_ct.py       # ANTs PET→CT 刚体配准（原始+2mm 两步）
│   │   ├── organ_extraction/
│   │   │   └── organ_segmentation.py   # TotalSegmentator 11 类器官分割
│   │   ├── export_nnunet.py         # 导出 nnU-Net 推理命名格式
│   │   ├── infer_nnunet.py          # AutoPET nnU-Net 推理（GPU）
│   │   └── segmentation.py          # 分割阶段统一入口（nnunet/threshold/both）
│   ├── visualization/
│   │   └── qc_segmentation.py       # 分割质控 MIP 图生成
│   ├── longitudinal/                # 跨时间点映射 + 代谢/组学
│   │   ├── catalog.py               # 索引 2 mm CT/PET、mask、同机配准
│   │   ├── session.py               # baseline/interim/end 会话 JSON
│   │   ├── ants_runner.py           # ANTs 单线程子进程封装
│   │   ├── interscan_register.py    # 基线 CT → 随访 CT 仿射 + mask 映射
│   │   ├── features.py              # SUVmax/MTV/TLG + pyradiomics
│   │   └── radiomics_params.yaml    # 组学：shape / firstorder / GLCM
│   ├── gui/                         # PySide6 纵向浏览软件
│   │   ├── app.py                   # 入口：python scripts/gui/app.py
│   │   ├── main_window.py           # 总布局、菜单导出
│   │   ├── patient_browser.py       # 患者树与完整性状态灯
│   │   ├── timepoint_panel.py       # 三个角色下拉框 + 映射按钮
│   │   ├── edit_panel.py            # AutoPET / 阈值 / 空白 / 载入 mask
│   │   ├── segment_editor.py        # 3×3 CT/PET/融合 × 轴/冠/矢 编辑弹窗
│   │   ├── mask_ops.py              # 连通域编号、二维画笔、膨胀腐蚀
│   │   ├── display_utils.py         # CT/PET/融合、窗宽窗位、显示坐标
│   │   ├── volume_io.py             # 读入编号 mask；另存 lesion_edited
│   │   ├── ortho_viewer.py          # 轴/冠/矢 + 窗宽窗位 + 十字线 + 画笔
│   │   ├── evolution_strip.py       # 三时间点冠状 MIP
│   │   ├── feature_panel.py         # 特征表与折线
│   │   └── workers.py               # QThread：映射 / 组学 / AutoPET
│   └── analysis/
│       └── plot_results.py          # 队列特征分布图（占位）
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
| `data-analysis` | convert / preprocess / register / organ_seg / longitudinal GUI / export / analyze | nibabel, SimpleITK, scipy, numpy, TotalSegmentator, ANTs, PySide6, pyqtgraph, pyradiomics |
| `autopet` | segment（nnU-Net GPU 推理） | torch 2.5.1+cu124, nnunetv2 2.5.1（editable）, nibabel, SimpleITK |

### 3.1 data-analysis 环境

```bash
conda create -n data-analysis python=3.11 -y
conda activate data-analysis
conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia -y
pip install nibabel SimpleITK scipy scikit-image pandas matplotlib seaborn TotalSegmentator
pip install PySide6 pyqtgraph
# pyradiomics 源码包需关闭 build isolation
pip install versioneer pykwalify
pip install --no-build-isolation pyradiomics
```

> **ANTs** 独立安装于 `/home/sun/ants-2.6.5/`，无需 conda 管理。
> 运行 `register_pet_ct.py` 时脚本自动定位该路径，无需手动配置 PATH。

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

### 3.3 TotalSegmentator 权重

首次运行前需下载模型权重到 `~/.totalsegmentator/`（约 3GB）：

```bash
# 若网络不稳定，手动下载后解压
totalseg_download_weights -t total
totalseg_download_weights -t head_glands_cavities
```

解压目标目录结构：
```
~/.totalsegmentator/nnunet/results/
    Dataset291_TotalSegmentator_part1_organs_1559subj/
    Dataset292_TotalSegmentator_part2_vertebrae_1532subj/
    ...
    Dataset300_TotalSegmentator_head_glands_cavities_1.0/
```

---

## 4. 完整流程使用方法

### 分阶段运行

```bash
DA=/home/sun/miniconda3/envs/data-analysis/bin/python
AP=/home/sun/miniconda3/envs/autopet/bin/python

# 1. DICOM 转换（需指定 dcm2niix 路径）
$DA scripts/tools/pacs_dicom_to_nifti_suv.py \
    --source data/raw/DICOM \
    --output-root data/interim \
    --dcm2niix-bin /home/sun/fsl/bin/dcm2niix

# 2. 预处理（CT 重采样 + PET 对齐）
$DA scripts/processing/preprocess.py --voxel-size 2.0

# 3. PET-CT 刚体配准（ANTs，data-analysis 环境）
$DA scripts/processing/register_pet_ct.py          # 全部时间点
$DA scripts/processing/register_pet_ct.py --first-study-only   # 仅每患者首次

# 4. 器官分割（TotalSegmentator，data-analysis 环境，建议 nohup 后台运行）
nohup $DA scripts/processing/organ_extraction/organ_segmentation.py \
    --totalseg-device gpu \
    > logs/organ_seg_run.log 2>&1 &

# 5. 导出 nnU-Net 格式
$AP scripts/processing/export_nnunet.py

# 6. 病灶分割（AutoPET III nnU-Net，autopet 环境，需 GPU）
$AP scripts/processing/segmentation.py --method nnunet

# 6b. 仅 SUV 阈值基线（无需 GPU）
$DA scripts/processing/segmentation.py --method threshold

# 7. 生成分割质控图（可选）
$DA scripts/visualization/qc_segmentation.py

# 8. 纵向浏览软件（指定基线/中期/末期，映射病灶床，计算 SUVmax/MTV/TLG）
$DA scripts/gui/app.py
```

### 调试单个病例

```bash
DA=/home/sun/miniconda3/envs/data-analysis/bin/python
AP=/home/sun/miniconda3/envs/autopet/bin/python

$DA scripts/processing/preprocess.py     --patient-id 00857723 --study-date 20180905 -v
$DA scripts/processing/register_pet_ct.py --patient-ids 00857723 -v
$DA scripts/processing/organ_extraction/organ_segmentation.py \
                                          --patient-id 00857723 --study-date 20180905 -v
$AP scripts/processing/export_nnunet.py  --patient-id 00857723 --study-date 20180905 -v
$AP scripts/processing/segmentation.py  --method nnunet --patient-id 00857723 -v

# 纵向：指定一对日期做基线病灶床映射，再提取代谢参数
$DA scripts/longitudinal/interscan_register.py \
    --patient-id 00136597 --baseline 20220425 --followup 20220728 -v
$DA scripts/longitudinal/features.py --patient-id 00136597 --no-radiomics
```

### 常用参数（各脚本通用）

| 参数 | 作用 | 默认值 |
|------|------|--------|
| `--overwrite` | 强制重新处理已有输出 | False |
| `--dry-run` | 只打印计划，不实际执行 | False |
| `-v` / `--verbose` | 输出 DEBUG 日志 | False |
| `--patient-id` | 只处理指定患者 | 全部 |
| `--study-date` | 只处理指定日期（需同时指定患者） | 全部 |

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

### 5.3 register — PET→CT 刚体配准（ANTs）

**脚本：** `scripts/processing/register_pet_ct.py`

**背景：**
PET/CT 同机采集但分段扫描，呼吸运动和胃肠道蠕动会造成 PET 与 CT 的空间不匹配，
尤其在肺底、膈肌、腹部等区域较明显。纯仿射重采样（preprocess.py）只对齐几何网格，
无法校正强度驱动的残余偏差。`register_pet_ct.py` 在原始分辨率图像上用 ANTs 求解
刚体变换，再将变换复用到 2mm 各向同性图，最小化插值误差。

**两步策略：**
1. **原始分辨率配准**：原始 CT（~1.4mm×1.4mm×3.3mm）为 fixed，原始 SUVbw 为 moving，
   ANTs Rigid + 互信息（MI）度量，得到 `.mat` 变换文件
2. **变换应用**：将同一 `.mat` 应用到已重采样的 2mm SUVbw，以 2mm CT 为参考网格，
   输出 `pet_iso_aligned.nii.gz`

**输出结构：**
```
data/processed/<PatientID>/<StudyDate>/registration/
    pet_to_ct_0GenericAffine.mat   ANTs 刚体变换（PET 空间 → CT 空间）
    pet_to_ct_affine.txt           可读文本：平移量（mm）+ 旋转角（rad）
    pet_orig_warped.nii.gz         原始 SUV 配准到原始 CT（质控，视觉检查偏差）
    pet_iso_aligned.nii.gz         2mm SUV 已配准对齐（影像组学工作图像）
    ct_iso_reference.nii.gz        2mm CT 副本（固定图像，与 pet_iso_aligned 一一对应）
```

**注意要点：**
- 依赖 ANTs 二进制文件（`/home/sun/ants-2.6.5/bin/`），脚本自动定位，无需设置 PATH
- 配准在**原始分辨率**（原始 CT 约 512×512×263，PET 约 192×192×263）上进行，
  信息更丰富，变换估计更准确
- 使用 `--float 1`（float32）以节省内存并避免极端值引起的数值不稳定
- 子进程环境自动限制为单线程（`ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1`），
  防止与 NumPy/OpenMP 产生 SIGFPE 竞争崩溃
- 若 ANTs 因信号终止（exit < 0），自动以保守 3 尺度参数重试一次
- 每例配准耗时约 **5~6 分钟**（RTX 4090 服务器，CPU-only ANTs），建议 `nohup` 后台运行

```bash
# 全部患者，每人取最早一个 study
nohup python scripts/processing/register_pet_ct.py --first-study-only \
    > logs/register_run.log 2>&1 &

# 全部患者全部时间点
nohup python scripts/processing/register_pet_ct.py \
    > logs/register_all.log 2>&1 &

# 指定患者试跑（逗号分隔）
python scripts/processing/register_pet_ct.py \
    --patient-ids 00136597,00500538,00555598 --first-study-only -v

# 干跑（只显示计划）
python scripts/processing/register_pet_ct.py --dry-run -v
```

---

### 5.4 organ_seg — 器官分割（TotalSegmentator）

**脚本：** `scripts/processing/organ_extraction/organ_segmentation.py`

**功能：**
使用 [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) 对原始分辨率 CT
进行全身器官分割，将多个解剖结构合并为 **11 类统一标签**（与 AutoPET III 双头模型设计对齐），
用于后续器官掩码分析及模型对齐验证。

**标签定义（与 autoPET3 原始脚本一致）：**

| 标签 | 器官 | TotalSeg 原始 ID |
|------|------|-----------------|
| 0 | Background | — |
| 1 | Spleen（脾） | 1 |
| 2 | Kidney（肾，左右合并） | 2, 3 |
| 3 | Liver（肝） | 5 |
| 4 | Urinary bladder（膀胱） | 21 |
| 5 | Lung（肺，五叶合并） | 10–14 |
| 6 | Brain（脑） | 90 |
| 7 | Heart（心脏） | 51 |
| 8 | Stomach（胃） | 6 |
| 9 | Prostate（前列腺） | 22 |
| 10 | Head/neck glands（头颈腺体，双侧腮腺+颌下腺） | 6–9（head_glands task） |

**输出结构：**
```
data/processed/<PatientID>/<StudyDate>/organs/
    organs.nii.gz    # 11 类器官标签（uint8），与原始 CT 同 affine
```

**注意要点：**
- **输入为原始 CT**（`data/interim/.../CT/`），不依赖 preprocess 阶段
- 每例运行两次 TotalSegmentator（`total` + `head_glands_cavities` 任务），
  结果合并重映射为统一 11 类
- 建议 `--totalseg-device gpu`（显著加速；无 GPU 时改为 `cpu`）
- 首次运行需下载权重（见 [3.3 节](#33-totalsegmentator-权重)）；权重约 3GB
- 中间文件存于临时目录，`--keep-staging` 可保留用于调试
- 每例耗时约 **2~5 分钟**（GPU），202 患者全部时间点约需 **14~35 小时**

```bash
# 全量后台运行（推荐）
nohup python scripts/processing/organ_extraction/organ_segmentation.py \
    --totalseg-device gpu \
    > logs/organ_seg_run.log 2>&1 &

# 指定单例
python scripts/processing/organ_extraction/organ_segmentation.py \
    --patient-id 00136597 --study-date 20220425 --totalseg-device gpu -v

# 强制重新分割
python scripts/processing/organ_extraction/organ_segmentation.py \
    --totalseg-device gpu --overwrite
```

---

### 5.5 export — nnU-Net 推理格式导出

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

### 5.6 segment — 病灶分割

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

### 5.7 qc — 分割结果质控可视化

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

### 5.8 analyze — 特征统计与绘图（占位）

**脚本：** `scripts/analysis/plot_results.py`

**当前状态：** `main.py --stage analyze` 仍调用占位绘图。单患者代谢参数与组学已由 [5.9](#59-longitudinal--纵向分析与桌面软件) 的 `features.py` 写出。

---

### 5.9 longitudinal — 纵向分析与桌面软件

**后端：** `scripts/longitudinal/`  
**GUI：** `scripts/gui/app.py`（PySide6 + pyqtgraph）  
**环境：** `data-analysis`（不进入 `main.py --stage all`，时间点需人工指定）

时间点不预写进清单：打开患者后在界面里指定 **Baseline / Interim / End**（可只选 1–3 个），写入

`data/processed/<PatientID>/longitudinal_session.json`

映射与演变至少需要 **baseline + 一个随访**。三个角色不能指向同一检查日期。

**读取与生成分割：**

- 自动读取 `{ID}_{Date}_lesion.nii.gz`；若存在 `{ID}_{Date}_lesion_edited.nii.gz` 则**优先**用调整副本
- 无 mask 时：右侧 **AutoPET 分割**、**SUV 阈值分割**、**空白手动分割**，或 **其他（载入已有 mask）**
- 四种入口都会打开 **3×3 编辑窗**（行：CT / PET / 融合；列：轴 / 冠 / 矢）。任一格左涂右擦，mask 实时同步到其余格
- **保存并关闭** 写入 sidecar `{ID}_{Date}_lesion_edited.nii.gz`；取消不改主窗口
- 形态学：膨胀 / 腐蚀 / 开 / 闭，半径 1–5，作用于当前层或三维、当前灶或全部
- 连通域自动编号，列表显示体素 / 体积 / SUVmax；可按体积重新编号

**显示：**

- 单选：仅 CT / 仅 PET / PET-CT 融合；PET 与 MIP 为 **灰度**（按 SUV 窗），mask 仍为红/黄/青
- CT 窗位 / 窗宽（默认 40 / 400）、PET SUV 上下限、融合透明度、50%–400% 缩放、**十字线**（可关）
- 启动时按当前显示器可用区域缩放窗口（小屏最大化），右侧栏可滚动
- **患者列表**：F2 或工具栏「患者」手动显隐；默认在选中一次检查后自动隐藏

**跨检查映射（CT→CT 刚体+仿射，不用 SyN）：**

按钮 **「将基线病灶映射到中期/末期」**。moving mask 为基线 `working_lesion`（edited 优先）。

DLBCL 病灶会消退或进展，强变形容易把病灶床拉碎，因此只用刚体+仿射把基线解剖位置对到随访 CT。

- 工作网格：2 mm 各向同性 CT（`ct_iso_reference.nii.gz` 优先，否则 `preprocessed/CT`）
- fixed = 随访 CT，moving = 基线 CT；`Rigid[0.1]` 再 `Affine[0.1]`，度量 MI
- `antsApplyTransforms -n GenericLabel` 把基线病灶 mask 拉到随访网格
- ANTs 子进程单线程（`ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1`），信号崩溃时自动降尺度重试

```
data/processed/<ID>/<FollowupDate>/longitudinal/
    baseline_<BLDate>_to_this_0GenericAffine.mat
    baseline_<BLDate>_to_this_affine.txt
    baseline_ct_warped.nii.gz          # 基线 CT 变到随访空间（质控）
    baseline_lesion_warped.nii.gz      # 映射后的基线病灶床
```

**特征（每个已指定时间点、每种 ROI）：**

| ROI | 含义 |
|-----|------|
| `native_lesion` | 该次检查工作 mask（edited 优先，否则 nnU-Net / 阈值） |
| `baseline_mapped` | 基线病灶床映射到该次检查（仅随访；用于看原病灶床内残留摄取） |

| 参数 | 说明 |
|------|------|
| SUVmax / SUVmean | ROI 内最大 / 平均 SUVbw |
| SUVpeak | 以 SUVmax 体素为球心的 1 cm³ 球均值 |
| MTV / TLG | 体积 (mL)；TLG = SUVmean × MTV |
| 肝 / 脾 SUVmean | 器官 mask 重采样到 PET 网格后的参考摄取 |
| 组学 | pyradiomics Original：shape + firstorder + GLCM（PET binWidth 0.25 SUV，CT 25 HU） |

写出 `data/processed/<ID>/longitudinal_features.csv`；批跑汇总 `results/tables/longitudinal_features.csv`。未安装 pyradiomics 或加 `--no-radiomics` 时只写代谢参数。

**GUI 布局与交互：**

```
┌────────────┬──────────────────────────────┬─────────────────┐
│ 患者列表    │ 当前检查 轴/冠/矢 三视图       │ 时间点 / 分割    │
│ 绿/黄/红灯  │ CT / PET / 融合 + 窗宽窗位    │ 画笔 / 形态学    │
│            │ 编辑 mask：左涂右擦            │ 病灶编号表       │
├────────────┼──────────────────────────────┼─────────────────┤
│            │ 最多三列冠状 MIP 演变条         │ 特征表 + 折线    │
└────────────┴──────────────────────────────┴─────────────────┘
```

| Overlay | 颜色 | 含义 |
|---------|------|------|
| 本底 mask | 红 | 当前检查分割（nnU-Net / 阈值 / 手动） |
| 当前选中灶 | 黄 | 编号列表中选中的病灶 |
| 映射 mask | 青 | 基线病灶床 warp 到当前检查 |

二者并存才能对比「旧病灶消退 vs 新病灶出现」。分割、映射与组学在 `QThread` 中运行。菜单可导出特征 CSV、MIP PNG、折线 PNG。

同机 PET–CT 配准不在 GUI 内重算；缺 `pet_iso_aligned.nii.gz` 时 PET 回退到 `preprocessed/PET`。患者树状态灯：绿 = CT+PET+lesion 齐全，黄 = 缺 mask 或同机配准，红 = 缺 CT 或 PET。

```bash
# 桌面软件（需图形界面；SSH 时请开 X11 转发）
conda activate data-analysis
python scripts/gui/app.py

# 仅 CLI：按会话 JSON 映射该患者全部随访
python scripts/longitudinal/interscan_register.py --patient-id 00136597

# 指定一对日期
python scripts/longitudinal/interscan_register.py \
    --patient-id 00136597 --baseline 20220425 --followup 20220728

# 提取特征
python scripts/longitudinal/features.py --patient-id 00136597
python scripts/longitudinal/features.py --no-radiomics   # 只要代谢参数
```

一期不做：三维画笔、层间插值、SyN 变形、覆盖原始 nnU-Net 文件、自动 Deauville / Lugano 分期。

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

### `scripts/processing/register_pet_ct.py` — ANTs PET-CT 刚体配准

| 项目 | 说明 |
|------|------|
| 核心类 | `PetCtRegistrar` |
| 配准框架 | ANTs `antsRegistration`（`/home/sun/ants-2.6.5/bin/`） |
| 变换类型 | 刚体（Rigid），MI 互信息度量，适合跨模态 PET/CT |
| 两步策略 | 原始分辨率估计变换 → 变换应用到 2mm 图像，避免双重插值 |
| 线程安全 | 子进程 env 设 `ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1`，防 SIGFPE |
| 自动重试 | 信号终止（exit < 0）时自动切换保守 3 尺度参数重试一次 |
| `--first-study-only` | 每患者仅处理最早的完整 study，用于基线分析 |
| `--patient-ids` | 逗号分隔的患者 ID 白名单 |

---

### `scripts/processing/organ_extraction/organ_segmentation.py` — 器官分割

| 项目 | 说明 |
|------|------|
| 工具 | TotalSegmentator（`total` + `head_glands_cavities` 两个 task） |
| 输入 | 原始分辨率 CT（`data/interim/.../CT/`，无需 preprocess） |
| 输出 | `organs.nii.gz`，11 类标签，与原始 CT 同 affine |
| 标签设计 | 与 autoPET3 双头模型原始脚本完全对齐（包含前列腺） |
| `--totalseg-device` | `gpu`（推荐）/ `cpu` |
| `--keep-staging` | 保留 TotalSegmentator 中间结果（默认删除） |

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

`plot_feature_distributions(table_csv, figure_out)` 当前仍抛出 `NotImplementedError`。
单患者代谢/组学表已由 `scripts/longitudinal/features.py` 写出；队列分布图可后续改为读取
`results/tables/longitudinal_features.csv`。

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

### `scripts/longitudinal/` — 跨时间点映射与特征

| 模块 | 说明 |
|------|------|
| `catalog.py` | 索引 2 mm CT/PET、同机配准、lesion / lesion_edited、器官 mask |
| `session.py` | `longitudinal_session.json`：baseline / interim / end |
| `ants_runner.py` | ANTs 单线程 env + 负退出码检测，供跨检查配准复用 |
| `interscan_register.py` | 基线 CT → 随访 CT 刚体+仿射；mask 用 working_lesion（edited 优先） |
| `features.py` | SUVmax / SUVpeak / MTV / TLG + 肝脾参考 + pyradiomics |
| `radiomics_params.yaml` | 组学：Original + shape/firstorder/GLCM |

---

### `scripts/gui/app.py` — 纵向浏览软件

| 项目 | 说明 |
|------|------|
| 技术 | PySide6 + pyqtgraph（切片）+ matplotlib（MIP / 折线） |
| 入口 | `python scripts/gui/app.py` |
| 交互 | 选患者 → 指定时间点 → 分割/微调 → 映射到随访 → MIP 与特征表 |
| Overlay | 红 = 本底 mask；黄 = 当前灶；青 = 映射的基线病灶床 |
| 分割 | AutoPET / SUV 阈值 / 空白 / 载入已有 mask；3×3 弹窗画笔同步；另存 edited |
| 显示 | 仅 CT / 仅 PET（灰度）/ 融合；十字线；窗宽窗位、SUV 窗、缩放；窗口按屏适配 |
| 患者栏 | F2 /「患者」按钮显隐；选中检查后可自动隐藏 |
| 后台线程 | `MappingWorker` / `FeatureWorker` / `SegmentWorker` |
| 导出 | 特征 CSV、MIP PNG、折线 PNG |

---

## 7. 数据说明

- 原始 DICOM 批次位于 `data/raw/`（`DICOM`、`DICOMDIS`–`DICOMDIY` 等）
- 所有数据目录（`data/raw/`、`data/interim/`、`data/processed/`、`data/nnunet_export/`、`data/qc/`）均已加入 `.gitignore`，请勿强制添加上传
- `autoPET/` 第三方模型目录（~4GB）同样已被 `.gitignore` 忽略
- 中间结果可安全删除后重新生成（所有阶段均为增量幂等）

**多时间点患者统计：**

| 项目 | 数量 |
|------|------|
| 患者总数 | 202 |
| 具有多个时间点的患者 | 125 |
| 总 study 数（所有时间点） | 427 |
| 每患者最多时间点数 | 8 |

**本地数据集运行结果（截至 2026-08-18）：**

| 阶段 | 结果 | 备注 |
|------|------|------|
| convert | 全部 DICOM 转换完成 | CT + PET `_ACT` / `_SUVbw` |
| preprocess | 427 studies 已发现 | 工作网格 2 mm 各向同性 |
| register（同机 PET→CT） | 199 / 427 `.mat` | 缺文件的 study 会跳过 |
| organ_seg | 422 `organs.nii.gz` | TotalSegmentator，11 类 |
| export | 419 pairs | 缺 CT 或 PET 的 study 跳过 |
| segment (nnunet) | 419 / 419 lesion mask | AutoPET III 5-fold 集成 |
| qc | 419 张质控 PNG | `data/qc/` |
| longitudinal | GUI / CLI 已就绪 | 跨检查映射按患者在界面中触发 |

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
