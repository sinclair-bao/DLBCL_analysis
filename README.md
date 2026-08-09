# DLBCL

弥漫大 B 细胞淋巴瘤（DLBCL）医学影像分析项目。

## 目录结构

```text
data/
  raw/         # 原始影像（DICOM / NIfTI），不纳入 Git
  interim/     # 中间处理产物
  processed/   # 最终处理结果（掩码、配准图等）
scripts/
  tools/       # 跨项目可复用的通用工具（如 PACS DICOM -> NIfTI/SUV 转换）
  common/      # 本项目内部各阶段共用的小工具（目录发现、日志等）
  processing/  # 预处理（重采样）、分割（SUV 阈值基线，可替换为自定义模型）
  analysis/    # 统计与绘图（特征提取步骤待补齐）
results/
  figures/     # 论文用图
  tables/      # 统计表
  models/      # 模型权重
docs/          # 实验记录、协议、数据字典
logs/          # 运行日志
main.py        # 流程入口
environment.yml
```

## 环境

```bash
conda env create -f environment.yml
conda activate dlbcl
```

## 运行

流程按阶段拆分为 `convert -> preprocess -> segment -> analyze`，每个阶段既可以
被 `main.py` 编排调用，也可以单独运行调试（各脚本头部有详细用法说明）。所有
阶段都是增量式的：输出已存在则自动跳过，单个病例失败不影响其他病例，可放心
重复执行来补跑失败项。

```bash
python main.py --help

# 完整跑一遍（需要 dcm2niix 可执行文件）
python main.py --stage all --dcm2niix-bin /home/sun/fsl/bin/dcm2niix

# 只跑单一阶段
python main.py --stage preprocess --voxel-size 1.5
python main.py --stage segment --threshold-mode relative --threshold 0.41

# 干跑，只看计划、不实际处理
python main.py --stage all --dry-run
```

阶段说明：

- `convert`：`scripts/tools/pacs_dicom_to_nifti_suv.py`，PACS DICOM 序列 ->
  NIfTI，PET 换算为 SUVbw，输出到 `data/interim/<PatientID>/<StudyDate>/`。
- `preprocess`：`scripts/processing/preprocess.py`，CT/PET 重采样为统一各向
  同性体素间距，输出到 `.../preprocessed/{CT,PET}/`。
- `segment`：`scripts/processing/segmentation.py`，对 PET 做 SUV 阈值分割
  基线，输出掩码到 `data/processed/<PatientID>/<StudyDate>/masks/`；后续可
  通过 `segmentation_fn` 注入自定义模型替换该基线算法。
- `analyze`：`scripts/analysis/plot_results.py` + `stats_analysis.R`，特征
  提取步骤尚未实现，当前会记录 warning 而不是报错。

## 数据说明

现有 DICOM 批次位于 `data/raw/`（如 `DICOM`、`DICOMDIS`–`DICOMDIY`）。  
原始数据已在 `.gitignore` 中屏蔽，请勿强制添加上传。

## Git 远程

- GitHub: `git@github.com:sinclair-bao/DLBCL_analysis.git`
- Gitea: `ssh://git@100.100.211.88:2222/sinclair/DLBCL_analysis.git`
