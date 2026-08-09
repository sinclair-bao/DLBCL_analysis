# DLBCL

弥漫大 B 细胞淋巴瘤（DLBCL）医学影像分析项目。

## 目录结构

```text
data/
  raw/         # 原始影像（DICOM / NIfTI），不纳入 Git
  interim/     # 中间处理产物
  processed/   # 最终处理结果（掩码、配准图等）
scripts/
  processing/  # 预处理、分割、特征提取
  analysis/    # 统计与绘图
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

```bash
python main.py --help
```

## 数据说明

现有 DICOM 批次位于 `data/raw/`（如 `DICOM`、`DICOMDIS`–`DICOMDIY`）。  
原始数据已在 `.gitignore` 中屏蔽，请勿强制添加上传。

## Git 远程

- GitHub: `git@github.com:sinclair-bao/DLBCL_analysis.git`
- Gitea: `ssh://git@100.100.211.88:2222/sinclair/DLBCL_analysis.git`
