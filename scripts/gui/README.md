# DLBCL 纵向 PET/CT 分析软件（GUI）

桌面软件，用来对同一患者的基线 / 中期 / 末期 PET/CT 做浏览、分割微调、跨检查病灶映射和代谢/组学提取。时间点不预写进处理流水线，必须在界面里人工指定。

项目总览、离线预处理与目录约定见仓库根目录 [`README.md`](../../README.md) §5.9。

---

## 1. 启动

**环境：** `data-analysis`（PySide6、pyqtgraph、nibabel、scipy）。AutoPET 推理另用 `autopet` 环境，由后台线程调用。GPU 切片合成为可选：`pip install cupy-cuda12x`（按本机 CUDA 版本选 `cupy-cuda11x` / `cupy-cuda12x`）；未安装时「渲染」下拉的 GPU 项禁用并回退 CPU。不把 CuPy 装进默认环境。

```bash
conda activate data-analysis
python scripts/gui/app.py

# 可选：覆盖数据根目录
python scripts/gui/app.py \
  --interim-root /path/to/data/interim \
  --processed-root /path/to/data/processed
```

默认读取项目下 `data/interim/` 与 `data/processed/`。需要图形界面；SSH 时请开 X11 转发。

**硬性依赖（本机路径）：**

| 用途 | 路径 |
|------|------|
| ANTs | `/home/sun/ants-2.6.5/bin`（`antsRegistration` / `antsApplyTransforms`） |
| AutoPET Python | `/home/sun/miniconda3/envs/autopet/bin/python` |

缺 CT/PET 的检查无法打开三视图。缺病灶 mask 仍可浏览影像，再用右侧四种分割入口补上。

---

## 2. 推荐工作流

```
选患者 → 打开一次检查 → 指定基线/中期/末期
  → （可选）分割或微调 mask 并保存
  → 「将基线病灶映射到中期/末期」
  → 在随访上核对青色 overlay
  → 「计算代谢 / 组学特征」
  → 导出 CSV / MIP / 折线
```

1. 左侧点患者，再点子检查日期。状态灯：绿 = CT+PET+lesion 齐全；黄 = 缺 mask 或同机配准；红 = 缺 CT 或 PET。
2. 右侧为 **基线 / 中期 / 末期** 各选一个日期（可只选 1–3 个；三个角色不能指向同一天）。写入 `data/processed/<PatientID>/longitudinal_session.json`。
3. 无 mask 或要改 mask：
   - **AutoPET**：先检查该次检查有 CT 与 PET；有进度条。完成后**回到主界面**看红色 overlay。若要手改，点完成框「手动调整」或「其他（载入 mask）」。
   - **SUV 阈值**：打开编辑窗**只显示图像**，不预填 41%；选「41% SUVmax」或「固定 SUV」后才算 mask。
   - **空白手动** / **其他（载入 mask）**：打开三格编辑窗。保存并关闭写入 sidecar。
4. 映射前必须有基线 mask + 至少一个随访、且随访有 2 mm CT。完成后界面会切到中期（没有则末期），勾上「映射 mask」。**青色只出现在随访上**，停在基线会误以为没映射。
5. 映射与 AutoPET、特征提取、三维形态学会弹出进度条（状态栏同时有文案）。未保存的 mask 映射前会先询问是否存盘。
6. 算特征后，右下角表和折线会刷新；菜单可导出。

主界面上的 **查看 基线 / 中期 / 末期** 只切换当前显示的检查，不负责指定角色。指定角色只用右侧下拉框。

---

## 3. 界面布局

```
┌────────────┬──────────────────────────────┬─────────────────┐
│ 患者列表    │ 轴 / 冠 / 矢 三视图            │ 时间点指定       │
│ F2 显隐    │ CT / PET / 融合、窗宽窗位      │ 映射 / 特征按钮  │
│            │ 本底 / 映射 / 编辑 mask        │ 分割四种入口     │
│            │ 十字线、缩放、笔刷、PET 配色    │ 形态学 + 病灶表  │
│            │ 渲染 CPU / GPU                │                 │
├────────────┼──────────────────────────────┼─────────────────┤
│            │ 最多三列冠状 MIP（基线/中/末）  │ 特征表 + 折线    │
└────────────┴──────────────────────────────┴─────────────────┘
```

启动时窗口按当前显示器可用区域缩放；右侧栏可滚动。

### 3.1 菜单与快捷键

| 位置 | 操作 |
|------|------|
| 文件 → 刷新目录 | 重新扫描 interim / processed |
| 文件 → 导出特征 CSV | `Ctrl+S`，复制该患者 `longitudinal_features.csv` |
| 文件 → 导出 MIP 演变图 | 把下方三列 MIP 存成 PNG |
| 文件 → 导出特征折线 | 把右下角折线存成 PNG |
| 视图 → 显示患者列表 | `F2`；工具栏「患者」按钮同等 |
| 视图 → 选择检查后自动隐藏患者列表 | 默认开启 |
| 帮助 → 关于 | 简要说明 overlay 颜色与映射 |

三视图滚轮改切片或缩放（视控件而定）；MIP 条滚轮改统一缩放（50%–400%）。

### 3.2 Overlay 与朝向

| Overlay | 颜色 | 含义 |
|---------|------|------|
| 本底 mask | 红 | 当前检查工作 mask（`*_lesion_edited.nii.gz` 优先，否则 nnU-Net / 阈值） |
| 当前选中灶 | 黄 | 编号表里选中的连通域 |
| 映射 mask | 青 | 基线病灶床 warp 到**当前随访** |

PET 与 MIP 默认 **PET 热金** 伪彩色（工具栏 **配色** 可选灰度 / Hot / Jet / Inferno）；mask 仍用红/黄/青。融合图里的 PET 与「仅 PET」共用该 LUT。工具栏 **渲染** 可在 CPU / GPU 间切换：仍画到现有 pyqtgraph `ImageItem`；GPU 用 CuPy 把整本 CT/PET/mask 留在设备上，滚层只改切片下标，合成 RGB 后再下载。无 CuPy/CUDA 时 GPU 项禁用，旁注「无 CUDA」。

朝向为放射科惯例（RAS 体积，画面左 = 患者右）：

- 轴位、冠状、MIP：左 **R**、右 **L**
- 矢状：左 **P**（后）、右 **A**（前）

标记贴在视口左右边，缩放平移时不跟着解剖跑。

### 3.3 编辑约定

- 分割编辑窗：并排 CT / PET / 融合；点选轴位 / 冠状 / 矢状。任一格**左键涂、右键擦**，三格 mask 同步。
- **SUV 阈值**：打开时**不算** mask；窗内点选「41% SUVmax / 固定 SUV」后才计算。固定模式用滑杆（松开后重算）。切换模式会覆盖当前 mask（可撤销）。
- 主窗口勾选「编辑 mask」后也可在当前三视图上涂擦（无三格对照，容易误改）。
- 形态学：膨胀 / 腐蚀 / 开 / 闭，半径 1–5；范围默认 **当前层 2D**（只改该切片）；**三维**在病灶边界盒内运算，并显示进度条。
- 保存写入 sidecar `{PatientID}_{StudyDate}_lesion_edited.nii.gz`，**从不覆盖** nnU-Net 的 `*_lesion.nii.gz`。
- 连通域自动编号；「按体积重新编号」按体素数从大到小变成 1..N。

---

## 4. 本目录脚本

入口与主窗负责装配；其余文件按「面板 / 视图 / 数据 / 后台」分组。GUI 还会调用 `scripts/longitudinal/` 与 `scripts/processing/`，那些脚本不在本目录。

### 4.1 入口与主窗

| 文件 | 作用 |
|------|------|
| [`app.py`](app.py) | 程序入口。在导入 matplotlib 画布前设置 `QtAgg`，检查 PySide6/pyqtgraph，解析 `--interim-root` / `--processed-root`，创建 `QApplication` 与 `MainWindow`。 |
| [`main_window.py`](main_window.py) | 主窗口：拼患者树、三视图、MIP、右侧时间点/分割/特征；装菜单与 F2；把信号接到映射/分割/形态学/存盘。映射前校验会话与文件、询问未保存 mask、成功后跳到随访并打开青色 overlay。体积缓存 `_vol_cache` 为上限 6 套的 LRU（换患者不整表清空；映射按患者失效，AutoPET 按检查失效）。耗时任务用 `QProgressDialog`。 |
| [`__init__.py`](__init__.py) | 包标记，无逻辑。 |

### 4.2 面板（左右栏）

| 文件 | 作用 |
|------|------|
| [`patient_browser.py`](patient_browser.py) | 左侧树：PatientID → StudyDate。按 `StudyAssets.completeness()` 涂绿/黄/红。发出 `patient_selected` / `study_selected`。 |
| [`timepoint_panel.py`](timepoint_panel.py) | 右侧「时间点指定」：三个下拉框 + 「将基线病灶映射到中期/末期」+ 「计算代谢 / 组学特征」。改下拉即保存会话（主窗校验通过后）。 |
| [`edit_panel.py`](edit_panel.py) | 右侧「分割」四按钮、形态学、撤销/重编号/保存、病灶表。只发信号，不碰磁盘。 |
| [`feature_panel.py`](feature_panel.py) | 右下特征表 + matplotlib 折线（native / mapped 的 SUVmax、MTV、TLG）。读 `longitudinal_features.csv`。 |

### 4.3 视图

| 文件 | 作用 |
|------|------|
| [`ortho_viewer.py`](ortho_viewer.py) | 轴/冠/矢三视图。点选十字时一次写入 ijk 再刷新，三平面同步；切片未变的平面只挪十字、不重绘、不改缩放。显示模式、窗位窗宽、SUV 窗、PET 配色、渲染 CPU/GPU、融合、十字线、本底/映射/编辑勾选、查看基线/中期/末期。 |
| [`evolution_strip.py`](evolution_strip.py) | 下方最多三列冠状 MIP，等比例、统一缩放。可叠本底（红）与映射（青）。`_MipView` 同样标 R/L。 |
| [`segment_editor.py`](segment_editor.py) | 模态分割编辑窗：三格 CT/PET/融合；SUV 阈值打开时不算默认 mask；PET 配色；渲染 CPU/GPU（打开时跟主窗）；画笔、形态学、撤销、重编号、病灶表。确定后把 mask 交回主窗并保存 sidecar。 |

### 4.4 显示与 mask 算法

| 文件 | 作用 |
|------|------|
| [`busy.py`](busy.py) | `QProgressDialog` 不确定进度条，供 AutoPET / 映射 / 特征 / 三维形态学 / 阈值重算。 |
| [`display_utils.py`](display_utils.py) | 切片朝向（`orient_*` / `slice_*`）、冠状 MIP、PET 配色 LUT（灰度/热金/Hot/Jet/Inferno）、CT 窗、`compose_rgb` 叠红/黄/青。`display_to_voxel` / `voxel_to_display` 把点击映回 RAS 体素。 |
| [`render_backend.py`](render_backend.py) | CPU/GPU 合成入口。`gpu_available()` 检测 CuPy+CUDA；GPU 把整本体积上传后切片+伪彩+融合，下载 RGB 给 pyqtgraph。失败回退 CPU。 |
| [`mask_ops.py`](mask_ops.py) | 连通域编号、`paint_disk`、新岛提升编号、`morph_labels`（2D 只改当前层，3D 用 bounding box）、`threshold_pet_mask`、`lesion_stats`（`unique` + `ndimage.maximum`）、按体积重编号。 |
| [`volume_io.py`](volume_io.py) | `VolumeSet` 数据类；把一次检查的 CT/PET/native/mapped 读到工作 CT 网格；任意 NIfTI mask 最近邻重采样；写出 uint16 `*_lesion_edited.nii.gz`。 |

### 4.5 后台线程

[`workers.py`](workers.py) 三个 `QThread`，避免 ANTs / nnU-Net / 组学卡住界面：

| 类 | 做什么 |
|----|--------|
| `MappingWorker` | 调 `InterscanRegistrar.map_session`。进度文案「正在配准 CT（约数分钟，请勿关闭）…」。任一随访成功则 `finished_ok`（可附带失败侧说明）；全部失败才 `failed`。 |
| `FeatureWorker` | 调 `features.extract_patient_features`，写出 `data/processed/<ID>/longitudinal_features.csv`。 |
| `SegmentWorker` | 仅 **AutoPET**：启动前校验 CT+PET；`export_nnunet.py` + `infer_nnunet.py`（`autopet` 解释器）。完成后回主界面。GUI 的 SUV 阈值在内存里算，不走此线程。 |

---

## 5. GUI 调用的后端（不在本目录）

| 模块 | 作用 |
|------|------|
| `scripts/longitudinal/catalog.py` | 扫描 interim/processed，给出 `working_ct` / `working_pet` / `working_lesion`（edited 优先）和 `warped_baseline_mask`。 |
| `scripts/longitudinal/session.py` | `longitudinal_session.json`：baseline / interim / end。 |
| `scripts/longitudinal/interscan_register.py` | 随访 CT 为 fixed、基线 CT 为 moving；刚体+仿射（**不用 SyN**）；`GenericLabel` 把基线 mask 拉到随访网格。 |
| `scripts/longitudinal/ants_runner.py` | ANTs 子进程，单线程，避免与 OpenMP 抢核。 |
| `scripts/longitudinal/features.py` | native_lesion 与 baseline_mapped 的 SUV / MTV / TLG / 组学。 |
| `scripts/processing/export_nnunet.py`、`infer_nnunet.py`、`segmentation.py` | AutoPET；CLI 仍可用 `segmentation.py --method threshold` 做批处理。GUI 阈值不调用该 CLI。 |

映射输出（写在**随访**检查目录，不在基线）：

```
data/processed/<ID>/<FollowupDate>/longitudinal/
    baseline_<BLDate>_to_this_0GenericAffine.mat
    baseline_<BLDate>_to_this_affine.txt
    baseline_ct_warped.nii.gz          # 基线 CT → 随访空间（质控，界面暂未显示）
    baseline_lesion_warped.nii.gz      # 青色 overlay 的来源
```

工作影像：2 mm 各向同性 CT（`ct_iso_reference.nii.gz` 优先），PET 优先 `pet_iso_aligned.nii.gz`，否则 `preprocessed/PET`。同机 PET–CT 配准不在 GUI 里重算。

特征 ROI：

| `roi_type` | 含义 |
|------------|------|
| `native_lesion` | 该次检查工作 mask |
| `baseline_mapped` | 基线病灶床映射到该随访（无 warped 文件则没有这些行） |

---

## 6. 运行逻辑（简图）

```
DataCatalog 扫描磁盘
        │
        ▼
  PatientBrowser ──► 选检查 ──► load_volume_set ──► OrthoViewer + EvolutionStrip
        │
        ▼
  TimepointPanel ──► save_session JSON
        │
        ├── AutoPET ──► 校验 CT+PET ──► 进度条 ──► 主界面结果（可选手动调整）
        ├── SUV 阈值 / 空白手动 / 载入 ──► SegmentEditorDialog ──► lesion_edited.nii.gz
        │
        ├── 映射按钮 ──► 预检 ──► MappingWorker ──► ANTs ──► 切到随访 + 青色 overlay
        │
        └── 特征按钮 ──► FeatureWorker ──► CSV ──► FeaturePanel
```

`working_lesion` = `lesion_edited` 若存在，否则 nnU-Net `*_lesion.nii.gz`。映射的 moving mask 始终是**磁盘上的**基线 `working_lesion`，因此未保存的编辑必须先存盘。

---

## 7. 进一步改进（Todo）

按优先级大致分组。已明确不做的写在最后。

### 数据安全与会话

- [ ] 关窗、换患者、换检查时，若 `VolumeSet.dirty` 未保存，弹出确认（现在只有映射前会问）。
- [ ] 关闭窗口时 `wait()` 尚未结束的 `MappingWorker` / `FeatureWorker` / `SegmentWorker`，避免子进程残留。
- [ ] ANTs 路径不要写死 `/home/sun/ants-2.6.5/bin`：环境变量或设置对话框；换机器才不挂。
- [ ] AutoPET 解释器路径同样可配置，不要写死 `autopet` conda 前缀。

### 映射与质控

- [ ] 随访上提供「基线 CT warped」叠加或配准互信息，避免盲信青色 mask。
- [ ] 「计算特征」旁提示：没有 `baseline_lesion_warped.nii.gz` 时 CSV 只有 `native_lesion`。
- [ ] 映射进行中把 ANTs 日志尾部刷到状态栏（现在只有一句「约数分钟」）。

### 分割与编辑

- [ ] 长期只保留分割编辑窗里的形态学/画笔；主窗那一套容易误改且无 CT/PET/融合对照。
- [ ] 跨时间点病灶对应：仿射后仍按连通域独立编号，基线灶 #1 与随访灶 #1 不是同一灶。
- [ ] 器官 mask（肝/脾）在三视图上可选叠加，便于 Deauville / 肝参照目测。
- [ ] Deauville 五分法：用已有肝/脾 SUV 与病灶 SUVmax 算分并写入特征表。

### 显示与性能

- [x] 病灶表：`lesion_stats` 一次 `np.unique` + `ndimage.maximum` 归约，不再对每个编号做 `mask == lid`。
- [x] 体积缓存：`OrderedDict` LRU，上限 6 套；换患者不清空；保存 edited 只更新当前 `VolumeSet`；映射丢掉该患者键，AutoPET 丢掉该检查键。
- [x] 渲染 CPU / GPU 二选一（仍画到 pyqtgraph `ImageItem`）。GPU 用可选 CuPy 在设备上切片+伪彩+融合；无 CUDA 时该项禁用并回退 CPU。不换 OpenGL/VisPy。

### 明确不做（保持现状）

- 不覆盖 nnU-Net `*_lesion.nii.gz`，只写 sidecar edited。
- 跨检查不用 SyN：DLBCL 病灶会消退/进展，强变形容易把病灶床拉碎。
- 不做 3D 笔刷（只做当前层圆盘）。
- 不在 GUI 里重跑同机 PET→CT 配准。
