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

# 其他纵向 PET 队列：以清单提供每次检查的实际文件路径
python scripts/gui/app.py --manifest /path/to/studies.csv \
  --processed-root /path/to/review_outputs
```

默认读取项目下 `data/interim/` 与 `data/processed/`。需要图形界面；SSH 时请开 X11 转发。

也可在运行中用 **文件 → 打开数据目录…** 选择 `data/`、`interim/` 或
`interim/<患者ID>/`。选择单个患者目录时，列表只显示该患者；结果写入同一
`data/processed/`。**文件 → 导入清单…** 可载入下述 CSV，无需固定文件树；
结果默认写入清单旁的 `<清单文件名>_review_outputs/`。切换数据源时会先处理
未保存的 mask，随后清除旧影像和缓存并刷新患者列表。取消选择或导入失败不会切换。

**可配置依赖：**

| 用途 | 路径 |
|------|------|
| ANTs | `ANTS_BIN` 环境变量，未设置时沿用 `/home/sun/ants-2.6.5/bin` |
| AutoPET Python | `AUTOPET_PYTHON` 环境变量，未设置时沿用原工作站路径 |

通用清单至少包含 `patient_id,study_date,ct_path,pet_path,pet_units`，可选
`lesion_mask_path,organ_mask_path,tracer`。相对路径相对于清单文件；检查编号只需在同一患者内唯一，建议用可按字典序排列的 ISO 日期。现有定量特征协议只接受 `pet_units=SUVbw`；其他单位可浏览和编辑，但须先实现并验证相应特征协议。
清单模式支持载入/编辑 mask、配准、质控与特征；本项目的 AutoPET 启动器不会自动应用于其他队列。

缺 CT/PET 的检查无法打开三视图。缺病灶 mask 仍可浏览影像，再用右侧四种分割入口补上。

### 1.1 输入目录树与文件名

目录模式以 `interim/` 为扫描根：其下**一级目录是患者 ID，二级目录是一次检查 ID**。
本研究约定患者 ID 为 8 位编号、检查 ID 为 `YYYYMMDD`；目录扫描本身不校验日期格式。
选择 `data/`、`interim/` 或 `interim/<患者ID>/` 均可；直接选择患者目录只显示该患者。
`processed/` 由软件按相同的患者和检查 ID 查找，不能只选择 `processed/` 来建立患者列表。

```text
data/
├── interim/
│   └── <PatientID>/
│       └── <StudyDate>/
│           ├── CT/*.nii.gz                         # 原始 CT；优先含 WB_Standard 的文件
│           ├── PET/*_SUVbw.nii.gz                 # 原始 SUVbw PET；*_ACT.nii.gz 不用于显示
│           └── preprocessed/
│               ├── CT/*.nii.gz                     # 工作 CT；优先含 WB_Standard 的文件
│               └── PET/*_SUVbw.nii.gz             # 工作 PET
└── processed/
    └── <PatientID>/
        └── <StudyDate>/
            ├── registration/
            │   ├── ct_iso_reference.nii.gz         # 优先于 preprocessed/CT
            │   └── pet_iso_aligned.nii.gz          # 优先于 preprocessed/PET
            ├── masks/
            │   ├── <PatientID>_<StudyDate>_lesion.nii.gz
            │   ├── <PatientID>_<StudyDate>_lesion_edited.nii.gz
            │   └── versions/                       # 每次保存的编辑稿及覆盖前备份
            └── organs/organs.nii.gz                # 可选
```

**最少可浏览输入**：`interim/<PatientID>/<StudyDate>/` 必须存在，且该次检查有一对
工作 CT/PET（两者分别来自 `registration/` 的优先文件，或 `preprocessed/` 的回退文件）。
仅有原始 `CT/` 和 `PET/` 会列在患者树中，但无法打开三视图。病灶 mask 可暂缺。
AutoPET 需要 `preprocessed/CT` 和 `preprocessed/PET` 两个文件，运行时重新导出当前影像、
再重新推理；已有模型原稿会被覆盖，人工编辑稿保留并优先显示。每个输入目录建议只放一个
符合规则的 NIfTI；若有多个，当前索引器按文件名排序取第一个，CT 优先选择名称含
`WB_Standard` 的文件。不要把不同重建序列混放在同一目录。

模型原稿必须精确命名为 `<PatientID>_<StudyDate>_lesion.nii.gz`；人工保存为
`<PatientID>_<StudyDate>_lesion_edited.nii.gz`，两者位于同一 `masks/` 目录。
输入 mask 可为 0/1 或 0/255 二值图，也可为正整数病灶编号图；读入时最近邻重采样到
工作 CT 网格。若已有 edited 文件，它总是优先于模型原稿。`*_mask.nii.gz` 是离线阈值
基线文件，**不会自动进入 GUI 的工作 mask**；要比较它，请用「其他（载入 mask）」。

### 1.2 清单模式与运行逻辑

对于其他纵向 PET 队列，用 **文件 → 导入清单…** 或启动参数 `--manifest`。
CSV 必需列：`patient_id,study_date,ct_path,pet_path,pet_units`；可选列：
`lesion_mask_path,organ_mask_path,tracer`。每一行是一位患者的一次检查；同一患者的
`study_date` 不能重复。路径可以是绝对路径，或相对 CSV 所在目录；所指文件必须存在。
患者和检查 ID 只能包含字母、数字、下划线、点和连字符。`pet_units=SUVbw` 才能进入
现有 SUV 阈值分割和定量特征协议；其他单位只支持浏览、手动编辑与质控。
清单模式不调用本项目的 AutoPET。

软件运行顺序是：扫描目录或清单 → 患者树显示全部检查及完整性状态 → 选择患者时读取其
`longitudinal_session.json` 并显示首个检查 → 人工指定 baseline / interim / end 与其他随访
→ 编辑并保存各次检查的 mask → 审阅 mask → 将基线病灶床映射到随访 CT 并审阅配准
→ 仅在所需质控均通过且输入文件指纹仍一致时计算 SUV/组学特征。
三视图使用工作 CT/PET；下方 MIP 在后台预取。切换数据源会先询问如何处理未保存的编辑，
随后清除旧图像、时间点和缓存；取消或输入无效时保持原数据源。

点击患者 ID 后，该行立即以蓝色显示选中状态；患者树下方的进度条显示**当前检查**的
读取阶段（索引与历史结果核对、CT、PET、病灶 mask、映射 mask）。载入完成后进度条短暂
显示 100% 并自动收起。进度百分比代表阶段完成情况，不是文件字节数；其他检查的 MIP
仍在后台预取，不计入此进度条。直接选择某次检查也显示该检查的读取进度。

编辑窗左键涂画、右键擦除，三格共享同一 3D mask；「当前层 2D」形态学只改正在看的层。
「保存并关闭」把结果写入 `*_lesion_edited.nii.gz`，先备份旧编辑稿，再写新稿与版本快照，
质控状态重置为待审阅；原始 `*_lesion.nii.gz` 不覆盖。无变化的已有 mask 不生成新版本。
取消编辑不落盘。阈值入口须先选阈值模式或实际手动修改，不能直接保存空稿。重新打开检查
优先读取 edited 文件；磁盘源 mask 被外部修改时拒绝覆盖并要求刷新。

---

## 2. 推荐工作流

```
选患者 → 打开一次检查 → 指定基线/中期/末期
  → 分割或微调 mask 并保存 → 审阅原稿/编辑稿，记录审阅者及结论
  → 「将基线病灶映射到中期/末期」
  → 在随访上核对青色 overlay 及基线 CT warped 棋盘格，记录结论
  → 「计算代谢 / 组学特征」
  → 导出 CSV / MIP / 折线
```

1. 左侧点患者，再点子检查日期。状态灯：绿 = CT+PET+lesion 齐全；黄 = 缺 mask 或同机配准；红 = 缺 CT 或 PET。
2. 右侧指定 **基线 / 中期 / 末期**，也可在「其他随访」多选更多检查；同一日期不可重复指定。写入 `data/processed/<PatientID>/longitudinal_session.json`。
3. 无 mask 或要改 mask：
   - **AutoPET**：先检查该次检查有 CT 与 PET；有进度条。完成后**回到主界面**看红色 overlay。若要手改，点完成框「手动调整」或「其他（载入 mask）」。
   - **SUV 阈值**：打开编辑窗**只显示图像**，不预填 41%；选「41% SUVmax」或「固定 SUV」后才算 mask。
   - **空白手动** / **其他（载入 mask）**：打开三格编辑窗。保存并关闭写入 sidecar。
4. 映射前必须有基线 mask + 至少一个随访、且随访有 2 mm CT。完成后界面会切到中期（没有则末期），勾上「映射 mask」。**青色只出现在随访上**，停在基线会误以为没映射。
5. 映射与 AutoPET、特征提取、三维形态学会弹出进度条（状态栏同时有文案）。未保存的 mask 映射前会先询问是否存盘。
6. GUI 仅在每个原生 mask 与随访配准均质控通过时计算特征。算特征后右下角表和折线刷新；菜单可导出。

每次保存 edited mask 均留一个 `masks/versions/` 快照，当前文件旁的
`*.review.json` 记录审阅者、理由与文件 SHA-256。修改文件后旧通过结论自动变为过期。
映射目录的 `mapping_qc.json` 记录两次 CT、基线 mask、映射 mask 与配准后 CT 的指纹；输入改变时重新映射。特征 CSV 同时记录 PET、CT 与 mask 的 SHA-256，历史文件只有在当前输入和质控状态均一致时才重新显示。
人工记录的病灶对应写在患者目录 `lesion_links.json`，与 mask 指纹绑定。

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
│            │ 多时间点冠状 MIP（可横向滚动） │ 特征表 + 折线    │
└────────────┴──────────────────────────────┴─────────────────┘
```

启动时窗口按当前显示器可用区域缩放；三视图下方的切片、显示模式、mask、窗位/SUV、配色及渲染选项横向排在同一行，宽度不足时使用该行下方的横向滚动条。坐标与 HU/SUV 状态独立显示在下一行。右侧栏保证至少 320 个逻辑像素宽，可纵向滚动。选择患者后先显示当前检查；即使尚未指定基线/随访，也显示当前检查的整幅冠状 MIP。默认 100% 按 MIP 面板尺寸完整适配图像，窗口大小变化时重新适配；提高百分比可放大。其他已指定检查的 MIP 在后台逐项载入，短暂出现“载入中”是正常状态。十字线仅出现在当前检查的 MIP 上，并与三视图的体素位置及显示开关同步；冠状 MIP 只表示左右和头脚坐标，前后方向在投影中合并，不能据此判断前后深度。

### 3.1 菜单与快捷键

| 位置 | 操作 |
|------|------|
| 文件 → 打开数据目录 | 选择 `data/`、`interim/` 或一个患者目录；`Ctrl+O` |
| 文件 → 导入清单 | 选择通用队列 CSV，并单独保存审阅结果 |
| 文件 → 刷新目录 | 重新扫描当前目录或清单及其结果目录 |
| 文件 → 导出特征 CSV | `Ctrl+S`，复制该患者 `longitudinal_features.csv` |
| 文件 → 导出 MIP 演变图 | 把下方所有时间点的 MIP 存成 PNG |
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

PET 与 MIP 默认 **PET 热金** 伪彩色（工具栏 **配色** 可选灰度 / Hot / Jet / Inferno）；mask 仍用红/黄/青。融合图里的 PET 与「仅 PET」共用该 LUT。工具栏 **渲染** 可在 CPU / GPU 间切换：仍画到现有 pyqtgraph `ImageItem`；GPU 用 CuPy 缓存最近两次检查的 CT/PET/mask，滚层只改切片下标，合成 RGB 后再下载。无 CuPy/CUDA 时 GPU 项禁用，旁注「无 CUDA」。

朝向为放射科惯例（RAS 体积，画面左 = 患者右）：

- 轴位、冠状、MIP：左 **R**、右 **L**
- 矢状：左 **P**（后）、右 **A**（前）

标记贴在视口左右边，缩放平移时不跟着解剖跑。

### 3.3 编辑约定

- 分割编辑窗：并排 CT / PET / 融合；点选轴位 / 冠状 / 矢状。任一格**左键涂、右键擦**，三格 mask 同步。
- **SUV 阈值**：打开时**不算** mask；将十字线放在目标灶附近，选择 10–100 mm 局部 ROI 半径，再点选「41% SUVmax / 固定 SUV」。41% 使用局部 ROI 内 SUVmax；结果仅为待审阅候选 mask。切换模式会覆盖当前 mask（可撤销）。
- 主窗口勾选「编辑 mask」后也可在当前三视图上涂擦（无三格对照，容易误改）。
- 形态学：膨胀 / 腐蚀 / 开 / 闭，半径 1–5；范围默认 **当前层 2D**（只改该切片）；**三维**在病灶边界盒内运算，并显示进度条。
- 保存写入 sidecar `{PatientID}_{StudyDate}_lesion_edited.nii.gz` 并保留历史快照，**从不覆盖** nnU-Net 的 `*_lesion.nii.gz`。编辑窗可切换原始 mask、当前 mask、增删差异。
- 连通域自动编号；「按体积重新编号」按体素数从大到小变成 1..N。

---

## 4. 本目录脚本

入口与主窗负责装配；其余文件按「面板 / 视图 / 数据 / 后台」分组。GUI 还会调用 `scripts/longitudinal/` 与 `scripts/processing/`，那些脚本不在本目录。

### 4.1 入口与主窗

| 文件 | 作用 |
|------|------|
| [`app.py`](app.py) | 程序入口。在导入 matplotlib 画布前设置 `QtAgg`，检查 PySide6/pyqtgraph，解析 `--interim-root` / `--processed-root`，创建 `QApplication` 与 `MainWindow`。 |
| [`main_window.py`](main_window.py) | 主窗口：拼患者树、三视图、MIP、右侧时间点/分割/特征；装菜单与 F2；把信号接到映射/分割/形态学/存盘。映射前校验会话与文件、询问未保存 mask、成功后跳到随访并打开青色 overlay。体积缓存 `_vol_cache` 为上限 6 套的 LRU，另保留轻量二维 MIP 缓存；其他时间点在后台读取。耗时任务用 `QProgressDialog`。 |
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
| [`evolution_strip.py`](evolution_strip.py) | 下方任意时间点冠状 MIP，可横向滚动、等比例、统一缩放。可叠本底（红）与映射（青）。`_MipView` 同样标 R/L。 |
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
| `scripts/longitudinal/session.py` | `longitudinal_session.json`：baseline / interim / end 与其他随访。 |
| `scripts/longitudinal/interscan_register.py` | 随访 CT 为 fixed、基线 CT 为 moving；刚体+仿射（**不用 SyN**）；`GenericLabel` 把基线 mask 拉到随访网格。 |
| `scripts/longitudinal/ants_runner.py` | ANTs 子进程，单线程，避免与 OpenMP 抢核。 |
| `scripts/longitudinal/features.py` | native_lesion 与 baseline_mapped 的 SUV / MTV / TLG / 组学。 |
| `scripts/processing/export_nnunet.py`、`infer_nnunet.py`、`segmentation.py` | AutoPET；CLI 仍可用 `segmentation.py --method threshold` 做批处理。GUI 阈值不调用该 CLI。 |

映射输出（写在**随访**检查目录，不在基线）：

```
data/processed/<ID>/<FollowupDate>/longitudinal/
    baseline_<BLDate>_to_this_0GenericAffine.mat
    baseline_<BLDate>_to_this_affine.txt
    baseline_ct_warped.nii.gz          # 基线 CT → 随访空间（质控窗显示）
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

- [x] 关窗、换患者、换检查时提示保存或放弃未保存 mask；后台任务运行中阻止关闭或切换。
- [x] ANTs 与 AutoPET Python 可通过环境变量配置。

### 映射与质控

- [x] 随访可打开当前 CT / 基线 CT warped / 棋盘格质控窗并记录通过或失败。
- [x] GUI 缺少原生 mask 审阅或随访映射质控时拒绝计算特征，并提示先完成质控。
- [ ] 映射进行中把 ANTs 日志尾部刷到状态栏（现在只有一句「约数分钟」）。

### 分割与编辑

- [ ] 长期只保留分割编辑窗里的形态学/画笔；主窗那一套容易误改且无 CT/PET/融合对照。
- [x] 支持人工记录病灶跨时间对应；持久 `lesion_uid` 独立于连通域编号，mask 改动会使旧对应失效。
- [ ] 器官 mask（肝/脾）在三视图上可选叠加，便于 Deauville / 肝参照目测。
- [ ] Deauville 五分法：用已有肝/脾 SUV 与病灶 SUVmax 算分并写入特征表。

### 显示与性能

- [x] 病灶表：用 `bincount` 统计标签，仅对病灶体素求 SUVmax；未修改 mask 时复用统计结果。
- [x] 体积缓存：`OrderedDict` LRU，上限 6 套；换患者不清空；保存 edited 只更新当前 `VolumeSet`；映射丢掉该患者键，AutoPET 丢掉该检查键。
- [x] 当前检查优先显示，其他时间点的 MIP 在后台预加载；MIP 缓存只保留二维投影，降低重复切换的等待与内存占用。
- [x] GPU 显存只保留最近两次检查的体积，来回切换时复用已上传的数据。
- [x] 渲染 CPU / GPU 二选一（仍画到 pyqtgraph `ImageItem`）。GPU 用可选 CuPy 在设备上切片+伪彩+融合；无 CUDA 时该项禁用并回退 CPU。不换 OpenGL/VisPy。

### 明确不做（保持现状）

- 不覆盖 nnU-Net `*_lesion.nii.gz`，只写 sidecar edited。
- 跨检查不用 SyN：DLBCL 病灶会消退/进展，强变形容易把病灶床拉碎。
- 不做 3D 笔刷（只做当前层圆盘）。
- 不在 GUI 里重跑同机 PET→CT 配准。
