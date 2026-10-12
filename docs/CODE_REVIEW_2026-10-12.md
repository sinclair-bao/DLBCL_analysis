# DLBCL 代码复核与 GUI mask 流程复盘（2026-10-12）

## 复核范围与方法

核对 `scripts/` 下 40 个 Python 文件及 `main.py` 的入口、目录索引、影像载入、分割编辑、版本保存、质控、映射和定量特征调用链；重点逐步检查 GUI 与离线管线之间的输入文件选择。运行静态编译、单元测试、离屏 GUI 冒烟测试及一例真实检查的只读载入。以下“通过”只表示软件行为符合当前代码约定，不代表 DICOM SUV 换算、AutoPET 模型或跨检查配准已完成临床准确性验证。

## 本次修正

| 位置 | 发现的问题 | 修正及验证 |
|---|---|---|
| `scripts/gui/feature_panel.py` | DejaVu Sans 无中文字符，启动后图表绘制重复报 Glyph missing；SUVmax、MTV、TLG 曾叠在同一纵轴上。 | 自动选可用 CJK 字体，三项指标各用独立纵轴；以 `PYTHONWARNINGS=error:Glyph` 完成离屏 GUI 测试。 |
| `scripts/gui/main_window.py`、`segment_editor.py`、`volume_io.py` | 阈值入口可不产生 mask 就保存空稿；无变化编辑会重置已批准质控；非 SUV PET 可被误用作 SUV 阈值；非法浮点 mask 标签可在转 uint16 时悄悄截断。 | 阈值入口要求实际生成或手工修改；已有 mask 无变化不保存；非 SUV 禁用阈值并标明单位；载入前检查标签范围和整数性。 |
| `scripts/gui/workers.py`、`scripts/processing/infer_nnunet.py` | AutoPET 曾使用固定项目目录及旧导出，可能重用旧影像/旧模型输出；路径含空格时推理命令拆错；阈值后台路径直接覆盖 edited mask；模型输出异常时可能影响旧原稿。 | 使用当前目录索引、每次重新导出与推理、独立临时输出、参数列表调用；取消绕过版本记录的阈值后台路径；新病灶 mask 提取成功后才替换原稿。 |
| `scripts/processing/preprocess.py`、`export_nnunet.py`、`segmentation.py` | CT 选择顺序不一致；ACT 可能被当成 SUV 输入；全零 PET 的相对阈值会选中全图。 | CT 优先 `WB_Standard`；模型导出和离线阈值只接受 `*_SUVbw.nii.gz`；全零或非有限 PET 得到空 mask。 |
| `scripts/tools/pacs_dicom_to_nifti_suv.py` | PET Series 被拆分时只取第一份，可能丢失床位覆盖。 | 改为报错并拒绝输出不完整 SUV 图；需按原始扫描协议完成拼接/验证。 |
| `scripts/gui/mask_ops.py`、`display_utils.py`、`render_backend.py`、`scripts/longitudinal/features.py` | 0/255 二值 mask 未分灶；非有限影像值可能影响显示和 SUVpeak。 | 0/255 按连通域编号；CPU/GPU 显示和 SUVpeak 过滤非有限值。 |
| `scripts/processing/register_pet_ct.py` | 缺 CT 参考图时仅凭 PET 和变换矩阵就跳过重跑。 | 三项产物均存在才可跳过。 |
| `scripts/visualization/qc_segmentation.py` | 离线 QC 图会回退到 ACT 图，且只核对 PET/mask 形状、不核对 affine；可能产生误导性 overlay。 | 只读取 SUVbw，并要求 PET/mask 几何一致；补充回归测试。 |
| `scripts/processing/organ_extraction/organ_segmentation.py` | 器官输入 CT 与主流程可能选不同重建；文件说明对 0–10 标签总数写法不一致。 | 同样优先 `WB_Standard`，明确输出为 11 个标签值（含背景）。 |

## mask 编辑、保存与质控复盘

1. 目录或清单索引一次检查后，工作图使用 `registration/ct_iso_reference.nii.gz` + `registration/pet_iso_aligned.nii.gz`，否则使用 `preprocessed/CT` + `preprocessed/PET`；清单模式使用指定路径。缺图像时不进入编辑。
2. 原始模型 mask 为 `*_lesion.nii.gz`；存在 `*_lesion_edited.nii.gz` 时始终优先载入编辑稿。载入会检查标签和 CT 空间关系，最近邻重采样，并对二值 mask 赋病灶编号。
3. 编辑窗在三正交面上修改同一 3D mask，可切换原稿、当前稿和增删差异；取消不写盘。阈值分割仅在 SUVbw 图像上可选。非 SUV 图像可手工编辑，但不得计算 SUV 阈值或现有 SUV 特征。
4. 保存前比较当前磁盘源 mask 与打开时的路径/SHA-256，发现外部修改即拒绝覆盖。已有编辑稿先归档，再原子替换 `*_lesion_edited.nii.gz`，生成版本快照与待审阅记录；模型原稿不改动。保存后重开以 edited 文件为准。无变化编辑不生成版本，也不取消既有批准状态。
5. 质控记录与 mask SHA-256、模型原稿来源绑定；任何文件变化使旧结论过期。GUI 特征工作线程要求原生 mask 质控通过，随访映射也需通过质控。

离屏测试覆盖了涂画、保存、模型原稿不变、版本数、批准后无变化保存、再次编辑使质控待审、重新载入、阈值空稿阻止、非 SUV 阈值阻止、目录/清单切换。真实检查仅进行只读载入与显示；没有在患者真实 mask 上做破坏性写入。

## 尚存问题与科研决策

| 优先级 | 位置 | 问题与影响 | 后续处理 |
|---|---|---|---|
| 高 | `scripts/tools/pacs_dicom_to_nifti_suv.py` | Series 级单一采集时刻计算 SUVbw，无法自动处理逐床位/逐帧时间差；多输出 PET 现在会安全报错，但没有自动重建功能。定量值和体积覆盖可能偏差。 | 逐扫描核对厂家 SUV 图、剂量/衰变/床位时间和完整覆盖；若包含分床位数据，实现受验证的拼接和逐床位换算后再作为科研终点。 |
| 高 | `scripts/longitudinal/interscan_register.py`、`scripts/gui/review_dialog.py` | 跨时间点 CT 映射提供视觉审阅，但没有预设量化接受标准、失败分级或双读者一致性流程；呼吸/体位差可能使基线病灶床定位偏移。 | 研究方案预先规定 QC 条件和失败处理，抽样记录解剖标志误差及读者一致性。 |
| 中 | `scripts/longitudinal/features.py` | SUVpeak 是以 SUVmax 体素为中心的约 1 cm³ 球均值，不是搜索全 ROI 的最大球均值；组学提取失败目前仅记录日志并省略对应列。 | 发表前明确为“近似 SUVpeak”或实现标准定义；增加每行组学成功/失败状态并锁定特征协议。 |
| 中 | `scripts/gui/mask_ops.py`、`scripts/gui/ortho_viewer.py` | 画笔和形态学半径按体素计，不按毫米计；不同体素间距的外部清单会改变实际编辑尺寸。 | 通用化时改为毫米输入，按 affine 换算各轴半径，并标明当前物理尺寸。 |
| 中 | `scripts/longitudinal/catalog.py` | 同目录有多个匹配 CT/PET 时按命名排序取一个，可能选错重建序列；清单 `study_date` 只作为 ID，不校验实际日期顺序。 | 对多候选给出可见选择/显式清单；纵向分析要求 ISO 日期并核对原始 DICOM 日期。 |
| 中 | `scripts/processing/organ_extraction/organ_segmentation.py` | 0–10 器官标签映射按项目代码配置，但尚未与实际 AutoPET 权重和训练集标签逐项实测。 | 若用于 organ head 微调，应先核查每个类别定义与模型输出通道。 |
| 中 | `scripts/longitudinal/features.py` | 命令行 `--require-qc` 默认关闭，可输出未审阅 mask 的探索性特征；GUI 工作线程则强制 QC。 | 正式分析命令固定使用 `--require-qc`，保留探索性命令输出与正式终点分开。 |
| 低 | `main.py`、`scripts/analysis/plot_results.py` | `--stage analyze` 是已声明的占位功能；`--stage all` 不等同于完成纵向分析。 | 队列统计方案确定后实现该阶段，或使用独立纵向特征命令。 |

科研投入前还需在预设病例样本上验证 AutoPET 推理、ANTs 配准、器官分割及组学参数的实际输出，并锁定图像预处理、SUV、分割和读者 QC 版本。当前测试不覆盖 GPU 模型推理、真实 DICOM 多床位重建、跨中心外部验证或临床终点统计。
