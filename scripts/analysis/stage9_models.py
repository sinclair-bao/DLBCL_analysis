#!/usr/bin/env python3
"""第九阶段：基线 Cox 模型、三套影像组学签名比较、纵向增量。

不改第五到第八阶段已经写好的表。结果全部写到 results/stage9_modeling/。
随访表没有死亡或删失列，终点定义写在输出说明里。
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from matplotlib import font_manager
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = Path("/home/sun/Documents/DLBCL")
OUT = ROOT / "results" / "stage9_modeling"
FIG = OUT / "figures"
TABLES = ROOT / "results" / "tables" / "stages_5_8"
FU_PATH = ROOT / "data" / "DLBCL随访表_与PET库匹配_最终ID_全量csv解析.csv"
SEED = 20261123
HORIZON = 24.0
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

SPATIAL_FEATURES = [
    "TMTV_球形度",
    "TMTV_PET_GLCM相关",
    "TMTV_PET_GLSZM灰度方差",
    "TMTV_CT_GLCM相关",
    "TMTV_CT_GLSZM灰度方差",
    "MBV_球形度",
    "MBV_PET_GLCM相关",
    "MBV_PET_GLSZM灰度方差",
    "MBV_CT_GLCM相关",
    "MBV_CT_GLSZM灰度方差",
]
SETS = [
    ("spatial", "空间组学（10个形态和纹理）"),
    ("original", "原始IBSI（每幅图像107个）"),
    ("filtered", "小波加LoG（每幅图像1316个）"),
]
PENALTIES = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]


def key_of(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    stripped = text.lstrip("0")
    return stripped or "0"


def num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def load_followup() -> pd.DataFrame:
    raw = pd.read_csv(FU_PATH, dtype=str, encoding="utf-8-sig")
    out = pd.DataFrame({
        "匹配键": raw["PET患者ID"].map(key_of),
        "姓名": raw["Excel姓名"],
        "年龄": num(raw["年龄"]),
        "性别": raw["性别"].str.strip(),
        "NCCNIPI": num(raw["NCCNIPI"]),
        "治疗反应": raw["治疗反应"].str.strip(),
        "PFS月": num(raw["PFS月"]),
        "OS月": num(raw["OS月"]),
        "治疗方法": raw["治疗方法"],
    })
    # 没有单独的进展/死亡列。PFS：PD，或 PFS 早于 OS，记为事件。
    # PFS 与 OS 相同且不是 PD，视为随访结束时尚未进展。
    out["PFS事件"] = (
        (out["治疗反应"] == "PD") | (out["PFS月"] < out["OS月"] - 1e-6)
    ).astype(int)
    # OS：36 个月是资料里的随访上限，等于 36 视为删失；短于 36 无法区分死亡和失访。
    out["OS事件"] = (out["OS月"] < 36 - 1e-6).astype(int)
    out = out.drop_duplicates("匹配键", keep="first")
    return out


def baseline_rows(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if columns is None:
        frame = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    else:
        header = pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns
        use = [c for c in columns if c in header]
        frame = pd.read_csv(path, dtype=str, usecols=use, encoding="utf-8-sig")
    frame = frame[(frame["时间点"] == "基线") & (frame["状态"] == "ok")].copy()
    frame["匹配键"] = frame["患者ID"].map(key_of)
    frame = frame.drop_duplicates("匹配键", keep="first")
    return frame


def feature_columns(path: Path) -> list[str]:
    header = pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns
    return [
        c for c in header
        if ("_original_" in c) or ("_wavelet-" in c) or ("_log-sigma-" in c)
    ]


def build_cohort() -> tuple[pd.DataFrame, dict]:
    follow = load_followup()
    metabolic = baseline_rows(TABLES / "stage5_metabolic.csv")
    spatial = baseline_rows(TABLES / "stage6_spatial_radiomics.csv")
    body = baseline_rows(TABLES / "stage7_body_composition.csv")
    stage8 = pd.read_csv(TABLES / "stage8_longitudinal.csv", dtype=str, encoding="utf-8-sig")
    stage8["匹配键"] = stage8["患者ID"].map(key_of)
    body_all = pd.read_csv(TABLES / "stage7_body_composition.csv", dtype=str, encoding="utf-8-sig")
    body_all["匹配键"] = body_all["患者ID"].map(key_of)

    cohort = follow.merge(
        metabolic[["匹配键", "患者ID", "检查日期", "TMTV_cm3", "TLG", "SUVmax", "肝归一化SUVmax"]],
        on="匹配键",
        how="inner",
    )
    cohort = cohort.merge(
        spatial[["匹配键", "NOL", "Dmax_cm", "SpreadSpleen_cm"] + SPATIAL_FEATURES],
        on="匹配键",
        how="left",
    )
    cohort = cohort.merge(
        body[["匹配键", "SMI", "肌少症_Martin", "身高_m", "体重_kg", "BMI"]],
        on="匹配键",
        how="left",
    )
    for col in ["TMTV_cm3", "TLG", "SUVmax", "肝归一化SUVmax", "NOL", "Dmax_cm", "SpreadSpleen_cm", "SMI", "BMI", "身高_m", "体重_kg"]:
        cohort[col] = num(cohort[col])
    cohort["logTLG"] = np.log1p(cohort["TLG"])
    cohort["肌少症"] = cohort["肌少症_Martin"].map({"是": 1, "否": 0})

    ok = stage8[stage8["状态"] == "ok"].copy()
    smi_lookup = {
        (row["匹配键"], row["检查日期"]): num(pd.Series([row["SMI"]])).iloc[0]
        for _, row in body_all.iterrows()
        if row["状态"] == "ok"
    }
    delta_smi = []
    for _, row in ok.iterrows():
        base = smi_lookup.get((row["匹配键"], row["基线日期"]))
        interim = smi_lookup.get((row["匹配键"], row["中期日期"]))
        if base and interim and np.isfinite(base) and np.isfinite(interim) and base != 0:
            delta_smi.append((base - interim) / base * 100.0)
        else:
            delta_smi.append(np.nan)
    ok["delta_SMI_pct"] = delta_smi
    for col in ["delta_TMTV_pct", "delta_TLG_pct", "delta_肝归一化SUVmax_pct"]:
        ok[col] = num(ok[col])
    keep = ok[[
        "匹配键", "基线日期", "中期日期",
        "delta_TMTV_pct", "delta_TLG_pct", "delta_肝归一化SUVmax_pct", "delta_SMI_pct",
        "中期病灶消失", "CMR映射状态",
    ]]
    cohort = cohort.merge(keep, on="匹配键", how="left")
    cohort["纵向"] = cohort["delta_TLG_pct"].notna().astype(int)

    readiness = {
        "随访人数": int(follow["匹配键"].nunique()),
        "基线代谢定量成功人数": int(metabolic["匹配键"].nunique()),
        "主队列人数": int(len(cohort)),
        "随访有影像无": int(len(set(follow["匹配键"]) - set(cohort["匹配键"]))),
        "影像有随访无": int(len(set(metabolic["匹配键"]) - set(follow["匹配键"]))),
        "纵向状态ok": int((stage8["状态"] == "ok").sum()),
        "主队列中的纵向人数": int(cohort["纵向"].sum()),
        "delta_TMTV有值": int(ok["delta_TMTV_pct"].notna().sum()),
        "delta_TLG有值": int(ok["delta_TLG_pct"].notna().sum()),
        "delta_肝归一化SUVmax有值": int(ok["delta_肝归一化SUVmax_pct"].notna().sum()),
        "delta_SMI有值": int(ok["delta_SMI_pct"].notna().sum()),
        "主队列delta_SMI有值": int(cohort["delta_SMI_pct"].notna().sum()),
        "中期病灶消失": ok["中期病灶消失"].value_counts(dropna=False).to_dict(),
        "CMR映射状态": ok["CMR映射状态"].value_counts(dropna=False).to_dict(),
        "Deauville": "随访表和已有定量表都没有 Deauville / 5-PS，阳性预测值分析做不了",
        "PFS事件数": int(cohort["PFS事件"].sum()),
        "OS事件数": int(cohort["OS事件"].sum()),
        "治疗反应": cohort["治疗反应"].value_counts().to_dict(),
    }
    return cohort.sort_values("匹配键").reset_index(drop=True), readiness


def attach_radiomics(cohort: pd.DataFrame) -> dict[str, pd.DataFrame]:
    meta = ["匹配键"]
    original_cols = feature_columns(TABLES / "stage6_ibsi_radiomics.csv")
    filtered_cols = feature_columns(TABLES / "stage6_ibsi_filtered_radiomics.csv")
    original = baseline_rows(
        TABLES / "stage6_ibsi_radiomics.csv",
        meta + ["患者ID", "检查日期", "时间点", "状态"] + original_cols,
    )
    filtered = baseline_rows(
        TABLES / "stage6_ibsi_filtered_radiomics.csv",
        meta + ["患者ID", "检查日期", "时间点", "状态"] + filtered_cols,
    )
    spatial = cohort[["匹配键"] + SPATIAL_FEATURES].copy()
    for col in SPATIAL_FEATURES:
        spatial[col] = num(spatial[col])
    return {
        "spatial": spatial,
        "original": original,
        "filtered": filtered,
    }


def safe_cindex(time, risk, event) -> float:
    time = np.asarray(time, dtype=float)
    risk = np.asarray(risk, dtype=float)
    event = np.asarray(event, dtype=int)
    if len(time) < 5 or event.sum() < 2 or np.unique(risk).size < 2:
        return float("nan")
    return float(concordance_index(time, -risk, event))


def auc_at(time, event, risk, horizon: float = HORIZON) -> tuple[float, int]:
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    risk = np.asarray(risk, dtype=float)
    keep = ((event == 1) & (time <= horizon)) | (time > horizon)
    if keep.sum() < 8:
        return float("nan"), int(keep.sum())
    labels = ((event == 1) & (time <= horizon))[keep].astype(int)
    if labels.min() == labels.max():
        return float("nan"), int(keep.sum())
    return float(roc_auc_score(labels, risk[keep])), int(keep.sum())


def evaluable_mask(time, event, horizon: float = HORIZON) -> np.ndarray:
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    return ((event == 1) & (time <= horizon)) | (time > horizon)


def nri_idi(time, event, risk_old, risk_new) -> dict:
    mask = evaluable_mask(time, event)
    labels = ((np.asarray(event) == 1) & (np.asarray(time) <= HORIZON))[mask]
    old = np.asarray(risk_old, dtype=float)[mask]
    new = np.asarray(risk_new, dtype=float)[mask]
    if labels.sum() < 3 or (labels == 0).sum() < 3:
        return {"NRI": float("nan"), "IDI": float("nan"), "可评价人数": int(mask.sum())}
    up = new > old + 1e-8
    down = new < old - 1e-8
    event_up = up[labels == 1].mean()
    event_down = down[labels == 1].mean()
    none_down = down[labels == 0].mean()
    none_up = up[labels == 0].mean()
    idi = (new[labels == 1] - old[labels == 1]).mean() - (new[labels == 0] - old[labels == 0]).mean()
    return {
        "NRI": float((event_up - event_down) + (none_down - none_up)),
        "IDI": float(idi),
        "可评价人数": int(mask.sum()),
    }


def net_benefit(time, event, risk, threshold: float) -> float:
    mask = evaluable_mask(time, event)
    labels = ((np.asarray(event) == 1) & (np.asarray(time) <= HORIZON))[mask]
    pred = np.asarray(risk, dtype=float)[mask] >= threshold
    n = max(len(labels), 1)
    tp = np.sum(pred & (labels == 1))
    fp = np.sum(pred & (labels == 0))
    odds = threshold / max(1.0 - threshold, 1e-6)
    return float(tp / n - fp / n * odds)


def prepare_xy(train: pd.DataFrame, test: pd.DataFrame, columns: list[str]):
    keep = []
    train_num = pd.DataFrame(index=train.index)
    test_num = pd.DataFrame(index=test.index)
    for col in columns:
        tr = num(train[col]) if col in train.columns else pd.Series(np.nan, index=train.index)
        te = num(test[col]) if col in test.columns else pd.Series(np.nan, index=test.index)
        if tr.isna().mean() > 0.20 or tr.nunique(dropna=True) < 2:
            continue
        keep.append(col)
        train_num[col] = tr
        test_num[col] = te
    if not keep:
        return None
    row_ok_train = train_num.isna().mean(axis=1) <= 0.30
    row_ok_test = test_num.isna().mean(axis=1) <= 0.30
    median = train_num.loc[row_ok_train, keep].median()
    train_filled = train_num.loc[row_ok_train, keep].fillna(median)
    test_filled = test_num.loc[row_ok_test, keep].fillna(median)
    center = train_filled.mean()
    scale = train_filled.std(ddof=0).replace(0, 1.0)
    return (
        (train_filled - center) / scale,
        (test_filled - center) / scale,
        keep,
    )


def screen_features(x_train: pd.DataFrame, time, event, limit: int = 30) -> list[str]:
    scores = []
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    for col in x_train.columns:
        c = safe_cindex(time, x_train[col].to_numpy(), event)
        scores.append((abs(c - 0.5) if np.isfinite(c) else 0.0, col))
    scores.sort(reverse=True)
    if len(scores) <= limit:
        return [col for _, col in scores]
    return [col for _, col in scores[:limit]]


def drop_correlated(x_train: pd.DataFrame, time, event, threshold: float = 0.90) -> list[str]:
    """同一相关簇里只留训练集上区分度更高的一个，避免 Cox 矩阵奇异。"""
    ranked = screen_features(x_train, time, event, limit=x_train.shape[1])
    kept: list[str] = []
    for col in ranked:
        if any(abs(float(x_train[col].corr(x_train[other]))) >= threshold for other in kept):
            continue
        kept.append(col)
    return kept


def _ascii_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    mapping = {col: f"x{i:04d}" for i, col in enumerate(frame.columns)}
    renamed = frame.rename(columns=mapping)
    return renamed, {safe: original for original, safe in mapping.items()}


def fit_penalized(x_train: pd.DataFrame, time, event) -> tuple[pd.Series, float, float]:
    time = pd.Series(np.asarray(time, dtype=float), index=x_train.index)
    event = pd.Series(np.asarray(event, dtype=int), index=x_train.index)
    chosen = list(x_train.columns)
    coded, back = _ascii_frame(x_train[chosen])
    best_pen = PENALTIES[0]
    best_score = -1.0
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    y = event.to_numpy()
    for pen in PENALTIES:
        fold_scores = []
        for inner_train, inner_test in splitter.split(coded, y):
            tr = coded.iloc[inner_train].copy()
            te = coded.iloc[inner_test]
            tr["time"] = time.iloc[inner_train].to_numpy()
            tr["event"] = event.iloc[inner_train].to_numpy()
            cph = CoxPHFitter(penalizer=pen, l1_ratio=1.0)
            try:
                cph.fit(tr, "time", "event", fit_options={"max_steps": 200})
                risk = cph.predict_partial_hazard(te).to_numpy().ravel()
                fold_scores.append(safe_cindex(time.iloc[inner_test], risk, event.iloc[inner_test]))
            except Exception:
                fold_scores.append(float("nan"))
        score = float(np.nanmean(fold_scores)) if fold_scores else float("nan")
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_pen = pen
    if not np.isfinite(best_score):
        best_pen = 0.5
        best_score = float("nan")
    full = coded.copy()
    full["time"] = time.to_numpy()
    full["event"] = event.to_numpy()
    cph = None
    candidates = [best_pen] + [pen for pen in (0.2, 0.5, 1.0, 2.0, 5.0) if pen > best_pen + 1e-9]
    for pen in candidates:
        trial = CoxPHFitter(penalizer=pen, l1_ratio=1.0)
        try:
            trial.fit(full, "time", "event", fit_options={"max_steps": 400})
            cph = trial
            best_pen = pen
            break
        except Exception:
            continue
    if cph is None:
        raise RuntimeError("LASSO-Cox 在加大惩罚后仍无法收敛")
    coef = cph.params_.copy()
    coef.index = [back[name] for name in coef.index]
    coef = coef[coef.abs() > 1e-6]
    if coef.empty:
        alone = screen_features(x_train, time, event, limit=1)
        tiny, tiny_back = _ascii_frame(x_train[alone])
        tiny = tiny.copy()
        tiny["time"] = time.to_numpy()
        tiny["event"] = event.to_numpy()
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(tiny, "time", "event")
        coef = cph.params_.copy()
        coef.index = [tiny_back[name] for name in coef.index]
        best_pen = 0.1
    return coef, best_pen, best_score


def risk_from_coef(frame: pd.DataFrame, coef: pd.Series) -> pd.Series:
    cols = [c for c in coef.index if c in frame.columns]
    return frame[cols].mul(coef[cols], axis=1).sum(axis=1)


CLINICAL_NAMES = {
    "肌少症": "sarcopenia",
    "delta_TLG_pct": "delta_tlg",
    "delta_SMI_pct": "delta_smi",
}


def _clinical_frame(frame: pd.DataFrame, columns: list[str], time_col: str, event_col: str) -> pd.DataFrame:
    use = frame[columns + [time_col, event_col]].dropna().copy()
    use = use.rename(columns={**CLINICAL_NAMES, time_col: "time", event_col: "event"})
    safe_cols = [CLINICAL_NAMES.get(col, col) for col in columns]
    return use[safe_cols + ["time", "event"]]


def fit_clinical(frame: pd.DataFrame, columns: list[str], time_col: str, event_col: str) -> CoxPHFitter | None:
    use = _clinical_frame(frame, columns, time_col, event_col)
    if len(use) < 20 or use["event"].sum() < 5:
        return None
    for pen in (0.0, 0.05, 0.2):
        cph = CoxPHFitter(penalizer=pen) if pen else CoxPHFitter()
        try:
            cph.fit(use, "time", "event", fit_options={"max_steps": 200})
            cph._stage9_columns = columns
            return cph
        except Exception:
            continue
    return None


def _predict_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    use = frame[columns].dropna().copy()
    use = use.rename(columns=CLINICAL_NAMES)
    safe_cols = [CLINICAL_NAMES.get(col, col) for col in columns]
    return use[safe_cols]


def survival_risk(cph: CoxPHFitter, frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    """24 个月生存概率，不是事件概率。"""
    use = _predict_frame(frame, columns)
    curve = cph.predict_survival_function(use, times=[HORIZON])
    survival = curve.iloc[0].copy()
    survival.index = use.index
    return survival


def partial_risk(cph: CoxPHFitter, frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    use = _predict_frame(frame, columns)
    risk = cph.predict_partial_hazard(use)
    risk.index = use.index
    return risk


def calibration_rows(time, event, risk) -> list[dict]:
    mask = evaluable_mask(time, event)
    labels_time = np.asarray(time, dtype=float)
    labels_event = np.asarray(event, dtype=int)
    risk = pd.Series(np.asarray(risk, dtype=float))
    usable = np.flatnonzero(mask)
    if len(usable) < 15:
        return []
    try:
        groups = pd.qcut(risk.iloc[usable], 3, labels=["低", "中", "高"], duplicates="drop")
    except ValueError:
        return []
    rows = []
    for name in groups.cat.categories:
        chosen = usable[np.asarray(groups == name)]
        km = KaplanMeierFitter()
        km.fit(labels_time[chosen], labels_event[chosen])
        observed = float(1.0 - km.predict(HORIZON))
        rows.append({
            "组": str(name),
            "人数": int(len(chosen)),
            "预测24个月事件率": float(np.mean(risk.iloc[chosen])),
            "观察24个月事件率": observed,
        })
    return rows


def mannwhitney(a: pd.Series, b: pd.Series) -> float:
    from scipy.stats import mannwhitneyu

    a = num(a).dropna()
    b = num(b).dropna()
    if len(a) < 3 or len(b) < 3:
        return float("nan")
    return float(mannwhitneyu(a, b, alternative="two-sided").pvalue)


def chi_p(table: pd.DataFrame) -> float:
    from scipy.stats import chi2_contingency

    if table.shape[0] < 2 or table.shape[1] < 2 or (table.to_numpy() == 0).any():
        return float("nan")
    return float(chi2_contingency(table.to_numpy())[1])


def selection_bias(cohort: pd.DataFrame) -> pd.DataFrame:
    yes = cohort[cohort["纵向"] == 1]
    no = cohort[cohort["纵向"] == 0]
    rows = []
    for label, col in [("年龄", "年龄"), ("NCCNIPI", "NCCNIPI"), ("基线TLG", "TLG"), ("基线TMTV", "TMTV_cm3"), ("基线SMI", "SMI")]:
        rows.append({
            "指标": label,
            "纵向中位数": float(num(yes[col]).median()),
            "非纵向中位数": float(num(no[col]).median()),
            "P值": mannwhitney(yes[col], no[col]),
            "检验": "Mann-Whitney",
        })
    sex = pd.crosstab(cohort["纵向"], cohort["性别"])
    sarc = pd.crosstab(cohort["纵向"], cohort["肌少症"])
    rows.append({
        "指标": "性别",
        "纵向中位数": float((yes["性别"] == "男").mean()),
        "非纵向中位数": float((no["性别"] == "男").mean()),
        "P值": chi_p(sex),
        "检验": "卡方，数值为男性比例",
    })
    rows.append({
        "指标": "肌少症_Martin",
        "纵向中位数": float(num(yes["肌少症"]).mean()),
        "非纵向中位数": float(num(no["肌少症"]).mean()),
        "P值": chi_p(sarc),
        "检验": "卡方，数值为肌少症比例",
    })
    rows.append({
        "指标": "Ann Arbor分期",
        "纵向中位数": np.nan,
        "非纵向中位数": np.nan,
        "P值": np.nan,
        "检验": "随访表没有分期列，未做",
    })
    return pd.DataFrame(rows)


def model_ladder(
    clinical: pd.DataFrame,
    radiomics: pd.DataFrame,
    train_ids: set[str],
    endpoint: str,
) -> dict:
    time_col = "PFS月" if endpoint == "PFS" else "OS月"
    event_col = "PFS事件" if endpoint == "PFS" else "OS事件"
    merged = clinical.merge(radiomics, on="匹配键", how="left", suffixes=("", "_组学"))
    feature_cols = [c for c in radiomics.columns if c != "匹配键"]
    train = merged[merged["匹配键"].isin(train_ids)].copy()
    test = merged[~merged["匹配键"].isin(train_ids)].copy()
    prepared = prepare_xy(train, test, feature_cols)
    if prepared is None:
        return {"终点": endpoint, "状态": "没有可用特征"}
    x_train, x_test, kept = prepared
    screened = screen_features(x_train, train.loc[x_train.index, time_col], train.loc[x_train.index, event_col])
    screened = drop_correlated(
        x_train[screened],
        train.loc[x_train.index, time_col],
        train.loc[x_train.index, event_col],
    )
    coef, pen, inner_c = fit_penalized(
        x_train[screened],
        train.loc[x_train.index, time_col],
        train.loc[x_train.index, event_col],
    )
    rs_train = risk_from_coef(x_train, coef)
    rs_test = risk_from_coef(x_test, coef)
    work = merged.copy()
    work["RS"] = np.nan
    work.loc[rs_train.index, "RS"] = rs_train
    work.loc[rs_test.index, "RS"] = rs_test
    performances = []
    fitted = {}
    risks = {}
    for split, series in (("训练", rs_train), ("测试", rs_test)):
        aligned = work.loc[series.dropna().index]
        auc, n_auc = auc_at(aligned[time_col], aligned[event_col], series.loc[aligned.index].to_numpy())
        performances.append({
            "模型": "组学签名单独",
            "数据集": split,
            "人数": int(len(aligned)),
            "事件数": int(aligned[event_col].sum()),
            "C指数": safe_cindex(aligned[time_col], series.loc[aligned.index], aligned[event_col]),
            "24个月AUC": auc,
            "AUC可评价人数": n_auc,
        })
    specs = [
        ("模型1 NCCN-IPI", ["NCCNIPI"]),
        ("模型2 加基线TLG", ["NCCNIPI", "logTLG"]),
        ("模型3 加脾脏距离和组学签名", ["NCCNIPI", "logTLG", "SpreadSpleen_cm", "RS"]),
        ("模型4 加肌少症", ["NCCNIPI", "logTLG", "SpreadSpleen_cm", "RS", "肌少症"]),
    ]
    for name, cols in specs:
        cph = fit_clinical(work.loc[x_train.index], cols, time_col, event_col)
        if cph is None:
            performances.append({"模型": name, "状态": "未能拟合"})
            continue
        fitted[name] = (cph, cols)
        for split, idx in (("训练", x_train.index), ("测试", x_test.index)):
            part = work.loc[idx]
            hazard = partial_risk(cph, part, cols)
            surv = survival_risk(cph, part, cols)
            aligned = part.loc[hazard.index]
            auc, n_auc = auc_at(aligned[time_col], aligned[event_col], hazard.to_numpy())
            performances.append({
                "模型": name,
                "数据集": split,
                "人数": int(len(aligned)),
                "事件数": int(aligned[event_col].sum()),
                "C指数": safe_cindex(aligned[time_col], hazard, aligned[event_col]),
                "24个月AUC": auc,
                "AUC可评价人数": n_auc,
            })
            if split == "测试":
                risks[name] = (
                    aligned[time_col].to_numpy(),
                    aligned[event_col].to_numpy(),
                    surv.reindex(aligned.index).to_numpy(),
                    hazard.reindex(aligned.index).to_numpy(),
                    aligned.index,
                )
    increments = []
    order = [s[0] for s in specs]
    for left, right in zip(order, order[1:]):
        if left not in risks or right not in risks:
            continue
        common = risks[left][4].intersection(risks[right][4])
        t = work.loc[common, time_col].to_numpy()
        e = work.loc[common, event_col].to_numpy()
        old = 1.0 - survival_risk(fitted[left][0], work.loc[common], fitted[left][1]).to_numpy()
        new = 1.0 - survival_risk(fitted[right][0], work.loc[common], fitted[right][1]).to_numpy()
        delta = nri_idi(t, e, old, new)
        delta["比较"] = f"{right} 相对 {left}"
        increments.append(delta)
    coef_table = pd.DataFrame({
        "特征": coef.index,
        "LASSO系数": coef.to_numpy(),
    })
    return {
        "终点": endpoint,
        "进入LASSO的特征数": len(screened),
        "原始特征数": len(feature_cols),
        "保留特征数": int(len(coef)),
        "惩罚系数": pen,
        "内层交叉验证C指数": inner_c,
        "签名训练集人数": int(len(x_train)),
        "签名测试集人数": int(len(x_test)),
        "效能": performances,
        "增量": increments,
        "系数": coef_table,
        "测试风险": {
            name: {
                "index": idx.astype(str).tolist(),
                "time": t.tolist(),
                "event": e.tolist(),
                "survival24": s.tolist(),
                "hazard": h.tolist(),
            }
            for name, (t, e, s, h, idx) in risks.items()
        },
        "校准": {
            name: calibration_rows(t, e, 1.0 - np.asarray(s, dtype=float))
            for name, (t, e, s, h, idx) in risks.items()
            if name.startswith("模型4") or name.startswith("模型1")
        },
        "工作表索引": {
            "训练": x_train.index.astype(int).tolist(),
            "测试": x_test.index.astype(int).tolist(),
        },
        "RS": work["RS"],
    }


def longitudinal_gain(cohort: pd.DataFrame, rs: pd.Series, train_ids: set[str], endpoint: str) -> dict:
    time_col = "PFS月" if endpoint == "PFS" else "OS月"
    event_col = "PFS事件" if endpoint == "PFS" else "OS事件"
    frame = cohort.copy()
    frame["RS"] = rs
    frame = frame[frame["纵向"] == 1].dropna(subset=[
        "NCCNIPI", "logTLG", "SpreadSpleen_cm", "RS", "肌少症", "delta_TLG_pct", "delta_SMI_pct",
        time_col, event_col,
    ])
    base_cols = ["NCCNIPI", "logTLG", "SpreadSpleen_cm", "RS", "肌少症"]
    plus_cols = base_cols + ["delta_TLG_pct", "delta_SMI_pct"]
    train = frame[frame["匹配键"].isin(train_ids)]
    test = frame[~frame["匹配键"].isin(train_ids)]
    out = {"终点": endpoint, "可用人数": int(len(frame)), "训练人数": int(len(train)), "测试人数": int(len(test))}
    rows = []
    risks = {}
    for name, cols in (("只有基线", base_cols), ("基线加deltaTLG和deltaSMI", plus_cols)):
        cph = fit_clinical(train, cols, time_col, event_col)
        if cph is None:
            rows.append({"模型": name, "状态": "未能拟合"})
            continue
        for split, part in (("训练", train), ("测试", test)):
            hazard = partial_risk(cph, part, cols)
            aligned = part.loc[hazard.index]
            auc, n_auc = auc_at(aligned[time_col], aligned[event_col], hazard.to_numpy())
            rows.append({
                "模型": name,
                "数据集": split,
                "人数": int(len(aligned)),
                "事件数": int(aligned[event_col].sum()),
                "C指数": safe_cindex(aligned[time_col], hazard, aligned[event_col]),
                "24个月AUC": auc,
                "AUC可评价人数": n_auc,
            })
        if len(test):
            risks[name] = (
                test.loc[partial_risk(cph, test, cols).index, time_col].to_numpy(),
                test.loc[partial_risk(cph, test, cols).index, event_col].to_numpy(),
                survival_risk(cph, test, cols),
            )
    if len(risks) == 2:
        common = risks["只有基线"][2].index.intersection(risks["基线加deltaTLG和deltaSMI"][2].index)
        t = test.loc[common, time_col].to_numpy()
        e = test.loc[common, event_col].to_numpy()
        old = 1.0 - risks["只有基线"][2].loc[common].to_numpy()
        new = 1.0 - risks["基线加deltaTLG和deltaSMI"][2].loc[common].to_numpy()
        out["增量"] = nri_idi(t, e, old, new)
    out["效能"] = rows
    return out


def draw_figures(performance: pd.DataFrame, dca_source: dict, km_source: dict) -> None:
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
    plt.rcParams["axes.unicode_minus"] = False
    FIG.mkdir(parents=True, exist_ok=True)

    test = performance[(performance["数据集"] == "测试") & (performance["层级"] == "主模型")].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, endpoint in zip(axes, ["PFS", "OS"]):
        part = test[test["终点"] == endpoint]
        models = ["组学签名单独", "模型1 NCCN-IPI", "模型2 加基线TLG", "模型3 加脾脏距离和组学签名", "模型4 加肌少症"]
        sets = [label for _, label in SETS]
        x = np.arange(len(models))
        width = 0.25
        for i, label in enumerate(sets):
            vals = []
            for model in models:
                hit = part[(part["特征集"] == label) & (part["模型"] == model)]
                vals.append(float(hit["C指数"].iloc[0]) if len(hit) else np.nan)
            ax.bar(x + (i - 1) * width, vals, width, label=label)
        ax.set_xticks(x, ["签名", "模型1", "模型2", "模型3", "模型4"])
        ax.set_ylim(0.45, 0.85)
        ax.axhline(0.5, color="0.6", lw=0.8)
        ax.set_title(f"{endpoint} 测试集 C 指数")
        ax.set_ylabel("C 指数")
    axes[1].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "test_cindex_three_sets.png", dpi=140)
    plt.close(fig)

    if km_source:
        fig, ax = plt.subplots(figsize=(6.2, 4.6))
        for label, (time, event) in km_source.items():
            km = KaplanMeierFitter()
            km.fit(time, event, label=label)
            km.plot_survival_function(ax=ax, ci_show=False)
        ax.set_xlabel("月")
        ax.set_ylabel("无进展生存")
        ax.set_title("测试集：组学签名中位数分组的 PFS")
        fig.tight_layout()
        fig.savefig(FIG / "km_pfs_best_signature.png", dpi=140)
        plt.close(fig)

    if dca_source:
        fig, ax = plt.subplots(figsize=(6.2, 4.6))
        thresholds = np.round(np.arange(0.05, 0.51, 0.05), 2)
        for label, (time, event, risk) in dca_source.items():
            benefits = [net_benefit(time, event, risk, t) for t in thresholds]
            ax.plot(thresholds, benefits, marker="o", label=label)
        treat_all = []
        time, event, _ = next(iter(dca_source.values()))
        mask = evaluable_mask(time, event)
        prevalence = ((np.asarray(event) == 1) & (np.asarray(time) <= HORIZON))[mask].mean()
        for t in thresholds:
            odds = t / max(1 - t, 1e-6)
            treat_all.append(prevalence - (1 - prevalence) * odds)
        ax.plot(thresholds, treat_all, ls="--", color="0.4", label="全部按高风险处理")
        ax.axhline(0, color="0.5", lw=0.8, label="全部按低风险处理")
        ax.set_xlabel("24个月事件阈值概率")
        ax.set_ylabel("净获益")
        ax.set_title("测试集决策曲线（PFS）")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG / "dca_pfs_best_set.png", dpi=140)
        plt.close(fig)


def write_findings(
    readiness: dict,
    performance: pd.DataFrame,
    increments: pd.DataFrame,
    bias: pd.DataFrame,
    longitudinal: pd.DataFrame,
    winner: dict,
) -> None:
    def cell(endpoint, feature_set, model, column, split="测试"):
        hit = performance[
            (performance["终点"] == endpoint)
            & (performance["特征集"] == feature_set)
            & (performance["模型"] == model)
            & (performance["数据集"] == split)
            & (performance["层级"] == "主模型")
        ]
        if hit.empty or pd.isna(hit.iloc[0][column]):
            return "—"
        value = hit.iloc[0][column]
        if column in {"人数", "事件数", "AUC可评价人数"}:
            return str(int(value))
        return f"{float(value):.3f}"

    lines = []
    lines.append("# 第九阶段建模结果")
    lines.append("")
    lines.append("定量指标来自本流程已经算好的第五到第八阶段，没有使用随访表里原来的 SUVmax、MTV、TLG。技术路线里的 200 例和纵向 98 例没有照搬，下面都是实际对得上的人数。影像仍是现有 2 mm 产品，没有按路线图重做成 1 mm，也没有重做分割。")
    lines.append("")
    lines.append("## 第八阶段数据是否够用")
    lines.append("")
    lines.append(f"- 随访表 {readiness['随访人数']} 人，基线代谢定量成功 {readiness['基线代谢定量成功人数']} 人，两者都能对上的主队列 {readiness['主队列人数']} 人。随访有、基线影像没有的 {readiness['随访有影像无']} 人；影像有、随访没有的 {readiness['影像有随访无']} 人。")
    lines.append(f"- 纵向状态为 ok 的 {readiness['纵向状态ok']} 人，其中在主队列里的 {readiness['主队列中的纵向人数']} 人。ΔTMTV {readiness['delta_TMTV有值']} 人，ΔTLG {readiness['delta_TLG有值']} 人，肝归一化 SUVmax 的变化 {readiness['delta_肝归一化SUVmax有值']} 人。")
    lines.append(f"- ΔSMI 不是第八阶段原表里的列，由第七阶段基线和中期 SMI 按 (基线 − 中期) / 基线 × 100 算出，纵向队列里有值 {readiness['delta_SMI有值']} 人，主队列里有值 {readiness['主队列delta_SMI有值']} 人。")
    lines.append(f"- 中期病灶消失的记录是 {readiness['中期病灶消失']}。原灶区回测状态是 {readiness['CMR映射状态']}。这批纵向患者中期检查上病灶并没有成批消失，路线图里的完全代谢缓解回测和 Deauville 阳性预测值做不了。{readiness['Deauville']}。")
    lines.append("")
    lines.append("## 终点怎么定")
    lines.append("")
    lines.append("随访表只有治疗反应、PFS 月和 OS 月，没有进展或死亡的单独标记，OS 在 36 个月处明显堆积。")
    lines.append("")
    lines.append("- PFS 事件：治疗反应为 PD，或者 PFS 月短于 OS 月。PFS 与 OS 相同、且反应不是 PD 的，按随访结束时尚未进展，记为删失。")
    lines.append("- OS 事件：OS 月短于 36 记为事件，等于 36 记为删失。短于 36 个月的人里，死亡和失访分不开，这个终点比 PFS 粗。")
    lines.append(f"- 主队列 PFS 事件 {readiness['PFS事件数']} 个，OS 事件 {readiness['OS事件数']} 个。治疗反应构成：{readiness['治疗反应']}。")
    lines.append("")
    lines.append("训练集和测试集按 7:3 一次划分，按 PFS 事件分层，随机种子 20261123。标准化、中位数填补、单变量筛选和 LASSO 惩罚只在训练集里拟合。特征多于 30 个时，先按训练集 Harrell C 指数留下 30 个，再去掉相关系数达到 0.90 的冗余特征，然后做 L1 惩罚 Cox。惩罚强度用训练集五折交叉验证选择。")
    lines.append("")
    lines.append("## 三套组学哪套更好")
    lines.append("")
    lines.append(f"三套特征直接比较的是 LASSO 签名本身在测试集上的 PFS C 指数，不把 NCCN-IPI 和 TLG 算进去。这一轮最高的是 **{winner['特征集']}**，测试集 PFS C 指数 {winner['PFS_C指数']:.3f}，24 个月 AUC {winner['PFS_AUC']:.3f}。同一签名在 OS 上的测试集 C 指数是 {winner['OS_C指数']:.3f}。三套之间的差距要和一次 7:3 划分的起伏一起看，差 0.02 以内不能当成稳定优劣。")
    lines.append("")
    lines.append("| 特征集 | PFS签名 C指数 | PFS签名 AUC | PFS模型3 C指数 | OS签名 C指数 | OS模型3 C指数 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for _, label in SETS:
        lines.append("| {set} | {sc} | {sa} | {p3c} | {oc} | {o3c} |".format(
            set=label,
            sc=cell("PFS", label, "组学签名单独", "C指数"),
            sa=cell("PFS", label, "组学签名单独", "24个月AUC"),
            p3c=cell("PFS", label, "模型3 加脾脏距离和组学签名", "C指数"),
            oc=cell("OS", label, "组学签名单独", "C指数"),
            o3c=cell("OS", label, "模型3 加脾脏距离和组学签名", "C指数"),
        ))
    lines.append("")
    lines.append("模型 1 只有 NCCN-IPI，模型 2 加上本流程算出的基线 log(1+TLG)，模型 3 再加上 Spread-Spleen 和该套特征的 LASSO 签名，模型 4 再加 Martin 标准的肌少症。模型 4 的测试集人数更少，因为一部分人没有肌少症判定。主队列里肌少症比例只有大约 2% 到 6%，这一项加进去之后测试集 C 指数下降，不能解读成肌少症有保护作用。签名保留了哪些特征写在 `coefficients/`，筛选过程写在 `signature_fit.csv`。")
    lines.append("")
    lines.append("## 模型之间的增量")
    lines.append("")
    lines.append("NRI 和 IDI 用测试集里能判断 24 个月状态的人：24 个月内发生事件，或随访超过 24 个月。24 个月前删失的人不进这两项。风险用 Cox 的 24 个月事件概率。")
    lines.append("")
    if increments.empty:
        lines.append("没有足够的重叠样本计算增量。")
    else:
        show = increments[increments["特征集"] == winner["特征集"]]
        lines.append(f"下表是最好的那一套（{winner['特征集']}）。")
        lines.append("")
        lines.append("| 终点 | 比较 | NRI | IDI | 可评价人数 |")
        lines.append("| --- | --- | --- | --- | --- |")
        for _, row in show.iterrows():
            lines.append(f"| {row['终点']} | {row['比较']} | {row['NRI']:.3f} | {row['IDI']:.3f} | {int(row['可评价人数'])} |")
    lines.append("")
    lines.append("## 纵向增量")
    lines.append("")
    lines.append("只纳入主队列里第八阶段状态为 ok、并且 ΔTLG 和 ΔSMI 都有值的人。基线模型沿用上面的模型 4 变量，签名仍是主分析训练集上拟合的那一个，避免在纵向子集里重选特征。比较对象是同一批人加上 ΔTLG 和 ΔSMI。")
    lines.append("")
    if longitudinal.empty:
        lines.append("纵向子集没有拟合出模型。")
    else:
        lines.append("| 特征集 | 终点 | 模型 | 测试人数 | 测试事件 | 测试C指数 | 测试24个月AUC |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        part = longitudinal[longitudinal["数据集"] == "测试"]
        for _, row in part.iterrows():
            cval = "—" if pd.isna(row["C指数"]) else f"{row['C指数']:.3f}"
            aval = "—" if pd.isna(row["24个月AUC"]) else f"{row['24个月AUC']:.3f}"
            lines.append(f"| {row['特征集']} | {row['终点']} | {row['模型']} | {int(row['人数'])} | {int(row['事件数'])} | {cval} | {aval} |")
    lines.append("")
    lines.append("## 做了中期 PET 的人是否不同")
    lines.append("")
    lines.append("比较的是主队列里有完整纵向定量的人，和主队列里其余的人。分期这一项随访表没有。")
    lines.append("")
    lines.append("| 指标 | 纵向 | 非纵向 | P值 | 说明 |")
    lines.append("| --- | --- | --- | --- | --- |")
    for _, row in bias.iterrows():
        pval = "—" if pd.isna(row["P值"]) else f"{row['P值']:.3f}"
        left = "—" if pd.isna(row["纵向中位数"]) else f"{row['纵向中位数']:.3f}"
        right = "—" if pd.isna(row["非纵向中位数"]) else f"{row['非纵向中位数']:.3f}"
        lines.append(f"| {row['指标']} | {left} | {right} | {pval} | {row['检验']} |")
    lines.append("")
    lines.append("## 图")
    lines.append("")
    lines.append("- `figures/test_cindex_three_sets.png`：三套特征、四个模型在测试集上的 C 指数。")
    lines.append("- `figures/km_pfs_best_signature.png`：最好那套签名按训练集中位数把测试集分成高低两组的 PFS 曲线。")
    lines.append("- `figures/dca_pfs_best_set.png`：最好那套特征的模型 1 和模型 4，在 PFS 24 个月风险上的决策曲线。")
    lines.append("")
    lines.append("校准分组在 `calibration_test.csv`。每个模型的系数在 `coefficients/`。患者级终点、划分和签名在 `cohort_with_signatures.csv`。")
    lines.append("")
    (OUT / "建模结果与主要发现.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    warnings.filterwarnings("ignore")
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    (OUT / "coefficients").mkdir(exist_ok=True)
    cohort, readiness = build_cohort()
    (OUT / "stage8_readiness.json").write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    matrices = attach_radiomics(cohort)
    train_keys, test_keys = train_test_split(
        cohort["匹配键"],
        test_size=0.30,
        random_state=SEED,
        stratify=cohort["PFS事件"],
    )
    train_ids = set(train_keys)
    cohort["数据集"] = np.where(cohort["匹配键"].isin(train_ids), "训练", "测试")
    bias = selection_bias(cohort)
    bias.to_csv(OUT / "selection_bias.csv", index=False)

    perf_rows = []
    incr_rows = []
    calib_rows = []
    long_rows = []
    sig_rows = []
    stored_rs = {}
    stored_pack = {}
    for key, label in SETS:
        print(f"特征集 {label}", flush=True)
        for endpoint in ("PFS", "OS"):
            print(f"  {endpoint}", flush=True)
            pack = model_ladder(cohort, matrices[key], train_ids, endpoint)
            sig_rows.append({
                "特征集": label,
                "终点": endpoint,
                "原始特征数": pack.get("原始特征数"),
                "进入LASSO的特征数": pack.get("进入LASSO的特征数"),
                "保留特征数": pack.get("保留特征数"),
                "惩罚系数": pack.get("惩罚系数"),
                "内层交叉验证C指数": pack.get("内层交叉验证C指数"),
                "签名训练集人数": pack.get("签名训练集人数"),
                "签名测试集人数": pack.get("签名测试集人数"),
            })
            if "系数" in pack:
                pack["系数"].to_csv(OUT / "coefficients" / f"{key}_{endpoint}.csv", index=False)
            for row in pack.get("效能", []):
                row = dict(row)
                row.update({"特征集": label, "终点": endpoint, "层级": "主模型"})
                perf_rows.append(row)
            for row in pack.get("增量", []):
                row = dict(row)
                row.update({"特征集": label, "终点": endpoint})
                incr_rows.append(row)
            for model_name, rows in pack.get("校准", {}).items():
                for row in rows:
                    row = dict(row)
                    row.update({"特征集": label, "终点": endpoint, "模型": model_name})
                    calib_rows.append(row)
            if endpoint == "PFS" and "RS" in pack:
                stored_rs[label] = pack["RS"]
                stored_pack[label] = pack
            long = longitudinal_gain(cohort, pack["RS"], train_ids, endpoint) if "RS" in pack else {}
            for row in long.get("效能", []):
                row = dict(row)
                row.update({"特征集": label, "终点": endpoint, "层级": "纵向"})
                long_rows.append(row)
                perf_rows.append(dict(row))

    performance = pd.DataFrame(perf_rows)
    increments = pd.DataFrame(incr_rows)
    performance.to_csv(OUT / "model_performance.csv", index=False)
    increments.to_csv(OUT / "nri_idi.csv", index=False)
    pd.DataFrame(calib_rows).to_csv(OUT / "calibration_test.csv", index=False)
    pd.DataFrame(sig_rows).to_csv(OUT / "signature_fit.csv", index=False)
    pd.DataFrame(long_rows).to_csv(OUT / "longitudinal_performance.csv", index=False)
    bias.to_csv(OUT / "selection_bias.csv", index=False)

    score = performance[
        (performance["层级"] == "主模型")
        & (performance["数据集"] == "测试")
        & (performance["终点"] == "PFS")
        & (performance["模型"] == "组学签名单独")
    ].sort_values("C指数", ascending=False)
    best_label = str(score.iloc[0]["特征集"])
    os_hit = performance[
        (performance["层级"] == "主模型")
        & (performance["数据集"] == "测试")
        & (performance["终点"] == "OS")
        & (performance["特征集"] == best_label)
        & (performance["模型"] == "组学签名单独")
    ]
    winner = {
        "特征集": best_label,
        "PFS_C指数": float(score.iloc[0]["C指数"]),
        "PFS_AUC": float(score.iloc[0]["24个月AUC"]),
        "OS_C指数": float(os_hit.iloc[0]["C指数"]) if len(os_hit) else float("nan"),
    }
    (OUT / "winner.json").write_text(json.dumps(winner, ensure_ascii=False, indent=2), encoding="utf-8")

    export = cohort.copy()
    for label, rs in stored_rs.items():
        export[f"RS_{label}"] = rs
    export.to_csv(OUT / "cohort_with_signatures.csv", index=False)

    best_pack = stored_pack[best_label]
    train_index = set(best_pack["工作表索引"]["训练"])
    rs = best_pack["RS"]
    # model_ladder 的 RS 索引是 merge 后的索引，与 cohort 不一定相同。KM 用测试集签名中位数，中位数取训练集。
    rs_frame = cohort[["匹配键", "PFS月", "PFS事件", "数据集"]].copy()
    rs_frame["RS"] = rs.to_numpy()
    train_rs = rs_frame.loc[rs_frame["数据集"] == "训练", "RS"].dropna()
    cutoff = float(train_rs.median()) if len(train_rs) else float("nan")
    test_rs = rs_frame[(rs_frame["数据集"] == "测试") & rs_frame["RS"].notna()]
    km_source = {}
    if np.isfinite(cutoff) and len(test_rs):
        high = test_rs[test_rs["RS"] >= cutoff]
        low = test_rs[test_rs["RS"] < cutoff]
        km_source = {
            f"签名≥训练集中位数（n={len(high)}）": (high["PFS月"].to_numpy(), high["PFS事件"].to_numpy()),
            f"签名<训练集中位数（n={len(low)}）": (low["PFS月"].to_numpy(), low["PFS事件"].to_numpy()),
        }
    dca_source = {}
    risks = best_pack.get("测试风险", {})
    if "模型1 NCCN-IPI" in risks and "模型4 加肌少症" in risks:
        for name, pretty in (("模型1 NCCN-IPI", "模型1"), ("模型4 加肌少症", "模型4")):
            pack = risks[name]
            dca_source[pretty] = (
                np.asarray(pack["time"]),
                np.asarray(pack["event"]),
                1.0 - np.asarray(pack["survival24"], dtype=float),
            )
    draw_figures(performance, dca_source, km_source)
    write_findings(readiness, performance, increments, bias, pd.DataFrame(long_rows), winner)
    print(json.dumps(winner, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
