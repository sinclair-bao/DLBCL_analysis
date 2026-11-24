#!/usr/bin/env python3
"""用随访表自带的 SUVmax、MTV、TLG 做 Cox，不使用影像流程定量。

结果写到 results/stage9_followup_sheet/，不覆盖 stage9_modeling。
终点定义与第九阶段主分析相同。
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from matplotlib import font_manager
from sklearn.model_selection import train_test_split

ROOT = Path("/home/sun/Documents/DLBCL")
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
import stage9_models as base  # noqa: E402

OUT = ROOT / "results" / "stage9_followup_sheet"
FIG = OUT / "figures"
FU_PATH = ROOT / "data" / "DLBCL随访表_与PET库匹配_最终ID_全量csv解析.csv"
SEED = base.SEED
HORIZON = base.HORIZON
FONT = base.FONT

MODELS = [
    ("模型1 NCCN-IPI", ["NCCNIPI"]),
    ("模型2 加表内TLG", ["NCCNIPI", "logTLG"]),
    ("模型3 加表内MTV", ["NCCNIPI", "logMTV"]),
    ("模型4 加表内SUVmax", ["NCCNIPI", "SUVmax"]),
    ("模型5 三个量化一起", ["NCCNIPI", "logTLG", "logMTV", "SUVmax"]),
]


def load_sheet() -> pd.DataFrame:
    raw = pd.read_csv(FU_PATH, dtype=str, encoding="utf-8-sig")
    out = pd.DataFrame({
        "患者ID": raw["PET患者ID"].map(base.key_of),
        "姓名": raw["Excel姓名"],
        "年龄": base.num(raw["年龄"]),
        "性别": raw["性别"].str.strip(),
        "SUVmax": base.num(raw["SUVmax"]),
        "MTV": base.num(raw["MTV"]),
        "TLG": base.num(raw["TLG"]),
        "NCCNIPI": base.num(raw["NCCNIPI"]),
        "治疗反应": raw["治疗反应"].str.strip(),
        "PFS月": base.num(raw["PFS月"]),
        "OS月": base.num(raw["OS月"]),
        "治疗方法": raw["治疗方法"].str.strip(),
    })
    out["logTLG"] = np.log1p(out["TLG"])
    out["logMTV"] = np.log1p(out["MTV"])
    out["PFS事件"] = (
        (out["治疗反应"] == "PD") | (out["PFS月"] < out["OS月"] - 1e-6)
    ).astype(int)
    out["OS事件"] = (out["OS月"] < 36 - 1e-6).astype(int)
    return out.drop_duplicates("患者ID", keep="first").reset_index(drop=True)


def standardize(train: pd.DataFrame, test: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train.copy()
    test = test.copy()
    for col in columns:
        center = train[col].mean()
        scale = train[col].std(ddof=0)
        if not np.isfinite(scale) or scale == 0:
            scale = 1.0
        train[col] = (train[col] - center) / scale
        test[col] = (test[col] - center) / scale
    return train, test


def evaluate_endpoint(cohort: pd.DataFrame, train_ids: set[str], endpoint: str) -> dict:
    time_col = "PFS月" if endpoint == "PFS" else "OS月"
    event_col = "PFS事件" if endpoint == "PFS" else "OS事件"
    cols_to_scale = ["NCCNIPI", "logTLG", "logMTV", "SUVmax"]
    train = cohort[cohort["患者ID"].isin(train_ids)].copy()
    test = cohort[~cohort["患者ID"].isin(train_ids)].copy()
    train, test = standardize(train, test, cols_to_scale)
    frame = pd.concat([train, test], axis=0)
    performances = []
    fitted = {}
    test_risk = {}
    coefficients = []
    for name, cols in MODELS:
        cph = base.fit_clinical(train, cols, time_col, event_col)
        if cph is None:
            performances.append({"模型": name, "数据集": "测试", "状态": "未能拟合"})
            continue
        fitted[name] = (cph, cols)
        for split, part in (("训练", train), ("测试", test)):
            hazard = base.partial_risk(cph, part, cols)
            aligned = part.loc[hazard.index]
            auc, n_auc = base.auc_at(aligned[time_col], aligned[event_col], hazard.to_numpy())
            performances.append({
                "终点": endpoint,
                "模型": name,
                "数据集": split,
                "人数": int(len(aligned)),
                "事件数": int(aligned[event_col].sum()),
                "C指数": base.safe_cindex(aligned[time_col], hazard, aligned[event_col]),
                "24个月AUC": auc,
                "AUC可评价人数": n_auc,
            })
        hazard = base.partial_risk(cph, test, cols)
        survival = base.survival_risk(cph, test, cols)
        common = hazard.index.intersection(survival.index)
        test_risk[name] = (
            test.loc[common, time_col].to_numpy(),
            test.loc[common, event_col].to_numpy(),
            survival.loc[common].to_numpy(),
            common,
        )
        summary = cph.summary
        for feature, row in summary.iterrows():
            coefficients.append({
                "终点": endpoint,
                "模型": name,
                "变量": feature,
                "系数": float(row["coef"]),
                "风险比": float(row["exp(coef)"]),
                "风险比下限": float(row["exp(coef) lower 95%"]),
                "风险比上限": float(row["exp(coef) upper 95%"]),
                "P值": float(row["p"]),
            })
    increments = []
    order = [name for name, _ in MODELS if name in test_risk]
    base_name = "模型1 NCCN-IPI"
    for name in order:
        if name == base_name or base_name not in test_risk:
            continue
        common = test_risk[base_name][3].intersection(test_risk[name][3])
        old = 1.0 - base.survival_risk(fitted[base_name][0], test.loc[common], fitted[base_name][1]).to_numpy()
        new = 1.0 - base.survival_risk(fitted[name][0], test.loc[common], fitted[name][1]).to_numpy()
        delta = base.nri_idi(
            test.loc[common, time_col].to_numpy(),
            test.loc[common, event_col].to_numpy(),
            old,
            new,
        )
        delta["比较"] = f"{name} 相对 {base_name}"
        delta["终点"] = endpoint
        increments.append(delta)
    calibrations = []
    for name in ("模型1 NCCN-IPI", "模型2 加表内TLG", "模型5 三个量化一起"):
        if name not in test_risk:
            continue
        time, event, survival, _ = test_risk[name]
        for row in base.calibration_rows(time, event, 1.0 - np.asarray(survival, dtype=float)):
            row = dict(row)
            row.update({"终点": endpoint, "模型": name})
            calibrations.append(row)
    return {
        "效能": performances,
        "增量": increments,
        "系数": coefficients,
        "测试风险": test_risk,
        "校准": calibrations,
        "训练": train,
        "测试": test,
    }


def draw(performance: pd.DataFrame, dca: dict, km: dict) -> None:
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
    plt.rcParams["axes.unicode_minus"] = False
    FIG.mkdir(parents=True, exist_ok=True)
    test = performance[performance["数据集"] == "测试"]
    labels = ["NCCN-IPI", "加TLG", "加MTV", "加SUVmax", "三个一起"]
    keys = [name for name, _ in MODELS]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
    for ax, endpoint in zip(axes, ["PFS", "OS"]):
        vals = []
        for key in keys:
            hit = test[(test["终点"] == endpoint) & (test["模型"] == key)]
            vals.append(float(hit["C指数"].iloc[0]) if len(hit) else np.nan)
        ax.bar(np.arange(len(labels)), vals, color=["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2"])
        ax.set_xticks(np.arange(len(labels)), labels, rotation=15)
        ax.set_ylim(0.45, 0.85)
        ax.axhline(0.5, color="0.6", lw=0.8)
        ax.set_title(f"{endpoint} 测试集 C 指数")
        ax.set_ylabel("C 指数")
        for i, val in enumerate(vals):
            if np.isfinite(val):
                ax.text(i, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "test_cindex_sheet_quant.png", dpi=140)
    plt.close(fig)

    if km:
        fig, ax = plt.subplots(figsize=(6.2, 4.6))
        for label, (time, event) in km.items():
            fitter = KaplanMeierFitter()
            fitter.fit(time, event, label=label)
            fitter.plot_survival_function(ax=ax, ci_show=False)
        ax.set_xlabel("月")
        ax.set_ylabel("无进展生存")
        ax.set_title("测试集：随访表 TLG 按训练集中位数分组")
        fig.tight_layout()
        fig.savefig(FIG / "km_pfs_sheet_tlg.png", dpi=140)
        plt.close(fig)

    if dca:
        fig, ax = plt.subplots(figsize=(6.2, 4.6))
        thresholds = np.round(np.arange(0.05, 0.51, 0.05), 2)
        for label, (time, event, risk) in dca.items():
            benefits = [base.net_benefit(time, event, risk, t) for t in thresholds]
            ax.plot(thresholds, benefits, marker="o", label=label)
        time, event, _ = next(iter(dca.values()))
        mask = base.evaluable_mask(time, event)
        prevalence = ((np.asarray(event) == 1) & (np.asarray(time) <= HORIZON))[mask].mean()
        treat_all = []
        for threshold in thresholds:
            odds = threshold / max(1 - threshold, 1e-6)
            treat_all.append(prevalence - (1 - prevalence) * odds)
        ax.plot(thresholds, treat_all, ls="--", color="0.4", label="全部按高风险处理")
        ax.axhline(0, color="0.5", lw=0.8, label="全部按低风险处理")
        ax.set_xlabel("24个月事件阈值概率")
        ax.set_ylabel("净获益")
        ax.set_title("测试集决策曲线（PFS，随访表定量）")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG / "dca_pfs_sheet_quant.png", dpi=140)
        plt.close(fig)


def write_findings(cohort: pd.DataFrame, performance: pd.DataFrame, increments: pd.DataFrame, winner: dict) -> None:
    def cell(endpoint, model, column, split="测试"):
        hit = performance[
            (performance["终点"] == endpoint)
            & (performance["模型"] == model)
            & (performance["数据集"] == split)
        ]
        if hit.empty or pd.isna(hit.iloc[0][column]):
            return "—"
        value = hit.iloc[0][column]
        if column in {"人数", "事件数", "AUC可评价人数"}:
            return str(int(value))
        return f"{float(value):.3f}"

    lines = [
        "# 随访表定量的 Cox 分析",
        "",
        "这一轮只用 `DLBCL随访表_与PET库匹配_最终ID_全量csv解析.csv` 里的 SUVmax、MTV、TLG 和 NCCN-IPI。没有使用第五到第八阶段算出的 TMTV、TLG、组学或 SMI。198 行定量和随访都完整，全部进入分析。",
        "",
        "终点与主分析相同。PFS 事件：治疗反应为 PD，或 PFS 月短于 OS 月；两者相同且不是 PD 的记为删失。OS 事件：OS 月短于 36 记为事件，等于 36 记为删失。短于 36 个月的人里，死亡和失访分不开。",
        "",
        f"人数 {len(cohort)}。PFS 事件 {int(cohort['PFS事件'].sum())} 个，OS 事件 {int(cohort['OS事件'].sum())} 个。治疗反应：{cohort['治疗反应'].value_counts().to_dict()}。",
        "",
        "训练集和测试集按 7:3 划分，按 PFS 事件分层，随机种子 20261123。TLG 和 MTV 取 log(1+x)。NCCN-IPI、log TLG、log MTV、SUVmax 都按训练集均值和标准差标准化，风险比表示每增加 1 个标准差。",
        "",
        f"测试集 PFS C 指数最高的是 **{winner['模型']}**，C 指数 {winner['PFS_C指数']:.3f}，24 个月 AUC {winner['PFS_AUC']:.3f}。同一模型的 OS C 指数是 {winner['OS_C指数']:.3f}。",
        "",
        "| 模型 | PFS C指数 | PFS 24个月AUC | OS C指数 | OS 24个月AUC | 测试人数 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, _ in MODELS:
        lines.append("| {name} | {pc} | {pa} | {oc} | {oa} | {n} |".format(
            name=name,
            pc=cell("PFS", name, "C指数"),
            pa=cell("PFS", name, "24个月AUC"),
            oc=cell("OS", name, "C指数"),
            oa=cell("OS", name, "24个月AUC"),
            n=cell("PFS", name, "人数"),
        ))
    lines += [
        "",
        "下面的 NRI 和 IDI 都是相对只有 NCCN-IPI 的模型，用测试集里能判断 24 个月状态的人。",
        "",
        "| 终点 | 比较 | NRI | IDI | 可评价人数 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for _, row in increments.iterrows():
        lines.append(
            f"| {row['终点']} | {row['比较']} | {row['NRI']:.3f} | {row['IDI']:.3f} | {int(row['可评价人数'])} |"
        )
    lines += [
        "",
        "TLG 近似等于摄取值和体积的乘积，所以模型 5 里三个量化指标高度相关，系数不要单独解释成互相独立的效应。训练集上的风险比在 `coefficients.csv`。",
        "",
        "## 图",
        "",
        "- `figures/test_cindex_sheet_quant.png`：五个模型在测试集上的 C 指数。",
        "- `figures/km_pfs_sheet_tlg.png`：随访表 TLG 按训练集中位数把测试集分成两组的 PFS 曲线。",
        "- `figures/dca_pfs_sheet_quant.png`：NCCN-IPI 和 PFS 上 C 指数最高的模型的决策曲线。",
        "",
        "校准分组在 `calibration_test.csv`。患者级数据和划分在 `cohort.csv`。",
        "",
    ]
    (OUT / "建模结果与主要发现.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    warnings.filterwarnings("ignore")
    OUT.mkdir(parents=True, exist_ok=True)
    cohort = load_sheet()
    train_ids, _ = train_test_split(
        cohort["患者ID"],
        test_size=0.30,
        random_state=SEED,
        stratify=cohort["PFS事件"],
    )
    train_ids = set(train_ids)
    cohort["数据集"] = np.where(cohort["患者ID"].isin(train_ids), "训练", "测试")
    packs = {}
    perf_rows = []
    incr_rows = []
    coef_rows = []
    calib_rows = []
    for endpoint in ("PFS", "OS"):
        pack = evaluate_endpoint(cohort, train_ids, endpoint)
        packs[endpoint] = pack
        perf_rows.extend(pack["效能"])
        incr_rows.extend(pack["增量"])
        coef_rows.extend(pack["系数"])
        calib_rows.extend(pack["校准"])
    performance = pd.DataFrame(perf_rows)
    increments = pd.DataFrame(incr_rows)
    performance.to_csv(OUT / "model_performance.csv", index=False)
    increments.to_csv(OUT / "nri_idi.csv", index=False)
    pd.DataFrame(coef_rows).to_csv(OUT / "coefficients.csv", index=False)
    pd.DataFrame(calib_rows).to_csv(OUT / "calibration_test.csv", index=False)
    cohort.to_csv(OUT / "cohort.csv", index=False)

    score = performance[
        (performance["数据集"] == "测试") & (performance["终点"] == "PFS")
    ].sort_values("C指数", ascending=False)
    best_name = str(score.iloc[0]["模型"])
    os_hit = performance[
        (performance["数据集"] == "测试")
        & (performance["终点"] == "OS")
        & (performance["模型"] == best_name)
    ]
    winner = {
        "模型": best_name,
        "PFS_C指数": float(score.iloc[0]["C指数"]),
        "PFS_AUC": float(score.iloc[0]["24个月AUC"]),
        "OS_C指数": float(os_hit.iloc[0]["C指数"]) if len(os_hit) else float("nan"),
        "人数": int(len(cohort)),
        "PFS事件": int(cohort["PFS事件"].sum()),
        "OS事件": int(cohort["OS事件"].sum()),
    }
    (OUT / "winner.json").write_text(json.dumps(winner, ensure_ascii=False, indent=2), encoding="utf-8")

    train = cohort[cohort["数据集"] == "训练"]
    test = cohort[cohort["数据集"] == "测试"]
    cutoff = float(train["TLG"].median())
    high = test[test["TLG"] >= cutoff]
    low = test[test["TLG"] < cutoff]
    km = {
        f"TLG≥训练集中位数（n={len(high)}）": (high["PFS月"].to_numpy(), high["PFS事件"].to_numpy()),
        f"TLG<训练集中位数（n={len(low)}）": (low["PFS月"].to_numpy(), low["PFS事件"].to_numpy()),
    }
    dca = {}
    risks = packs["PFS"]["测试风险"]
    for name, pretty in ((best_name, best_name), ("模型1 NCCN-IPI", "NCCN-IPI")):
        if name not in risks or pretty in dca:
            continue
        time, event, survival, _ = risks[name]
        dca[pretty.replace("模型1 NCCN-IPI", "NCCN-IPI")] = (
            time,
            event,
            1.0 - np.asarray(survival, dtype=float),
        )
    draw(performance, dca, km)
    write_findings(cohort, performance, increments, winner)
    print(json.dumps(winner, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
