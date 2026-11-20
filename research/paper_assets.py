#!/usr/bin/env python3
"""Numbers, tables and figures of the manuscript, taken from the result files.

Nothing is estimated here.  The script reads the CSV files written by the
analysis scripts and turns them into

    manuscript/generated/numbers.json   every number that may appear in the text, under a readable key
    manuscript/generated/tables.json    LaTeX source of the tables of the paper and the supplement
    manuscript/figures/fig*.pdf|png     the four greyscale figures of the paper

The text (manuscript/body.tex, manuscript/supplement_body.tex) refers to a
number as [[key]]; research/build_paper.py substitutes the value, so that the
manuscript always agrees with the result files.

Run from the repository root:  python3 research/paper_assets.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RT, RC = Path("results_trend"), Path("results_cause")
GEN, FIG = Path("manuscript/generated"), Path("manuscript/figures")
MINUS = "−"
SEXES = (("Female", "F", "女性", "女"), ("Male", "M", "男性", "男"))
FOCUS = (("TWN", "臺灣"), ("JPN", "日本"), ("KOR", "韓國"), ("HKG", "香港"))
BANDS = ("30-49", "50-64", "65-79", "80+")
BAND_ZH = {"30-49": "30–49 歲", "50-64": "50–64 歲", "65-79": "65–79 歲", "80+": "80 歲以上"}
NAMES = {
    "AUT": "奧地利", "BEL": "比利時", "CHE": "瑞士", "CHL": "智利", "DEUTNP": "德國", "DNK": "丹麥", "EST": "愛沙尼亞",
    "FIN": "芬蘭", "FRATNP": "法國", "HKG": "香港", "HRV": "克羅埃西亞", "JPN": "日本", "KOR": "韓國", "LTU": "立陶宛",
    "LVA": "拉脫維亞", "NLD": "荷蘭", "NOR": "挪威", "PRT": "葡萄牙", "SVK": "斯洛伐克", "SWE": "瑞典", "TWN": "臺灣", "USA": "美國",
}
MEASURE_ZH = {"e": "30 歲平均餘命", "q25": "第 25 百分位", "q50": "中位數", "q75": "第 75 百分位"}

# Supplement tables in their order of appearance.  The text refers to them as [[ts.name]] and places them with
# "% TABLE tableS_name", so the numbers follow this list.
SUPP_ORDER = ("pops", "agerates", "baselines", "bands", "backtestgap", "backtest", "bridge", "registration", "categories", "codes",
              "broad", "vague", "context")
TS = {name: str(i + 1) for i, name in enumerate(SUPP_ORDER)}   # 附表 1, 附表 2, ... of the online appendix

# cause groups of results_cause/* condensed for Table 4 (order of presentation)
CAUSE_ROWS = [
    ("covid", "COVID-19", ["covid19"]),
    ("cancer", "惡性腫瘤", ["cancer_oral_esophagus", "cancer_liver", "cancer_lung", "cancer_colorectal", "cancer_breast", "cancer_other"]),
    ("ihd", "缺血性心臟病", ["heart_ischemic"]),
    ("stroke", "腦血管疾病", ["cerebrovascular"]),
    ("heartother", "其他心臟疾病", ["heart_other"]),
    ("hyper", "高血壓性疾病", ["hypertensive"]),
    ("diabetes", "糖尿病", ["diabetes"]),
    ("pneumonia", "肺炎、流感及吸入性肺炎", ["pneumonia_influenza", "aspiration"]),
    ("clrd", "慢性下呼吸道疾病", ["chronic_lower_respiratory"]),
    ("liver", "慢性肝病、肝硬化及酒精所致疾病", ["liver_nonalcohol", "alcohol_specific"]),
    ("kidney", "腎炎、腎病症候群及腎病變", ["kidney"]),
    ("sepsis", "敗血症", ["sepsis"]),
    ("dementia", "失智症、帕金森病及阿茲海默病", ["dementia_parkinson"]),
    ("illdef", "症狀、徵候及診斷欠明（含衰老）", ["senility_illdefined"]),
    ("accident", "事故傷害", ["accidents"]),
    ("suicide", "蓄意自我傷害", ["suicide"]),
    ("other", "其他", ["other"]),
]
BROAD_ROWS = [   # key, label, classes of results_cause/chapter_rates_by_band.csv
    ("circ", "循環系統疾病（I00–I99）", ["circulatory (I00-I99)"]),
    ("ill", "症狀、徵候及診斷欠明（R00–R99）", ["ill-defined (R00-R99)"]),
    ("circill", "以上兩類合計", ["circulatory (I00-I99)", "ill-defined (R00-R99)"]),
    ("neo", "腫瘤（C00–D48）", ["neoplasms (C00-D48)"]),
    ("resp", "呼吸系統疾病（J00–J99）", ["respiratory (J00-J99)"]),
    ("dig", "消化系統疾病（K00–K93）", ["digestive (K00-K93)"]),
    ("endo", "內分泌及代謝疾病（E00–E90）", ["endocrine and metabolic (E00-E90)"]),
    ("ext", "外因（V01–Y98）", ["external (V01-Y98)"]),
    ("covid", "COVID-19", ["COVID-19"]),
    ("oth", "其他", ["infectious (A00-B99)", "mental and nervous (F00-G99)", "genitourinary (N00-N99)", "all other"]),
]


# ----------------------------------------------------------------------------
# formatting
# ----------------------------------------------------------------------------
def signed(x: float, d: int = 2) -> str:
    s = f"{x:+.{d}f}"
    if float(s) == 0:
        s = f"{0:.{d}f}"
    return s.replace("-", MINUS)


def plain(x: float, d: int = 2) -> str:
    s = f"{x:.{d}f}"
    if float(s) == 0:
        s = f"{0:.{d}f}"
    return s.replace("-", MINUS)


def pct(ratio: float, d: int = 0) -> str:
    """Percentage by which a ratio exceeds one."""
    return plain((ratio - 1.0) * 100.0, d)


def count(x: float) -> str:
    return f"{int(round(x)):,}"


def pvalue(p: float) -> str:
    return "小於 0.01" if p < 0.01 else f"{p:.2f}"


# ----------------------------------------------------------------------------
# numbers
# ----------------------------------------------------------------------------
def load() -> dict[str, pd.DataFrame]:
    d = {
        "exp": pd.read_csv(RT / "expected_observed.csv"),
        "t25": pd.read_csv(RT / "taiwan_2025_deficits.csv"),
        "bands": pd.read_csv(RT / "age_band_contributions.csv"),
        "bands25": pd.read_csv(RT / "taiwan_2025_age_bands.csv"),
        "groups": pd.read_csv(RT / "age_contributions.csv"),
        "groups25": pd.read_csv(RT / "taiwan_2025_age_contributions.csv"),
        "rr": pd.read_csv(RT / "age_rate_ratios.csv"),
        "obs": pd.read_csv(RT / "observed_measures.csv"),
        "bt": pd.read_csv(RT / "backtest_point.csv"),
        "bti": pd.read_csv(RT / "backtest_interval.csv"),
        "coh_bt": pd.read_csv(RT / "cohort_check_backtest.csv"),
        "coh_gap": pd.read_csv(RT / "cohort_check_gaps.csv"),
        "coh_band": pd.read_csv(RT / "cohort_check_age_bands.csv"),
        "coh_rr": pd.read_csv(RT / "cohort_check_rate_ratios.csv"),
        "bridge": pd.read_csv(RT / "taiwan_2025_bridge_validation.csv"),
        "sources": pd.read_csv(RT / "taiwan_2025_source_comparison.csv"),
        "reg": pd.read_csv(RT / "taiwan_registration_check.csv"),
        "fit": pd.read_csv(RT / "lc_fit_summary.csv"),
        "cc": pd.read_csv(RC / "cause_contributions_by_band.csv"),
        "cr": pd.read_csv(RC / "cause_rates_by_band.csv"),
        "chap": pd.read_csv(RC / "chapter_rates_by_band.csv"),
        "code3": pd.read_csv(RC / "code_rates_by_band.csv"),
        "code4": pd.read_csv(RC / "code4_deaths_by_band.csv"),
        "check": pd.read_csv(RC / "decomposition_checks.csv"),
        "rep": pd.read_csv(RT / "replication_dowd.csv"),
        "ctx_ems": pd.read_csv(RC / "context_ems_ohca.csv"),
        "ctx_place": pd.read_csv(RC / "context_place_of_death.csv"),
        "ctx_inq": pd.read_csv(RC / "context_inquests.csv"),
        "by_cause": pd.read_csv(RC / "deaths_by_cause.csv"),
        "condensed": pd.read_csv(RC / "condensed_category_check.csv"),
    }
    # Taiwan 2025 joins the 2020-2024 rows of the main file
    t = d["t25"].copy()
    t.insert(0, "population", "TWN")
    d["gaps"] = pd.concat([d["exp"][d["exp"].observed.notna()][["population", "sex", "spec", "year", "measure", "observed", "expected",
                                                                 "deficit", "deficit_low", "deficit_high", "p_value",
                                                                 "deficit_low_mcse", "deficit_high_mcse", "p_value_mcse"]], t], ignore_index=True)
    b = d["bands25"].copy()
    b.insert(0, "population", "TWN")
    b.insert(2, "spec", "main")
    d["bands_all"] = pd.concat([d["bands"], b], ignore_index=True)
    g = d["groups25"].copy()
    g.insert(0, "population", "TWN")
    g.insert(2, "spec", "main")
    d["groups_all"] = pd.concat([d["groups"], g], ignore_index=True)
    return d


def build_numbers(d: dict[str, pd.DataFrame]) -> dict[str, str]:
    N: dict[str, str] = {}
    sex_code = {s[0]: s[1] for s in SEXES}

    # gaps to trend ---------------------------------------------------------
    for r in d["gaps"].itertuples():
        spec = "" if r.spec == "main" else f"{r.spec}."
        k = f"{spec}{r.population}.{sex_code[r.sex]}.{r.measure}.{r.year}"
        N[f"gap.{k}"] = signed(r.deficit)
        N[f"gapabs.{k}"] = plain(abs(r.deficit))
        N[f"gaplo.{k}"] = signed(r.deficit_low)
        N[f"gaphi.{k}"] = signed(r.deficit_high)
        N[f"gapp.{k}"] = pvalue(r.p_value)
        N[f"obs.{k}"] = plain(r.observed)
        N[f"exp.{k}"] = plain(r.expected)
    main24 = d["gaps"][(d["gaps"].spec == "main") & (d["gaps"].year == 2024)]
    for m in ("e", "q25", "q50", "q75"):
        x = main24[main24.measure == m]
        N[f"n.series.{m}.2024"] = str(len(x))
        N[f"n.neg.{m}.2024"] = str(int((x.deficit < 0).sum()))
        N[f"n.sig.{m}.2024"] = str(int((x.deficit_high < 0).sum()))
    q = main24.pivot_table(index=["population", "sex"], columns="measure", values="deficit")
    N["n.q25worse.2024"] = str(int((q["q25"] < q["q75"]).sum()))
    N["n.populations"] = str(main24.population.nunique())
    names = [NAMES[p] for p in sorted(main24.population.unique())]                    # in the order of the HMD codes
    N["populations.names"] = "、".join(names[:-1]) + "與" + names[-1]

    # death-rate ratios -----------------------------------------------------
    rr = d["rr"][d["rr"].grouping == "band"]
    for r in rr.itertuples():
        k = f"{r.population}.{sex_code[r.sex]}.{r.age_group}.{r.year}"
        N[f"r19.{k}"] = signed((r.ratio_to_2019 - 1) * 100, 1)
        N[f"r19abs.{k}"] = plain(abs(r.ratio_to_2019 - 1) * 100, 1)
        if np.isfinite(r.ratio_to_trend):
            N[f"rr.{k}"] = plain(r.ratio_to_trend)
            N[f"rrpct.{k}"] = pct(r.ratio_to_trend)
            N[f"rrpctlo.{k}"] = pct(r.ratio_low)
            N[f"rrpcthi.{k}"] = pct(r.ratio_high)
            N[f"rrrate.{k}"] = plain(r.observed_rate_per_100k, 1)
            N[f"rrexp.{k}"] = plain(r.expected_rate_per_100k, 1)
    five = d["rr"][(d["rr"].grouping == "five_year") & (d["rr"].population == "TWN")]
    for r in five.itertuples():
        k = f"TWN.{sex_code[r.sex]}.{r.age_group}.{r.year}"
        N[f"r19five.{k}"] = signed((r.ratio_to_2019 - 1) * 100, 1)
        N[f"r19fiveabs.{k}"] = plain(abs(r.ratio_to_2019 - 1) * 100, 1)
        if np.isfinite(r.ratio_to_trend):
            N[f"rrfivepct.{k}"] = pct(r.ratio_to_trend)

    # age-band contributions to the e30 gap -----------------------------------
    for r in d["bands_all"][(d["bands_all"].spec == "main")].itertuples():
        k = f"{r.population}.{sex_code[r.sex]}.{r.measure}.{r.age_band}.{r.year}"
        N[f"band.{k}"] = signed(r.contribution)
        N[f"bandabs.{k}"] = plain(abs(r.contribution))
        if np.isfinite(r.low):
            N[f"bandlo.{k}"] = signed(r.low)
            N[f"bandhi.{k}"] = signed(r.high)

    gap_e = {(r.population, r.sex, r.year): r.deficit for r in d["gaps"][(d["gaps"].spec == "main") & (d["gaps"].measure == "e")].itertuples()}
    for r in d["bands_all"][(d["bands_all"].spec == "main") & (d["bands_all"].measure == "e")].itertuples():
        N[f"bandshare.{r.population}.{sex_code[r.sex]}.{r.age_band}.{r.year}"] = plain(100.0 * r.contribution / gap_e[(r.population, r.sex, r.year)], 0)

    # cohort check ------------------------------------------------------------
    base = {"APC": "apc", "Lee-Carter": "lc", "Lee-Carter, 1990-2019": "lc30", "Lee-Carter, 2010-2019": "lc10"}
    for r in d["coh_band"].itertuples():
        N[f"coh.{base[r.baseline]}.{sex_code[r.sex]}.{r.age_band}.{r.year}"] = signed(r.contribution_to_e30_gap)
    for r in d["coh_rr"].itertuples():
        N[f"cohrr.{base[r.baseline]}.{sex_code[r.sex]}.{r.age_band}.{r.year}"] = pct(r.observed_over_expected_rate)
    for r in d["coh_gap"].itertuples():
        for col, b in (("gap_apc", "apc"), ("gap_lee_carter", "lc"), ("gap_lee_carter_1990_2019", "lc30"), ("gap_lee_carter_2010_2019", "lc10")):
            N[f"cohgap.{b}.{sex_code[r.sex]}.{r.measure}.{r.year}"] = signed(getattr(r, col))
    model = {"APC, 30 years": "apc", "Lee-Carter, 17 years": "lc", "Lee-Carter, 30 years": "lc30", "Lee-Carter, 10 years": "lc10"}
    cb = d["coh_bt"]
    for (h, sex, m, mod), g in cb.groupby(["horizon", "sex", "measure", "model"]):
        is_rate = m.startswith("log rate ")
        mk = "rate" + m.replace("log rate ", "") if is_rate else m          # e.g. rate50-64: errors as percentages (100 x log ratio)
        k = f"h{h}.{model[mod]}.{sex_code[sex]}.{mk}"                         # h5: five years ahead (10 cases), h6: six years ahead (9 cases)
        scale, dd = (100.0, 1) if is_rate else (1.0, 2)
        N[f"cohbt.n.{k}"] = str(len(g))
        N[f"cohbt.bias.{k}"] = signed(g.error.mean() * scale, dd)
        N[f"cohbt.biasabs.{k}"] = plain(abs(g.error.mean()) * scale, dd)
        N[f"cohbt.rmse.{k}"] = plain(float(np.sqrt((g.error ** 2).mean())) * scale, dd)
        N[f"cohbt.min.{k}"] = signed(g.error.min() * scale, dd)
        N[f"cohbt.max.{k}"] = signed(g.error.max() * scale, dd)
        N[f"cohbt.minabs.{k}"] = plain(abs(g.error.min()) * scale, dd)
        N[f"cohbt.maxabs.{k}"] = plain(abs(g.error.max()) * scale, dd)
        N[f"cohbt.nneg.{k}"] = str(int((g.error < 0).sum()))
        if is_rate:                                   # mean error as an ordinary percentage: each error converted, then averaged
            N[f"cohbt.biaspct.{k}"] = plain(float(((np.exp(g.error) - 1.0) * 100.0).mean()), 1)
    # how the band contributions moved between 2022 and 2025 under the four baselines
    cbnd = d["coh_band"].pivot_table(index=["baseline", "sex", "age_band"], columns="year", values="contribution_to_e30_gap")
    old_age = cbnd.xs("80+", level="age_band")
    shrink = (1 - old_age[2025] / old_age[2022]) * 100
    N["cohsum.80.minshrink"] = plain(shrink.min(), 0)
    N["cohsum.80.nshrunk"] = str(int((old_age[2025].abs() < old_age[2022].abs()).sum()))
    mid = cbnd.xs("50-64", level="age_band")
    N["cohsum.5064.maxabschg"] = plain((mid[2025] - mid[2022]).abs().max())
    N["cohsum.5064.nworse"] = str(int((mid[2025] < mid[2022]).sum()))
    N["cohsum.n"] = str(len(mid))
    logs = {}
    for r in d["coh_rr"].itertuples():                                     # the same scale for the gaps: 100 x log(observed / expected)
        v = 100.0 * float(np.log(r.observed_over_expected_rate))
        logs[(base[r.baseline], r.sex, r.age_band, r.year)] = v
        N[f"cohlog.{base[r.baseline]}.{sex_code[r.sex]}.{r.age_band}.{r.year}"] = signed(v, 1)
        N[f"cohlogabs.{base[r.baseline]}.{sex_code[r.sex]}.{r.age_band}.{r.year}"] = plain(abs(v), 1)
    rr25 = d["coh_rr"][d["coh_rr"].year == 2025].copy()
    rr25["pct"] = (rr25.observed_over_expected_rate - 1.0) * 100.0
    mid25 = rr25[rr25.age_band == "50-64"]
    N["cohsum.5064.rrmin"], N["cohsum.5064.rrmax"] = plain(mid25.pct.min(), 0), plain(mid25.pct.max(), 0)
    for sex, sc, *_ in SEXES:
        x = mid25[mid25.sex == sex].pct
        N[f"cohsum.5064.rrmin.{sc}"], N[f"cohsum.5064.rrmax.{sc}"] = plain(x.min(), 0), plain(x.max(), 0)
        N[f"cohsum.5064.rrspread.{sc}"] = plain(x.max() - x.min(), 0)
    young_m = rr25[(rr25.age_band == "30-49") & (rr25.sex == "Male") & (rr25.baseline != "Lee-Carter")]
    N["cohsum.3049.other.rrmin.M"], N["cohsum.3049.other.rrmax.M"] = plain(young_m.pct.min(), 0), plain(young_m.pct.max(), 0)
    old25 = rr25[(rr25.age_band == "80+") & rr25.baseline.isin(["APC", "Lee-Carter, 1990-2019"])]
    N["cohsum.80.long.rrmin"], N["cohsum.80.long.rrmax"] = plain(old25.pct.min(), 0), plain(old25.pct.max(), 0)
    # In how many of the eight baseline-by-sex combinations is the gap larger than the largest
    # under-prediction of the backtests with the same horizon (2025 is six years ahead, 2024 five)?
    for h, year in ((6, 2025), (5, 2024)):
        for band in BANDS:
            above = 0
            for mod, b in model.items():
                for sex, sc, *_ in SEXES:
                    g = cb[(cb.horizon == h) & (cb.sex == sex) & (cb.model == mod) & (cb.measure == f"log rate {band}")]
                    above += int(logs[(b, sex, band, year)] > 100.0 * g.error.max())
            N[f"cohsum.above.{band}.{year}"] = str(above)

    # backtest ----------------------------------------------------------------
    bt = d["bt"][d["bt"].origin.between(2005, 2014) & d["bt"].window.isin(["17y", "10y"])]
    bti = d["bti"]
    baltic = ["EST", "LTU", "LVA"]
    scopes = {"pool": lambda x: x[~x.population.isin(baltic)], "all": lambda x: x, "TWN": lambda x: x[x.population == "TWN"]}
    for scope, pick in scopes.items():
        a, c = pick(bt), pick(bti)
        for (w, j, m, h), g in a.groupby(["window", "jump", "measure", "horizon"]):
            k = f"{w}.{j}.{scope}.{m}.{h}"
            N[f"bt.bias.{k}"] = signed(g.error.mean())
            N[f"bt.biasabs.{k}"] = plain(abs(g.error.mean()))
            N[f"bt.rmse.{k}"] = plain(float(np.sqrt((g.error ** 2).mean())))
            N[f"bt.n.{k}"] = str(len(g))
        for (w, j, m, h), g in c.groupby(["window", "jump", "measure", "horizon"]):
            N[f"bt.cover.{w}.{j}.{scope}.{m}.{h}"] = plain(g.covered.mean() * 100, 1)
    pool17 = scopes["pool"](bt)
    pool17 = pool17[(pool17.window == "17y") & (pool17.jump == "corrected")]
    N["bt.n.populations.17y"] = str(pool17.population.nunique())
    N["bt.n.populations.10y"] = str(scopes["pool"](bt[bt.window == "10y"]).population.nunique())
    cov = bti[(bti.window == "17y") & (bti.jump == "corrected") & (bti.horizon == 5) & (bti.measure == "e")].groupby("population").covered.mean()
    for pop, v in cov.items():
        N[f"bt.coverpop.{pop}"] = plain(v * 100, 0)
    x5 = bti[(bti.window == "17y") & (bti.jump == "corrected") & (bti.horizon == 5) & (bti.measure == "e")]
    for pop, v in (x5.observed < x5.lower).groupby(x5.population).mean().items():      # share of misses below the interval
        N[f"bt.belowpop.{pop}"] = plain(v * 100, 0)

    # causes: contributions to the change in e30 ------------------------------------
    cc = d["cc"][d["cc"].measure == "e"]
    periods = {(2012, 2018): "1218", (2019, 2025): "1925", (2019, 2022): "1922", (2022, 2025): "2225", (2019, 2024): "1924"}
    for sex, sc, *_ in SEXES:
        s = cc[cc.sex == sex]
        for (y0, y1), pk in periods.items():
            p = s[(s.from_year == y0) & (s.to_year == y1)]
            N[f"tot.{sc}.{pk}"] = signed(p.contribution.sum())
            N[f"totabs.{sc}.{pk}"] = plain(abs(p.contribution.sum()))
            for band in BANDS:
                N[f"totband.{sc}.{band}.{pk}"] = signed(p[p.age_band == band].contribution.sum())
            for key, _, members in CAUSE_ROWS:
                v = p[p.cause_group.isin(members)].contribution.sum()
                N[f"cause.{sc}.{key}.{pk}"] = signed(v)
                N[f"causeabs.{sc}.{key}.{pk}"] = plain(abs(v))
                for band in BANDS:
                    N[f"causeband.{sc}.{band}.{key}.{pk}"] = signed(p[p.cause_group.isin(members) & (p.age_band == band)].contribution.sum())
        N[f"totdiff.{sc}"] = signed(float(N[f"tot.{sc}.1925"].replace(MINUS, "-")) - float(N[f"tot.{sc}.1218"].replace(MINUS, "-")))
        for key, _, members in CAUSE_ROWS:
            a = s[(s.from_year == 2019) & (s.to_year == 2025) & s.cause_group.isin(members)].contribution.sum()
            b = s[(s.from_year == 2012) & (s.to_year == 2018) & s.cause_group.isin(members)].contribution.sum()
            N[f"causediff.{sc}.{key}"] = signed(a - b)

    # causes: death rates by broad class and by code ------------------------------------
    chap = d["chap"]
    for sex, sc, *_ in SEXES:
        for band in BANDS:
            t = chap[(chap.sex == sex) & (chap.age_band == band)].pivot(index="broad_class", columns="year", values="rate_per_100k").fillna(0.0)
            total = t.sum()
            series = {key: t.loc[members].sum() for key, _, members in BROAD_ROWS}
            series["all"] = total
            series["rest"] = total - series["circill"]
            series["restnocovid"] = total - series["circill"] - series["covid"]
            for key, v in series.items():
                for year in range(2008, 2026):
                    N[f"cr.{sc}.{band}.{key}.{year}"] = plain(v[year], 1)
                for (y0, y1), pk in (((2019, 2025), "1925"), ((2012, 2018), "1218"), ((2019, 2022), "1922")):
                    N[f"crd.{sc}.{band}.{key}.{pk}"] = signed(v[y1] - v[y0], 1)
                    if v[y0] > 0:
                        N[f"crpct.{sc}.{band}.{key}.{pk}"] = signed((v[y1] / v[y0] - 1) * 100, 1)
                        N[f"crpctabs.{sc}.{band}.{key}.{pk}"] = plain(abs(v[y1] / v[y0] - 1) * 100, 1)
    c4 = d["code4"]
    for r in c4.itertuples():
        N[f"code.{sex_code[r.sex]}.{r.age_band}.{r.ICD}.{r.year}"] = count(r.deaths)
    for (icd, year), v in c4.groupby(["ICD", "year"]).deaths.sum().items():       # national totals, ages 30+, both sexes
        N[f"natcode.{icd}.{year}"] = count(v)
    c3 = d["code3"]
    watch3 = ["R57", "R09", "R54", "R99", "I51", "I50", "I10", "I11", "I21", "I25", "I46", "R96", "I64", "J18", "C22"]
    for r in c3[c3.icd3.isin(watch3)].itertuples():
        N[f"code3.{sex_code[r.sex]}.{r.age_band}.{r.icd3}.{r.year}"] = count(r.deaths)
        N[f"code3rate.{sex_code[r.sex]}.{r.age_band}.{r.icd3}.{r.year}"] = plain(r.rate_per_100k, 1)
    # ischaemic heart diseases (I20-I25) as a death rate, and the two codes that literally mean sudden death
    # (I46 cardiac arrest, R96 other sudden death of unknown cause), by age band
    ihd = c3[c3.icd3.isin([f"I{n}" for n in range(20, 26)])].groupby(["sex", "age_band", "year"])[["deaths", "rate_per_100k"]].sum()
    for (sex, band, year), r in ihd.iterrows():
        N[f"ihd.rate.{sex_code[sex]}.{band}.{year}"] = plain(r.rate_per_100k, 1)
        N[f"ihd.deaths.{sex_code[sex]}.{band}.{year}"] = count(r.deaths)
    for sex, sc, *_ in SEXES:
        for band in BANDS:
            a, b = ihd.loc[(sex, band, 2019), "rate_per_100k"], ihd.loc[(sex, band, 2025), "rate_per_100k"]
            N[f"ihd.pct.{sc}.{band}.1925"] = signed((b / a - 1) * 100, 1)
            N[f"ihd.pctabs.{sc}.{band}.1925"] = plain(abs(b / a - 1) * 100, 1)
    sudden = c3[c3.icd3.isin(["I46", "R96"])]
    nat_sudden = sudden.groupby("year").deaths.sum()                              # ages 30 and over, both sexes
    for year in range(2019, 2026):
        N[f"natsudden.{year}"] = count(nat_sudden[year])
    N["natsudden.pct"] = plain((nat_sudden[2025] / nat_sudden[2019] - 1) * 100, 1)
    nat_all = c3.groupby("year").deaths.sum()                                       # every code, ages 30 and over, same source
    N["nat30.deaths.2019"], N["nat30.deaths.2025"] = count(nat_all[2019]), count(nat_all[2025])
    N["nat30.deaths.pct"] = plain((nat_all[2025] / nat_all[2019] - 1) * 100, 1)
    for band in BANDS:
        x = sudden[sudden.age_band == band].groupby("year").deaths.sum()
        for year in range(2019, 2026):
            N[f"sudden.{band}.{year}"] = count(x.get(year, 0.0))
        N[f"sudden.{band}.min"], N[f"sudden.{band}.max"] = count(x.loc[2019:2025].min()), count(x.loc[2019:2025].max())
        for sex, sc, *_ in SEXES:
            y = sudden[(sudden.age_band == band) & (sudden.sex == sex)].groupby("year").deaths.sum()
            for year in range(2019, 2026):
                N[f"sudden.{sc}.{band}.{year}"] = count(y.get(year, 0.0))
    five_codes = ["R570", "R092", "I516", "I509", "I10"]
    f5 = c4[c4.ICD.isin(five_codes) & c4.year.isin([2019, 2025])].pivot_table(index=["ICD", "sex", "age_band"], columns="year", values="deaths", aggfunc="sum").fillna(0.0)
    N["five.cells"] = str(len(f5))                                                    # 5 codes x 2 sexes x 4 age bands
    N["five.cells.up"] = str(int((f5[2025] > f5[2019]).sum()))
    older = f5[f5.index.get_level_values("age_band") != "30-49"]
    N["five.cells50.up"], N["five.cells50"] = str(int((older[2025] > older[2019]).sum())), str(len(older))
    f5tot = f5.groupby(level=["sex", "age_band"]).sum()
    N["five.groups.up"], N["five.groups"] = str(int((f5tot[2025] > f5tot[2019]).sum())), str(len(f5tot))
    # from 2019 (2016 selection rules) every death coded I46 is I46.1; before that I46.9 also occurs
    i46_all = c3[(c3.icd3 == "I46") & (c3.year >= 2019)].groupby("year").deaths.sum()
    i461 = c4[(c4.ICD == "I461") & (c4.year >= 2019)].groupby("year").deaths.sum()
    if not i46_all.round(6).equals(i461.reindex(i46_all.index).round(6)):
        raise ValueError("from 2019 on, deaths coded I46 are not all I46.1; the text calls them I46.1")
    cond = d["condensed"]
    N["cond.cells"] = count(len(cond))
    N["cond.categories"] = str(cond["cat"].nunique())
    N["cond.maxdiff"] = str(int(cond["difference"].abs().max()))
    for sex, sc, *_ in SEXES:                 # the five codes named in the text, taken together (ages 50-64)
        x = c4[(c4.sex == sex) & (c4.age_band == "50-64") & c4.ICD.isin(["R570", "R092", "I516", "I509", "I10"])].groupby("year").deaths.sum()
        N[f"fivecodes.{sc}.2019"], N[f"fivecodes.{sc}.2025"] = count(x[2019]), count(x[2025])
        N[f"fivecodes.{sc}.pct"] = plain((x[2025] / x[2019] - 1) * 100, 0)
        y = c4[(c4.sex == sex) & (c4.age_band == "50-64") & c4.ICD.isin(["I219", "I251", "I259"])].groupby("year").deaths.sum()
        N[f"ihdcodes.{sc}.2019"], N[f"ihdcodes.{sc}.2025"] = count(y[2019]), count(y[2025])

    # observed levels and changes since 2019 -----------------------------------------
    o = d["obs"]
    lv = {(r.population, r.sex, r.measure, r.year): r.own_value for r in o.itertuples()}
    for r in d["t25"][d["t25"].spec == "main"].itertuples():
        lv[("TWN", r.sex, r.measure, 2025)] = r.observed
    for (pop, sex, m, year), v in lv.items():
        if m not in ("e", "q25", "q50", "q75", "iqr"):
            continue
        k = f"{pop}.{sex_code[sex]}.{m}.{year}"
        N[f"lvl.{k}"] = plain(v)
        if year >= 2020:
            N[f"chg19.{k}"] = signed(v - lv[(pop, sex, m, 2019)])
            N[f"chg19abs.{k}"] = plain(abs(v - lv[(pop, sex, m, 2019)]))
    for pop, _ in FOCUS:
        for sex, sc, *_ in SEXES:
            N[f"trendgain.{pop}.{sc}"] = plain((lv[(pop, sex, "e", 2019)] - lv[(pop, sex, "e", 2003)]) / 16.0)
    for sex, sc, *_ in SEXES:                                    # trend-expected rise 2019 -> 2025 in Taiwan
        e25 = float(d["t25"][(d["t25"].spec == "main") & (d["t25"].sex == sex) & (d["t25"].measure == "e")].expected.iloc[0])
        N[f"exprise.TWN.{sc}.2025"] = plain(e25 - lv[("TWN", sex, "e", 2019)])

    # changes in standardised death rates between years (Taiwan, age bands) ------------------
    tw = d["rr"][(d["rr"].grouping == "band") & (d["rr"].population == "TWN")]
    for sex, sc, *_ in SEXES:
        for band in BANDS:
            x = tw[(tw.sex == sex) & (tw.age_group == band)].set_index("year").ratio_to_2019
            for y0, y1 in ((2012, 2019), (2012, 2018), (2019, 2022), (2019, 2025), (2022, 2025)):
                N[f"chgrate.{sc}.{band}.{y0}.{y1}"] = signed((x[y1] / x[y0] - 1) * 100, 1)
                N[f"chgrateabs.{sc}.{band}.{y0}.{y1}"] = plain(abs(x[y1] / x[y0] - 1) * 100, 1)

    # replication of published deficits -----------------------------------------------
    rep = d["rep"]
    for r in rep.itertuples():
        N[f"rep.own.{r.population}.{r.year}"] = signed(r.deficit_this_study)
        if np.isfinite(r.deficit_dowd_et_al):
            N[f"rep.pub.{r.population}.{r.year}"] = signed(r.deficit_dowd_et_al)
    for r in rep.itertuples():
        if np.isfinite(r.difference):
            N[f"rep.absdiff.{r.population}.{r.year}"] = plain(abs(r.difference))
    N["rep.maxdiff.2022_2024"] = plain(rep[rep.year >= 2022].difference.abs().max(), 3)
    N["rep.maxdiff.all"] = plain(rep.difference.abs().max(), 3)

    # indicators that do not depend on cause-of-death certification ---------------------
    ems = d["ctx_ems"].set_index("year")
    for year, v in ems["ohca_non_traumatic"].items():
        N[f"ctx.ohca.{year}"] = count(v)
    N["ctx.ohca.pct"] = signed((ems.loc[2025, "ohca_non_traumatic"] / ems.loc[2019, "ohca_non_traumatic"] - 1) * 100, 1)
    N["ctx.ohcaabs.pct"] = plain(abs(ems.loc[2025, "ohca_non_traumatic"] / ems.loc[2019, "ohca_non_traumatic"] - 1) * 100, 1)
    inq = d["ctx_inq"].set_index("year")
    for year, r in inq.iterrows():
        N[f"ctx.inquest.{year}"] = count(r.judicial_inquests)
        N[f"ctx.regdeaths.{year}"] = count(r.deaths_registered)
    N["ctx.inquest.pct"] = plain((inq.loc[2025, "judicial_inquests"] / inq.loc[2019, "judicial_inquests"] - 1) * 100, 1)
    N["ctx.deaths.pct"] = plain((inq.loc[2025, "deaths_registered"] / inq.loc[2019, "deaths_registered"] - 1) * 100, 1)
    for r in d["ctx_place"].itertuples():
        sc = {"Male": "M", "Female": "F", "Total": "T"}[r.sex]
        N[f"ctx.home.{sc}.{r.year}"] = plain(r.share_residence * 100, 1)
        N[f"ctx.outside.{sc}.{r.year}"] = plain(r.share_outside_hospital * 100, 1)
    bc = d["by_cause"]
    tot = bc.groupby("year").deaths.sum()
    ill = bc[bc.cause_group == "senility_illdefined"].groupby("year").deaths.sum()
    for year in tot.index:
        N[f"illshare.{year}"] = plain(ill[year] / tot[year] * 100, 1)
        N[f"illdeaths.{year}"] = count(ill[year])

    # checks -------------------------------------------------------------------
    N["check.hmd.maxdiff.e"] = plain(o[o.measure == "e"].difference.abs().max(), 3)
    N["check.hmd.maxdiff.q"] = plain(o[o.measure.isin(["q25", "q50", "q75"])].difference.abs().max(), 3)
    N["check.bridge.maxdiff"] = plain(d["bridge"][d["bridge"].measure.isin(["e", "q25", "q50", "q75"])].difference.abs().max(), 3)
    N["check.bridge.maxdiff.q90"] = plain(d["bridge"][d["bridge"].measure == "q90"].difference.abs().max(), 3)
    cmeta = json.loads((RT / "cohort_check_metadata.json").read_text())
    N["apc.rule.maxgap"] = plain(cmeta["max_abs_difference_between_rules_in_gap_years"], 3)
    N["apc.rule.maxband"] = plain(cmeta["max_abs_difference_between_rules_in_band_contribution_years"], 3)
    for rule in ("main", "alternative"):
        w = cmeta["cohort_windows_2019_fit"][rule]
        key = "main" if rule == "main" else "alt"
        N[f"apc.rule.{key}.lastkept"] = str(w["last_cohort_kept"])
        N[f"apc.rule.{key}.firstout"] = str(w["last_cohort_kept"] + 1)
        N[f"apc.rule.{key}.line0"], N[f"apc.rule.{key}.line1"] = (str(v) for v in w["line_fitted_to_cohorts"])

    # Monte Carlo error of the interval end points (Taiwan, main specification) ----------------
    def largest(frame: pd.DataFrame, ends: list[str], errors: list[str], rows: int, what: str) -> float:
        """Largest standard error of the interval end points.  A maximum does not show that rows are missing, so
        the rows with an interval are counted first and each must carry both standard errors."""
        have = frame[frame[ends].notna().all(axis=1)]
        without = int(have[errors].isna().any(axis=1).sum())
        if len(have) != rows or without:
            raise ValueError(f"simulation error of {what}: {len(have)} rows with an interval where {rows} are expected, {without} of them without a standard error")
        return float(have[errors].max().max())

    # rows expected: 2 sexes x the years with an interval (x the age groups)
    g = d["gaps"][(d["gaps"].population == "TWN") & (d["gaps"].spec == "main") & (d["gaps"].measure == "e")]
    N["mcse.gap.TWN.max"] = plain(largest(g, ["deficit_low", "deficit_high"], ["deficit_low_mcse", "deficit_high_mcse"], 2 * 6, "the gap in e30, 2020-2025"), 3)
    b = d["bands_all"][(d["bands_all"].population == "TWN") & (d["bands_all"].spec == "main") & (d["bands_all"].measure == "e")]
    N["mcse.band.TWN.max"] = plain(largest(b, ["low", "high"], ["low_mcse", "high_mcse"], 2 * 3 * 4, "the contributions of the four age bands, 2022, 2024 and 2025"), 3)
    r4 = d["rr"][(d["rr"].population == "TWN") & (d["rr"].grouping == "band") & d["rr"].ratio_to_trend.notna()]
    N["mcse.rr.TWN.max"] = plain(100.0 * largest(r4, ["ratio_low", "ratio_high"], ["ratio_low_mcse", "ratio_high_mcse"], 2 * 6 * 4, "the rate ratios of the four age bands, 2020-2025"), 2)
    N["mcse.rrlow.TWN.max"] = plain(100.0 * float(r4["ratio_low_mcse"].max()), 2)
    r5 = d["rr"][(d["rr"].population == "TWN") & (d["rr"].grouping == "five_year") & d["rr"].ratio_to_trend.notna()]
    N["mcse.rr5.TWN.max"] = plain(100.0 * largest(r5, ["ratio_low", "ratio_high"], ["ratio_low_mcse", "ratio_high_mcse"], 2 * 6 * 14, "the rate ratios of the five-year age groups, 2020-2025"), 2)
    for r in d["rr"][(d["rr"].grouping == "band") & d["rr"].ratio_to_trend.notna()].itertuples():
        k = f"{r.population}.{sex_code[r.sex]}.{r.age_group}.{r.year}"
        N[f"rrp.{k}"] = pvalue(r.p_value)
        N[f"rrpmcse.{k}"] = f"{r.p_value_mcse:.3f}"
        N[f"mcse.rrlo.{k}"] = plain(100.0 * r.ratio_low_mcse, 2)
        N[f"mcse.rrhi.{k}"] = plain(100.0 * r.ratio_high_mcse, 2)
        N[f"rrpctlo1.{k}"] = plain((r.ratio_low - 1.0) * 100.0, 1)          # one decimal, for bounds close to zero
        N[f"rrpct1.{k}"] = plain((r.ratio_to_trend - 1.0) * 100.0, 1)
    chk = d["check"]
    N["check.cause.residual.e"] = f"{chk[chk.measure == 'e'].residual.abs().max():.6f}"          # the paper decomposes e30 only
    N["check.cause.residual.all"] = plain(chk.residual.abs().max(), 3)
    # central decompositions only (the simulated life tables behind the intervals are not stored): the contributions are
    # summed again from the rows and compared with the gap itself
    age_residual = 0.0
    for x in (d["bands"], d["bands25"], d["groups"], d["groups25"]):
        e_rows = x[x.measure == "e"]
        keys = [c for c in ("population", "sex", "spec", "year") if c in e_rows.columns]
        again = e_rows.groupby(keys).agg(total=("total_deficit", "first"), summed=("contribution", "sum"), stored=("sum_of_contributions", "first"))
        age_residual = max(age_residual, float((again.total - again.summed).abs().max()), float((again.summed - again.stored).abs().max()))
    if age_residual >= 1e-4:
        raise ValueError(f"the age decomposition of e30 leaves a residual of {age_residual} years; the text says less than 0.0001")
    N["check.age.residual.e"] = f"{age_residual:.6f}"
    N["check.age.residual.bound"] = "0.0001"
    for name, number in TS.items():                                              # numbers of the supplement tables
        N[f"ts.{name}"] = number
    reg = d["reg"]
    for r in reg[reg.year >= 2020].itertuples():
        N[f"reg.cum.{sex_code[r.sex]}.{r.age_band_at_start}.{r.year}"] = signed(r.cumulative_since_end_2019_per_100)
        N[f"reg.year.{sex_code[r.sex]}.{r.age_band_at_start}.{r.year}"] = signed(r.per_100_population)
    return N


def rise_in_registered_deaths(d: dict[str, pd.DataFrame], sex: str) -> tuple[float, float]:
    """Ages 50-64, 2019 to 2025: rise in the number of deaths coded R57.0, R09.2 or I51.6, and rise in the number
    of deaths of the two broad classes (circulatory, ill-defined) to which these codes belong."""
    def rise(frame: pd.DataFrame) -> float:
        by_year = frame[(frame.sex == sex) & (frame.age_band == "50-64")].groupby("year").deaths.sum()
        return float(by_year[2025] - by_year[2019])
    chap = d["chap"]
    two = chap[chap.broad_class.isin(["circulatory (I00-I99)", "ill-defined (R00-R99)"])]
    code = d["code4"]
    return rise(code[code.ICD.isin(["R570", "R092", "I516"])]), rise(two)


def check_claims(N: dict[str, str], d: dict[str, pd.DataFrame]) -> None:
    """Statements of the text that are words, not numbers.  If a rerun changes one of them, the build
    stops here instead of producing a text that no longer matches the results.

    Conditions on intervals and thresholds use the unrounded results (both ends of an interval where the
    text says "inside"); conditions on quantities that only exist as formatted numbers read those.  This
    is a consistency check between results and wording.  It cannot tell whether an interpretation is
    warranted, and it does not read the sentences themselves.
    """
    def v(key: str) -> float:
        return float(N[key].replace(MINUS, "-").replace(",", ""))

    sex_of = {"F": "Female", "M": "Male"}
    rr = d["rr"][d["rr"].grouping == "band"].set_index(["population", "sex", "age_group", "year"]).sort_index()
    gaps = d["gaps"][(d["gaps"].spec == "main") & (d["gaps"].measure == "e")].set_index(["population", "sex", "year"]).sort_index()
    bands = d["bands_all"][(d["bands_all"].spec == "main") & (d["bands_all"].measure == "e")].set_index(["population", "sex", "age_band", "year"]).sort_index()
    base = {"APC": "apc", "Lee-Carter": "lc", "Lee-Carter, 1990-2019": "lc30", "Lee-Carter, 2010-2019": "lc10"}
    model = {"apc": "APC, 30 years", "lc": "Lee-Carter, 17 years", "lc30": "Lee-Carter, 30 years", "lc10": "Lee-Carter, 10 years"}
    logs = {(base[r.baseline], r.sex, r.age_band, r.year): 100.0 * float(np.log(r.observed_over_expected_rate)) for r in d["coh_rr"].itertuples()}
    pct = {(base[r.baseline], r.sex, r.age_band, r.year): 100.0 * (r.observed_over_expected_rate - 1.0) for r in d["coh_rr"].itertuples()}
    cb = d["coh_bt"]

    def ratio(pop, s, band, year):                       # (point, low, high) of observed / expected
        r = rr.loc[(pop, sex_of[s], band, year)]
        return float(r.ratio_to_trend), float(r.ratio_low), float(r.ratio_high)

    def inside(pop, s, band, year) -> bool:
        _, lo, hi = ratio(pop, s, band, year)
        return lo < 1.0 < hi

    def above(pop, s, band, year) -> bool:
        return ratio(pop, s, band, year)[1] > 1.0

    def gap(pop, s, year):                               # (gap, low, high) of e30
        r = gaps.loc[(pop, sex_of[s], year)]
        return float(r.deficit), float(r.deficit_low), float(r.deficit_high)

    def band(s, b, year):                                # (contribution, low, high), Taiwan
        r = bands.loc[("TWN", sex_of[s], b, year)]
        return float(r.contribution), float(r.low), float(r.high)

    def errors(key, s, b, h):                            # backtest errors (100 x log ratio) of one baseline
        g = cb[(cb.horizon == h) & (cb.sex == sex_of[s]) & (cb.model == model[key]) & (cb.measure == f"log rate {b}")]
        return 100.0 * g.error.to_numpy()

    keys4 = ("lc", "lc10", "lc30", "apc")
    c4 = d["code4"]
    five_codes = ["R570", "R092", "I516", "I509", "I10"]
    f5 = c4[c4.ICD.isin(five_codes) & c4.year.isin([2019, 2025])].pivot_table(index=["ICD", "sex", "age_band"], columns="year", values="deaths", aggfunc="sum").fillna(0.0)
    not_up = sorted((code, sex, b) for (code, sex, b), r in f5.iterrows() if not r[2025] > r[2019])
    f5tot = f5.groupby(level=["sex", "age_band"]).sum()

    def series(code, b):                                 # deaths of one code in one age band, both sexes, by year
        return c4[(c4.ICD == code) & (c4.age_band == b)].groupby("year").deaths.sum()

    five = d["rr"][(d["rr"].grouping == "five_year") & (d["rr"].population == "TWN")].set_index(["sex", "age_group", "year"]).sort_index()
    r19 = rr["ratio_to_2019"]
    cond = d["condensed"]
    claims = {
        "the 2022 gap is about 1.2 years and the 2025 gap about 0.6 years, both sexes":
            all(round(abs(gap("TWN", s, 2022)[0]), 1) == 1.2 and round(abs(gap("TWN", s, 2025)[0]), 1) == 0.6 for s in "FM"),
        "e30 in Taiwan: 2020, 2021 and 2025 inside the interval, 2022 and 2023 below it":
            all(gap("TWN", s, y)[1] < 0 < gap("TWN", s, y)[2] for s in "FM" for y in (2020, 2021, 2025))
            and all(gap("TWN", s, y)[2] < 0 for s in "FM" for y in (2022, 2023)),
        "ages 50-64: inside the interval in 2020 and 2021 with a positive point estimate in 2021, above the interval in 2022-2025":
            all(inside("TWN", s, "50-64", y) for s in "FM" for y in (2020, 2021))
            and all(ratio("TWN", s, "50-64", 2021)[0] > 1.0 for s in "FM")
            and all(above("TWN", s, "50-64", y) for s in "FM" for y in (2022, 2023, 2024, 2025)),
        "ages 50-64: more than 10 per cent above expectation in every year 2022-2025":
            all(ratio("TWN", s, "50-64", y)[0] > 1.10 for s in "FM" for y in (2022, 2023, 2024, 2025)),
        "ages 50-64: the gap to expectation is larger in 2025 than in 2022, both sexes, under the main baseline":
            all(ratio("TWN", s, "50-64", 2025)[0] > ratio("TWN", s, "50-64", 2022)[0] for s in "FM"),
        "men aged 50-64: death rate above its 2019 level in every year 2022-2025":
            all(float(r19.loc[("TWN", "Male", "50-64", y)]) > 1.0 for y in (2022, 2023, 2024, 2025)),
        "ages 50-64, 2019-2025: deaths of the two broad classes rose and more than half of the rise is in three codes, both sexes":
            all(two > 0 and three > 0.5 * two for three, two in (rise_in_registered_deaths(d, s) for s in ("Female", "Male"))),
        "ages 80+ in 2022: the same rounded percentage above expectation for women and men":
            N["rrpct.TWN.F.80+.2022"] == N["rrpct.TWN.M.80+.2022"],
        "women in 2022: ages 80+ account for more than half of the gap in e30":
            band("F", "80+", 2022)[0] / gap("TWN", "F", 2022)[0] > 0.5,
        "R09.2, both sexes, ages 30+: more than 4,000 and fewer than 5,000 deaths in each year 2023-2025":
            all(4000 < v(f"natcode.R092.{y}") < 5000 for y in (2023, 2024, 2025)),
        # research/taiwan_2025.py stops if an age is missing in either source, so each row stands for all 71 ages
        # (30-99 and 100+); here every sex and year must be present once, with a finite difference below 0.1
        "deaths by single year of age, 2015-2024: the two sources differ by less than 0.1 at every age, both sexes":
            sorted(zip(d["sources"].sex, d["sources"].year)) == [(s, y) for s in ("Female", "Male") for y in range(2015, 2025)]
            and bool(np.isfinite(d["sources"].max_abs_death_difference_by_age).all())
            and bool((d["sources"].max_abs_death_difference_by_age < 0.1).all()),
        "the line of the cohort effects is fitted to 15 birth cohorts (main rule)":
            v("apc.rule.main.line1") - v("apc.rule.main.line0") + 1 == 15,
        "ages 80+ and 65-79 in 2025: rate ratios and band contributions inside their intervals":
            all(inside("TWN", s, b, 2025) and band(s, b, 2025)[1] < 0 < band(s, b, 2025)[2] for s in "FM" for b in ("80+", "65-79")),
        "ages 50-64: band contribution below zero with its whole interval in 2022, 2024 and 2025, and not smaller in 2025 than in 2022":
            all(band(s, "50-64", y)[2] < 0 for s in "FM" for y in (2022, 2024, 2025))
            and all(band(s, "50-64", 2025)[0] <= band(s, "50-64", 2022)[0] for s in "FM"),
        "ages 80+ and 65-79: contributions smaller in 2025 than in 2022":
            all(abs(band(s, b, 2025)[0]) < abs(band(s, b, 2022)[0]) for s in "FM" for b in ("80+", "65-79")),
        "ages 65-79 make up about three tenths of the 2025 gap of women and about four tenths of that of men (point estimates)":
            25 <= 100 * band("F", "65-79", 2025)[0] / gap("TWN", "F", 2025)[0] < 35
            and 35 <= 100 * band("M", "65-79", 2025)[0] / gap("TWN", "M", 2025)[0] < 45,
        "women aged 30-49 above the interval in 2025, men inside":
            above("TWN", "F", "30-49", 2025) and inside("TWN", "M", "30-49", 2025),
        "2024: Japan and Korea above expectation in all four bands, 50-64 by 10 per cent or more, 80+ above the interval; "
        "Hong Kong inside in all four; Taiwan 80+ inside":
            all(ratio(pp, s, b, 2024)[0] > 1.0 for pp in ("JPN", "KOR") for s in "FM" for b in BANDS)
            and all(ratio(pp, s, "50-64", 2024)[0] >= 1.10 and above(pp, s, "80+", 2024) for pp in ("JPN", "KOR") for s in "FM")
            and all(inside("HKG", s, b, 2024) for s in "FM" for b in BANDS) and all(inside("TWN", s, "80+", 2024) for s in "FM"),
        "e30: Japan's gap widens every year and Japan and Korea are below the interval in 2022-2024; Hong Kong's 2024 gap is within 0.1 years of zero":
            all(gap("JPN", s, 2022)[0] > gap("JPN", s, 2023)[0] > gap("JPN", s, 2024)[0] for s in "FM")
            and all(gap(pp, s, y)[2] < 0 for pp in ("JPN", "KOR") for s in "FM" for y in (2022, 2023, 2024))
            and all(abs(gap("HKG", s, 2024)[0]) < 0.1 for s in "FM"),
        "main specification, six years ahead: positive mean backtest error at 50-64 and 80+; the 2025 gap is above the largest error at 50-64 "
        "and between zero and the mean error at 80+":
            all(errors("lc", s, b, 6).mean() > 0 for s in "FM" for b in ("50-64", "80+"))
            and all(logs[("lc", sex_of[s], "50-64", 2025)] > errors("lc", s, "50-64", 6).max() > 0 for s in "FM")
            and all(0 < logs[("lc", sex_of[s], "80+", 2025)] < errors("lc", s, "80+", 6).mean() for s in "FM"),
        "ages 50-64, 2025: seven of eight gaps exceed the largest six-year backtest error; men with the 2010-2019 window lie inside the "
        "range of errors; women exceed it under all four baselines; none of the eight exceeds it at 80+":
            sum(logs[(k, sex_of[s], "50-64", 2025)] > errors(k, s, "50-64", 6).max() for k in keys4 for s in "FM") == 7
            and errors("lc10", "M", "50-64", 6).min() < logs[("lc10", "Male", "50-64", 2025)] < errors("lc10", "M", "50-64", 6).max()
            and all(logs[(k, "Female", "50-64", 2025)] > errors(k, "F", "50-64", 6).max() for k in keys4)
            and sum(logs[(k, sex_of[s], "80+", 2025)] > errors(k, s, "80+", 6).max() for k in keys4 for s in "FM") == 0
            and N["cohsum.n"] == "8",
        "men aged 50-64: the gap is positive under all four baselines and smallest with the 2010-2019 window; most APC backtest errors are "
        "negative while the 2025 gap is positive":
            all(pct[(k, "Male", "50-64", 2025)] > 0 for k in keys4)
            and pct[("lc10", "Male", "50-64", 2025)] == min(pct[(k, "Male", "50-64", 2025)] for k in keys4)
            and (errors("apc", "M", "50-64", 6) < 0).sum() > len(errors("apc", "M", "50-64", 6)) / 2
            and logs[("apc", "Male", "50-64", 2025)] > 0,
        "ages 65-79, 2025: all eight gaps positive, seven above the largest backtest error; women aged 30-49 above it under all four baselines":
            all(logs[(k, sex_of[s], "65-79", 2025)] > 0 for k in keys4 for s in "FM") and N["cohsum.above.65-79.2025"] == "7"
            and all(logs[(k, "Female", "30-49", 2025)] > errors(k, "F", "30-49", 6).max() for k in keys4),
        "ages 80+ in 2025: within 4 per cent of expectation under the main and 2010-2019 baselines, 4 per cent or more above it under the other two":
            all(abs(pct[(k, sex_of[s], "80+", 2025)]) <= 4 for k in ("lc", "lc10") for s in "FM")
            and all(pct[(k, sex_of[s], "80+", 2025)] >= 4 for k in ("lc30", "apc") for s in "FM"),
        "2022 to 2025 under four baselines: the 80+ contribution shrank in all eight cases, the 50-64 contribution widened in seven":
            N["cohsum.80.nshrunk"] == "8" and N["cohsum.5064.nworse"] == "7",
        "men aged 50-64: death rate above 2019 in 2025, women within 1 per cent of 2019; men fell by less than 5 per cent in 2012-2019; "
        "the other age bands of men fell in 2019-2025":
            N["chgrate.M.50-64.2019.2025"].startswith("+") and abs(v("chgrate.F.50-64.2019.2025")) < 1 and v("chgrateabs.M.50-64.2012.2019") < 5
            and all(N[f"chgrate.M.{b}.2019.2025"].startswith(MINUS) for b in ("30-49", "65-79", "80+")),
        "e30 rose in both six-year periods, less in 2019-2025":
            all(v(f"tot.{s}.1218") > v(f"tot.{s}.1925") > 0 for s in "FM"),
        "ages 50-64: circulatory plus ill-defined rose every year from 2021 (men) and 2022 (women); other causes fell; among men "
        "neoplasms fell and respiratory diseases rose":
            all(v(f"cr.M.50-64.circill.{y}") > v(f"cr.M.50-64.circill.{y - 1}") for y in range(2021, 2026))
            and all(v(f"cr.F.50-64.circill.{y}") > v(f"cr.F.50-64.circill.{y - 1}") for y in range(2022, 2026))
            and all(N[f"crpct.{s}.50-64.rest.1925"].startswith(MINUS) for s in "FM") and N["crpct.M.50-64.resp.1925"].startswith("+")
            and N["crpct.M.50-64.neo.1925"].startswith(MINUS),
        "men aged 50-64: each of the five codes rose; ischaemic heart disease changed by less than 5 per cent in both sexes; the two "
        "sudden-death codes: 2025 within 10 per cent of 2019, never above 200 a year, national change close to that of all deaths at 30+":
            all(v(f"code.M.50-64.{c}.2025") > v(f"code.M.50-64.{c}.2019") for c in ("R570", "R092", "I516", "I509", "I10"))
            and all(v(f"ihd.pctabs.{s}.50-64.1925") < 5 for s in "FM") and abs(v("sudden.50-64.2025") / v("sudden.50-64.2019") - 1) < 0.1
            and v("sudden.50-64.max") < 200 and abs(v("natsudden.pct") - v("nat30.deaths.pct")) < 5
            and v("code.M.50-64.R570.2019") < v("code.M.50-64.R570.2018"),
        "ill-defined deaths rose at ages 65+ in both sexes while circulatory, respiratory and all-cause death rates of men aged 80+ fell":
            all(N[f"crpct.{s}.{b}.ill.1925"].startswith("+") for s in "FM" for b in ("65-79", "80+"))
            and all(N[f"crpct.M.80+.{k}.1925"].startswith(MINUS) for k in ("circ", "resp", "all")),
        "national series: ambulance-attended arrests fell, judicial inquests rose less than deaths, the share of home deaths rose in 2021 "
        "and is below 2019 in 2025":
            v("ctx.ohca.2025") < v("ctx.ohca.2019") and v("ctx.inquest.pct") < v("ctx.deaths.pct")
            and v("ctx.home.T.2021") > v("ctx.home.T.2019") > v("ctx.home.T.2025"),
        "registration at ages 50-64: net change negative by 2022 and within 0.5 per cent of zero by 2025":
            all(v(f"reg.cum.{s}.50-64.2022") < 0 and abs(v(f"reg.cum.{s}.50-64.2025")) < 0.5 for s in "FM"),
        "replication of the published deficits for 2022-2024 within 0.05 years":
            v("rep.maxdiff.2022_2024") < 0.05,
        "simulation error is small next to the intervals: at most 0.05 years for gaps and contributions, 1 point for the four age bands":
            v("mcse.gap.TWN.max") < 0.05 and v("mcse.band.TWN.max") < 0.05 and v("mcse.rr.TWN.max") < 1.0,
        "six-year backtests of the main specification: mean under-prediction about ten per cent for men aged 80+, under 3 per cent for men aged 50-64":
            8 < v("cohbt.biaspct.h6.lc.M.rate80+") < 13 and 0 < v("cohbt.biaspct.h6.lc.M.rate50-64") < 3,
        "supplement: 2024 against five-year backtests, all eight 50-64 gaps above the largest error":
            N["cohsum.above.50-64.2024"] == "8",
        "supplement: ages 80+ below the interval in 2020, above it in 2022 and 2023, inside it in 2024 and 2025":
            all(ratio("TWN", s, "80+", 2020)[2] < 1 and above("TWN", s, "80+", 2022) and above("TWN", s, "80+", 2023)
                and inside("TWN", s, "80+", 2024) and inside("TWN", s, "80+", 2025) for s in "FM"),
        "supplement: circulatory plus ill-defined at 65+ rose among men and fell among women while all-cause rates fell; registration sums "
        "within 0.25 per cent of zero at ages 50-79 by 2025":
            all(N[f"crpct.M.{b}.circill.1925"].startswith("+") and N[f"crpct.F.{b}.circill.1925"].startswith(MINUS)
                and N[f"crpct.{s}.{b}.all.1925"].startswith(MINUS) for b in ("65-79", "80+") for s in "FM")
            and all(abs(v(f"reg.cum.{s}.{b}.2025")) <= 0.25 for s in "FM" for b in ("50-64", "65-79")),
        "the five codes, 2019 to 2025: together they rose in all eight sex-by-age groups; each rose in every group aged 50 and over; "
        "the only cells without a rise are I10 among women aged 30-49 and R09.2 among men aged 30-49":
            bool((f5tot[2025] > f5tot[2019]).all()) and not_up == [("I10", "Female", "30-49"), ("R092", "Male", "30-49")],
        "R09.2 jumped in 2021 and again in 2022 in each of the three age bands from 50 up and did not fall back; "
        "R57.0 jumped in 2022 and in 2024 in all four age bands and did not fall back":
            all(series("R092", b)[2021] > 1.2 * series("R092", b)[2020] and series("R092", b)[2022] > 1.2 * series("R092", b)[2021]
                and series("R092", b)[2025] > 1.3 * series("R092", b)[2020] for b in ("50-64", "65-79", "80+"))
            and all(series("R570", b)[2022] > 1.3 * series("R570", b)[2021] and series("R570", b)[2024] > 1.3 * series("R570", b)[2023]
                    and series("R570", b)[2025] >= series("R570", b)[2022] for b in BANDS),
        "women aged 50-64 against 2019: 2 to 4 per cent lower in 2020 and 2021, higher in 2022, within 1 per cent in 2025":
            all(0.96 <= r19[("TWN", "Female", "50-64", y)] <= 0.98 for y in (2020, 2021)) and r19[("TWN", "Female", "50-64", 2022)] > 1.0
            and abs(r19[("TWN", "Female", "50-64", 2025)] - 1.0) < 0.01,
        "ages 50-64 in 2025: positive under all four baselines; smaller than the main specification under APC and the 2010-2019 window in both "
        "sexes; under the 1990-2019 window larger than the main specification for men and within one point of it for women":
            all(pct[(k, sx, "50-64", 2025)] > 0 for k in keys4 for sx in ("Female", "Male"))
            and all(pct[(k, sx, "50-64", 2025)] < pct[("lc", sx, "50-64", 2025)] for k in ("apc", "lc10") for sx in ("Female", "Male"))
            and pct[("lc30", "Male", "50-64", 2025)] > pct[("lc", "Male", "50-64", 2025)]
            and abs(pct[("lc30", "Female", "50-64", 2025)] - pct[("lc", "Female", "50-64", 2025)]) < 1.0,
        "men aged 30-49 in 2025: within 3 per cent of expectation under the main specification; below it under one and above it under "
        "another of the other three baselines":
            abs(pct[("lc", "Male", "30-49", 2025)]) < 3
            and min(pct[(k, "Male", "30-49", 2025)] for k in ("lc10", "lc30", "apc")) < 0 < max(pct[(k, "Male", "30-49", 2025)] for k in ("lc10", "lc30", "apc")),
        "men, five-year groups in 2025: point estimates above expectation from 55-59 to 70-74, with an interval that includes no difference at 70-74":
            all(five.loc[("Male", g, 2025), "ratio_to_trend"] > 1.0 for g in ("55-59", "60-64", "65-69", "70-74"))
            and five.loc[("Male", "70-74", 2025), "ratio_low"] < 1.0 < five.loc[("Male", "70-74", 2025), "ratio_high"],
        "the Ministry's condensed categories rebuilt from the detailed codes agree with the published file in every cell":
            int(cond["difference"].abs().max()) == 0 and len(cond) == 444 and cond["cat"].nunique() == 13,
        "deaths coded to COVID-19 contribute a small negative amount to the 2019-2025 change in e30 in both sexes":
            all(-0.1 < v(f"cause.{sx}.covid.1925") < 0 for sx in "FM"),
    }
    failed = [name for name, ok in claims.items() if not ok]
    if failed:
        raise ValueError("the text states things the results no longer support:\n  " + "\n  ".join(failed))
    print(f"{len(claims)} statements of the text checked against the results")


# ----------------------------------------------------------------------------
# tables
# ----------------------------------------------------------------------------
def tex_escape(v) -> str:
    """Escape %, & and _ in a table cell; characters that are already escaped stay as they are."""
    return re.sub(r"(?<!\\)([%&_])", r"\\\1", str(v))


KEEP_PANEL = 12             # a panel of a long table with at most this many rows stays on one page


class TableSource(str):
    """LaTeX of a table.  `cells` keeps what every row is meant to show, so that the typeset files can be
    checked against the intended content and not against the LaTeX (research/check_outputs.py)."""
    cells: list
    note: str


def plain_text(v) -> str:
    return str(v).replace(r"\%", "%").replace(r"\&", "&").replace(r"\_", "_").strip()


def table(title: str, headers: list[str], rows: list[list], note: str = "", align: str | None = None, long: bool = False,
          size: str = r"\small", supplement: bool = False, group_header: str | None = None, tabcolsep: str | None = None,
          label: str | None = None) -> TableSource:
    """One table; `group_header` is an optional LaTeX line above the column heads."""
    cells = [[plain_text(title)]]
    if group_header:
        spanning = re.findall(r"\\multicolumn\{\d+\}\{[^{}]*\}\{([^{}]*)\}", group_header)
        if spanning:
            cells.append([plain_text(x) for x in spanning])
    cells.append([plain_text(h) for h in headers])
    for row in rows:
        if isinstance(row, tuple) and row[0] == "panel":
            cells.append([plain_text(row[1])])
        elif isinstance(row, tuple) and row[0] == "raw":
            content = re.sub(r"\\multicolumn\{\d+\}\{[^{}]*\}\{([^{}]*)\}", r"\1", row[1].split(r"\\")[0])
            cells.append([plain_text(c) for c in content.split("&")])
        elif row != "midrule":
            cells.append([plain_text(v) for v in row])
    env = "longtable" if long else "tabular"
    align = align or ("l" + "r" * (len(headers) - 1))
    cap = r"\caption*{" if supplement else r"\caption{"
    if label:
        title = title + r"\label{" + label + "}"
    # LaTeX resets the font size at the start of a float, so the size command goes inside the table environment
    lines = [r"\begingroup" + size + r"\singlespacing"]
    if not long:
        lines += [r"\begin{table}[htbp]\centering" + size, cap + title + "}"]
    if tabcolsep:
        lines.append(r"\setlength{\tabcolsep}{" + tabcolsep + "}")
    if long:                                    # do not start a long table in the last lines of a page
        lines.append(r"\Needspace{14\baselineskip}% PDFONLY")
    lines.append(r"\begin{" + env + "}{" + align + "}")
    if long:
        lines.append(cap + title + r"}\\")
    head = [r"\toprule"] + ([group_header] if group_header else []) + [" & ".join(headers) + r" \\", r"\midrule"]
    lines += head
    if long:
        plain_title = title.split(r"\label{")[0]
        lines += [r"\endfirsthead", r"\multicolumn{" + str(len(headers)) + r"}{l}{" + plain_title + r"（續）}\\"] + head + [r"\endhead"]
    data_lines: list[int] = []                  # where the ordinary rows are in `lines`
    panel_rows: list[list[int]] = []            # the same, panel by panel
    for row in rows:
        if row == "midrule":
            lines.append(r"\midrule")
        elif isinstance(row, tuple) and row[0] == "panel":          # a line spanning the table, e.g. the name of a panel
            # in a long table the name of a panel stays on the same page as its first row
            lines.append(r"\multicolumn{" + str(len(headers)) + r"}{l}{" + row[1] + r"} \\" + ("*" if long else ""))
            panel_rows.append([])
        elif isinstance(row, tuple) and row[0] == "raw":
            lines.append(row[1])
        else:
            lines.append(" & ".join(tex_escape(v) for v in row) + r" \\")
            data_lines.append(len(lines) - 1)
            if panel_rows:
                panel_rows[-1].append(len(lines) - 1)
    if long:
        # A panel of up to KEEP_PANEL rows is not split between two pages: a continuation page repeats the title
        # and the column heads but not the name of the panel, so its rows would stand there without it.
        for members in panel_rows:
            if len(members) <= KEEP_PANEL:
                for k in members[:-1]:
                    lines[k] += "*"
    note_line = r"\par\smallskip{\footnotesize\raggedright " + note + r"\par}"
    if long and note:
        # In the PDF the note of a long table is the last row of the table, as wide as the text and held together
        # with the rule and the rows above it (the last panel, or the last three rows of a table without
        # panels): a page break can neither split the note nor part it from the end of the table.  The Word
        # version keeps the note as a paragraph after the table.  It drops the lines marked PDFONLY, and with
        # them the \iffalse ... \fi around the two parts that LaTeX skips.
        last_panel = panel_rows[-1] if panel_rows and len(panel_rows[-1]) <= KEEP_PANEL else []
        for k in (last_panel or data_lines[-3:]):                   # the whole last panel, or the last three rows
            if not lines[k].endswith("*"):
                lines[k] += "*"
        span = r"\multicolumn{" + str(len(headers)) + r"}{@{}c@{}}"
        # the closing rule is a row of its own, as wide as the table; the two skips take back the height and the
        # depth that every row of a table has, so that the rule sits as \bottomrule would
        lines += [r"\noalign{\nobreak\vskip\dimexpr\aboverulesep-\arraystretch\ht\strutbox\relax}% PDFONLY",
                  span + r"{\leavevmode\leaders\hrule height\heavyrulewidth\hfill\kern0pt} \\*% PDFONLY",
                  r"\noalign{\nobreak\vskip\dimexpr\belowrulesep-\arraystretch\dp\strutbox+\smallskipamount\relax\nobreak}% PDFONLY",
                  span + r"{\makebox[0pt][c]{\parbox[t]{\linewidth}{\footnotesize\raggedright " + note + r"}}} \\% PDFONLY",
                  r"\iffalse% PDFONLY", r"\bottomrule", r"\fi% PDFONLY", r"\end{" + env + "}",
                  r"\iffalse% PDFONLY", note_line, r"\fi% PDFONLY"]
    else:
        lines += [r"\bottomrule", r"\end{" + env + "}"]
        if note:
            lines.append(note_line)
    if not long:
        lines.append(r"\end{table}")
    lines.append(r"\endgroup")
    out = TableSource("\n".join(lines))
    out.cells, out.note = cells, plain_text(note)
    # rows of `cells` that are names of panels (the title, the spanning heads and the column heads come first)
    n_head = len(cells) - len([r for r in rows if r != "midrule"])
    out.panels = [n_head + k for k, r in enumerate([r for r in rows if r != "midrule"]) if isinstance(r, tuple) and r[0] == "panel"]
    return out


def par_col(width: str) -> str:
    """Column of fixed width with ragged-right text (justified text stretches the spaces in narrow columns)."""
    return r">{\raggedright\arraybackslash}p{" + width + "}"


def star(value: str, low: float, high: float) -> str:
    return value + ("*" if (low > 0 or high < 0) else "")


def build_tables(d: dict[str, pd.DataFrame], N: dict[str, str]) -> dict[str, str]:
    T: dict[str, str] = {}
    gaps = d["gaps"][d["gaps"].spec == "main"]

    # Table 1: gap in e30, four populations -----------------------------------------
    rows = []
    for pop, zh in FOCUS:
        for sex, sc, _, short in SEXES:
            g = gaps[(gaps.population == pop) & (gaps.sex == sex) & (gaps.measure == "e")].set_index("year")
            cells = [star(signed(g.loc[y, "deficit"]), g.loc[y, "deficit_low"], g.loc[y, "deficit_high"]) if y in g.index else "—" for y in range(2020, 2026)]
            last = g.index.max()
            rows.append([zh, short] + cells + [f"{signed(g.loc[last, 'deficit_low'])} 至 {signed(g.loc[last, 'deficit_high'])}"])
    T["table1"] = table("東亞四地 30 歲平均餘命與趨勢預期值的差距，2020–2025 年", ["地區", "性別", "2020", "2021", "2022", "2023", "2024", "2025", r"最近一年缺口的 95\% 區間"], rows,
                        "註：表中數值為 30 歲平均餘命的缺口，也就是實際值減去趨勢預期值，單位為年；負值表示實際平均餘命低於疫情前趨勢延續時的水準。"
                        "趨勢預期值由 2003–2019 年資料配適的 Lee–Carter 模型推估。"
                        r"* 表示實際值落在 95\% 預測區間之外。最後一欄是各地最近一年缺口的 95\% 區間；最近一年在臺灣為 2025 年，在其餘三地為 2024 年。"
                        "臺灣 2025 年數值以內政部原始統計重建，其餘數值取自 HMD。",
                        align="llrrrrrrc", label="tab:gaps")

    # Table 2: backtest, five years ahead ----------------------------------------------
    rows = []
    for scope, label in (("pool", "各地區彙整（不含波羅的海三國）"), ("TWN", "臺灣")):
        rows.append(("panel", label))
        for m in ("e", "q25", "q50", "q75"):
            row = ["\u3000" + MEASURE_ZH[m]]
            for w in ("17y", "10y"):
                k = f"{w}.corrected.{scope}.{m}.5"
                row += [N[f"bt.bias.{k}"], N[f"bt.rmse.{k}"], N[f"bt.cover.{k}"]]
            rows.append(row)
        if scope == "pool":
            rows.append("midrule")
    T["table2"] = table("疫情前回測：提前 5 年預測的誤差與區間涵蓋率", ["指標", "平均誤差", "均方根誤差", "涵蓋率", "平均誤差", "均方根誤差", "涵蓋率"], rows,
                        "註：回測以 2005–2014 年每一年為起點，只用起點當年及以前的資料配適模型，再把提前 5 年的預測值與實際值比較。"
                        "誤差為實際值減預測值，單位為年；"
                        r"涵蓋率是實際值落在 95\% 預測區間之內的回測所占百分比。「各地區彙整」把各地區的回測合併計算，不包含愛沙尼亞、拉脫維亞與立陶宛，因為這三國在回測期間死亡率變動劇烈。"
                        f"資料起始年較晚的地區，只納入配適期完整的起點。納入這三國的結果見附表 {TS['backtest']}。",
                        align="lrrrrrr", tabcolsep="4.5pt", label="tab:backtest",
                        group_header=r" & \multicolumn{3}{c}{配適期 17 年} & \multicolumn{3}{c}{配適期 10 年} \\ \cmidrule(lr){2-4}\cmidrule(lr){5-7}")

    # Table 3: death rates against the trend by age band ---------------------------------
    rr = d["rr"][d["rr"].grouping == "band"]

    def cell(pop: str, sex: str, band: str, year: int) -> str:
        r = rr[(rr.population == pop) & (rr.sex == sex) & (rr.age_group == band) & (rr.year == year)].iloc[0]
        s = f"{pct(r.ratio_to_trend)}（{pct(r.ratio_low)}，{pct(r.ratio_high)}）"
        return s + ("*" if (r.ratio_low > 1 or r.ratio_high < 1) else "")

    rows = [("panel", "（一）臺灣各年")]
    for sex, sc, _, short in SEXES:
        for band in BANDS:
            rows.append([short, BAND_ZH[band]] + [cell("TWN", sex, band, y) for y in (2022, 2023, 2024, 2025)])
    rows.append("midrule")
    rows.append(("panel", "（二）2024 年四地比較"))
    rows.append(("raw", r"性別 & 年齡 & 臺灣 & 日本 & 韓國 & 香港 \\ \midrule"))
    for sex, sc, _, short in SEXES:
        for band in BANDS:
            rows.append([short, BAND_ZH[band]] + [cell(p, sex, band, 2024) for p in ("TWN", "JPN", "KOR", "HKG")])
    T["table3"] = table(r"各年齡組死亡率與趨勢預期值的差距（\%）", ["性別", "年齡", "2022", "2023", "2024", "2025"], rows,
                        r"註：表中數值為死亡率的差距，也就是（實際死亡率 ÷ 趨勢預期死亡率 − 1）× 100；正值表示實際死亡率高於預期值，負值表示低於預期值。括號內為差距的 95\% 區間，* 表示區間不包含 0。"
                        "各年齡組死亡率，以該地區、該性別 2019 年暴露人年的年齡結構標準化。",
                        align="llcccc", label="tab:ratios")

    # Table 4: causes and the change in e30 ----------------------------------------------
    rows = []
    for key, label, _ in CAUSE_ROWS:
        row = [label]
        for _, sc, *_ in SEXES[::-1]:
            row += [N[f"cause.{sc}.{key}.1218"], N[f"cause.{sc}.{key}.1925"], N[f"causediff.{sc}.{key}"]]
        rows.append(row)
    rows.append("midrule")
    rows.append(["合計"] + [v for _, sc, *_ in SEXES[::-1] for v in (N[f"tot.{sc}.1218"], N[f"tot.{sc}.1925"], N[f"totdiff.{sc}"])])
    T["table4"] = table("各死因對臺灣 30 歲平均餘命變動的貢獻：2012–2018 年與 2019–2025 年", ["死因", "2012–2018", "2019–2025", "差", "2012–2018", "2019–2025", "差"], rows,
                        "註：表中數值為各死因對 30 歲平均餘命變動的貢獻，單位為年。正值表示該死因死亡率下降，使平均餘命增加；負值表示該死因死亡率上升，使平均餘命減少。"
                        "兩段期間各為六年：2012–2018 年適用舊版死因選取準則，2019–2025 年適用 2016 年版準則，兩段期間各自在同一套準則之內計算。"
                        "「差」為後一段期間的貢獻減去前一段期間的貢獻；兩段期間適用不同的死因選取準則，解讀「差」時需要留意。各列數值加總與「合計」列之間的差異，來自四捨五入。",
                        align=par_col("11.4em") + "rrrrrr", tabcolsep="2.6pt", label="tab:causes",
                        group_header=r" & \multicolumn{3}{c}{男性} & \multicolumn{3}{c}{女性} \\ \cmidrule(lr){2-4}\cmidrule(lr){5-7}")

    # Table 5: ages 50-64, death rates by broad class ----------------------------------------
    rows = []
    for key, label, _ in BROAD_ROWS:
        row = [label]
        for _, sc, *_ in SEXES[::-1]:
            change = N.get(f"crpct.{sc}.50-64.{key}.1925", "—") if key != "covid" else "—"
            row += [N[f"cr.{sc}.50-64.{key}.2019"], N[f"cr.{sc}.50-64.{key}.2025"], change]
        rows.append(row)
        if key == "circill":
            rows.append("midrule")
    rows.append("midrule")
    rows.append(["全死因"] + [v for _, sc, *_ in SEXES[::-1] for v in (N[f"cr.{sc}.50-64.all.2019"], N[f"cr.{sc}.50-64.all.2025"], N[f"crpct.{sc}.50-64.all.1925"])])
    T["table5"] = table("臺灣 50–64 歲死因別死亡率：2019 年與 2025 年", ["死因（ICD-10）", "2019", "2025", r"變動（\%）", "2019", "2025", r"變動（\%）"], rows,
                        "註：2019 與 2025 兩欄為年齡標準化死亡率，單位為每十萬人年的死亡人數，以各性別 2019 年 50–64 歲暴露人年的年齡結構為標準。"
                        "全死因死亡率取自 HMD 與本文重建的 2025 年數值，再依衛生福利部死因統計中各年齡組的死因比率分配到各類死因。"
                        "「其他」指表中所列各類以外的全部死因。「變動」為 2025 年相對於 2019 年的百分比變動。",
                        align=par_col("13em") + "rrrrrr", tabcolsep="4pt", label="tab:midlife",
                        group_header=r" & \multicolumn{3}{c}{男性} & \multicolumn{3}{c}{女性} \\ \cmidrule(lr){2-4}\cmidrule(lr){5-7}")

    # ---------------- supplement ----------------
    FS, TC = r"\footnotesize", "3pt"
    LC_NOTE = "LC 表示 Lee–Carter 模型，APC 表示年齡—時期—世代模型；模型名稱之後的年份是配適期。"

    def cap(name: str, title: str) -> str:
        return f"附表 {TS[name]}　{title}"

    # 22 populations, 2024
    rows = []
    order = gaps[(gaps.year == 2024) & (gaps.measure == "e")].groupby("population").deficit.mean().sort_values().index
    for pop in order:
        for sex, sc, _, short in SEXES:
            g = gaps[(gaps.population == pop) & (gaps.sex == sex) & (gaps.year == 2024)].set_index("measure")
            rows.append([NAMES[pop], short] + [star(signed(g.loc[m, "deficit"]), g.loc[m, "deficit_low"], g.loc[m, "deficit_high"]) for m in ("e", "q25", "q50", "q75")]
                        + [f"{signed(g.loc['e', 'deficit_low'])} 至 {signed(g.loc['e', 'deficit_high'])}"])
    T["tableS_pops"] = table(cap("pops", N["n.populations"] + " 個地區 2024 年與趨勢預期值的差距"), ["地區", "性別", "平均餘命", "第 25 百分位", "中位數", "第 75 百分位", r"平均餘命缺口的 95\% 區間"], rows,
                             r"註：表中數值為各指標的缺口（實際值減趨勢預期值），單位為年；「平均餘命」指 30 歲平均餘命。各地區依兩性平均的 30 歲平均餘命缺口由低到高排列。* 表示實際值落在 95\% 預測區間之外。",
                             align="llrrrrc", long=True, supplement=True, size=FS)

    # Taiwan, death rates against the trend in every year 2020-2025 (main specification)
    rows = []
    for k, years3 in enumerate(((2020, 2021, 2022), (2023, 2024, 2025))):
        if k:
            rows += ["midrule", ("raw", "性別 & 年齡 & " + " & ".join(str(y) for y in years3) + r" \\ \midrule")]
        for sex, sc, _, short in SEXES:
            for band in BANDS:
                rows.append([short, BAND_ZH[band]] + [cell("TWN", sex, band, y) for y in years3])
    T["tableS_agerates"] = table(cap("agerates", r"臺灣各年齡組死亡率與趨勢預期值的差距，2020–2025 年（\%）"), ["性別", "年齡", "2020", "2021", "2022"], rows,
                                 r"註：表中數值為死亡率的差距，定義與正文表 3 相同，也就是（實際死亡率 ÷ 趨勢預期死亡率 − 1）× 100；括號內為差距的 95\% 區間，* 表示區間不包含 0。各年齡組死亡率以該性別 2019 年暴露人年的年齡結構標準化，趨勢預期值來自主要基準。",
                                 align="llccc", supplement=True, size=FS)

    # Taiwan under alternative baselines
    rows = []
    specs = [("LC 2003–2019（主要基準）", lambda sc, m, y: N[f"gap.TWN.{sc}.{m}.{y}"]),
             ("LC 2010–2019", lambda sc, m, y: N[f"gap.window2010.TWN.{sc}.{m}.{y}"]),
             ("LC 2003–2019，不校正起點", lambda sc, m, y: N[f"gap.fitted_start.TWN.{sc}.{m}.{y}"]),
             ("LC 1990–2019", lambda sc, m, y: N[f"cohgap.lc30.{sc}.{m}.{y}"]),
             ("APC 1990–2019", lambda sc, m, y: N[f"cohgap.apc.{sc}.{m}.{y}"])]
    for sex, sc, sex_zh, short in SEXES:
        rows.append(("panel", sex_zh))
        for label, f in specs:
            rows.append(["　" + label] + [f(sc, m, y) for m in ("e", "q25", "q75") for y in (2022, 2024, 2025)])
    T["tableS_baselines"] = table(cap("baselines", "不同趨勢基準下臺灣的缺口"), ["基準", "2022", "2024", "2025", "2022", "2024", "2025", "2022", "2024", "2025"], rows,
                                  "註：表中數值為各指標的缺口（實際值減趨勢預期值），單位為年；兩個百分位都指活到 30 歲者的死亡年齡百分位。" + LC_NOTE + "除了標明「不校正起點」的一列之外，各基準都以配適期最後三年的實際死亡數校正預測起點。",
                                  align="lrrrrrrrrr", supplement=True, size=FS, tabcolsep="1.6pt",
                                  group_header=r" & \multicolumn{3}{c}{30 歲平均餘命} & \multicolumn{3}{c}{第 25 百分位} & \multicolumn{3}{c}{第 75 百分位} \\ \cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}")

    # age-band contributions and rate ratios under four baselines (Taiwan)
    baselines = (("lc", "LC 2003–2019"), ("lc10", "LC 2010–2019"), ("lc30", "LC 1990–2019"), ("apc", "APC 1990–2019"))
    short_band = [BAND_ZH[b].replace(" 歲", "").replace("以上", "+") for b in BANDS]
    rows = []
    for sex, sc, _, short in SEXES:
        for b, label in baselines:
            for y in (2022, 2024, 2025):
                rows.append([short, label, str(y)] + [N[f"coh.{b}.{sc}.{band}.{y}"] for band in BANDS] + [N[f"cohrr.{b}.{sc}.{band}.{y}"] for band in BANDS])
        if sex == "Female":
            rows.append("midrule")
    T["tableS_bands"] = table(cap("bands", "四種基準下臺灣各年齡組對平均餘命缺口的貢獻與死亡率差距"), ["性別", "基準", "年"] + short_band * 2, rows,
                              r"註：每個年齡組各有兩個數值：對 30 歲平均餘命缺口的貢獻（單位為年），以及死亡率的差距（單位為 \%，負值表示實際死亡率低於預期值）。" + LC_NOTE, align="lllrrrrrrrr", supplement=True, size=FS, long=True,
                              group_header=r" & & & \multicolumn{4}{c}{對平均餘命缺口的貢獻（年）} & \multicolumn{4}{c}{死亡率的差距（\%）} \\ \cmidrule(lr){4-7}\cmidrule(lr){8-11}")

    # the 2025 gap in death rates against pre-pandemic forecast errors of the same quantity and the same horizon
    rows = []
    band_pairs = (("30-49", "50-64"), ("65-79", "80+"))
    stat_head = "基準 & " + " & ".join(["平均", "最小", "最大", "2025"] * 2)
    for k, pair in enumerate(band_pairs):
        if k:
            rows += ["midrule",
                     ("raw", " & " + " & ".join(r"\multicolumn{4}{c}{" + BAND_ZH[b] + "}" for b in pair) + r" \\ \cmidrule(lr){2-5}\cmidrule(lr){6-9}"),
                     ("raw", stat_head + r" \\ \midrule")]
        for sex, sc, sex_zh, short in SEXES:
            rows.append(("panel", sex_zh))
            for b, label in baselines:
                row = ["　" + label]
                for band in pair:
                    key = f"h6.{b}.{sc}.rate{band}"
                    row += [N[f"cohbt.bias.{key}"], N[f"cohbt.min.{key}"], N[f"cohbt.max.{key}"], N[f"cohlog.{b}.{sc}.{band}.2025"]]
                rows.append(row)
    n6 = N["cohbt.n.h6.lc.M.rate50-64"]
    T["tableS_backtestgap"] = table(cap("backtestgap", "臺灣各年齡組 2025 年死亡率差距與疫情前回測誤差的比較（對數尺度）"), stat_head.split(" & "), rows,
                                    "註：表中數值為 100 × ln（實際死亡率 ÷ 預期死亡率），接近以百分比表示的差距；正值表示實際死亡率高於預期，也就是預測低估了死亡率。"
                                    f"「平均」「最小」「最大」三欄，是以 2005–2013 年為起點、提前 6 年預測的 {n6} 次回測誤差的平均值、最小值與最大值；"
                                    "「2025」一欄是 2025 年實際死亡率相對於各基準預期值的差距，同樣是提前 6 年的預測。"
                                    "各個起點所用資料互相重疊，各次回測並不獨立；回測的最大值是樣本中的極值，不是具有固定涵蓋率的界限。" + LC_NOTE,
                                    align="l" + "r" * 8, supplement=True, size=FS, tabcolsep="4pt",
                                    group_header=" & " + " & ".join(r"\multicolumn{4}{c}{" + BAND_ZH[b] + "}" for b in band_pairs[0])
                                    + r" \\ \cmidrule(lr){2-5}\cmidrule(lr){6-9}")

    # backtest in full
    rows = []
    for scope, label in (("pool", "不含波羅的海三國"), ("all", "納入波羅的海三國"), ("TWN", "臺灣")):
        for w, wl in (("17y", "17 年"), ("10y", "10 年")):
            rows.append(("panel", f"{label}，配適期 {wl}"))
            for m in ("e", "q25", "q75"):
                row = ["　" + MEASURE_ZH[m]]
                for h in (1, 3, 5):
                    k = f"{w}.corrected.{scope}.{m}.{h}"
                    row += [N[f"bt.bias.{k}"], N[f"bt.rmse.{k}"], N[f"bt.cover.{k}"]]
                rows.append(row)
    T["tableS_backtest"] = table(cap("backtest", "以 2005–2014 年各年為起點、提前 1、3、5 年的回測結果"), ["指標", "平均", "均方根", "涵蓋率", "平均", "均方根", "涵蓋率", "平均", "均方根", "涵蓋率"], rows,
                                 r"註：「平均」為平均誤差，「均方根」為均方根誤差；誤差為實際值減預測值，單位為年。涵蓋率是實際值落在 95\% 預測區間之內的回測所占百分比。17 年配適期不含資料年數不足的韓國與克羅埃西亞。",
                                 align="lrrrrrrrrr", supplement=True, size=FS, long=True,
                                 group_header=r" & \multicolumn{3}{c}{提前 1 年} & \multicolumn{3}{c}{提前 3 年} & \multicolumn{3}{c}{提前 5 年} \\ \cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}")

    # 2025 rebuilt against HMD
    br = d["bridge"]
    rows = []
    for sex, sc, _, short in SEXES:
        for m in ("e", "q25", "q50", "q75"):
            g = br[(br.sex == sex) & (br.measure == m)].set_index("year")
            rows.append([short, MEASURE_ZH[m]] + [signed(g.loc[y, "difference"], 3) for y in range(2018, 2025)])
    T["tableS_bridge"] = table(cap("bridge", "以內政部原始統計重建的數值與 HMD 數值的差"), ["性別", "指標"] + [str(y) for y in range(2018, 2025)], rows,
                               "註：表中差值為重建數值減去由當年 HMD 死亡數與暴露人年編算的數值，單位為年；百分位指活到 30 歲者的死亡年齡百分位。重建每一年的數值時，都只用該年之前三年的 HMD 資料換算暴露人年，再與該年的 HMD 數值比較；2025 年的重建採用同一程序。",
                               align="llrrrrrrr", supplement=True, size=FS)

    # registration
    reg = d["reg"]
    rows = []
    for sex, sc, _, short in SEXES:
        for band in ("30-49", "50-64", "65-79", "80-94"):
            g = reg[(reg.sex == sex) & (reg.age_band_at_start == band)].set_index("year")
            rows.append([short, band.replace("-", "–")] + [signed(g.loc[y, "per_100_population"]) for y in range(2018, 2026)] + [signed(g.loc[y, "cumulative_since_end_2019_per_100"]) for y in (2022, 2025)])
    T["tableS_registration"] = table(cap("registration", "戶籍人口中死亡以外的淨變動（每百人）"), ["性別", "年齡"] + [str(y) for y in range(2018, 2026)] + ["至 2022", "至 2025"], rows,
                                     "註：各年數值按年初年齡分組計算：把各出生世代相鄰兩個年底的人口差加回當年死亡數（取相鄰兩個年齡的死亡數各半），加總後除以該組年初人口，再乘以 100。"
                                     "負值表示加回死亡數之後，戶籍人口仍然減少。最後兩欄是 2020 年起各年數值的加總；各年的年齡組由不同出生世代組成，這項加總只是概略的累計。",
                                     align="llrrrrrrrrrr", supplement=True, size=FS, tabcolsep=TC)

    # cause categories and their ICD-10 codes
    rows = [["COVID-19", "U07、U09、U10"], ["惡性腫瘤", "C00–C97"], ["缺血性心臟病", "I20–I25"], ["腦血管疾病", "I60–I69"],
            ["其他心臟疾病", "I01、I02.0、I05–I09、I27、I30–I52"], ["高血壓性疾病", "I10–I15"], ["糖尿病", "E10–E14"],
            ["肺炎、流感及吸入性肺炎", "J10–J18、J66–J69"], ["慢性下呼吸道疾病", "J40–J47"],
            ["慢性肝病、肝硬化及酒精所致疾病", "K73、K74、B15–B19；F10、K70、X45、K86.0、I42.6、G31.2、G62.1、G72.1、K29.2"],
            ["腎炎、腎病症候群及腎病變", "N00–N07、N17–N19、N25–N27"], ["敗血症", "A40、A41"], ["失智症、帕金森病及阿茲海默病", "F01–F03、G20、G21、G30"],
            ["症狀、徵候及診斷欠明（含衰老）", "R00–R99"], ["事故傷害", "V01–X59、Y85、Y86"], ["蓄意自我傷害", "X60–X84、Y87.0"],
            ["其他", "其餘代碼，含司法相驗中的案件"]]
    if [r[0] for r in rows] != [label for _, label, _ in CAUSE_ROWS]:
        raise ValueError("the category table and CAUSE_ROWS list different categories")
    T["tableS_categories"] = table(cap("categories", "死因類別與 ICD-10 代碼"), ["類別", "ICD-10 代碼"], rows,
                                   "註：「肺炎、流感及吸入性肺炎」除了 J10–J18 之外，另含 J66–J69，也就是吸入有機粉塵、化學物質、固體或液體引起的呼吸系統疾病。"
                                   "「慢性肝病、肝硬化及酒精所致疾病」另含病毒性肝炎（B15–B19）與酒精意外中毒（X45）。", align="l" + par_col("24em"), supplement=True, size=FS)

    # codes at ages 50-64
    c3 = d["code3"]
    rows = []
    for sex, sc, sex_zh, short in SEXES[::-1]:
        p = c3[(c3.sex == sex) & (c3.age_band == "50-64")].pivot(index="icd3", columns="year", values="rate_per_100k").fillna(0.0)
        dth = c3[(c3.sex == sex) & (c3.age_band == "50-64")].pivot(index="icd3", columns="year", values="deaths").fillna(0.0)
        p["change"] = p[2025] - p[2019]
        p = p[~p.index.isin(["ZZZ", "U07"])]
        for label, part in (("上升最多的 8 個代碼", p.sort_values("change", ascending=False).head(8)), ("下降最多的 8 個代碼", p.sort_values("change").head(8))):
            rows.append(("panel", f"{sex_zh}，{label}"))
            for code, r in part.iterrows():
                if code not in ICD_ZH:
                    raise KeyError(f"no Chinese name for {code}")
                rows.append(["　" + code, ICD_ZH[code], count(dth.loc[code, 2019]), count(dth.loc[code, 2025]), plain(r[2019], 1), plain(r[2025], 1), signed(r["change"], 1)])
    T["tableS_codes"] = table(cap("codes", "臺灣 50–64 歲死亡率變動最大的 ICD-10 代碼，2019–2025 年"), ["代碼", "名稱", "2019", "2025", "2019", "2025", "變動"], rows,
                              "註：死亡率是年齡標準化死亡率（每十萬人年；以各性別 2019 年 50–64 歲暴露人年的年齡結構為標準），「變動」為 2025 年死亡率減 2019 年死亡率。COVID-19（U07）與司法相驗中的案件沒有列入排序。",
                              align="l" + par_col("15em") + "rrrrr", supplement=True, size=FS, long=True,
                              group_header=r" & & \multicolumn{2}{c}{死亡人數} & \multicolumn{3}{c}{死亡率} \\ \cmidrule(lr){3-4}\cmidrule(lr){5-7}")

    # broad classes by age band
    rows = []
    for sex, sc, sex_zh, short in SEXES[::-1]:
        for band in BANDS:
            rows.append(("panel", f"{sex_zh} {BAND_ZH[band]}"))
            for key, label in (("circill", "循環系統疾病與診斷欠明"), ("restnocovid", "其他死因（不含 COVID-19）"), ("all", "全死因")):
                rows.append(["　" + label, N[f"cr.{sc}.{band}.{key}.2012"], N[f"cr.{sc}.{band}.{key}.2018"], N[f"crpct.{sc}.{band}.{key}.1218"],
                             N[f"cr.{sc}.{band}.{key}.2019"], N[f"cr.{sc}.{band}.{key}.2025"], N[f"crpct.{sc}.{band}.{key}.1925"]])
    T["tableS_broad"] = table(cap("broad", "臺灣各年齡組兩大類死因的死亡率"), ["死因", "2012", "2018", r"變動（\%）", "2019", "2025", r"變動（\%）"], rows,
                              "註：各年度欄為年齡標準化死亡率，單位為每十萬人年的死亡人數，以同一性別、同一年齡組 2019 年暴露人年的年齡結構為標準；「變動」欄為該段期間的百分比變動。循環系統疾病指 I00–I99，診斷欠明指 R00–R99。「其他死因」指這兩大類與 COVID-19 以外的全部死因，"
                              "全死因則包含 COVID-19。死因選取準則在 2018 年與 2019 年之間改版，因此 2012–2018 年與 2019–2025 年兩段期間各自比較。",
                              align="lrrrrrr", supplement=True, size=FS, long=True)

    # national counts of the codes discussed in the text
    c4 = d["code4"]
    nat = c4.groupby(["ICD", "year"]).deaths.sum().unstack("year").fillna(0)
    labels = [("R570", "R57.0 心因性休克"), ("R092", "R09.2 呼吸停止、心肺衰竭"), ("R54", "R54 衰老"), ("I516", "I51.6 未明示心血管疾病"),
              ("I519", "I51.9 未明示心臟病"), ("I509", "I50.9 未明示心臟衰竭"), ("I10", "I10 本態性高血壓"), ("I219", "I21.9 未明示急性心肌梗塞")]
    years = list(range(2015, 2026))
    rows = [[lab] + [count(nat.loc[code, y]) for y in years] for code, lab in labels]
    rows.append("midrule")
    nat3 = d["code3"].groupby(["icd3", "year"]).deaths.sum().unstack("year").fillna(0)          # three-character codes, ages 30+
    for code, lab in (("I46", "I46 心跳停止"), ("R96", "R96 原因不明猝死")):
        rows.append([lab] + [count(nat3.loc[code, y]) for y in years])
    rows.append("midrule")
    rows.append(["R00–R99 占全部死亡（%）"] + [N[f"illshare.{y}"] for y in years])
    T["tableS_vague"] = table(cap("vague", "正文討論的 ICD-10 代碼歷年死亡人數，2015–2025 年"), ["代碼"] + [str(y) for y in years], rows,
                              "註：表中數值為 30 歲以上、兩性合計的死亡人數；只有最後一列是全年齡 R00–R99 的死亡人數占全部死亡人數的百分比。臺灣自 2019 年起改用 2016 年版死因選取準則。I21.9 列為對照。"
                              "I46 在 2019 年以後全部是 I46.1（心因性猝死），2018 年以前另包含 I46.9（未明示的心跳停止）。I46.1 與 R96 是字面上表示猝死的代碼；"
                              "依原死因選取準則，這兩個代碼屬於診斷不明確的情況，證明書上另有可選疾病時不選為原死因，已知病因的猝死則歸入該病因。",
                              align=par_col("12.5em") + "rrrrrrrrrrr", supplement=True, size=FS, tabcolsep="1.6pt")

    # national indicators that do not use the cause-of-death codes
    ems = d["ctx_ems"].set_index("year")
    inq = d["ctx_inq"].set_index("year")
    place = d["ctx_place"]
    rows = []
    for year in range(2019, 2026):
        pt = place[(place.year == year) & (place.sex == "Total")].iloc[0]
        pm = place[(place.year == year) & (place.sex == "Male")].iloc[0]
        rows.append([str(year), count(inq.loc[year, "deaths_registered"]), count(ems.loc[year, "ohca_non_traumatic"]), count(inq.loc[year, "judicial_inquests"]),
                     plain(inq.loc[year, "inquests_per_death"] * 100, 1), plain(pt.share_residence * 100, 1), plain(pm.share_residence * 100, 1),
                     plain(pt.share_outside_hospital * 100, 1)])
    T["tableS_context"] = table(cap("context", "不使用死因代碼的三項全國指標，2019–2025 年"), ["年", "死亡人數", "心肺功能停止", "相驗案件", r"相驗率（\%）", r"兩性（\%）", r"男性（\%）", r"醫院外（\%）"], rows,
                                "註：「死亡人數」是戶籍登記的死亡人數。「心肺功能停止」是消防機關救護送醫的非創傷類到院前心肺功能停止人數，只計入送醫的人。"
                                "「相驗案件」是全國地方檢察署司法相驗案件數，不包含衛生所等辦理的行政相驗；「相驗率」是司法相驗案件數占死亡人數的比率。"
                                "「兩性」與「男性」兩欄，是在住居所死亡的人數占死亡人數的比率。「醫院外」一欄是兩性合計在醫院以外各場所死亡的人數占死亡人數的比率。這三項統計都沒有公布年齡別數字。",
                                align="lrrrrrrr", supplement=True, size=FS, tabcolsep="2pt",
                                group_header=r" & & & & & \multicolumn{2}{c}{在住居所死亡} & \\ \cmidrule(lr){6-7}")
    missing = [name for name in SUPP_ORDER if f"tableS_{name}" not in T]
    if missing:
        raise ValueError(f"supplement tables not built: {missing}")
    return T


ICD_ZH = {
    "R57": "休克，他處未歸類", "R09": "循環及呼吸系統的其他症狀", "I51": "診斷欠明的心臟病及其併發症", "I50": "心臟衰竭", "I10": "本態性高血壓",
    "J18": "肺炎，病原體未明示", "C18": "結腸惡性腫瘤", "I61": "腦內出血", "E11": "第 2 型糖尿病", "I63": "腦梗塞", "X80": "由高處跳下之蓄意自我傷害",
    "C22": "肝及肝內膽管惡性腫瘤", "C15": "食道惡性腫瘤", "C16": "胃惡性腫瘤", "K74": "肝纖維化及肝硬化", "X70": "以懸吊、勒縊及窒息之蓄意自我傷害",
    "C13": "下咽惡性腫瘤", "E14": "未明示的糖尿病", "K70": "酒精性肝病", "C34": "支氣管及肺惡性腫瘤", "C11": "鼻咽惡性腫瘤", "V89": "未明示車輛事故",
    "A41": "其他敗血症", "I64": "中風，未明示出血或梗塞", "X68": "農藥中毒之蓄意自我傷害", "C85": "非何杰金氏淋巴瘤，其他及未明示", "C53": "子宮頸惡性腫瘤",
    "C80": "未明示部位惡性腫瘤", "I21": "急性心肌梗塞", "C50": "乳房惡性腫瘤", "C55": "子宮惡性腫瘤，部位未明示", "J45": "氣喘", "C64": "腎惡性腫瘤",
    "I11": "高血壓性心臟病", "K85": "急性胰臟炎", "J40": "支氣管炎，未明示急性或慢性", "V23": "機車騎士與汽車碰撞受傷", "F20": "思覺失調症",
    "I71": "主動脈瘤及剝離", "I60": "蜘蛛膜下腔出血", "I27": "其他肺性心臟病", "I25": "慢性缺血性心臟病", "C54": "子宮體惡性腫瘤", "C56": "卵巢惡性腫瘤",
    "N18": "慢性腎臟病", "G93": "腦的其他疾患", "X67": "其他氣體及蒸氣中毒之蓄意自我傷害", "R99": "其他診斷欠明及未明示原因的死亡", "K92": "消化系統其他疾病",
    "C06": "口腔其他及未明示部位惡性腫瘤", "C25": "胰臟惡性腫瘤", "C61": "攝護腺惡性腫瘤", "I46": "心跳停止", "R54": "衰老",
    "R96": "其他猝死，原因不明",
}


# ----------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------
def style() -> None:
    plt.rcParams.update({
        "font.family": ["Liberation Sans", "cwTeX Q Hei"], "axes.unicode_minus": False, "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 600, "figure.dpi": 120, "pdf.fonttype": 42,
    })


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def figure1(d) -> None:
    """Observed e30 and the trend, four populations."""
    obs, gaps = d["obs"], d["gaps"][(d["gaps"].spec == "main") & (d["gaps"].measure == "e")]
    fig, axes = plt.subplots(2, 4, figsize=(10.5, 5.0), sharex=True)
    for j, (pop, zh) in enumerate(FOCUS):
        for i, (sex, _, sex_zh, _) in enumerate(SEXES):
            ax = axes[i, j]
            o = obs[(obs.population == pop) & (obs.sex == sex) & (obs.measure == "e")].sort_values("year")
            g = gaps[(gaps.population == pop) & (gaps.sex == sex)].sort_values("year")
            years = list(o.year) + [y for y in g.year if y > o.year.max()]
            values = list(o.own_value) + [float(g[g.year == y].observed.iloc[0]) for y in g.year if y > o.year.max()]
            last19 = float(o[o.year == 2019].own_value.iloc[0])
            ax.fill_between(g.year, g.observed - g.deficit_high, g.observed - g.deficit_low, color="0.82", linewidth=0, label="95% 預測區間")
            ax.plot([2019] + list(g.year), [last19] + list(g.expected), color="0.35", linestyle="--", linewidth=1.2, label="趨勢預期值")
            ax.plot(years, values, color="black", marker="o", markersize=2.6, linewidth=1.0, label="實際值")
            ax.axvline(2019.5, color="0.6", linewidth=0.6)
            if i == 0:
                ax.set_title(zh, fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{sex_zh}\n30 歲平均餘命（年）")
            ax.set_xticks([2005, 2010, 2015, 2020, 2025])
            ax.set_xlim(2002.3, 2025.7)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, "fig1_trajectories")


def figure2(d) -> None:
    """Age-group contributions to Taiwan's e30 gap, 2022, 2024 and 2025."""
    g = d["groups_all"]
    g = g[(g.population == "TWN") & (g.spec == "main") & (g.measure == "e") & g.year.isin([2022, 2024, 2025])]
    groups = [f"{a}-{a + 4}" for a in range(30, 100, 5)] + ["100+"]
    labels = [f"{a}–{a + 4}" for a in range(30, 100, 5)] + ["100+"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7), sharey=True)
    width = 0.27
    x = np.arange(len(groups))
    for ax, (sex, _, sex_zh, _) in zip(axes, SEXES):
        for k, (year, color, hatch) in enumerate(((2022, "0.80", None), (2024, "0.50", None), (2025, "0.10", None))):
            v = g[(g.sex == sex) & (g.year == year)].set_index("age_group").loc[groups, "contribution"]
            ax.bar(x + (k - 1) * width, v, width=width, color=color, label=f"{year} 年", hatch=hatch, linewidth=0)
        ax.axhline(0, color="0.3", linewidth=0.6)
        ax.set_title(sex_zh)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=60, ha="right")
        ax.set_xlabel("年齡組（歲）")
        ax.grid(axis="y", color="0.9", linewidth=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("對 30 歲平均餘命缺口的貢獻（年）")
    axes[0].legend(frameon=False, loc="lower left")
    fig.tight_layout()
    save(fig, "fig2_age_gap")


def figure3(d) -> None:
    """Age-standardised death rates by age band relative to 2019, Taiwan 2012-2025."""
    rr = d["rr"]
    rr = rr[(rr.population == "TWN") & (rr.grouping == "band")]
    styles = {"30-49": ("0.60", "-", "o"), "50-64": ("black", "-", "D"), "65-79": ("0.35", "--", "s"), "80+": ("0.35", ":", "^")}
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), sharey=True)
    for ax, (sex, _, sex_zh, _) in zip(axes, SEXES):
        for band in BANDS:
            s = rr[(rr.sex == sex) & (rr.age_group == band)].sort_values("year")
            color, ls, mk = styles[band]
            ax.plot(s.year, s.ratio_to_2019, color=color, linestyle=ls, marker=mk, markersize=3.2, linewidth=1.6 if band == "50-64" else 1.1, label=BAND_ZH[band])
        ax.axhline(1, color="0.5", linewidth=0.6)
        ax.axvline(2019, color="0.7", linewidth=0.6)
        ax.set_title(sex_zh)
        ax.set_xlabel("年")
        ax.set_xticks([2012, 2014, 2016, 2018, 2020, 2022, 2024])
        ax.grid(axis="y", color="0.92", linewidth=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("年齡標準化死亡率（2019 年 = 1）")
    axes[1].legend(frameon=False, loc="upper right", ncol=2)
    fig.tight_layout()
    save(fig, "fig3_rates_2019")


def figure4(d) -> None:
    """Ages 50-64: death rates in three groups of causes, 2008-2025."""
    chap = d["chap"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    series_style = (("circill", "循環系統疾病與診斷欠明（I00–I99、R00–R99）", dict(color="black", linestyle="-", marker="o")),
                    ("neo", "腫瘤（C00–D48）", dict(color="0.50", linestyle="--", marker="s")),
                    ("rest", "其他死因（不含腫瘤與 COVID-19）", dict(color="0.30", linestyle=":", marker="^")))
    for ax, (sex, _, sex_zh, _) in zip(axes, SEXES):
        t = chap[(chap.sex == sex) & (chap.age_band == "50-64")].pivot(index="year", columns="broad_class", values="rate_per_100k").fillna(0.0)
        circill = t["circulatory (I00-I99)"] + t["ill-defined (R00-R99)"]
        neo = t["neoplasms (C00-D48)"]
        rest = t.sum(axis=1) - circill - neo - t["COVID-19"]
        data = {"circill": circill, "neo": neo, "rest": rest}
        for key, label, st in series_style:
            s = data[key]
            for seg in (s.loc[:2018], s.loc[2019:]):
                ax.plot(seg.index, seg.values, markersize=3, linewidth=1.5 if key == "circill" else 1.1, label=label if seg.index[0] == 2019 else None, **st)
        ax.axvline(2018.5, color="0.6", linewidth=0.6)
        ax.set_title(f"{sex_zh} 50–64 歲")
        ax.set_xlabel("年")
        ax.set_xticks([2008, 2011, 2014, 2017, 2019, 2022, 2025])
        ax.set_ylim(0, None)
        ax.grid(axis="y", color="0.92", linewidth=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("年齡標準化死亡率（每十萬人）")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "fig4_causes_5064")


def main() -> None:
    GEN.mkdir(parents=True, exist_ok=True)
    d = load()
    N = build_numbers(d)
    check_claims(N, d)
    T = build_tables(d, N)
    (GEN / "numbers.json").write_text(json.dumps(N, ensure_ascii=False, indent=0, sort_keys=True))
    (GEN / "tables.json").write_text(json.dumps(T, ensure_ascii=False, indent=0))
    (GEN / "table_cells.json").write_text(json.dumps({name: {"cells": t.cells, "note": t.note, "panels": t.panels} for name, t in T.items()}, ensure_ascii=False, indent=0))
    style()
    figure1(d)
    figure2(d)
    figure3(d)
    figure4(d)
    print(f"{len(N)} numbers, {len(T)} tables, 4 figures")


if __name__ == "__main__":
    main()
