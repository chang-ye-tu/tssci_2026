#!/usr/bin/env python3
"""Death rates by ICD-10 code and age band, Taiwan 2008-2025.

The cause decomposition (research/cause_decomposition.py) works with 24 cause
groups.  This script keeps the Ministry's ICD-10 codes so that one can see
which codes carry a change and whether a rise in one code is matched by a fall
in a neighbouring one (a sign of certification or coding drift).

Rates are built exactly as in the decomposition: the all-cause death rate by
single year of age (HMD; 2025 rebuilt) is split by code in proportion to the
Ministry's deaths within each five-year age group, and standardised within the
age band to the 2019 exposure of the same sex.

Outputs (results_cause/):
    code_rates_by_band.csv     sex, age band, year, three-character code: deaths and standardised rate
    code4_deaths_by_band.csv   the same at four characters, deaths only, for the codes in WATCH
    chapter_rates_by_band.csv  the same by ICD-10 chapter-like broad class

Run from the repository root (reads downloaded files, hence -I):
    python3 -I research/cause_codes.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cause_prepare as cp  # noqa: E402
import trendlib as tl  # noqa: E402
from cause_decomposition import AGE0, AGES, BANDS, GROUP_LOWER, GROUP_OF_AGE, all_cause_rates  # noqa: E402

OUT = Path("results_cause")
YEARS = range(2008, 2026)
WATCH = ("R57", "R09", "R96", "R98", "R99", "R54", "I46", "I50", "I51", "I10", "I11", "I21", "I25", "I64", "J96", "J98")


def broad_class(icd: str) -> str:
    """Broad classes that keep the circulatory system and the ill-defined chapter separate."""
    L, n, code = cp.parse(icd)
    if L == "U" and n in (7, 9, 10):
        return "COVID-19"
    if L in ("A", "B"):
        return "infectious (A00-B99)"
    if L == "C" or (L == "D" and n <= 48):
        return "neoplasms (C00-D48)"
    if L == "E":
        return "endocrine and metabolic (E00-E90)"
    if L in ("F", "G"):
        return "mental and nervous (F00-G99)"
    if L == "I":
        return "circulatory (I00-I99)"
    if L == "J":
        return "respiratory (J00-J99)"
    if L == "K":
        return "digestive (K00-K93)"
    if L == "N":
        return "genitourinary (N00-N99)"
    if L == "R":
        return "ill-defined (R00-R99)"
    if L in ("V", "W", "X", "Y"):
        return "external (V01-Y98)"
    return "all other"


def main() -> None:
    detail = cp.read_detail()
    detail["icd3"] = detail["ICD"].str[:3]
    long = detail.melt(id_vars=["year", "sex", "ICD", "icd3"], value_vars=list(cp.AGE_COLUMNS), var_name="age_col", value_name="deaths")
    long["age_lower"] = long["age_col"].map(cp.AGE_COLUMNS)
    long = long[(long.age_lower >= AGE0) & (long.deaths > 0)].copy()
    long["broad"] = long["ICD"].map(broad_class)

    band_of_group = {lower: next(b for b, (lo, hi) in BANDS.items() if lo <= lower <= hi) for lower in GROUP_LOWER}
    rows3, rows4, rowsb = [], [], []
    for sex in tl.SEXES:
        m_all, expo = all_cause_rates(sex)
        w19 = expo[2019]
        # standardised all-cause rate contributed by each five-year group, per death in that group and year
        factor = {}                                                    # (year, group lower) -> rate per 100k per death
        for year in YEARS:
            tot = long[(long.sex == sex) & (long.year == year)].groupby("age_lower")["deaths"].sum()
            for band, (lo, hi) in BANDS.items():
                idx = (AGES >= lo) & (AGES <= hi)
                w = np.zeros(len(AGES))
                w[idx] = w19[idx] / w19[idx].sum()
                for g, lower in enumerate(GROUP_LOWER):
                    if band_of_group[lower] != band:
                        continue
                    in_g = GROUP_OF_AGE == g
                    factor[(year, lower)] = float((m_all[year][in_g] * w[in_g]).sum() * 1e5 / tot.loc[lower])
        d = long[long.sex == sex].copy()
        d["band"] = d["age_lower"].map(band_of_group)
        d["rate"] = d["deaths"] * [factor[(y, a)] for y, a in zip(d["year"], d["age_lower"])]
        g3 = d.groupby(["band", "year", "icd3"], as_index=False)[["deaths", "rate"]].sum()
        g3.insert(0, "sex", sex)
        rows3.append(g3)
        gb = d.groupby(["band", "year", "broad"], as_index=False)[["deaths", "rate"]].sum()
        gb.insert(0, "sex", sex)
        rowsb.append(gb)
        g4 = d[d.icd3.isin(WATCH)].groupby(["band", "year", "ICD"], as_index=False)["deaths"].sum()
        g4.insert(0, "sex", sex)
        rows4.append(g4)

    r3 = pd.concat(rows3, ignore_index=True).rename(columns={"band": "age_band", "rate": "rate_per_100k"})
    rb = pd.concat(rowsb, ignore_index=True).rename(columns={"band": "age_band", "rate": "rate_per_100k", "broad": "broad_class"})
    r4 = pd.concat(rows4, ignore_index=True).rename(columns={"band": "age_band"})
    r3.to_csv(OUT / "code_rates_by_band.csv", index=False)
    rb.to_csv(OUT / "chapter_rates_by_band.csv", index=False)
    r4.to_csv(OUT / "code4_deaths_by_band.csv", index=False)

    # consistency with the grouped rates used in the decomposition
    grouped = pd.read_csv(OUT / "cause_rates_by_band.csv")
    a = grouped.groupby(["sex", "age_band", "year"])["rate_per_100k"].sum()
    b = r3.groupby(["sex", "age_band", "year"])["rate_per_100k"].sum()
    print("max |all-cause rate here - all-cause rate in cause_rates_by_band.csv| =", float((a - b).abs().max()))

    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 400)
    pd.set_option("display.max_columns", 40)
    years = [2012, 2015, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
    for sex in tl.SEXES:
        for band in ("50-64",):
            t = rb[(rb.sex == sex) & (rb.age_band == band)].pivot(index="broad_class", columns="year", values="rate_per_100k")[years]
            t.loc["ALL CAUSES"] = t.sum()
            t["2019->2025"] = t[2025] - t[2019]
            t["2012->2018"] = t[2018] - t[2012]
            print(f"\n== {sex} {band}: standardised death rate per 100,000 by broad class ==")
            print(t.round(1).to_string())
            p = r3[(r3.sex == sex) & (r3.age_band == band)].pivot(index="icd3", columns="year", values="rate_per_100k").fillna(0.0)
            p["change"] = p[2025] - p[2019]
            top = pd.concat([p.sort_values("change").head(15), p.sort_values("change").tail(20)])
            print(f"\n== {sex} {band}: three-character codes with the largest falls and rises, 2019 -> 2025 (rate per 100,000) ==")
            print(top[years + ["change"]].round(1).to_string())


if __name__ == "__main__":
    main()
