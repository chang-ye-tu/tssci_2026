#!/usr/bin/env python3
"""Extend Taiwan's observed adult mortality to 2025 with official raw counts.

The HMD release ends in 2024 for Taiwan.  Splicing the Ministry of the
Interior's published 2025 life table onto the HMD series would mix two
methods (in 2019 the two sources differ by 0.3 years of life expectancy at
birth).  Instead, 2025 is rebuilt from raw counts with the same construction
as every other year:

    deaths      MOI/RIS, by sex and single year of age, by date of occurrence
    exposure    mean of the year-end populations of 2024 and 2025 (RIS),
                multiplied by an age-specific factor that brings it onto the
                HMD scale (HMD re-estimates the population at high ages)

The factor is the ratio of HMD exposure to the raw mid-year population,
averaged over the three latest HMD years.  The procedure is validated by
"predicting" each of 2018-2024 from the three years before it and comparing
the resulting life-table measures with the HMD-based values.

Run from the repository root (reads downloaded workbooks, hence -I):
    python3 -I research/taiwan_2025.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xlrd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trendlib as tl  # noqa: E402

MOI = Path("data/moi/2026-11-17/downloads")
DEATHS_XLS = MOI / "deaths_single_age" / "y3s3-00000.xls"
POP_XLS = MOI / "population_single_age" / "y1s6-00000.xls"
OUT = Path("results_trend")
AGE0, OPEN_AGE = 30, 100
AGES = np.arange(AGE0, OPEN_AGE + 1)
YEARS = np.arange(2015, 2026)            # reconstruction years
HMD_LAST = 2024
CAL_YEARS = 3
SPECS = {"main": ((2003, 2019), "corrected"), "window2010": ((2010, 2019), "corrected"), "fitted_start": ((2003, 2019), "fit")}
N_BOOT, N_PATHS, SEED = 2000, 4, 20261119
MCSE_SEED = 20261198            # separate stream for the Monte Carlo error of interval end points
DECOMP_STEPS = 40
MEASURE_NAMES = ("e", "q10", "q25", "q50", "q75", "q90", "iqr")
BANDS = [(30, 49), (50, 64), (65, 79), (80, 200)]
BAND_LABELS = ["30-49", "50-64", "65-79", "80+"]
SEX_ZH = {"男": "Male", "女": "Female"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_ris(path: Path, sheet: str) -> pd.DataFrame:
    """Rows come in triplets (total, male, female); the Gregorian year sits on
    the male row.  Columns 4.. are ages 0-99 and the open group 100+."""
    ws = xlrd.open_workbook(str(path)).sheet_by_name(sheet)
    header = ws.row_values(2)
    # the age of a column is taken from its position, so the heads of all 101 age columns are compared: 0 to 99, then 100+
    ages = [str(int(x)) if isinstance(x, float) and x.is_integer() else str(x).strip() for x in header[4:]]
    if ws.ncols != 105 or ages != [str(a) for a in range(100)] + ["100+"]:
        raise ValueError(f"unexpected layout in {path.name}/{sheet}: the columns are not ages 0 to 99 followed by 100+")
    records = []
    r = 3
    while r + 2 < ws.nrows:
        rows = [ws.row_values(r + i) for i in range(3)]
        if [str(x[2]).strip() for x in rows] != ["計", "男", "女"]:
            break
        year = int(float(rows[1][1]))
        for row in rows[1:]:
            vals = row[4:105]
            if any(v == "" for v in vals):
                continue                                  # early years without single ages
            records.append((year, SEX_ZH[str(row[2]).strip()], np.array(vals, dtype=float), float(row[3])))
        r += 3
    out = []
    for year, sex, vals, total in records:
        if abs(vals.sum() - total) > 0.5:
            raise ValueError(f"{path.name}/{sheet} {year} {sex}: ages do not sum to the total")
        for age, v in enumerate(vals):
            out.append((year, sex, age, v))
    return pd.DataFrame(out, columns=["year", "sex", "age", "value"])


def to_matrix(df: pd.DataFrame, sex: str, years) -> np.ndarray:
    """(ages 30..99 + 100+) x years."""
    piv = df[df.sex == sex].pivot(index="age", columns="year", values="value").reindex(columns=list(years))
    if piv.isna().any().any():
        raise ValueError("missing single-age data")
    return piv.loc[AGE0:OPEN_AGE].to_numpy(dtype=float)


def main() -> None:
    rng = np.random.default_rng(SEED)
    deaths_ris = read_ris(DEATHS_XLS, "死亡單齡")
    pop_ris = read_ris(POP_XLS, "單齡")
    hmd_d = tl.read_hmd_table("TWN", "Deaths")
    hmd_e = tl.read_hmd_table("TWN", "Exposures")
    hmd_years = range(YEARS[0], HMD_LAST + 1)
    band_idx = np.zeros(len(AGES), dtype=int)
    for i, (lo, hi) in enumerate(BANDS):
        band_idx[(AGES >= lo) & (AGES <= hi)] = i

    rate_rows, valid_rows, result_rows, band_rows, bridge_rows, group_rows = [], [], [], [], [], []
    for sex in tl.SEXES:
        D_own = to_matrix(deaths_ris, sex, YEARS)                                  # (A, 11)
        P_end = to_matrix(pop_ris, sex, range(YEARS[0] - 1, YEARS[-1] + 1))        # year-end 2014..2025
        E_raw = 0.5 * (P_end[:, :-1] + P_end[:, 1:])                               # mid-year population
        D_hmd = tl.age_year_matrix(hmd_d, sex, hmd_years, AGE0, OPEN_AGE)
        E_hmd = tl.age_year_matrix(hmd_e, sex, hmd_years, AGE0, OPEN_AGE)
        n_h = D_hmd.shape[1]
        ratio = E_hmd / E_raw[:, :n_h]
        for j, year in enumerate(hmd_years):
            bridge_rows.append((sex, int(year), float(D_own[:, j].sum()), float(D_hmd[:, j].sum()),
                                float(np.abs(D_own[:, j] - D_hmd[:, j]).max()),
                                float(ratio[:50, j].mean()), float(ratio[50:60, j].mean()), float(ratio[60:70, j].mean()), float(ratio[70, j])))

        def rebuilt_rates(target_col: int) -> np.ndarray:
            factor = ratio[:, target_col - CAL_YEARS:target_col].mean(axis=1)
            return D_own[:, target_col] / (E_raw[:, target_col] * factor)

        # validation: rebuild each HMD year from the three years before it
        for j in range(CAL_YEARS, n_h):
            m_true = D_hmd[:, j] / E_hmd[:, j]
            m_reb = rebuilt_rates(j)
            for name in MEASURE_NAMES:
                a, b = float(tl.measure(m_true, name, AGE0)), float(tl.measure(m_reb, name, AGE0))
                valid_rows.append((sex, int(YEARS[j]), name, a, b, b - a))

        j25 = len(YEARS) - 1
        factor = ratio[:, n_h - CAL_YEARS:n_h].mean(axis=1)
        E_2025 = E_raw[:, j25] * factor
        m_2025 = D_own[:, j25] / E_2025
        for i, age in enumerate(AGES):
            rate_rows.append((sex, int(age), D_own[i, j25], E_raw[i, j25], factor[i], E_2025[i], m_2025[i]))

        # deficits against the trend, with predictive interval for horizon 6
        D_fit_all = tl.age_year_matrix(hmd_d, sex, range(2003, 2020), AGE0, OPEN_AGE)
        E_fit_all = tl.age_year_matrix(hmd_e, sex, range(2003, 2020), AGE0, OPEN_AGE)
        for spec, ((y0, y1), jump) in SPECS.items():
            sl = slice(y0 - 2003, y1 - 2003 + 1)
            fc = tl.forecast(D_fit_all[:, sl], E_fit_all[:, sl], np.array([6]), jump=jump, n_boot=N_BOOT, n_paths=N_PATHS, rng=rng)
            central = fc["central"][:, 0]
            sims = tl.predictive_rates(fc["draws"], E_2025[:, None], rng)[:, 0, :]
            for name in MEASURE_NAMES:
                o = float(tl.measure(m_2025, name, AGE0))
                e = float(tl.measure(central, name, AGE0))
                vals = tl.measure(sims, name, AGE0)
                lo, hi = np.quantile(vals, [0.025, 0.975])
                se_lo, se_hi = tl.quantile_mcse(vals, [0.025, 0.975], cluster=N_PATHS, seed=MCSE_SEED)
                tail = min(np.mean(vals <= o), np.mean(vals >= o))
                result_rows.append((sex, spec, 2025, name, o, e, o - e, o - hi, o - lo, min(1.0, max(2 * tail, 1 / len(vals))),
                                    float(se_hi), float(se_lo), tl.two_sided_p_mcse(vals, o, N_PATHS)))
            if spec == "main":
                for name in ("e", "q25", "q50", "q75"):
                    func = lambda x, n=name: tl.measure(x, n, AGE0)
                    c = tl.horiuchi(func, central, m_2025, DECOMP_STEPS)
                    total = float(tl.measure(m_2025, name, AGE0) - tl.measure(central, name, AGE0))     # the gap itself
                    banded = np.bincount(band_idx, weights=c, minlength=len(BANDS))
                    # interval: the same simulation as for the whole gap, one simulated life table per refit,
                    # each decomposed against the observed rates with the same number of steps
                    cs = tl.horiuchi_batch(func, sims[::N_PATHS], m_2025, DECOMP_STEPS)
                    sims_b = np.stack([cs[:, band_idx == g].sum(axis=1) for g in range(len(BANDS))], axis=1)
                    lo_b, hi_b = np.quantile(sims_b, [0.025, 0.975], axis=0)
                    se_lo_b, se_hi_b = tl.quantile_mcse(sims_b, [0.025, 0.975], cluster=1, seed=MCSE_SEED)
                    for g, label in enumerate(BAND_LABELS):
                        band_rows.append((sex, 2025, name, label, banded[g], lo_b[g], hi_b[g], total, float(c.sum()), se_lo_b[g], se_hi_b[g]))
                    five = np.bincount(np.minimum((AGES - AGE0) // 5, (OPEN_AGE - AGE0) // 5), weights=c)
                    for g, v in enumerate(five):
                        lower = AGE0 + 5 * g
                        group_rows.append((sex, 2025, name, f"{lower}-{lower + 4}" if lower < OPEN_AGE else f"{OPEN_AGE}+", v, total, float(c.sum())))

    OUT.mkdir(exist_ok=True)
    pd.DataFrame(rate_rows, columns=["sex", "age", "deaths_moi", "midyear_population", "hmd_scale_factor", "exposure", "rate"]).to_csv(OUT / "taiwan_2025_rates.csv", index=False)
    valid = pd.DataFrame(valid_rows, columns=["sex", "year", "measure", "hmd_based", "rebuilt", "difference"])
    valid.to_csv(OUT / "taiwan_2025_bridge_validation.csv", index=False)
    pd.DataFrame(bridge_rows, columns=["sex", "year", "deaths_moi_30plus", "deaths_hmd_30plus", "max_abs_death_difference_by_age",
                                       "exposure_ratio_30_79", "exposure_ratio_80_89", "exposure_ratio_90_99", "exposure_ratio_100plus"]).to_csv(OUT / "taiwan_2025_source_comparison.csv", index=False)
    res = pd.DataFrame(result_rows, columns=["sex", "spec", "year", "measure", "observed", "expected", "deficit", "deficit_low", "deficit_high", "p_value",
                                             "deficit_low_mcse", "deficit_high_mcse", "p_value_mcse"])
    res.to_csv(OUT / "taiwan_2025_deficits.csv", index=False)
    pd.DataFrame(band_rows, columns=["sex", "year", "measure", "age_band", "contribution", "low", "high", "total_deficit",
                                     "sum_of_contributions", "low_mcse", "high_mcse"]).to_csv(OUT / "taiwan_2025_age_bands.csv", index=False)
    pd.DataFrame(group_rows, columns=["sex", "year", "measure", "age_group", "contribution", "total_deficit",
                                      "sum_of_contributions"]).to_csv(OUT / "taiwan_2025_age_contributions.csv", index=False)
    meta = {"script": "research/taiwan_2025.py", "seed": SEED, "calibration_years": CAL_YEARS,
            "poisson_bootstrap_fits": N_BOOT, "paths_per_fit": N_PATHS, "decomposition_steps": DECOMP_STEPS,
            "age_band_interval": "one simulated life table per refit (Poisson variation of the 2025 deaths included), decomposed against the observed rates",
            "not_included_in_intervals": "uncertainty of the exposure scale factor used to rebuild 2025",
            "monte_carlo_error": "columns *_mcse: standard error due to the finite number of simulations.  Interval end points: 100 resamples "
                                 "of whole refits (separate random stream).  p-values: analytic clustered approximation, not resampled "
                                 "(trendlib.two_sided_p_mcse)", "mcse_seed": MCSE_SEED,
            "inputs_sha256": {str(DEATHS_XLS): sha256(DEATHS_XLS), str(POP_XLS): sha256(POP_XLS)},
            "code_sha256": tl.code_sha256("research/taiwan_2025.py", "research/trendlib.py"),
            "decomposition_precision": "contributions add up to the gap up to the finite-step residual (total_deficit against sum_of_contributions); "
                                       "negligible for life expectancy, up to a few thousandths of a year for the death-age quantiles",
            "max_abs_bridge_error": valid.groupby("measure")["difference"].agg(lambda x: float(np.abs(x).max())).round(4).to_dict()}
    (OUT / "taiwan_2025_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    pd.set_option("display.width", 220)
    print(pd.DataFrame(bridge_rows, columns=["sex", "year", "D_moi", "D_hmd", "max|dD|", "E 30-79", "E 80-89", "E 90-99", "E 100+"]).round(3).to_string(index=False))
    print("\nbridge validation, max |rebuilt - HMD-based| by measure:", meta["max_abs_bridge_error"])
    print(valid[valid.measure.isin(["e", "q25", "q50", "q75"])].pivot_table(index=["sex", "year"], columns="measure", values="difference").round(3).to_string())
    print(res[res.measure.isin(["e", "q25", "q50", "q75"])].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
