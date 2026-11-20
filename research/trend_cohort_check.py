#!/usr/bin/env python3
"""Does the mid-life gap survive a baseline that allows for cohort effects?

The main baseline (Lee-Carter) has age and period terms only.  Taiwanese men
born about 1955-1969 show weak mortality improvement along the cohort
diagonal, so part of the post-2019 "gap to trend" at ages 50-69 could be a
cohort effect that an age-period model cannot anticipate.

This check fits a Poisson age-period-cohort model

    log m(x, t) = alpha_x + kappa_t + gamma_(t - x)

to Taiwan, ages 30-100+, 1990-2019, projects kappa with a random walk with
drift and applies the same 2017-2019 jump-off correction as the main
specification.  Cohort effects keep their estimates, except for the youngest
cohorts observed in fewer than five age-year cells and the cohorts not yet
observed: their effects are extrapolated linearly from the 15 cohorts before
them (an estimate from one to four cells is too noisy to carry forward).  With
the 1990-2019 fit the line is therefore fitted to the cohorts born 1971-1985
and applied from the 1986 cohort on.  The alternative rule keeps the estimate
of every observed cohort; it changes two things at once, the cohorts that are
extrapolated (only those born after 1989) and the cohorts the line is fitted
to (1975-1989).  It is run as a sensitivity check.  Fitted values and these
forecasts do not depend on how the linear trend is split between period and
cohort.

It reports (i) pre-pandemic forecast errors of the APC and Lee-Carter
baselines, five years ahead (origins 2005-2014, ten cases) and six years
ahead (origins 2005-2013, nine cases; 2025 is six years after the last
fitting year), and (ii) the 2020-2025 gaps, their age-band contributions and
the ratio of observed to expected death rates by age band under four
baselines: the main specification (Lee-Carter, 2003-2019), Lee-Carter fitted
to 2010-2019, Lee-Carter fitted to the same 30 years as the APC model
(1990-2019) and the APC model.

Run from the repository root:  python3 research/trend_cohort_check.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trendlib as tl  # noqa: E402

OUT = Path("results_trend")
AGE0, OPEN_AGE = 30, 100
AGES = np.arange(AGE0, OPEN_AGE + 1)
FIRST_DATA_YEAR = 1975                    # single-age deaths start in 1975
APC_WINDOW = 30
BANDS = [(30, 49), (50, 64), (65, 79), (80, 200)]
BAND_LABELS = ["30-49", "50-64", "65-79", "80+"]
MEASURES = ("e", "q25", "q50", "q75")
BACKTEST_HORIZONS = (5, 6)


def fit_apc(D: np.ndarray, E: np.ndarray, years: np.ndarray, n_iter: int = 200, tol: float = 1e-10):
    """Poisson APC by iteratively reweighted least squares (minimum-norm solution)."""
    A, T = D.shape
    age_i = np.repeat(np.arange(A), T)
    per_i = np.tile(np.arange(T), A)
    cohort = (years[per_i] - AGES[age_i])
    cohorts = np.arange(cohort.min(), cohort.max() + 1)
    coh_i = cohort - cohorts[0]
    n = A * T
    X = np.zeros((n, A + T + len(cohorts)))
    X[np.arange(n), age_i] = 1.0
    X[np.arange(n), A + per_i] = 1.0
    X[np.arange(n), A + T + coh_i] = 1.0
    y, off = D.reshape(n), np.log(E.reshape(n))
    eta = np.log((y + 0.5) / np.exp(off))
    beta = np.zeros(X.shape[1])
    dev_old = np.inf
    for _ in range(n_iter):
        mu = np.exp(eta + off)
        z = eta + (y - mu) / mu
        w = np.sqrt(mu)
        beta, *_ = np.linalg.lstsq(X * w[:, None], z * w, rcond=None)
        eta = X @ beta
        mu = np.exp(eta + off)
        dev = 2.0 * np.sum(np.where(y > 0, y * np.log(y / mu), 0.0) - (y - mu))
        if abs(dev_old - dev) <= tol * (abs(dev) + 1.0):
            break
        dev_old = dev
    return {"alpha": beta[:A], "kappa": beta[A:A + T], "gamma": beta[A + T:], "cohorts": cohorts, "deviance": dev,
            "fitted": np.exp(eta).reshape(A, T)}


def kept_cohorts(years: np.ndarray, cohorts: np.ndarray, min_cells: int) -> tuple[np.ndarray, np.ndarray]:
    """Indices of the cohorts whose estimates are kept (observed in at least `min_cells` cells) and of
    the last 15 of them, to which the line for the remaining cohorts is fitted."""
    n_obs = np.array([np.sum((years[None, :] - AGES[:, None]) == c) for c in cohorts])
    ok = np.flatnonzero(n_obs >= min_cells)
    return ok, ok[-15:]


def apc_forecast(D: np.ndarray, E: np.ndarray, years: np.ndarray, horizons: np.ndarray, last: int = 3, min_cells: int = 5) -> np.ndarray:
    fit = fit_apc(D, E, years)
    kappa, gamma, cohorts = fit["kappa"], fit["gamma"], fit["cohorts"]
    drift = (kappa[-1] - kappa[0]) / (len(kappa) - 1)
    # cohorts observed in fewer than `min_cells` cells, and cohorts not yet observed, take the linear
    # trend of the 15 cohorts before them
    ok, tail = kept_cohorts(years, cohorts, min_cells)
    slope, intercept = np.polyfit(cohorts[tail], gamma[tail], 1)
    Dh = E[:, -last:] * fit["fitted"][:, -last:]
    corr = np.log((D[:, -last:].sum(axis=1) + 0.5) / (Dh.sum(axis=1) + 0.5))
    out = np.empty((len(AGES), len(horizons)))
    for j, h in enumerate(horizons):
        year = years[-1] + h
        c = year - AGES
        g = np.where(c <= cohorts[ok[-1]], gamma[np.clip(c - cohorts[0], 0, len(gamma) - 1)], intercept + slope * c)
        out[:, j] = np.exp(fit["alpha"] + corr + kappa[-1] + drift * h + g)
    return out


def main() -> None:
    deaths = tl.read_hmd_table("TWN", "Deaths")
    expos = tl.read_hmd_table("TWN", "Exposures")
    all_years = np.arange(FIRST_DATA_YEAR, 2025)
    band_idx = np.zeros(len(AGES), dtype=int)
    for i, (lo, hi) in enumerate(BANDS):
        band_idx[(AGES >= lo) & (AGES <= hi)] = i
    back_rows, gap_rows, band_rows, ratio_rows = [], [], [], []
    rule_diffs, rule_band_diffs = [], []
    r25 = pd.read_csv(OUT / "taiwan_2025_rates.csv")

    for sex in tl.SEXES:
        D = tl.age_year_matrix(deaths, sex, range(FIRST_DATA_YEAR, 2025), AGE0, OPEN_AGE)
        E = tl.age_year_matrix(expos, sex, range(FIRST_DATA_YEAR, 2025), AGE0, OPEN_AGE)
        m_obs = D / E
        col = lambda y: y - FIRST_DATA_YEAR

        # (i) pre-pandemic forecast errors, origins 2005-2014: five years ahead (ten cases) and six years
        #     ahead (nine cases, targets up to 2019; six years is the horizon of 2025)
        for origin in range(2005, 2015):
            hz = np.array([h for h in BACKTEST_HORIZONS if origin + h <= 2019])
            sl30 = slice(col(origin - APC_WINDOW + 1), col(origin) + 1)
            apc = apc_forecast(D[:, sl30], E[:, sl30], all_years[sl30], hz)
            sl17 = slice(col(origin - 16), col(origin) + 1)
            lc17 = tl.forecast(D[:, sl17], E[:, sl17], hz, jump="corrected")["central"]
            lc30 = tl.forecast(D[:, sl30], E[:, sl30], hz, jump="corrected")["central"]
            sl10 = slice(col(origin - 9), col(origin) + 1)
            lc10 = tl.forecast(D[:, sl10], E[:, sl10], hz, jump="corrected")["central"]
            for j, h in enumerate(hz):
                target = origin + int(h)
                obs = {n: float(tl.measure(m_obs[:, col(target)], n, AGE0)) for n in MEASURES}
                for model, fc in (("APC, 30 years", apc), ("Lee-Carter, 17 years", lc17), ("Lee-Carter, 30 years", lc30), ("Lee-Carter, 10 years", lc10)):
                    rates = fc[:, j]
                    for n in MEASURES:
                        back_rows.append((sex, model, origin, int(h), n, obs[n] - float(tl.measure(rates, n, AGE0))))
                    # error in the death rate of each age band (log ratio, 2019 exposure weights)
                    for label, (lo, hi) in zip(BAND_LABELS, BANDS):
                        idx = (AGES >= lo) & (AGES <= hi)
                        w = E[idx, col(2019)] / E[idx, col(2019)].sum()
                        back_rows.append((sex, model, origin, int(h), f"log rate {label}",
                                          float(np.log((m_obs[idx, col(target)] * w).sum() / (rates[idx] * w).sum()))))

        # (ii) gaps 2022-2025 under the APC baseline
        sl = slice(col(2019 - APC_WINDOW + 1), col(2019) + 1)
        horizons = np.arange(1, 7)
        apc = apc_forecast(D[:, sl], E[:, sl], all_years[sl], horizons)
        apc_alt = apc_forecast(D[:, sl], E[:, sl], all_years[sl], horizons, min_cells=1)     # sensitivity: keep every observed cohort
        lc = tl.forecast(D[:, col(2003):col(2019) + 1], E[:, col(2003):col(2019) + 1], horizons, jump="corrected")["central"]
        lc30 = tl.forecast(D[:, sl], E[:, sl], horizons, jump="corrected")["central"]
        lc10 = tl.forecast(D[:, col(2010):col(2019) + 1], E[:, col(2010):col(2019) + 1], horizons, jump="corrected")["central"]
        obs_rates = {y: m_obs[:, col(y)] for y in range(2020, 2025)}
        obs_rates[2025] = r25[r25.sex == sex].sort_values("age")["rate"].to_numpy()
        w19 = E[:, col(2019)]
        baselines = (("APC", apc), ("Lee-Carter", lc), ("Lee-Carter, 1990-2019", lc30), ("Lee-Carter, 2010-2019", lc10))
        for j, year in enumerate(range(2020, 2026)):
            for n in MEASURES:
                o = float(tl.measure(obs_rates[year], n, AGE0))
                gap_rows.append((sex, year, n, o, o - float(tl.measure(apc[:, j], n, AGE0)), o - float(tl.measure(lc[:, j], n, AGE0)),
                                 o - float(tl.measure(lc30[:, j], n, AGE0)), o - float(tl.measure(lc10[:, j], n, AGE0))))
            for model, rates in baselines:
                for g, (lo, hi) in enumerate(BANDS):
                    idx = (AGES >= lo) & (AGES <= hi)
                    w = w19[idx] / w19[idx].sum()
                    ratio_rows.append((sex, year, model, BAND_LABELS[g], float((obs_rates[year][idx] * w).sum() / (rates[idx, j] * w).sum())))
            for n in MEASURES:                                   # how much the cohort-extrapolation rule matters
                rule_diffs.append(abs(float(tl.measure(apc[:, j], n, AGE0)) - float(tl.measure(apc_alt[:, j], n, AGE0))))
            if year in (2022, 2024, 2025):
                ca = tl.horiuchi(lambda x: tl.measure(x, "e", AGE0), apc[:, j], obs_rates[year], 40)
                cb = tl.horiuchi(lambda x: tl.measure(x, "e", AGE0), apc_alt[:, j], obs_rates[year], 40)
                rule_band_diffs.append(float(np.abs(np.bincount(band_idx, weights=ca - cb, minlength=len(BANDS))).max()))
            if year in (2022, 2024, 2025):
                for model, rates in baselines:
                    c = tl.horiuchi(lambda x: tl.measure(x, "e", AGE0), rates[:, j], obs_rates[year], 40)
                    b = np.bincount(band_idx, weights=c, minlength=len(BANDS))
                    for g, label in enumerate(BAND_LABELS):
                        band_rows.append((sex, year, model, label, b[g]))

    back = pd.DataFrame(back_rows, columns=["sex", "model", "origin", "horizon", "measure", "error"])
    origins = {}
    for h in BACKTEST_HORIZONS:                       # every group must hold each origin exactly once
        expected = list(range(2005, 2019 - h + 1))
        for key, g in back[back.horizon == h].groupby(["sex", "model", "measure"]):
            if sorted(g.origin) != expected:
                raise ValueError(f"backtest {h} years ahead, {key}: origins {sorted(g.origin)} instead of {expected}")
        origins[str(h)] = [expected[0], expected[-1]]
    gaps = pd.DataFrame(gap_rows, columns=["sex", "year", "measure", "observed", "gap_apc", "gap_lee_carter", "gap_lee_carter_1990_2019",
                                           "gap_lee_carter_2010_2019"])
    ratios = pd.DataFrame(ratio_rows, columns=["sex", "year", "baseline", "age_band", "observed_over_expected_rate"])
    ratios.to_csv(OUT / "cohort_check_rate_ratios.csv", index=False)
    bands = pd.DataFrame(band_rows, columns=["sex", "year", "baseline", "age_band", "contribution_to_e30_gap"])
    back.to_csv(OUT / "cohort_check_backtest.csv", index=False)
    gaps.to_csv(OUT / "cohort_check_gaps.csv", index=False)
    bands.to_csv(OUT / "cohort_check_age_bands.csv", index=False)
    summary = back.groupby(["horizon", "sex", "measure", "model"])["error"].agg(
        n="size", bias="mean", rmse=lambda x: float(np.sqrt(np.mean(x ** 2))), lowest="min", highest="max").round(3)
    fit_years = np.arange(2019 - APC_WINDOW + 1, 2020)                    # the fitting window of the 2020-2025 baseline
    fit_cohorts = np.arange(fit_years[0] - AGES[-1], fit_years[-1] - AGES[0] + 1)
    windows = {}
    for rule, cells in (("main", 5), ("alternative", 1)):
        ok, tail = kept_cohorts(fit_years, fit_cohorts, cells)
        windows[rule] = {"last_cohort_kept": int(fit_cohorts[ok[-1]]), "line_fitted_to_cohorts": [int(fit_cohorts[tail[0]]), int(fit_cohorts[tail[-1]])]}
    (OUT / "cohort_check_metadata.json").write_text(json.dumps({
        "script": "research/trend_cohort_check.py", "apc_window_years": APC_WINDOW, "population": "TWN", "ages": "30-100+",
        "cohort_rule": "cohorts with fewer than 5 observed cells and unobserved cohorts: linear extrapolation from the 15 cohorts before them",
        "alternative_rule": "every observed cohort keeps its estimate; only unobserved cohorts are extrapolated, from the last 15 observed cohorts "
                            "(the rule changes both the cohorts extrapolated and the cohorts the line is fitted to)",
        "cohort_windows_2019_fit": windows,
        "max_abs_difference_between_rules_in_gap_years": max(rule_diffs),
        "max_abs_difference_between_rules_in_band_contribution_years": max(rule_band_diffs),
        "backtest": {"origins_first_last_by_horizon": origins, "horizons": list(BACKTEST_HORIZONS),
                     "cases": {str(h): int(back[(back.horizon == h) & (back.sex == "Male") & (back.measure == "e") & (back.model == "APC, 30 years")].shape[0])
                               for h in BACKTEST_HORIZONS},
                     "error": "observed minus forecast for life-table measures; log(observed rate / forecast rate) for age bands"},
        "age_band_backtest_weights": "2019 exposure (known only after the forecast origin; used for scoring, not for fitting)",
        "code_sha256": tl.code_sha256("research/trend_cohort_check.py", "research/trendlib.py")}, indent=2))
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 200)
    print("== forecast errors five and six years ahead, origins 2005-2014 (observed minus forecast; log ratio for rates) ==")
    print(summary.to_string())
    print("\n== gaps (observed minus expected, years) ==")
    print(gaps[gaps.measure.isin(["e", "q25", "q75"]) & (gaps.year >= 2022)].round(2).to_string(index=False))
    print("\n== age-band contributions to the e30 gap ==")
    print(bands.pivot_table(index=["sex", "year", "baseline"], columns="age_band", values="contribution_to_e30_gap")[BAND_LABELS].round(2).to_string())
    print("\n== observed / expected death rate by age band ==")
    print(ratios[ratios.year.isin([2022, 2024, 2025])].pivot_table(index=["sex", "year", "baseline"], columns="age_band",
                                                                   values="observed_over_expected_rate")[BAND_LABELS].round(3).to_string())


if __name__ == "__main__":
    main()
