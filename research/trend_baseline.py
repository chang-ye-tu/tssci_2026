#!/usr/bin/env python3
"""Observed adult mortality 2020-2024 against the pre-pandemic trend.

For each of the 22 populations and each sex a Poisson Lee-Carter model is
fitted to ages 30-100+ in the pre-pandemic years and projected forward with a
random walk with drift.  The projected ("expected") death rates give the life
table that would have been observed had the earlier trend continued.  The gap
between the observed and the expected value of a life-table measure is called
the deficit (observed minus expected; negative = worse than trend).

The observed measures, computed here from deaths and exposures, are also set
beside the ones implied by the period life tables that HMD publishes
(results_trend/observed_measures.csv).  That comparison checks the life-table
construction; it enters no other result.

Specifications
    main          fit 2003-2019, forecast anchored to the 2017-2019 observed level
    window2010    fit 2010-2019, same anchoring
    fitted_start  fit 2003-2019, forecast starts from the fitted 2019 level
The choice of the main specification rests on research/trend_backtest.py.

Run from the repository root:  python3 research/trend_baseline.py
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trendlib as tl  # noqa: E402

OUT = Path("results_trend")
AGE0, OPEN_AGE = 30, 100
AGES = np.arange(AGE0, OPEN_AGE + 1)
SPECS = {
    "main": {"window": (2003, 2019), "jump": "corrected"},
    "window2010": {"window": (2010, 2019), "jump": "corrected"},
    "fitted_start": {"window": (2003, 2019), "jump": "fit"},
}
LAST_FIT_YEAR = 2019
OBS_YEARS = np.arange(2020, 2025)          # observed in the HMD release
ALL_TARGETS = np.arange(2020, 2026)        # expected rates are also stored for 2025
N_BOOT, N_PATHS = 2000, 4
SEED = 20261118
MCSE_SEED = 20261198            # separate stream for the Monte Carlo error of interval end points
MEASURE_NAMES = ("e", "q10", "q25", "q50", "q75", "q90", "iqr")
DECOMP_MEASURES = ("e", "q25", "q50", "q75")
DECOMP_STEPS = 40
SPEC_WINDOW = {"main": "17y", "fitted_start": "17y", "window2010": "10y"}     # backtest window that matches each specification
FOCUS = ("TWN", "JPN", "KOR", "HKG")
BANDS = [(30, 49), (50, 64), (65, 79), (80, 200)]
BAND_LABELS = ["30-49", "50-64", "65-79", "80+"]
LIFE_TABLES = {"Female": "fltper", "Male": "mltper"}     # HMD's period life tables, read only for the comparison in observed_measures.csv


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def five_year_groups() -> tuple[np.ndarray, list[str]]:
    """Index of the five-year group of each single age (100+ is its own group)."""
    idx = np.minimum((AGES - AGE0) // 5, (OPEN_AGE - AGE0) // 5)
    labels = [f"{a}-{a + 4}" for a in range(AGE0, OPEN_AGE, 5)] + [f"{OPEN_AGE}+"]
    return idx, labels


def band_index() -> np.ndarray:
    idx = np.zeros(len(AGES), dtype=int)
    for i, (lo, hi) in enumerate(BANDS):
        idx[(AGES >= lo) & (AGES <= hi)] = i
    return idx


def hmd_life_table_measures(table: pd.DataFrame, year: int) -> dict[str, float]:
    """The measures of one year as they follow from the period life table that HMD publishes.

    Life expectancy at age 30 is T(30) / l(30).  The quantiles of the age at death among those alive
    at 30 are read from l(x), with linear interpolation within a year of age.  HMD prints l(x) and
    T(x) as whole numbers per 100,000 births and builds its tables with its own assumptions at the
    oldest ages, so these values differ slightly from the ones computed here from deaths and exposures.
    """
    g = table[table["Year"] == year].sort_values("Age")
    ages, lx, Tx = g["Age"].to_numpy(), g["lx"].to_numpy(), g["Tx"].to_numpy()
    if not np.array_equal(ages, np.arange(111)):
        raise ValueError(f"published life table of {year}: ages 0 to 110 are not all present")
    if not (np.all(np.isfinite(lx)) and np.isfinite(Tx[AGE0]) and lx[AGE0] > 0 and np.all(np.diff(lx) <= 0)):
        raise ValueError(f"published life table of {year}: l(x) or T(x) is missing, or l(x) increases with age")
    alive = lx[AGE0]
    out = {"e": float(Tx[AGE0] / alive)}
    for p in tl.QUANTILES:
        target = alive * (1 - p)
        reached = np.flatnonzero((ages >= AGE0) & (lx <= target))          # exact ages by which the share p has died
        if not len(reached):
            raise ValueError(f"published life table of {year}: the {p:.0%} quantile lies beyond age 110")
        k = reached[0]
        out[f"q{int(round(p * 100))}"] = float(ages[k]) if lx[k] == target else float(ages[k - 1] + (lx[k - 1] - target) / (lx[k - 1] - lx[k]))
    out["iqr"] = out["q75"] - out["q25"]
    return out


def placebo_reference() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pre-pandemic forecast errors from the backtest, used as a yardstick.
    Each specification is matched to the backtest with its own window length."""
    bt = pd.read_csv(OUT / "backtest_point.csv")
    bt = bt[bt["window"].isin(["17y", "10y"]) & (bt["origin"].between(2005, 2014))]
    by_pop = (
        bt.groupby(["population", "sex", "window", "jump", "measure", "horizon"])["error"]
        .agg(placebo_n="size", placebo_mean="mean", placebo_min="min", placebo_max="max",
             placebo_rmse=lambda x: float(np.sqrt(np.mean(np.square(x)))))
        .reset_index()
    )
    # the Baltic mortality crisis and recovery make their forecast errors
    # unrepresentative; they are left out of the pooled yardstick only
    stable = bt[~bt["population"].isin(["EST", "LTU", "LVA"])]
    pooled = (
        stable.groupby(["window", "jump", "measure", "horizon"])["error"]
        .agg(pooled_rmse=lambda x: float(np.sqrt(np.mean(np.square(x)))), pooled_mean="mean")
        .reset_index()
    )
    return by_pop, pooled


def main() -> None:
    started = time.time()
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    group_idx, group_labels = five_year_groups()
    bands = band_index()

    obs_rows, exp_rows, rate_rows, par_rows, fit_rows, dec_rows, band_rows, kappa_rows = [], [], [], [], [], [], [], []
    inputs = {}
    checks = {"max_abs_decomposition_residual": 0.0, "open_group_fallback_cells": 0, "simulated_open_group_fallbacks": 0,
              "simulated_open_groups": 0}
    all_years = np.arange(2003, 2025)

    for pop in tl.TREND_POPULATIONS:
        deaths = tl.read_hmd_table(pop, "Deaths")
        expos = tl.read_hmd_table(pop, "Exposures")
        for kind in ("Deaths", "Exposures", *LIFE_TABLES.values()):
            inputs[str(tl.hmd_path(pop, kind))] = sha256(tl.hmd_path(pop, kind))
        for sex in tl.SEXES:
            D = tl.age_year_matrix(deaths, sex, range(2003, 2025), AGE0, OPEN_AGE)
            E = tl.age_year_matrix(expos, sex, range(2003, 2025), AGE0, OPEN_AGE)
            pooled = (D.sum(axis=1) / E.sum(axis=1))[:, None]
            m_obs, replaced = tl.observed_rates(D, E, fallback=pooled, age_axis=0)
            checks["open_group_fallback_cells"] += replaced
            obs = {name: tl.measure(m_obs.T, name, AGE0) for name in MEASURE_NAMES}
            life_table = tl.read_hmd_table(pop, LIFE_TABLES[sex])
            for j, year in enumerate(all_years):
                published = hmd_life_table_measures(life_table, int(year))
                for name in MEASURE_NAMES:
                    obs_rows.append((pop, sex, int(year), name, obs[name][j], published[name]))

            for spec, cfg in SPECS.items():
                y0, y1 = cfg["window"]
                sl = slice(y0 - 2003, y1 - 2003 + 1)
                horizons = ALL_TARGETS - LAST_FIT_YEAR
                fc = tl.forecast(D[:, sl], E[:, sl], horizons, jump=cfg["jump"], n_boot=N_BOOT, n_paths=N_PATHS, rng=rng)
                fit = fc["fit"]
                drift, sigma, _ = tl.drift_and_sigma(fit["k"])
                fit_rows.append((pop, sex, spec, y0, y1, cfg["jump"], float(fit["deviance"]), int(fit["n_iter"]), bool(fit["converged"]),
                                 float(drift), float(sigma), fc["boot_converged"], float(np.abs(fc["correction"]).max()), fc["capped_share"], fc["failed_share"]))
                for i, age in enumerate(AGES):
                    par_rows.append((pop, sex, spec, int(age), fit["a"][i], fit["b"][i], fc["correction"][i]))
                for t, year in enumerate(range(y0, y1 + 1)):
                    kappa_rows.append((pop, sex, spec, int(year), fit["k"][t]))
                central = fc["central"]                                   # (A, H)
                for h, year in enumerate(ALL_TARGETS):
                    for i, age in enumerate(AGES):
                        rate_rows.append((pop, sex, spec, int(year), int(age), central[i, h]))
                central_measures = {name: tl.measure(central.T, name, AGE0) for name in MEASURE_NAMES}

                n_obs = len(OBS_YEARS)
                cols = OBS_YEARS - 2003
                sims, n_fallback = tl.predictive_rates(fc["draws"][:, :n_obs, :], E[:, cols], rng, return_fallback=True)   # (S, H, A)
                checks["simulated_open_group_fallbacks"] += n_fallback
                checks["simulated_open_groups"] += sims.shape[0] * sims.shape[1]
                for name in MEASURE_NAMES:
                    vals = tl.measure(sims, name, AGE0)                                     # (S, H)
                    lo, med, hi = np.quantile(vals, [0.025, 0.5, 0.975], axis=0)
                    se = tl.quantile_mcse(vals, [0.025, 0.975], cluster=N_PATHS, seed=MCSE_SEED)   # (2, H)
                    for h, year in enumerate(OBS_YEARS):
                        o = obs[name][cols[h]]
                        tail = min(np.mean(vals[:, h] <= o), np.mean(vals[:, h] >= o))
                        p = max(2.0 * tail, 1.0 / vals.shape[0])
                        exp_rows.append((pop, sex, spec, int(year), int(year - LAST_FIT_YEAR), name, o, central_measures[name][h],
                                         med[h], lo[h], hi[h], o - central_measures[name][h], o - hi[h], o - lo[h], min(p, 1.0),
                                         se[1, h], se[0, h], tl.two_sided_p_mcse(vals[:, h], o, N_PATHS)))
                for h, year in enumerate(ALL_TARGETS[n_obs:], start=n_obs):               # 2025: expectation only
                    for name in MEASURE_NAMES:
                        exp_rows.append((pop, sex, spec, int(year), int(year - LAST_FIT_YEAR), name, np.nan, central_measures[name][h],
                                         np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))

                # age decomposition of the deficit (expected -> observed)
                if spec != "main" and pop not in FOCUS:
                    continue
                years_to_do = OBS_YEARS if pop in FOCUS else np.array([2022, 2024])
                if spec != "main":
                    years_to_do = np.array([2024])
                for year in years_to_do:
                    h = int(year - 2020)
                    m_from, m_to = central[:, h], m_obs[:, year - 2003]
                    for name in DECOMP_MEASURES:
                        func = lambda x, n=name: tl.measure(x, n, AGE0)
                        c = tl.horiuchi(func, m_from, m_to, DECOMP_STEPS)
                        total = float(tl.measure(m_to, name, AGE0) - tl.measure(m_from, name, AGE0))
                        checks["max_abs_decomposition_residual"] = max(checks["max_abs_decomposition_residual"], abs(c.sum() - total))
                        grouped = np.bincount(group_idx, weights=c, minlength=len(group_labels))
                        for g, label in enumerate(group_labels):
                            dec_rows.append((pop, sex, spec, int(year), name, label, grouped[g], total, float(c.sum())))
                        banded = np.bincount(bands, weights=c, minlength=len(BANDS))
                        lo_b = hi_b = se_lo = se_hi = np.full(len(BANDS), np.nan)
                        if spec == "main" and pop in FOCUS and year in (2022, 2024) and name == "e":
                            # Same simulation as the interval of the whole gap: one simulated life table per
                            # refit (N_BOOT independent draws, Poisson variation of the target-year deaths
                            # included).  Each is decomposed against the observed rates with the same number
                            # of steps as the central decomposition, so that the band contributions of a
                            # draw add up to "observed minus that draw".
                            cs = tl.horiuchi_batch(func, sims[::N_PATHS, h, :], m_to, DECOMP_STEPS)
                            sims_b = np.stack([cs[:, bands == g].sum(axis=1) for g in range(len(BANDS))], axis=1)
                            lo_b, hi_b = np.quantile(sims_b, [0.025, 0.975], axis=0)
                            se_lo, se_hi = tl.quantile_mcse(sims_b, [0.025, 0.975], cluster=1, seed=MCSE_SEED)
                        for g, label in enumerate(BAND_LABELS):
                            band_rows.append((pop, sex, spec, int(year), name, label, banded[g], lo_b[g], hi_b[g], total, float(c.sum()),
                                              se_lo[g], se_hi[g]))
        print(f"{pop}: done ({time.time() - started:.0f}s)", flush=True)

    observed = pd.DataFrame(obs_rows, columns=["population", "sex", "year", "measure", "own_value", "hmd_official"])
    observed["difference"] = observed["own_value"] - observed["hmd_official"]
    observed.to_csv(OUT / "observed_measures.csv", index=False)

    expected = pd.DataFrame(exp_rows, columns=["population", "sex", "spec", "year", "horizon", "measure", "observed", "expected",
                                               "predictive_median", "predictive_low", "predictive_high", "deficit", "deficit_low",
                                               "deficit_high", "p_value", "deficit_low_mcse", "deficit_high_mcse", "p_value_mcse"])
    by_pop, pooled = placebo_reference()
    expected["jump"] = expected["spec"].map({k: v["jump"] for k, v in SPECS.items()})
    expected["window"] = expected["spec"].map(SPEC_WINDOW)
    expected = expected.merge(by_pop, on=["population", "sex", "window", "jump", "measure", "horizon"], how="left")
    expected = expected.merge(pooled, on=["window", "jump", "measure", "horizon"], how="left")
    expected.to_csv(OUT / "expected_observed.csv", index=False)

    # mtime=0: without it the gzip header carries the time of writing and the file differs between identical runs
    pd.DataFrame(rate_rows, columns=["population", "sex", "spec", "year", "age", "expected_rate"]).to_csv(
        OUT / "expected_rates.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    pd.DataFrame(par_rows, columns=["population", "sex", "spec", "age", "a", "b", "jumpoff_correction"]).to_csv(OUT / "lc_age_parameters.csv", index=False)
    pd.DataFrame(kappa_rows, columns=["population", "sex", "spec", "year", "k"]).to_csv(OUT / "lc_period_index.csv", index=False)
    pd.DataFrame(fit_rows, columns=["population", "sex", "spec", "fit_start", "fit_end", "jump", "deviance", "iterations", "converged",
                                    "drift", "sigma", "bootstrap_converged_share", "max_abs_correction", "capped_share", "failed_share"]).to_csv(OUT / "lc_fit_summary.csv", index=False)
    pd.DataFrame(dec_rows, columns=["population", "sex", "spec", "year", "measure", "age_group", "contribution", "total_deficit",
                                    "sum_of_contributions"]).to_csv(OUT / "age_contributions.csv", index=False)
    pd.DataFrame(band_rows, columns=["population", "sex", "spec", "year", "measure", "age_band", "contribution", "low", "high", "total_deficit",
                                     "sum_of_contributions", "low_mcse", "high_mcse"]).to_csv(OUT / "age_band_contributions.csv", index=False)

    diff = observed.groupby("measure")["difference"].agg(lambda x: float(np.abs(x).max()))
    meta = {
        "script": "research/trend_baseline.py",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hmd_release": "20261111",
        "populations": list(tl.TREND_POPULATIONS),
        "excluded_small_populations": list(tl.SMALL_POPULATIONS),
        "age_range": f"{AGE0}-{OPEN_AGE}+ (single years, open group {OPEN_AGE}+)",
        "life_table": "piecewise-constant force of mortality; conditional on survival to age 30",
        "specifications": {k: {"window": list(v["window"]), "jump": v["jump"]} for k, v in SPECS.items()},
        "simulation": {"poisson_bootstrap_fits": N_BOOT, "paths_per_fit": N_PATHS, "seed": SEED,
                       "interval": "95% predictive: refit uncertainty, drift uncertainty, random-walk innovations, Poisson variation of target-year deaths",
                       "monte_carlo_error": "columns *_mcse: standard error due to the finite number of simulations.  Interval end points: "
                                            "100 resamples of whole refits (separate random stream, seed MCSE_SEED); the estimate itself has a "
                                            "relative error of roughly 7%.  p-values: analytic clustered approximation, not resampled "
                                            "(trendlib.two_sided_p_mcse)",
                       "mcse_seed": MCSE_SEED,
                       "age_band_interval": "same simulation, one simulated life table per refit, decomposed against the observed rates (measure e, main specification, focus populations, 2022 and 2024)"},
        "decomposition": {"method": "Horiuchi, Wilmoth & Pletcher (2008) line integral", "steps": DECOMP_STEPS,
                          "max_abs_residual_years": checks["max_abs_decomposition_residual"],
                          "precision": "midpoint rule with a finite number of steps: contributions add up to the gap only up to the residual "
                                       "(total_deficit is the gap itself, sum_of_contributions the sum); the residual is negligible for "
                                       "life expectancy and up to a few thousandths of a year for the death-age quantiles, which the paper does not decompose"},
        "open_group_fallback_cells": checks["open_group_fallback_cells"],
        "simulated_open_group_fallbacks": checks["simulated_open_group_fallbacks"],
        "simulated_open_groups": checks["simulated_open_groups"],
        "guards": "ages without any death in the fitting window get 0.25 deaths in total; non-finite resampled fits are replaced by finite ones (share in lc_fit_summary.csv); simulated rates are capped at 2 per person-year",
        "hmd_official": "column of observed_measures.csv: the same measures taken from HMD's period life tables (fltper_1x1, mltper_1x1) "
                        "as T(30)/l(30) and as quantiles of l(x) with linear interpolation within a year of age; a check only",
        "max_abs_difference_from_hmd_official": diff.round(4).to_dict(),
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "seconds": round(time.time() - started, 1),
        "inputs_sha256": inputs,
        "code_sha256": {p: sha256(Path(p)) for p in ("research/trend_baseline.py", "research/trendlib.py", "research/trend_backtest.py")},
    }
    (OUT / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in meta.items() if k not in ("inputs_sha256",)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
