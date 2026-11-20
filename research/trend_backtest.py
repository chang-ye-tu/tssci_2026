#!/usr/bin/env python3
"""Pre-pandemic test of the trend baseline.

Question: when adult mortality is forecast five years ahead with a Poisson
Lee-Carter model, which specification would have predicted the years up to
2019 best -- a 17-year or a 10-year fitting window, starting from the fitted
level or from the recent observed level?

For every population, sex and forecast origin T0 (targets T0+1..T0+5 <= 2019)
the model is fitted to the w years ending in T0 and compared with what was
actually observed.  Nothing from 2020 onwards is used.

Run from the repository root:
    python3 research/trend_backtest.py            # point forecasts only
    python3 research/trend_backtest.py --intervals  # adds interval coverage
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trendlib as tl  # noqa: E402

OUT = Path("results_trend")
AGE0, OPEN_AGE = 30, 100
WINDOWS = (17, 10)
JUMPS = ("fit", "corrected")
HORIZONS = np.arange(1, 6)
LAST_TARGET = 2019
FIRST_ORIGIN = 1990
MEASURE_NAMES = ("e", "q25", "q50", "q75")
INTERVAL_ORIGINS = range(2005, 2015)
N_BOOT, N_PATHS = 200, 4
SEED = 20261117


def load_population(pop: str):
    deaths = tl.read_hmd_table(pop, "Deaths")
    expos = tl.read_hmd_table(pop, "Exposures")
    first = int(max(deaths["Year"].min(), expos["Year"].min(), FIRST_ORIGIN - max(WINDOWS) + 1))
    years = range(first, LAST_TARGET + 1)
    data = {}
    for sex in tl.SEXES:
        D = tl.age_year_matrix(deaths, sex, years, AGE0, OPEN_AGE)
        E = tl.age_year_matrix(expos, sex, years, AGE0, OPEN_AGE)
        pooled = (D.sum(axis=1) / E.sum(axis=1))[:, None]
        m_obs, replaced = tl.observed_rates(D, E, fallback=pooled, age_axis=0)
        obs = {name: tl.measure(m_obs.T, name, AGE0) for name in MEASURE_NAMES}
        data[sex] = {"D": D, "E": E, "obs": obs, "replaced": replaced}
    return first, data


def window_slice(first: int, origin: int, width: int) -> slice:
    return slice(origin - width + 1 - first, origin - first + 1)


def run(with_intervals: bool) -> None:
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    point_rows, interval_rows, skipped = [], [], []
    started = time.time()
    for pop in tl.TREND_POPULATIONS:
        first, data = load_population(pop)
        for sex in tl.SEXES:
            D, E, obs = data[sex]["D"], data[sex]["E"], data[sex]["obs"]
            # (label, width, origins): rolling windows plus the 2003-2014 window
            # that every population can supply
            plans = [(f"{w}y", w, range(max(first + w - 1, FIRST_ORIGIN), LAST_TARGET - 4)) for w in WINDOWS]
            plans.append(("2003-2014", 12, range(2014, 2015)))
            for label, width, origins in plans:
                for origin in origins:
                    if origin - width + 1 < first:
                        continue
                    sl = window_slice(first, origin, width)
                    Dw, Ew = D[:, sl], E[:, sl]
                    targets = origin + HORIZONS
                    cols = targets - first
                    for jump in JUMPS:
                        do_int = with_intervals and origin in INTERVAL_ORIGINS and label != "2003-2014"
                        try:
                            fc = tl.forecast(Dw, Ew, HORIZONS, jump=jump, n_boot=N_BOOT if do_int else 0, n_paths=N_PATHS, rng=rng)
                        except ValueError as err:
                            skipped.append({"population": pop, "sex": sex, "window": label, "origin": origin, "reason": str(err)})
                            break
                        central = fc["central"]
                        for name in MEASURE_NAMES:
                            pred = tl.measure(central.T, name, AGE0)
                            for i, h in enumerate(HORIZONS):
                                point_rows.append((pop, sex, label, jump, origin, int(h), int(targets[i]), name, obs[name][cols[i]], pred[i]))
                        if do_int:
                            sims = tl.predictive_rates(fc["draws"], E[:, cols], rng)
                            for name in MEASURE_NAMES:
                                vals = tl.measure(sims, name, AGE0)            # (S, H)
                                lo, hi = np.quantile(vals, [0.025, 0.975], axis=0)
                                for i, h in enumerate(HORIZONS):
                                    o = obs[name][cols[i]]
                                    interval_rows.append((pop, sex, label, jump, origin, int(h), name, o, lo[i], hi[i], bool(lo[i] <= o <= hi[i]), float(vals[:, i].std())))
        print(f"{pop}: done ({time.time() - started:.0f}s)", flush=True)

    point = pd.DataFrame(point_rows, columns=["population", "sex", "window", "jump", "origin", "horizon", "target_year", "measure", "observed", "forecast"])
    point["error"] = point["observed"] - point["forecast"]
    point.to_csv(OUT / "backtest_point.csv", index=False)
    pd.DataFrame(skipped).to_csv(OUT / "backtest_skipped.csv", index=False)
    if with_intervals:
        interval = pd.DataFrame(interval_rows, columns=["population", "sex", "window", "jump", "origin", "horizon", "measure", "observed", "lower", "upper", "covered", "predictive_sd"])
        interval.to_csv(OUT / "backtest_interval.csv", index=False)
    meta = {
        "script": "research/trend_backtest.py",
        "age_range": f"{AGE0}-{OPEN_AGE}+",
        "windows_years": list(WINDOWS),
        "jump_off": list(JUMPS),
        "first_origin": FIRST_ORIGIN,
        "last_target": LAST_TARGET,
        "interval_origins": [INTERVAL_ORIGINS.start, INTERVAL_ORIGINS.stop - 1] if with_intervals else None,
        "n_boot": N_BOOT, "n_paths": N_PATHS, "seed": SEED,
        "rows_point": int(len(point)), "skipped": int(len(skipped)),
        "seconds": round(time.time() - started, 1),
    }
    (OUT / "backtest_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--intervals", action="store_true")
    run(parser.parse_args().intervals)
