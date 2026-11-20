#!/usr/bin/env python3
"""Shared building blocks for the trend-baseline (counterfactual) analysis.

Everything here is a pure function of arrays so that the main analysis
(`trend_baseline.py`), the cause-of-death analysis and the independent checks
can share one definition of each quantity.

Conventions
-----------
* Ages are single years from `age0` to `open_age - 1`, followed by one open
  group `open_age`+.  A rate vector therefore has ``open_age - age0 + 1``
  entries, the last one being the open group.
* Life tables use a piecewise-constant force of mortality within each age
  interval (the same assumption as the "piecewise-exponential" tables in
  Dowd et al. 2026), so survival over one year of age is ``exp(-m)``.
* All "conditional" quantities are conditional on survival to `age0`.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

SNAPSHOT = Path("data/hmd/2026-11-17/raw/core")

POPULATIONS = (
    "AUT BEL CHE CHL DEUTNP DNK EST FIN FRATNP HKG HRV ISL JPN KOR LTU LUX "
    "LVA NLD NOR PRT SVK SWE TWN USA"
).split()

# Iceland and Luxembourg have only about 2,000 adult deaths a year.  A
# single-year-of-age Lee-Carter model is not estimable for them (fits and
# resampled fits diverge), so the trend analysis uses the other 22.
SMALL_POPULATIONS = ("ISL", "LUX")
TREND_POPULATIONS = tuple(p for p in POPULATIONS if p not in SMALL_POPULATIONS)

SEXES = ("Female", "Male")

QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
MAX_RATE = 2.0   # cap for simulated death rates (observed rates at 100+ are about 0.4-0.6)
MEASURES = ("e", "q10", "q25", "q50", "q75", "q90", "iqr")


# --------------------------------------------------------------------------
# Data access
# --------------------------------------------------------------------------
def code_sha256(*paths: str) -> dict[str, str]:
    """SHA-256 of the scripts behind a result file, for its metadata."""
    return {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}


def read_hmd_table(population: str, kind: str, root: Path = SNAPSHOT) -> pd.DataFrame:
    """Read an HMD 1x1 text product (``Deaths`` or ``Exposures``).

    The open age group ``110+`` is returned as integer age 110.
    """
    path = root / population / "STATS" / f"{kind}_1x1.txt"
    df = pd.read_csv(path, sep=r"\s+", skiprows=2, na_values=".")
    df["Age"] = df["Age"].astype(str).str.rstrip("+").astype(int)
    return df


def hmd_path(population: str, kind: str, root: Path = SNAPSHOT) -> Path:
    return root / population / "STATS" / f"{kind}_1x1.txt"


def age_year_matrix(
    table: pd.DataFrame,
    column: str,
    years: range,
    age0: int = 30,
    open_age: int = 100,
) -> np.ndarray:
    """Return an (ages x years) matrix with ages ``age0..open_age-1`` and one
    final row holding the sum over ``open_age`` and above."""
    sub = table[(table["Year"] >= years[0]) & (table["Year"] <= years[-1]) & (table["Age"] >= age0)]
    piv = sub.pivot(index="Age", columns="Year", values=column)
    piv = piv.reindex(columns=list(years))
    if piv.isna().any().any():
        raise ValueError(f"missing cells in {column} for years {years[0]}-{years[-1]}")
    closed = piv.loc[age0 : open_age - 1].to_numpy(dtype=float)
    if closed.shape[0] != open_age - age0:
        raise ValueError("incomplete single-year age range")
    opened = piv.loc[open_age:].to_numpy(dtype=float).sum(axis=0, keepdims=True)
    return np.vstack([closed, opened])


# --------------------------------------------------------------------------
# Life-table functionals
# --------------------------------------------------------------------------
def survival(m: np.ndarray) -> np.ndarray:
    """Survivorship l(x) at exact ages age0..open_age (first entry 1).

    `m` has shape (..., A) with the open group last; the result has the same
    shape (the last entry is survival to the start of the open group).
    """
    m = np.asarray(m, dtype=float)
    cum = np.cumsum(m[..., :-1], axis=-1)
    ones = np.ones(m.shape[:-1] + (1,))
    return np.concatenate([ones, np.exp(-cum)], axis=-1)


def life_expectancy(m: np.ndarray) -> np.ndarray:
    """Remaining life expectancy at age0 under piecewise-constant hazards."""
    m = np.asarray(m, dtype=float)
    l = survival(m)
    closed_m = m[..., :-1]
    drop = l[..., :-1] - l[..., 1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        person_years = np.where(closed_m > 0, drop / closed_m, l[..., :-1])
        open_years = l[..., -1] / m[..., -1]
    return person_years.sum(axis=-1) + open_years


def death_age_quantile(m: np.ndarray, p: float, age0: int = 30) -> np.ndarray:
    """Age by which a share `p` of those alive at age0 have died."""
    m = np.asarray(m, dtype=float)
    l = survival(m)
    target = 1.0 - p
    idx = (l >= target).sum(axis=-1) - 1          # interval that contains the quantile
    l_at = np.take_along_axis(l, idx[..., None], axis=-1)[..., 0]
    m_at = np.take_along_axis(m, idx[..., None], axis=-1)[..., 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        within = np.log(l_at / target) / m_at
    return age0 + idx + within


def measures(m: np.ndarray, age0: int = 30) -> dict[str, np.ndarray]:
    """All summary measures used in the paper for rate vectors `m`."""
    out = {"e": life_expectancy(m)}
    for p in QUANTILES:
        out[f"q{int(round(p * 100))}"] = death_age_quantile(m, p, age0)
    out["iqr"] = out["q75"] - out["q25"]
    return out


def measure(m: np.ndarray, name: str, age0: int = 30) -> np.ndarray:
    if name == "e":
        return life_expectancy(m)
    if name == "iqr":
        return death_age_quantile(m, 0.75, age0) - death_age_quantile(m, 0.25, age0)
    if name.startswith("q"):
        return death_age_quantile(m, int(name[1:]) / 100.0, age0)
    raise KeyError(name)


def observed_rates(D: np.ndarray, E: np.ndarray, fallback: np.ndarray | None = None, age_axis: int = -1):
    """Occurrence/exposure rates with the age dimension on `age_axis` (open
    group last).  A cell without exposure, or an open group without deaths
    (which would make the open-interval life expectancy infinite), takes the
    `fallback` rate.  Returns (rates, number of cells replaced)."""
    D = np.asarray(D, dtype=float)
    E = np.asarray(E, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        m = np.where(E > 0, D / E, np.nan)
    need = ~np.isfinite(m)
    is_open = np.zeros(m.shape[age_axis], dtype=bool)
    is_open[-1] = True
    shape = [1] * m.ndim
    shape[age_axis] = m.shape[age_axis]
    need = need | (is_open.reshape(shape) & (np.nan_to_num(m, nan=0.0) <= 0))
    replaced = int(need.sum())
    if replaced:
        if fallback is None:
            raise ValueError("rate undefined and no fallback supplied")
        m = np.where(need, fallback, m)
    return m, replaced


# --------------------------------------------------------------------------
# Poisson Lee-Carter
# --------------------------------------------------------------------------
def _normalise(a: np.ndarray, b: np.ndarray, k: np.ndarray):
    """Impose sum(b)=1 and sum(k)=0 (works on the last axes of batched arrays)."""
    kbar = k.mean(axis=-1, keepdims=True)
    a = a + b * kbar
    k = k - kbar
    s = b.sum(axis=-1, keepdims=True)
    return a, b / s, k * s


def _deviance(D: np.ndarray, Dhat: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(D > 0, D * np.log(D / Dhat), 0.0)
    return 2.0 * (term - (D - Dhat)).sum(axis=(-2, -1))


def _guard_empty_rows(D: np.ndarray, E: np.ndarray, pseudo: float = 0.25) -> np.ndarray:
    """An age with no deaths at all in the window has no finite maximum-
    likelihood level.  Give such a row a total of `pseudo` deaths, spread in
    proportion to exposure (affects only tiny populations)."""
    empty = D.sum(axis=-1, keepdims=True) <= 0
    if not empty.any():
        return D
    share = E / E.sum(axis=-1, keepdims=True)
    return np.where(empty, pseudo * share, D)


def lc_initial(D: np.ndarray, E: np.ndarray):
    """SVD starting values (classic Lee-Carter on log rates with a continuity
    correction).  Supports a leading batch dimension."""
    # continuity correction: add half a death and the matching exposure at the
    # year's overall rate, so empty cells get a finite starting value
    per_death = E.sum(axis=-2, keepdims=True) / np.maximum(D.sum(axis=-2, keepdims=True), 1.0)
    logm = np.log((D + 0.5) / (E + 0.5 * per_death))
    a = logm.mean(axis=-1)
    resid = logm - a[..., None]
    u, s, vt = np.linalg.svd(resid, full_matrices=False)
    b = u[..., :, 0]
    k = s[..., 0, None] * vt[..., 0, :]
    sign = np.sign(b.sum(axis=-1, keepdims=True))
    sign = np.where(sign == 0, 1.0, sign)
    b, k = b * sign, k * sign
    return _normalise(a, b, k)


def fit_lee_carter(
    D: np.ndarray,
    E: np.ndarray,
    start: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    tol: float = 1e-9,
    max_iter: int = 20000,
):
    """Maximum-likelihood Poisson Lee-Carter fit (Brouhns, Denuit & Vermunt 2002).

    D, E : (..., A, T) deaths and exposures; a leading batch dimension is
    allowed and every batch member is iterated until all have converged.
    Returns a dict with a (..., A), b (..., A), k (..., T), deviance, n_iter,
    converged.
    """
    D = np.asarray(D, dtype=float)
    E = np.asarray(E, dtype=float)
    if np.any(E.sum(axis=-1) <= 0):
        raise ValueError("an age has no exposure in the fitting window")
    D = np.where(E > 0, D, 0.0)
    D = _guard_empty_rows(D, E)
    a, b, k = lc_initial(D, E) if start is None else tuple(np.array(v, dtype=float) for v in start)
    if start is not None:
        a, b, k = np.broadcast_to(a, D.shape[:-1]).copy(), np.broadcast_to(b, D.shape[:-1]).copy(), np.broadcast_to(k, D.shape[:-2] + D.shape[-1:]).copy()

    def fitted(a, b, k):
        return E * np.exp(a[..., :, None] + b[..., :, None] * k[..., None, :])

    dev_old = _deviance(D, fitted(a, b, k))
    converged = np.zeros(np.shape(dev_old), dtype=bool)
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        Dh = fitted(a, b, k)
        a = a + (D - Dh).sum(axis=-1) / Dh.sum(axis=-1)
        Dh = fitted(a, b, k)
        k = k + ((D - Dh) * b[..., :, None]).sum(axis=-2) / (Dh * b[..., :, None] ** 2).sum(axis=-2)
        Dh = fitted(a, b, k)
        b = b + ((D - Dh) * k[..., None, :]).sum(axis=-1) / (Dh * k[..., None, :] ** 2).sum(axis=-1)
        a, b, k = _normalise(a, b, k)
        dev = _deviance(D, fitted(a, b, k))
        converged = np.abs(dev_old - dev) <= tol * (np.abs(dev) + 1.0)
        dev_old = dev
        if np.all(converged):
            break
    return {"a": a, "b": b, "k": k, "deviance": dev_old, "n_iter": n_iter, "converged": converged}


def drift_and_sigma(k: np.ndarray):
    """Random-walk-with-drift estimates for the period index."""
    dk = np.diff(k, axis=-1)
    n = dk.shape[-1]
    drift = dk.mean(axis=-1)
    sigma2 = ((dk - drift[..., None]) ** 2).sum(axis=-1) / (n - 1)
    return drift, np.sqrt(sigma2), n


def central_forecast(fit: dict, horizons: np.ndarray) -> np.ndarray:
    """Central forecast rates (A, H) from the fitted jump-off."""
    drift, _, _ = drift_and_sigma(fit["k"])
    k_future = fit["k"][..., -1, None] + drift[..., None] * horizons
    return np.exp(fit["a"][..., :, None] + fit["b"][..., :, None] * k_future[..., None, :])


def simulate_period_index(k: np.ndarray, horizons: np.ndarray, n_paths: int, rng: np.random.Generator) -> np.ndarray:
    """Simulate future values of the period index for each batch member.

    k : (B, T).  Returns (B, n_paths, H).  Each path draws its own drift
    (estimation uncertainty, sigma/sqrt(n)) and cumulative innovations.
    """
    drift, sigma, n = drift_and_sigma(k)
    hmax = int(horizons.max())
    B = k.shape[0]
    z0 = rng.standard_normal((B, n_paths))
    drift_star = drift[:, None] + sigma[:, None] / np.sqrt(n) * z0
    innov = rng.standard_normal((B, n_paths, hmax)).cumsum(axis=-1) * sigma[:, None, None]
    steps = np.arange(1, hmax + 1)
    paths = k[:, -1, None, None] + drift_star[:, :, None] * steps + innov
    return paths[:, :, horizons - 1]


def jumpoff_correction(D: np.ndarray, E: np.ndarray, a, b, k, last: int = 3, shrink: float = 0.5):
    """Log ratio of observed to fitted deaths by age over the last `last`
    years of the fitting window.  Adding it to the fitted level anchors the
    forecast to the recent observed age pattern (in the spirit of Lee &
    Miller 2001) without relying on a single year."""
    Dh = E[..., -last:] * np.exp(a[..., :, None] + b[..., :, None] * k[..., None, -last:])
    obs = np.where(E[..., -last:] > 0, D[..., -last:], 0.0)
    return np.log((obs.sum(axis=-1) + shrink) / (Dh.sum(axis=-1) + shrink))


def forecast(
    D: np.ndarray,
    E: np.ndarray,
    horizons: np.ndarray,
    jump: str = "fit",
    n_boot: int = 0,
    n_paths: int = 4,
    rng: np.random.Generator | None = None,
    last: int = 3,
):
    """Fit Poisson Lee-Carter to (A, T) deaths/exposures and forecast rates.

    jump = "fit"        forecast starts from the fitted level
    jump = "corrected"  fitted level plus the recent observed/fitted log ratio

    Returns a dict with the fit, the correction, `central` rates (A, H) and,
    if n_boot > 0, `draws` of shape (n_boot * n_paths, H, A) that combine
    (i) Poisson resampling of the fitting data with refitting,
    (ii) uncertainty of the drift and (iii) random-walk innovations.
    """
    horizons = np.asarray(horizons)
    fit = fit_lee_carter(D, E)
    a, b, k = fit["a"], fit["b"], fit["k"]
    if jump == "corrected":
        r = jumpoff_correction(D, E, a, b, k, last)
    elif jump == "fit":
        r = np.zeros_like(a)
    else:
        raise ValueError(jump)
    drift, _, _ = drift_and_sigma(k)
    k_future = k[-1] + drift * horizons
    central = np.exp((a + r)[:, None] + b[:, None] * k_future[None, :])
    out = {"fit": fit, "correction": r, "central": central}
    if n_boot > 0:
        Db = rng.poisson(np.where(E > 0, D, 0.0), size=(n_boot,) + D.shape).astype(float)
        fb = fit_lee_carter(Db, E, start=(a, b, k), tol=1e-8, max_iter=500)
        rb = jumpoff_correction(Db, E, fb["a"], fb["b"], fb["k"], last) if jump == "corrected" else np.zeros_like(fb["a"])
        paths = simulate_period_index(fb["k"], horizons, n_paths, rng)            # (B, J, H)
        logm = (fb["a"] + rb)[:, None, None, :] + fb["b"][:, None, None, :] * paths[..., None]
        # In very small populations a resampled fit can give an erratic age
        # response and a simulated rate that explodes.  Rates are capped at
        # MAX_RATE per person-year; the share of capped cells is reported.
        failed = ~np.isfinite(logm).all(axis=(1, 2, 3))
        out["failed_share"] = float(failed.mean())
        if failed.all():
            raise ValueError("all resampled fits failed")
        if failed.any():                                   # replace by good replicates
            good = np.flatnonzero(~failed)
            logm[failed] = logm[rng.choice(good, size=int(failed.sum()))]
        capped = logm > np.log(MAX_RATE)
        out["capped_share"] = float(capped.mean())
        draws = np.exp(np.minimum(logm, np.log(MAX_RATE))).reshape(n_boot * n_paths, len(horizons), D.shape[0])
        out["draws"] = draws
        out["boot_converged"] = float(np.mean(fb["converged"]))
    return out


def predictive_rates(draws: np.ndarray, E_future: np.ndarray, rng: np.random.Generator, return_fallback: bool = False):
    """Add Poisson variation of the death counts to simulated expected rates.

    draws : (S, H, A) expected rates; E_future : (A, H) exposures actually
    observed in the target years.  Cells without exposure, and open groups
    that draw zero deaths (which would make the open-interval life expectancy
    infinite), keep the expected rate.  With `return_fallback` the number of
    simulated open groups that needed this fallback is returned as well.
    """
    Ef = E_future.T[None, :, :]
    deaths = rng.poisson(draws * Ef)
    with np.errstate(divide="ignore", invalid="ignore"):
        m = np.where(Ef > 0, deaths / Ef, draws)
    open_zero = m[..., -1] <= 0
    m[..., -1] = np.where(open_zero, draws[..., -1], m[..., -1])
    if return_fallback:
        return m, int(open_zero.sum())
    return m


# --------------------------------------------------------------------------
# Decomposition (Horiuchi, Wilmoth & Pletcher 2008)
# --------------------------------------------------------------------------
def quantile_mcse(values: np.ndarray, probs, cluster: int = 1, n_rep: int = 100, seed: int = 0) -> np.ndarray:
    """Monte Carlo standard error of sample quantiles of simulated values.

    values : (S, ...) simulations along the first axis.  The `cluster` paths that come from one refit are
    consecutive rows and are resampled together, because they share the refit.  Returns an array of shape
    (len(probs), ...).  The resampling has its own random stream, so calling this function does not change
    any other simulated number.
    """
    values = np.asarray(values, dtype=float)
    rest = values.shape[1:]
    v = values.reshape((values.shape[0] // cluster, cluster) + rest)
    n = v.shape[0]
    rng = np.random.default_rng(seed)
    out = np.empty((n_rep, len(probs)) + rest)
    for r in range(n_rep):
        out[r] = np.quantile(v[rng.integers(0, n, size=n)].reshape((-1,) + rest), probs, axis=0)
    return out.std(axis=0, ddof=1)


def two_sided_p_mcse(values: np.ndarray, observed: float, cluster: int = 1) -> float:
    """Approximate Monte Carlo standard error of the two-sided p-value 2 x min(share below, share above).

    Analytic, not resampled: the tail share is computed within each cluster (the paths of one refit) and
    its standard error is the standard deviation across clusters divided by the square root of their
    number, times two.  The smaller tail is fixed at the one observed, so the approximation is rough for
    p-values close to one, and it is zero when no simulated value falls in the tail (which does not mean
    the p-value is known exactly)."""
    v = np.asarray(values, dtype=float).reshape(-1, cluster)
    below, above = (v <= observed).mean(axis=1), (v >= observed).mean(axis=1)
    tail = below if below.mean() <= above.mean() else above
    return float(2.0 * tail.std(ddof=1) / np.sqrt(len(tail)))


def horiuchi(func, m_from: np.ndarray, m_to: np.ndarray, n_steps: int = 50) -> np.ndarray:
    """Line-integral decomposition of ``func(m_to) - func(m_from)`` into one
    contribution per element of the rate vector.

    `func` must accept an array of shape (N, A) and return shape (N,).
    """
    m_from = np.asarray(m_from, dtype=float)
    m_to = np.asarray(m_to, dtype=float)
    A = m_from.shape[-1]
    delta = (m_to - m_from) / n_steps
    half = 0.5 * np.diag(delta)
    contrib = np.zeros(A)
    for step in range(n_steps):
        mid = m_from + (step + 0.5) * delta
        contrib += func(mid[None, :] + half) - func(mid[None, :] - half)
    return contrib


def horiuchi_batch(func, M_from: np.ndarray, m_to: np.ndarray, n_steps: int = 40, chunk: int = 250) -> np.ndarray:
    """`horiuchi` for many starting schedules at once.

    M_from : (S, A) rate schedules; m_to : (A,) common end point.  Returns the
    (S, A) contributions of each age to ``func(m_to) - func(M_from[s])``.
    """
    M_from = np.asarray(M_from, dtype=float)
    m_to = np.asarray(m_to, dtype=float)
    S, A = M_from.shape
    out = np.empty((S, A))
    eye = np.eye(A)
    for lo in range(0, S, chunk):
        start = M_from[lo:lo + chunk]                              # (s, A)
        s = start.shape[0]
        delta = (m_to[None, :] - start) / n_steps                  # (s, A)
        half = 0.5 * delta[:, :, None] * eye[None, :, :]           # (s, A, A): row i perturbs age i
        contrib = np.zeros((s, A))
        for step in range(n_steps):
            mid = start + (step + 0.5) * delta
            plus = (mid[:, None, :] + half).reshape(s * A, A)
            minus = (mid[:, None, :] - half).reshape(s * A, A)
            contrib += (func(plus) - func(minus)).reshape(s, A)
        out[lo:lo + chunk] = contrib
    return out


def horiuchi_matrix(func, r_from: np.ndarray, r_to: np.ndarray, n_steps: int = 50) -> np.ndarray:
    """Same as `horiuchi` for a matrix of inputs (for example age x cause
    rates).  `func` receives arrays of shape (N,) + r_from.shape."""
    r_from = np.asarray(r_from, dtype=float)
    r_to = np.asarray(r_to, dtype=float)
    shape = r_from.shape
    size = r_from.size
    delta = ((r_to - r_from) / n_steps).reshape(size)
    half = 0.5 * np.diag(delta)
    contrib = np.zeros(size)
    base = r_from.reshape(size)
    for step in range(n_steps):
        mid = base + (step + 0.5) * delta
        plus = (mid[None, :] + half).reshape((size,) + shape)
        minus = (mid[None, :] - half).reshape((size,) + shape)
        contrib += func(plus) - func(minus)
    return contrib.reshape(shape)


def sensitivity_integral(func, m_from: np.ndarray, m_to: np.ndarray, n_steps: int = 40, rel_step: float = 1e-4) -> np.ndarray:
    """Average derivative of `func` with respect to each element of the rate
    vector along the straight path from m_from to m_to.

    With a straight path, the contribution of cause c at age x to
    func(m_to) - func(m_from) is this value times the change in the
    cause-specific rate, because the life table depends on causes only through
    their sum.  (Summed over causes it reproduces `horiuchi`.)
    """
    m_from = np.asarray(m_from, dtype=float)
    m_to = np.asarray(m_to, dtype=float)
    delta = m_to - m_from
    total = np.zeros_like(m_from)
    for step in range(n_steps):
        mid = m_from + (step + 0.5) / n_steps * delta
        h = rel_step * np.maximum(mid, 1e-8)
        bump = np.diag(h)
        total += (func(mid[None, :] + bump) - func(mid[None, :] - bump)) / (2.0 * h)
    return total / n_steps
