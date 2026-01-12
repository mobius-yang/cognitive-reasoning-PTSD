from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from statsmodels.tsa.arima.model import ARIMA


@dataclass(frozen=True)
class _ArimaFit:
    p: int
    d: int
    q: int
    aic: float
    arparams: np.ndarray
    maparams: np.ndarray
    sigma2: float


@dataclass(frozen=True)
class _GarchFit:
    omega: float
    alpha1: float
    beta1: float
    loglik: float
    sigma2: np.ndarray


def _safe_float(x) -> float:
    return float(x)


def _poly_slope(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return 0.0
    mask = np.isfinite(x) & np.isfinite(y)
    x2 = x[mask].astype(float, copy=False)
    y2 = y[mask].astype(float, copy=False)
    if x2.size < 2:
        return 0.0
    return float(np.polyfit(x2, y2, deg=1)[0])


def _extract_ordered_series(group: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    sessions = pd.to_numeric(group.get("session"), errors="coerce")
    suds_after = pd.to_numeric(group.get("suds_after"), errors="coerce")
    tmp = pd.DataFrame({"session": sessions, "suds_after": suds_after}).dropna()
    if tmp.empty:
        return np.array([], dtype=float), np.array([], dtype=float)
    tmp = tmp.groupby("session", as_index=False)["suds_after"].mean().sort_values("session")
    x = tmp["session"].to_numpy(dtype=float)
    y = tmp["suds_after"].to_numpy(dtype=float)
    return x, y


def _fit_best_arima_by_aic(
    y: np.ndarray,
    *,
    max_p: int = 2,
    max_d: int = 1,
    max_q: int = 2,
) -> _ArimaFit | None:
    best: _ArimaFit | None = None
    n = int(y.size)
    if n < 4:
        return None

    for d in range(max_d + 1):
        trend = "n" if d > 0 else "c"
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                if p == 0 and q == 0:
                    continue
                # Roughly ensure enough observations for estimation.
                if n <= (p + q + d + 1):
                    continue
                res = ARIMA(
                    y,
                    order=(p, d, q),
                    trend=trend,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit()

                aic = _safe_float(getattr(res, "aic", np.nan))
                if not np.isfinite(aic):
                    continue

                arparams = np.asarray(getattr(res, "arparams", np.array([], dtype=float)), dtype=float)
                maparams = np.asarray(getattr(res, "maparams", np.array([], dtype=float)), dtype=float)
                resid = np.asarray(getattr(res, "resid"), dtype=float)
                sigma2 = float(np.nanvar(resid, ddof=0))

                fit = _ArimaFit(p=p, d=d, q=q, aic=aic, arparams=arparams, maparams=maparams, sigma2=sigma2)
                if best is None or fit.aic < best.aic:
                    best = fit
    return best


def _fit_garch11(e: np.ndarray) -> _GarchFit | None:
    e = np.asarray(e, dtype=float)
    e = e[np.isfinite(e)]
    if e.size < 4:
        return None

    e = e - float(np.mean(e))
    var0 = float(np.var(e, ddof=0))
    if not np.isfinite(var0) or var0 <= 0:
        return None

    def _rec_sigma2(omega: float, alpha1: float, beta1: float) -> np.ndarray | None:
        if omega <= 0 or alpha1 < 0 or beta1 < 0 or (alpha1 + beta1) >= 0.999:
            return None
        sigma2 = np.empty(e.size, dtype=float)
        sigma2[0] = var0
        for t in range(1, e.size):
            sigma2[t] = omega + alpha1 * (e[t - 1] ** 2) + beta1 * sigma2[t - 1]
            if not np.isfinite(sigma2[t]) or sigma2[t] <= 0:
                return None
        return sigma2

    def _nll(params: np.ndarray) -> float:
        omega, alpha1, beta1 = [float(x) for x in params]
        sigma2 = _rec_sigma2(omega, alpha1, beta1)
        if sigma2 is None:
            return 1e18
        ll = -0.5 * np.sum(np.log(2 * np.pi) + np.log(sigma2) + (e**2) / sigma2)
        if not np.isfinite(ll):
            return 1e18
        return -float(ll)

    alpha0 = 0.05
    beta0 = 0.90
    omega0 = max(1e-8, var0 * (1.0 - alpha0 - beta0))
    x0 = np.array([omega0, alpha0, beta0], dtype=float)

    bounds = [(1e-12, var0 * 10.0), (0.0, 0.999), (0.0, 0.999)]
    cons = ({"type": "ineq", "fun": lambda p: 0.999 - float(p[1]) - float(p[2])},)

    res = minimize(
        _nll,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 500},
    )
    if not res.success:
        return None

    omega, alpha1, beta1 = [float(x) for x in np.asarray(res.x, dtype=float)]
    sigma2 = _rec_sigma2(omega, alpha1, beta1)
    if sigma2 is None:
        return None

    return _GarchFit(omega=omega, alpha1=alpha1, beta1=beta1, loglik=-_nll(np.array([omega, alpha1, beta1])), sigma2=sigma2)


def extract_suds_dynamics_features(df_task1: pd.DataFrame) -> pd.DataFrame:
    """
    Extract subject-level time-series features from session-level SUDS measurements.

    Expected columns:
      - Name: subject identifier
      - session: time index (numeric-like)
      - suds_after: SUDS rating (numeric-like)
    """
    if "Name" not in df_task1.columns:
        raise KeyError("Missing `Name` column for SUDS time-series feature extraction.")
    if "session" not in df_task1.columns or "suds_after" not in df_task1.columns:
        return pd.DataFrame({"Name": df_task1["Name"].astype(str).unique()})

    rows: list[dict] = []
    for name, group in df_task1.groupby("Name", dropna=False):
        x, y = _extract_ordered_series(group)
        out: dict[str, object] = {"Name": str(name)}

        out["SUDS_n_obs"] = int(y.size)
        out["SUDS_after_mean"] = float(np.nanmean(y)) if y.size else 0.0
        out["SUDS_after_std"] = float(np.nanstd(y, ddof=0)) if y.size else 0.0
        out["SUDS_after_slope"] = _poly_slope(x, y) if y.size else 0.0

        # ARIMA(p,d,q): fit per subject and treat AR(1) coefficient as inertia proxy.
        fit = _fit_best_arima_by_aic(y)
        if fit is None:
            out["SUDS_ARIMA_p"] = np.nan
            out["SUDS_ARIMA_d"] = np.nan
            out["SUDS_ARIMA_q"] = np.nan
            out["SUDS_ARIMA_aic"] = np.nan
            out["SUDS_ARIMA_sigma2"] = np.nan
            out["SUDS_ARIMA_ar1"] = np.nan
            out["SUDS_ARIMA_ma1"] = np.nan
            out["SUDS_ARIMA_ar_sum"] = np.nan
            out["SUDS_ARIMA_ma_sum"] = np.nan
        else:
            out["SUDS_ARIMA_p"] = int(fit.p)
            out["SUDS_ARIMA_d"] = int(fit.d)
            out["SUDS_ARIMA_q"] = int(fit.q)
            out["SUDS_ARIMA_aic"] = float(fit.aic)
            out["SUDS_ARIMA_sigma2"] = float(fit.sigma2)
            out["SUDS_ARIMA_ar1"] = float(fit.arparams[0]) if fit.arparams.size >= 1 else 0.0
            out["SUDS_ARIMA_ma1"] = float(fit.maparams[0]) if fit.maparams.size >= 1 else 0.0
            out["SUDS_ARIMA_ar_sum"] = float(np.sum(fit.arparams)) if fit.arparams.size else 0.0
            out["SUDS_ARIMA_ma_sum"] = float(np.sum(fit.maparams)) if fit.maparams.size else 0.0

        # GARCH(1,1): model conditional heteroskedasticity of SUDS changes (volatility clustering).
        dy = np.diff(y) if y.size else np.array([], dtype=float)
        gfit = _fit_garch11(dy)
        if gfit is None:
            out["SUDS_GARCH_omega"] = np.nan
            out["SUDS_GARCH_alpha1"] = np.nan
            out["SUDS_GARCH_beta1"] = np.nan
            out["SUDS_GARCH_persistence"] = np.nan
            out["SUDS_GARCH_loglik"] = np.nan
            out["SUDS_GARCH_sigma2_mean"] = np.nan
            out["SUDS_GARCH_sigma2_max"] = np.nan
            out["SUDS_GARCH_sigma2_last"] = np.nan
        else:
            out["SUDS_GARCH_omega"] = float(gfit.omega)
            out["SUDS_GARCH_alpha1"] = float(gfit.alpha1)
            out["SUDS_GARCH_beta1"] = float(gfit.beta1)
            out["SUDS_GARCH_persistence"] = float(gfit.alpha1 + gfit.beta1)
            out["SUDS_GARCH_loglik"] = float(gfit.loglik)
            out["SUDS_GARCH_sigma2_mean"] = float(np.mean(gfit.sigma2)) if gfit.sigma2.size else np.nan
            out["SUDS_GARCH_sigma2_max"] = float(np.max(gfit.sigma2)) if gfit.sigma2.size else np.nan
            out["SUDS_GARCH_sigma2_last"] = float(gfit.sigma2[-1]) if gfit.sigma2.size else np.nan

        rows.append(out)

    df_out = pd.DataFrame(rows)
    if df_out.empty:
        return df_out

    feature_cols = [c for c in df_out.columns if c != "Name"]
    df_out[feature_cols] = df_out[feature_cols].replace([np.inf, -np.inf], np.nan)
    return df_out
