"""The fit measures Epoch reports for its Bayesian ECI (report, Nov 4 2026, "Similar accuracy and calibration" and
Appendix B "Quality of fit"), computed in sample and by PSIS-LOO for one or more scripts/bayes_eci.py fits.

Usage: python3 scripts/fit_metrics.py ARCHIVE_EPOCH_DIR LABEL=FIT_DIR [LABEL=FIT_DIR ...] [--draws 1000] [--seed 1] [--csv OUT]

Fits: Epoch's clipped beta (any clip) and the censored stretched beta (--sbeta, --sbeta-shift). Every fit is judged against the
observed scores as they are (exact 0s and 1s included). Posterior predictive draws:
  clipped beta: X ~ Beta(mu kappa, (1 - mu) kappa), read through the fit's own clip (X <= c -> 0, X >= 1 - c -> 1),
    the same convention as scripts/loo_compare.py at c* = c, so exact 0s and 1s can be covered;
  stretched beta: clamp(-delta + (1 + 2 delta) X, 0, 1).
LOO versions weight each score's draws by its PSIS weights from the likelihood the fit actually conditioned on
(clipped beta: beta density of the clipped score; stretched beta: its censored likelihood).
Measures (Epoch's definitions): RMSE, MAE, median absolute error of the predictive mean; beta-scaled RMSE (squared
error / (4 yhat (1 - yhat)), yhat the fit's own prediction); mean bias (predicted - observed); coverage of central
50/90/99% predictive intervals, and of 90% intervals for scores <= 0.05 and >= 0.95; CRPS; randomized PIT
(KS distance from uniform); posterior predictive p-value for the variance of all scores (in sample
only); predicted vs observed share of scores below 0.1%.
"""
import argparse
import json
import numpy as np
import pandas as pd
from scipy.special import betaln, expit
from scipy.stats import beta as B, kstest
from arviz_stats.base.array import array_stats

ap = argparse.ArgumentParser()
ap.add_argument("data_dir"); ap.add_argument("fits", nargs="+")
ap.add_argument("--draws", type=int, default=1000); ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--csv", help="also write the table here")
a = ap.parse_args()
d = pd.read_csv(f"{a.data_dir}/processed_data_for_eci.csv")
y = d.performance.to_numpy(float); n = len(y)
rng = np.random.default_rng(a.seed)


def wquantile(x, w, q):
    """Weighted quantiles of each column (x, w: draws x scores)."""
    o = np.argsort(x, axis=0); xs = np.take_along_axis(x, o, 0); ws = np.take_along_axis(w, o, 0)
    cw = np.cumsum(ws, 0) - 0.5 * ws
    return np.array([[np.interp(qq, cw[:, j], xs[:, j]) for j in range(x.shape[1])] for qq in q])


def wcrps(x, w, obs):
    """CRPS of a weighted sample: E|X - y| - 0.5 E|X - X'| (sorted-sample formula for the second term)."""
    t1 = (w * np.abs(x - obs)).sum(0)
    o = np.argsort(x, axis=0); xs = np.take_along_axis(x, o, 0); ws = np.take_along_axis(w, o, 0)
    W = np.cumsum(ws, 0)
    t2 = (2 * ws * xs * (2 * W - ws - 1)).sum(0)
    return t1 - 0.5 * t2


def setup(fit):
    dr = np.load(f"{fit}/draws.npz"); diag = json.load(open(f"{fit}/diagnostics.json"))
    models, benches = list(dr["models"]), list(dr["benchmarks"])
    mi = np.array([models.index(m) for m in d.Model]); bi = np.array([benches.index(b) for b in d.benchmark])
    idx = np.linspace(0, dr["eci"].shape[0] - 1, a.draws).astype(int)
    eci, edi, sl, sg = (dr[k][idx].astype(float) for k in ("eci", "edi", "slope", "sigma"))
    mu = np.clip(expit(sl[:, bi] * (eci[:, mi] - edi[:, bi])), 1e-6, 1 - 1e-6)
    k = 1 / (4 * sg[:, bi] ** 2) - 1; al, be = mu * k, (1 - mu) * k
    X = rng.beta(al, be)
    if diag.get("sbeta"):
        dl = dr["hyper_delta"][idx].astype(float)[:, None]
        if diag.get("sbeta_shift"):  # delta_b = delta (sigma_b / sigma_median)^gamma, capped at 0.1
            g, sm = (dr[k][idx].astype(float)[:, None] for k in ("hyper_gamma", "hyper_sigma_median"))
            dl = np.minimum(dl * (sg[:, bi] / sm) ** g, 0.1)
        rep = np.clip(-dl + (1 + 2 * dl) * X, 0, 1)
        tr = lambda v: (v + dl) / (1 + 2 * dl)
        cat = np.where(y <= 0, 0, np.where(y >= 1, 2, 1))
        ll = np.where(cat == 0, B.logcdf(tr(0.0), al, be), np.where(cat == 2, B.logsf(tr(1.0), al, be),
                      B.logpdf(tr(np.clip(y, 1e-12, 1 - 1e-12)), al, be) - np.log1p(2 * dl)))
        p_lt = lambda v: np.where(v <= 0, 0.0, np.where(v >= 1, B.cdf(tr(1.0), al, be), B.cdf(tr(v), al, be)))  # P(Y < v)
        p_eq = lambda v: np.where(v <= 0, B.cdf(tr(0.0), al, be), np.where(v >= 1, B.sf(tr(1.0), al, be), 0.0))
    else:
        c = diag.get("clip", 1e-3)
        rep = np.where(X <= c, 0.0, np.where(X >= 1 - c, 1.0, X))
        yc = np.clip(y, c, 1 - c)
        ll = (al - 1) * np.log(yc) + (be - 1) * np.log1p(-yc) - betaln(al, be)   # what the fit conditioned on
        p_lt = lambda v: np.where(v <= 0, 0.0, np.where(v >= 1, B.cdf(1 - c, al, be), np.where(v <= c, 0.0, B.cdf(v, al, be))))
        p_eq = lambda v: np.where(v <= 0, B.cdf(c, al, be), np.where(v >= 1, B.sf(1 - c, al, be), 0.0))
    lw, khat = array_stats.psislw(ll.T, axis=-1)  # takes the log-likelihood (negated inside)
    return dict(rep=rep, w_loo=np.exp(lw).T, khat=khat, p_lt=p_lt, p_eq=p_eq)


def metrics(f, w):
    rep = f["rep"]
    pred = (w * rep).sum(0); err = pred - y
    out = {"RMSE": np.sqrt(np.mean(err ** 2)), "MAE": np.mean(np.abs(err)), "median abs error": np.median(np.abs(err)),
           "beta-scaled RMSE": np.sqrt(np.mean(err ** 2 / (4 * np.clip(pred, 1e-9, 1 - 1e-9) * (1 - np.clip(pred, 1e-9, 1 - 1e-9))))),
           "mean bias": err.mean()}
    lv = (0.5, 0.9, 0.99)
    qs = wquantile(rep, w, sum(([(1 - l) / 2, (1 + l) / 2] for l in lv), []))
    for i, l in enumerate(lv):
        cov = (qs[2 * i] <= y) & (y <= qs[2 * i + 1])
        out[f"coverage {int(l * 100)}%"] = cov.mean()
        if l == 0.9:
            out["coverage 90%, scores <= 0.05"] = cov[y <= 0.05].mean()
            out["coverage 90%, scores >= 0.95"] = cov[y >= 0.95].mean()
    out["CRPS"] = wcrps(rep, w, y).mean()
    u = rng.uniform(size=n)
    pit = (w * f["p_lt"](y)).sum(0) + u * (w * f["p_eq"](y)).sum(0)
    out["PIT KS"] = kstest(pit, "uniform").statistic
    out["predicted share < 0.1%"] = (w * (rep < 1e-3)).sum(0).mean()
    return out


res = {}
for arg in a.fits:
    lab, fit = arg.split("=", 1)
    f = setup(fit)
    eq = np.full_like(f["rep"], 1 / f["rep"].shape[0])
    ins, loo = metrics(f, eq), metrics(f, f["w_loo"])
    rep = f["rep"]
    ins["PPP variance"] = np.mean(rep.var(1) >= y.var())
    res[(lab, "in sample")] = ins; res[(lab, "PSIS-LOO")] = loo
    print(f"{lab}: Pareto k > 0.7 for {(f['khat'] > 0.7).sum()} scores", flush=True)
t = pd.DataFrame(res)
print(f"\nobserved share < 0.1%: {np.mean(y < 1e-3):.4f}; {n} scores; {a.draws} posterior draws per fit\n")
with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 200):
    print(t.to_string())
if a.csv:
    t.columns = [f"{l}|{k}" for l, k in t.columns]
    t.rename_axis("measure").to_csv(a.csv)
