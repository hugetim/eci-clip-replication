"""The fit measures Epoch reports for its Bayesian ECI (report, Nov 4 2026, "Similar accuracy and calibration" and
Appendix B "Quality of fit"), computed in sample and by PSIS-LOO for one or more scripts/bayes_eci.py fits.

Usage: python3 scripts/fit_metrics.py ARCHIVE_EPOCH_DIR LABEL=FIT_DIR [LABEL=FIT_DIR ...] [--draws 1000] [--seed 1] [--csv OUT]

Fits: Epoch's clipped beta (any clip) and the extended-support beta, XB (bayes_eci.py --sbeta, --sbeta-shift). Every
fit is judged against the observed scores as they are (exact 0s and 1s included). Posterior predictive draws:
  clipped beta: X ~ Beta(mu kappa, (1 - mu) kappa), read through the fit's own clip (X <= c -> 0, X >= 1 - c -> 1),
    the same convention as scripts/loo_compare.py at c* = c, so exact 0s and 1s can be covered;
  XB: clamp(-delta + (1 + 2 delta) X, 0, 1).
LOO versions weight each score's draws by its PSIS weights from the likelihood the fit actually conditioned on
(clipped beta: beta density of the clipped score; XB: its censored likelihood).
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
from scipy.stats import beta as beta_dist, kstest
from arviz_stats.base.array import array_stats

ap = argparse.ArgumentParser()
ap.add_argument("data_dir")
ap.add_argument("fits", nargs="+")
ap.add_argument("--draws", type=int, default=1000)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--csv", help="also write the table here")
args = ap.parse_args()
data = pd.read_csv(f"{args.data_dir}/processed_data_for_eci.csv")
y = data.performance.to_numpy(float)
n_scores = len(y)
rng = np.random.default_rng(args.seed)


def by_position(at_zero, inside, at_one):
    """Pick, score by score, the value for a score of exactly 0, one strictly inside (0, 1), or one of exactly 1."""
    return np.where(y <= 0, at_zero, np.where(y >= 1, at_one, inside))


def weighted_quantiles(x, w, q):
    """Weighted quantiles of each column (x, w: draws x scores)."""
    order = np.argsort(x, axis=0)
    xs, ws = np.take_along_axis(x, order, 0), np.take_along_axis(w, order, 0)
    cum = np.cumsum(ws, 0) - 0.5 * ws
    return np.array([[np.interp(level, cum[:, j], xs[:, j]) for j in range(x.shape[1])] for level in q])


def weighted_crps(x, w, obs):
    """CRPS of a weighted sample: E|X - y| - 0.5 E|X - X'| (sorted-sample formula for the second term)."""
    distance_to_obs = (w * np.abs(x - obs)).sum(0)
    order = np.argsort(x, axis=0)
    xs, ws = np.take_along_axis(x, order, 0), np.take_along_axis(w, order, 0)
    cum = np.cumsum(ws, 0)
    spread = (2 * ws * xs * (2 * cum - ws - 1)).sum(0)
    return distance_to_obs - 0.5 * spread


def predictive(fit):
    """One fit's posterior predictive for every score, from args.draws evenly spaced posterior draws. All arrays are
    draws x scores:
      rep: a replicated score per draw; w_loo: each draw's PSIS-LOO weight for that score;
      p_below, p_at: P(Y < y) and P(Y = y) at the observed score y (for the randomized PIT);
    and khat: each score's Pareto k."""
    dr = np.load(f"{fit}/draws.npz")
    diag = json.load(open(f"{fit}/diagnostics.json"))
    models, benches = list(dr["models"]), list(dr["benchmarks"])
    model_idx = np.array([models.index(m) for m in data.Model])
    bench_idx = np.array([benches.index(b) for b in data.benchmark])
    keep = np.linspace(0, dr["eci"].shape[0] - 1, args.draws).astype(int)
    eci, edi, slope, sigma = (dr[k][keep].astype(float) for k in ("eci", "edi", "slope", "sigma"))
    # Epoch's beta for each score: mean mu on the benchmark's logistic curve, precision kappa from its sigma_b
    mu = np.clip(expit(slope[:, bench_idx] * (eci[:, model_idx] - edi[:, bench_idx])), 1e-6, 1 - 1e-6)
    kappa = 1 / (4 * sigma[:, bench_idx] ** 2) - 1
    a, b = mu * kappa, (1 - mu) * kappa
    x = rng.beta(a, b)
    if diag.get("sbeta"):  # XB: the beta stretched to [-delta, 1 + delta], then clamped to [0, 1]
        delta = dr["hyper_delta"][keep].astype(float)[:, None]
        if diag.get("sbeta_shift"):  # delta_b = delta (sigma_b / sigma_median)^gamma, capped at 0.1
            gamma, sigma_median = (dr[k][keep].astype(float)[:, None] for k in ("hyper_gamma", "hyper_sigma_median"))
            delta = np.minimum(delta * (sigma[:, bench_idx] / sigma_median) ** gamma, 0.1)
        on_beta_scale = lambda score: (score + delta) / (1 + 2 * delta)  # where a score sits on the beta's own [0, 1]
        lo, hi = on_beta_scale(0.0), on_beta_scale(1.0)
        rep = np.clip(-delta + (1 + 2 * delta) * x, 0, 1)
        density = beta_dist.logpdf(on_beta_scale(np.clip(y, 1e-12, 1 - 1e-12)), a, b) - np.log1p(2 * delta)
        loglik = by_position(beta_dist.logcdf(lo, a, b), density, beta_dist.logsf(hi, a, b))
        p_below = by_position(0.0, beta_dist.cdf(on_beta_scale(y), a, b), beta_dist.cdf(hi, a, b))
        p_at = by_position(beta_dist.cdf(lo, a, b), 0.0, beta_dist.sf(hi, a, b))
    else:  # Epoch's clipped beta: a draw at or beyond the clip counts as an exact 0 or 1
        c = diag.get("clip", 1e-3)
        rep = np.where(x <= c, 0.0, np.where(x >= 1 - c, 1.0, x))
        y_clipped = np.clip(y, c, 1 - c)
        loglik = (a - 1) * np.log(y_clipped) + (b - 1) * np.log1p(-y_clipped) - betaln(a, b)  # what the fit conditioned on
        p_below = by_position(0.0, np.where(y <= c, 0.0, beta_dist.cdf(y, a, b)), beta_dist.cdf(1 - c, a, b))
        p_at = by_position(beta_dist.cdf(c, a, b), 0.0, beta_dist.sf(1 - c, a, b))
    log_weights, khat = array_stats.psislw(loglik.T, axis=-1)  # takes the log-likelihood (negated inside)
    return dict(rep=rep, w_loo=np.exp(log_weights).T, khat=khat, p_below=p_below, p_at=p_at)


def metrics(pred, w):
    """Epoch's fit measures for one fit's predictive draws under weights w (draws x scores; each column sums to 1)."""
    rep = pred["rep"]
    mean = (w * rep).sum(0)
    err = mean - y
    mean_inside = np.clip(mean, 1e-9, 1 - 1e-9)
    out = {"RMSE": np.sqrt(np.mean(err ** 2)), "MAE": np.mean(np.abs(err)), "median abs error": np.median(np.abs(err)),
           "beta-scaled RMSE": np.sqrt(np.mean(err ** 2 / (4 * mean_inside * (1 - mean_inside)))),
           "mean bias": err.mean()}
    levels = (0.5, 0.9, 0.99)
    edges = weighted_quantiles(rep, w, sum(([(1 - level) / 2, (1 + level) / 2] for level in levels), []))
    for i, level in enumerate(levels):
        covered = (edges[2 * i] <= y) & (y <= edges[2 * i + 1])
        out[f"coverage {int(level * 100)}%"] = covered.mean()
        if level == 0.9:
            out["coverage 90%, scores <= 0.05"] = covered[y <= 0.05].mean()
            out["coverage 90%, scores >= 0.95"] = covered[y >= 0.95].mean()
    out["CRPS"] = weighted_crps(rep, w, y).mean()
    u = rng.uniform(size=n_scores)
    pit = (w * pred["p_below"]).sum(0) + u * (w * pred["p_at"]).sum(0)
    out["PIT KS"] = kstest(pit, "uniform").statistic
    out["predicted share < 0.1%"] = (w * (rep < 1e-3)).sum(0).mean()
    return out


results = {}
for arg in args.fits:
    label, fit = arg.split("=", 1)
    pred = predictive(fit)
    equal_weights = np.full_like(pred["rep"], 1 / pred["rep"].shape[0])
    in_sample, loo = metrics(pred, equal_weights), metrics(pred, pred["w_loo"])
    in_sample["PPP variance"] = np.mean(pred["rep"].var(1) >= y.var())
    results[(label, "in sample")] = in_sample
    results[(label, "PSIS-LOO")] = loo
    print(f"{label}: Pareto k > 0.7 for {(pred['khat'] > 0.7).sum()} scores", flush=True)
table = pd.DataFrame(results)
print(f"\nobserved share < 0.1%: {np.mean(y < 1e-3):.4f}; {n_scores} scores; {args.draws} posterior draws per fit\n")
with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 200):
    print(table.to_string())
if args.csv:
    table.columns = [f"{label}|{kind}" for label, kind in table.columns]
    table.rename_axis("measure").to_csv(args.csv)
