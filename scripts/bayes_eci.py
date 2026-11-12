"""Fit Epoch's Bayesian ECI with Epoch's own model code, sampled by nutpie.

The model is built by eci.bayesian.build_bayesian_model from Epoch's public
package (https://github.com/epoch-research/eci-public, cloned into
eci-public/ at the repo root; install with `pip install -e eci-public`), so
priors, constraints, clipping and the likelihood are exactly Epoch's. Only the
sampler differs: nutpie with a low-rank-plus-diagonal mass matrix instead of
Epoch's PyMC NUTS ladder (diagonal pilot, then a dense mass matrix). Both
target the same posterior. The per-draw rescaling to the ECI scale
(Claude 3.5 Sonnet = 130, GPT-5 = 150) and the summaries (posterior mean,
5th-95th percentiles) follow eci.bayesian.fit_bayesian_eci.

Unlike Epoch's released files, the output includes the per-benchmark noise
levels sigma_b and the hyperparameters.

Usage:
    python3 scripts/bayes_eci.py DATA_DIR OUT_DIR [--chains 4 --tune 1000 --draws 1000]
DATA_DIR is an archive's epoch_capabilities_index/ folder (Epoch's processed
fit input, processed_data_for_eci.csv, is used as is).
Outputs in OUT_DIR: eci.csv, edi.csv (edi, slope, sigma with 90% intervals),
hyper.json, diagnostics.json, draws.npz.
"""
import argparse
import hashlib
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
import arviz as az  # noqa: E402
from eci.bayesian import (  # noqa: E402
    DEFAULT_ANCHOR_ECI_HIGH, DEFAULT_ANCHOR_ECI_LOW, DEFAULT_ANCHOR_MODEL_HIGH,
    DEFAULT_ANCHOR_MODEL_LOW, build_bayesian_model)

# Epoch's convergence standards (eci.bayesian: _MIN_ESS, _MAX_RHAT) and the
# variables it checks.
MIN_ESS, MAX_RHAT = 400, 1.01
DIAG_VARS = ["capability", "difficulty", "slope", "sigma", "tau_alpha", "sigma_median", "tau_sigma"]
HYPER = ["tau_alpha", "sigma_median", "tau_sigma"]
ECI_PUBLIC_COMMIT = "e04929a"  # the eci-public commit this was checked against (setup.sh pins it)


def beta_logcdf_small(a, b, x, n_terms=60):
    """log I_x(a, b), the beta CDF at a small x, as a PyTensor expression with stable gradients.

    PyTensor's own Beta logcdf (betainc) returns NaN gradients when a is small and b large, which is exactly where
    a weak model's censored zero sits (a = mu * kappa ~ 0.02), so every NUTS transition diverged. Instead, DLMF 8.17.8:
        I_x(a, b) = x^a (1 - x)^b / (a B(a, b)) * sum_n (a + b)_n / (a + 1)_n x^n,
    whose terms are all positive, summed in logs. Successive terms have ratio (a + b + n - 1) x / (a + n), which tends
    to x, so the series converges for any x < 1; the terms first grow while (a + b) x > a + n, so the larger
    (a + b) x is, the more terms are needed.

    Accuracy against scipy.special.betainc, in logs, for mu = a / (a + b) up to 0.1 and kappa = a + b up to 1,000
    (Epoch's fit has kappa up to ~770): better than 1e-11 with the default 60 terms at x <= 1e-3, and with 200 terms
    (what sbeta_model passes) at x up to 0.084, the largest x its cap on delta allows. Gradients agree to 1e-9
    (checked for a in [1e-4, 30], b up to 600, x = 1e-3 and 1e-4). At kappa = 2,000 and x = 0.084, 200 terms are no
    longer enough (log error 0.1).
    """
    import pytensor.tensor as pt
    n = np.arange(1, n_terms + 1, dtype=float)
    x = pt.as_tensor_variable(x)  # a scalar, or one x per (a, b)
    lx = pt.log(x)
    logt = pt.cumsum(pt.log(a[:, None] + b[:, None] + n[None, :] - 1) - pt.log(a[:, None] + n[None, :])
                     + (lx[:, None] if x.ndim else lx), axis=1)
    return (a * lx + b * pt.log1p(-x) - pt.log(a) - (pt.gammaln(a) + pt.gammaln(b) - pt.gammaln(a + b))
            + pt.log1p(pt.exp(logt).sum(axis=1)))



def init_from_fit(fit_dir, model):
    """Posterior means of another fit, on this model's raw scale, as nutpie initial points. draws.npz holds ECI-scale
    draws plus each draw's map ECI = a + b * raw, so raw capability = (eci - a) / b, difficulty likewise, and
    raw slope = scaled slope * b (both fits share Epoch's identification: sum D = 0, sum log slope = 0)."""
    dr = np.load(Path(fit_dir) / "draws.npz")
    a, b = dr["a"][:, None].astype(float), dr["b"][:, None].astype(float)
    src_m, src_b = list(dr["models"]), list(dr["benchmarks"])
    mods, bens = list(model.coords["model"]), list(model.coords["benchmark"])
    mi, bi = [src_m.index(m) for m in mods], [src_b.index(x) for x in bens]
    cap = ((dr["eci"].astype(float) - a) / b).mean(0)[mi]
    dif = ((dr["edi"].astype(float) - a) / b).mean(0)[bi]
    lsl = np.log(dr["slope"].astype(float) * b).mean(0)[bi]
    init = {"capability": cap, "difficulty": dif - dif.mean(), "log_slope": lsl - lsl.mean(),
            "sigma": dr["sigma"].astype(float).mean(0)[bi]}
    for k in ("tau_alpha", "sigma_median", "tau_sigma", "gamma", "delta", "nu"):
        if "hyper_" + k in dr.files and k in model.named_vars:
            init[k] = float(np.median(dr["hyper_" + k]))
    return init


def sbeta_model(df, shift=False, delta_prior="halfnormal0.05", xbx_nodes=0):
    """Epoch's model with the extended-support beta likelihood, XB(mu, phi, u), of Kosmidis and Zeileis (2026, JRSS C
    75: 139-157), with phi = kappa and u = delta. (This code calls it "sbeta", for stretched beta: the name predates
    the post's "XB".)

    The model. A latent score y* = -delta + (1 + 2 delta) x, with x ~ Beta(mu kappa_b, (1 - mu) kappa_b) and
    mu = sigmoid(slope_b (C_m - D_b)): Epoch's curve and beta (same clip on mu, same kappa_b = 1 / (4 sigma_b^2) - 1),
    stretched to [-delta, 1 + delta]. The observed score is y = clamp(y*, 0, 1). So:
      - interior scores keep Epoch's beta shape and spread;
      - exact 0s and 1s get the probabilities P(x <= x0) and P(x >= 1 - x0), x0 = delta / (1 + 2 delta);
      - the mean curve -delta + (1 + 2 delta) mu crosses 0 and 1 at finite capability.
    Scores are used as they are (no clip on scores). Every prior and constraint other than delta's is Epoch's.

    delta_prior: "halfnormal0.05" (the default: HalfNormal(0.05)), "halfnormal0.01", "uniform" (Uniform(0, 0.1)) or
    "loguniform" (density proportional to 1 / delta on [1e-5, 0.1]). All are capped at 0.1, which keeps x0 < 0.084,
    where the series for the boundary probabilities is accurate (see beta_logcdf_small).

    shift=True ties the stretch to each benchmark's scatter: delta_b = delta (sigma_b / sigma_median)^gamma, with
    gamma estimated (prior Normal(0, 2)). gamma > 0 gives widely scattered benchmarks more 0s and 100s than plain XB
    (gamma = 0). One extra parameter. delta_b is capped at 0.1 too; the fit never comes near it.

    xbx_nodes > 0 fits their XBX instead: a continuous mixture of XB over the exceedance u ~ Exponential(mean nu), so
    each score's likelihood is f_XBX(y) = int_0^inf f_XB(y | u) e^(-u / nu) / nu du. nu takes delta's prior (it is
    the mean exceedance). The integral is computed by quadrature:
      - interior scores: Gauss-Laguerre with xbx_nodes points (their eq. 9), sum_t W_t f_XB(y | u = nu Q_t);
      - exact 0s and 1s: the trapezoid rule in s = log(u / nu), 32 points on [-14, 4]. There f_XB(y | u) behaves like
        u^a near u = 0, with a = mu kappa as small as ~0.02, which Gauss-Laguerre handles poorly (error ~2e-3 per
        score at 32 nodes).
    Against a 600-point reference at the XB fit's estimates, nu = 0.0013 to 0.011, the total log-likelihood error is
    under 0.015 with 12 Laguerre nodes. u is capped at 0.1, which only matters for nodes whose weight is negligible
    at any nu the data support.
    """
    import pymc as pm
    import pytensor.tensor as pt
    from pymc.distributions.transforms import ZeroSumTransform
    from eci.bayesian import BayesianPriors, _index_data, _lognormal, _CD_BOUND, _MAX_SIGMA
    priors = BayesianPriors()
    _, model_names, _, bench_names, model_idx, bench_idx = _index_data(df)
    y = df.performance.to_numpy(float)
    assert ((y >= 0) & (y <= 1)).all()
    cat = np.where(y <= 0, 0, np.where(y >= 1, 2, 1))
    inner = cat == 1
    with pm.Model(coords={"model": model_names, "benchmark": bench_names}) as model:
        tau_alpha = _lognormal("tau_alpha", priors.tau_alpha)
        sigma_median = _lognormal("sigma_median", priors.sigma_median)
        tau_sigma = _lognormal("tau_sigma", priors.tau_sigma)
        sigma = pm.Truncated("sigma", pm.LogNormal.dist(mu=pm.math.log(sigma_median), sigma=tau_sigma),
                             lower=0.0, upper=_MAX_SIGMA, dims="benchmark")
        capability = pm.Uniform("capability", -_CD_BOUND, _CD_BOUND, dims="model")
        difficulty = pm.Flat("difficulty", dims="benchmark", transform=ZeroSumTransform([-1]))
        pm.Potential("difficulty_bound", pm.math.switch(pm.math.all(pm.math.abs(difficulty) <= _CD_BOUND), 0.0, -np.inf))
        log_slope = pm.ZeroSumNormal("log_slope", sigma=tau_alpha * np.sqrt(len(bench_names) / (len(bench_names) - 1)),
                                     dims="benchmark")
        slope = pm.Deterministic("slope", pm.math.exp(log_slope), dims="benchmark")
        name = "nu" if xbx_nodes else "delta"
        if delta_prior == "halfnormal0.05":
            delta = pm.Truncated(name, pm.HalfNormal.dist(0.05), lower=0.0, upper=0.1)
        elif delta_prior == "halfnormal0.01":
            delta = pm.Truncated(name, pm.HalfNormal.dist(0.01), lower=0.0, upper=0.1)
        elif delta_prior == "uniform":
            delta = pm.Uniform(name, 0.0, 0.1)
        elif delta_prior == "loguniform":
            delta = pm.Uniform(name, 1e-5, 0.1)
            pm.Potential("loguniform_prior", -pt.log(delta))
        else:
            raise ValueError(delta_prior)
        mu = pt.clip(pm.math.sigmoid(slope[bench_idx] * (capability[model_idx] - difficulty[bench_idx])), 1e-6, 1 - 1e-6)
        kappa = 1 / (4 * sigma[bench_idx] ** 2) - 1
        a, b = mu * kappa, (1 - mu) * kappa
        lo, hi = cat == 0, cat == 2
        if xbx_nodes:  # XBX: mixture over u = nu Q_t, log-sum-exp over the quadrature nodes
            assert not shift
            Q, W = np.polynomial.laguerre.laggauss(xbx_nodes)
            parts = []
            for q, w in zip(Q, W):  # interior scores: Gauss-Laguerre
                u = pt.minimum(delta * q, 0.1)
                x_in = (y[inner] + u) / (1 + 2 * u)
                parts.append(pm.logp(pm.Beta.dist(a[inner], b[inner]), x_in) - pt.log1p(2 * u) + np.log(w))
            lp_in = pm.math.logsumexp(pt.stack(parts), axis=0)
            svals = np.linspace(-14.0, 4.0, 32); h = svals[1] - svals[0]
            parts = []
            for sv in svals:  # exact 0s and 1s: trapezoid in s = log(u / nu), weight e^(s - e^s) h
                u = pt.minimum(delta * np.exp(sv), 0.1)
                parts.append(pt.concatenate([beta_logcdf_small(a[lo], b[lo], u / (1 + 2 * u), n_terms=200),
                                             beta_logcdf_small(b[hi], a[hi], u / (1 + 2 * u), n_terms=200)])
                             + (sv - np.exp(sv) + np.log(h)))
            lp_bd = pm.math.logsumexp(pt.stack(parts), axis=0)
            pm.Potential("xbx", lp_in.sum() + lp_bd.sum())
            print(f"XBX ({xbx_nodes} Gauss-Laguerre nodes, nu prior {delta_prior}): {int(lo.sum())} scores at 0, "
                  f"{int(hi.sum())} at 1, {int(inner.sum())} interior", flush=True)
            return model
        if shift:  # one delta per score, from its benchmark's sigma_b
            gamma = pm.Normal("gamma", 0.0, 2.0)
            d_all = pt.minimum(delta * pt.exp(gamma * pt.log(sigma[bench_idx] / sigma_median)), 0.1)
            d_in, d_lo, d_hi = d_all[inner], d_all[lo], d_all[hi]
        else:
            d_in = d_lo = d_hi = delta
        x_in = (y[inner] + d_in) / (1 + 2 * d_in)
        lp_in = pm.logp(pm.Beta.dist(a[inner], b[inner]), x_in) - pt.log1p(2 * d_in)
        lp_lo = beta_logcdf_small(a[lo], b[lo], d_lo / (1 + 2 * d_lo), n_terms=200)
        lp_hi = beta_logcdf_small(b[hi], a[hi], d_hi / (1 + 2 * d_hi), n_terms=200)   # P(x >= 1 - x0) for Beta(a, b) = P(x' <= x0), Beta(b, a)
        pm.Potential("sbeta", lp_in.sum() + lp_lo.sum() + lp_hi.sum())
    print(f"censored stretched beta (XB){' (stretch shifted by sigma_b)' if shift else ''}, delta prior {delta_prior}: {int(lo.sum())} scores at 0, {int(hi.sum())} at 1, {int(inner.sum())} interior", flush=True)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--cores", type=int, default=2)
    ap.add_argument("--tune", type=int, default=1000)
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--adaptation", default="low_rank")
    ap.add_argument("--target-accept", type=float, default=0.95, help="NUTS target acceptance rate (Epoch: 0.95)")
    ap.add_argument("--seed", type=int, default=20261104)
    ap.add_argument("--clip", type=float, default=1e-3,
                    help="clip scores to [clip, 1 - clip] (Epoch's performance_clip_eps; default 0.001)")
    ap.add_argument("--sbeta", action="store_true",
                    help="extended-support beta (XB): Epoch's beta on [-delta, 1 + delta], clamped to [0, 1] (see sbeta_model)")
    ap.add_argument("--sbeta-shift", action="store_true",
                    help="XB with each benchmark's delta (sigma_b / sigma_median)^gamma, gamma estimated")
    ap.add_argument("--delta-prior", default="halfnormal0.05",
                    choices=["halfnormal0.05", "halfnormal0.01", "uniform", "loguniform"],
                    help="with --sbeta or --xbx: prior on delta (or nu), see sbeta_model")
    ap.add_argument("--xbx", type=int, default=0, metavar="NODES",
                    help="Kosmidis-Zeileis XBX (exponential mixture of XB over the exceedance), with this many "
                         "Gauss-Laguerre nodes (see sbeta_model)")
    ap.add_argument("--init-from", metavar="FIT_DIR",
                    help="start chains at this fit's posterior means (its draws.npz; ECI scale converted to raw)")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    import nutpie
    src = Path(args.data_dir) / "processed_data_for_eci.csv"
    df = pd.read_csv(src)
    if args.xbx:
        model = sbeta_model(df, delta_prior=args.delta_prior, xbx_nodes=args.xbx)
        DIAG_VARS.append("nu"); HYPER.append("nu")
    elif args.sbeta or args.sbeta_shift:
        model = sbeta_model(df, shift=args.sbeta_shift, delta_prior=args.delta_prior)
        DIAG_VARS.append("delta"); HYPER.append("delta")
        if args.sbeta_shift:
            DIAG_VARS.append("gamma"); HYPER.append("gamma")
    else:
        model = build_bayesian_model(df, performance_clip_eps=args.clip)
    models = list(model.coords["model"])
    benches = list(model.coords["benchmark"])
    print(f"{len(df)} scores, {len(models)} models, {len(benches)} benchmarks", flush=True)

    # Start capabilities and difficulties at 0 on the raw scale; jitter only the
    # slopes, noise and hyperparameters. nutpie's default jitter (+-1 on the
    # unconstrained scale) puts Uniform(-50, 50) starts up to +-23 away, where
    # the curve saturates and Epoch's 1e-6 clip on mu flattens the gradient.
    jitter = {model[v] for v in ("log_slope", "sigma", *HYPER)}
    init = None
    if args.init_from:  # start every chain at another fit's posterior means (raw scale), jittering only new parameters
        init = init_from_fit(args.init_from, model)
        jitter = {model[v] for v in HYPER if v not in init}
    t0 = time.time()
    compiled = nutpie.compile_pymc_model(model, jitter_rvs=jitter, initial_points=init)
    t1 = time.time()
    trace = nutpie.sample(compiled, chains=args.chains, cores=args.cores, tune=args.tune,
                          draws=args.draws, seed=args.seed, adaptation=args.adaptation, target_accept=args.target_accept,
                          progress_bar=False)
    t2 = time.time()
    post = trace.posterior.to_dataset() if hasattr(trace.posterior, "to_dataset") else trace.posterior
    stats = trace.sample_stats.to_dataset() if hasattr(trace.sample_stats, "to_dataset") else trace.sample_stats

    summ = az.summary(post, var_names=DIAG_VARS, kind="diagnostics")
    div = int(stats["diverging"].sum())
    diag = dict(divergences=div, min_ess_bulk=float(summ["ess_bulk"].min()),
                min_ess_tail=float(summ["ess_tail"].min()), max_rhat=float(summ["r_hat"].max()),
                worst_rhat=str(summ["r_hat"].idxmax()), worst_ess=str(summ["ess_bulk"].idxmin()),
                chains=args.chains, tune=args.tune, draws=args.draws, adaptation=args.adaptation, target_accept=args.target_accept,
                compile_s=round(t1 - t0, 1), sample_s=round(t2 - t1, 1),
                n_scores=len(df), n_models=len(models), n_benchmarks=len(benches),
                processed_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
                clip=args.clip, sbeta=args.sbeta or args.sbeta_shift, sbeta_shift=args.sbeta_shift, delta_prior=args.delta_prior, xbx_nodes=args.xbx,
                eci_public_commit=ECI_PUBLIC_COMMIT, fitted_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    diag["passes_epoch_checks"] = bool(div == 0 and diag["min_ess_bulk"] >= MIN_ESS
                                       and diag["max_rhat"] <= MAX_RHAT)
    print(json.dumps(diag, indent=1), flush=True)
    (out / "diagnostics.json").write_text(json.dumps(diag, indent=1))

    def stack(name, dim):
        return post[name].stack(s=("chain", "draw")).transpose("s", dim).values

    C, D = stack("capability", "model"), stack("difficulty", "benchmark")
    al, sg = stack("slope", "benchmark"), stack("sigma", "benchmark")
    lo, hi = models.index(DEFAULT_ANCHOR_MODEL_LOW), models.index(DEFAULT_ANCHOR_MODEL_HIGH)
    b = (DEFAULT_ANCHOR_ECI_HIGH - DEFAULT_ANCHOR_ECI_LOW) / (C[:, hi] - C[:, lo])
    a = DEFAULT_ANCHOR_ECI_LOW - b * C[:, lo]
    s = dict(a=a, b=b, eci=a[:, None] + b[:, None] * C, edi=a[:, None] + b[:, None] * D,
             slope=al / b[:, None], sigma=sg)

    def summarise(x):
        return x.mean(0), np.quantile(x, 0.05, axis=0), np.quantile(x, 0.95, axis=0)

    m, l, h = summarise(s["eci"])
    pd.DataFrame({"Model": models, "eci": m, "eci_q05": l, "eci_q95": h}).to_csv(out / "eci.csv", index=False)
    rows = {"benchmark": benches}
    for k in ("edi", "slope", "sigma"):
        rows[k], rows[k + "_q05"], rows[k + "_q95"] = summarise(s[k])
    pd.DataFrame(rows).to_csv(out / "edi.csv", index=False)
    hyp = {k: dict(zip(("mean", "q05", "q95"),
                       (float(v[0]) for v in summarise(post[k].values.reshape(-1, 1)))))
           for k in HYPER}
    (out / "hyper.json").write_text(json.dumps(hyp, indent=1))
    np.savez_compressed(out / "draws.npz", models=np.array(models), benchmarks=np.array(benches),
                        **{k: v.astype(np.float32) for k, v in s.items()},
                        **{"hyper_" + k: post[k].values.reshape(-1).astype(np.float32) for k in HYPER})


if __name__ == "__main__":
    main()
