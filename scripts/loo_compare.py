"""PSIS-LOO comparison of ECI fits that handle 0% and 100% scores differently, on one common outcome space.

Usage: python3 scripts/loo_compare.py ARCHIVE_EPOCH_DIR EVAL_C LABEL=FIT_DIR [LABEL=FIT_DIR ...]
Fits are scripts/bayes_eci.py outputs: Epoch's clipped beta (any clip), XB (--sbeta, --sbeta-shift) or XBX (--xbx).
The first fit is the reference. --save=CSV also writes each score's elpd under every fit.

Every fit is scored on the same outcomes, set by EVAL_C (c*): a score <= c* is the event "bottom", a score >= 1 - c*
is "top", and any other score is scored by its density. For a beta fit, P(bottom) = P(Y <= c*) and P(top) =
P(Y >= 1 - c*) under its fitted beta; for XB, the same events under its stretched and clamped beta. This is a proper predictive distribution for each
fit, so the comparison is fair whatever clip a fit used; c* only decides how finely boundary outcomes are told apart.
Uses every 4th draw of each of 4 chains.
"""
import json
import sys
import arviz as az
import numpy as np
import pandas as pd
SAVE = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--save=")), None)  # pointwise elpd CSV
sys.argv = [a for a in sys.argv if not a.startswith("--save=")]
from scipy.special import betaln, expit
from scipy.stats import beta as B
from median_vm import beta_params

d = pd.read_csv(f"{sys.argv[1]}/processed_data_for_eci.csv")
c_eval = float(sys.argv[2])
y = d.performance.to_numpy()
cat = np.where(y <= c_eval, 0, np.where(y >= 1 - c_eval, 2, 1))


def pointwise(fit):
    dr = np.load(f"{fit}/draws.npz"); diag = json.load(open(f"{fit}/diagnostics.json"))
    models, benches = list(dr["models"]), list(dr["benchmarks"])
    mi = np.array([models.index(m) for m in d.Model]); bi = np.array([benches.index(b) for b in d.benchmark])
    n = dr["eci"].shape[0]; per = n // 4
    idx = np.concatenate([np.arange(c * per, (c + 1) * per, 4) for c in range(4)])
    eci, edi, sl, sg = (dr[k][idx].astype(float) for k in ("eci", "edi", "slope", "sigma"))
    eta = sl[:, bi] * (eci[:, mi] - edi[:, bi]); sig = sg[:, bi]
    mu = np.clip(expit(eta), 1e-6, 1 - 1e-6)
    a, b = beta_params(dr, diag, idx, bi, sig, sl, mu)  # Epoch's beta, or median-centred / variance-matched (median_vm.py)
    yi = np.clip(y, c_eval, 1 - c_eval)
    dens = (a - 1) * np.log(yi) + (b - 1) * np.log1p(-yi) - betaln(a, b)
    lcdf, lsf = B.logcdf(c_eval, a, b), B.logsf(1 - c_eval, a, b)
    if diag.get("xbx_nodes"):  # Kosmidis-Zeileis XBX: XB mixed over u ~ Exp(mean nu); same quadrature as bayes_eci.py
        from scipy.special import logsumexp
        nu = dr["hyper_nu"][idx].astype(float)[:, None]
        def ll_u(u):  # XB log-likelihood at exceedance u (array like nu), on the c* outcome space
            u = np.minimum(u, 0.1); tr = lambda v: (v + u) / (1 + 2 * u)
            return np.where(cat == 0, B.logcdf(tr(c_eval), a, b), np.where(cat == 2, B.logsf(tr(1 - c_eval), a, b),
                            B.logpdf(tr(np.clip(y, 1e-12, 1 - 1e-12)), a, b) - np.log1p(2 * u)))
        Q, W = np.polynomial.laguerre.laggauss(int(diag["xbx_nodes"]))
        l_in = logsumexp(np.stack([ll_u(nu * q) + np.log(w) for q, w in zip(Q, W)]), axis=0)
        sv = np.linspace(-14.0, 4.0, 32); h = sv[1] - sv[0]
        l_bd = logsumexp(np.stack([ll_u(nu * np.exp(t)) + t - np.exp(t) + np.log(h) for t in sv]), axis=0)
        ll = np.where(cat == 1, l_in, l_bd)
        return ll.reshape(4, -1, len(y))
    if diag.get("sbeta"):  # censored stretched beta (bayes_eci.sbeta_model): Epoch's beta on [-delta, 1 + delta], clamped
        dl = dr["hyper_delta"][idx].astype(float)[:, None]
        if diag.get("sbeta_shift"):  # delta_b = delta (sigma_b / sigma_median)^gamma, capped at 0.1
            g, sm = (dr[k][idx].astype(float)[:, None] for k in ("hyper_gamma", "hyper_sigma_median"))
            dl = np.minimum(dl * (sig / sm) ** g, 0.1)
        tr = lambda v: (v + dl) / (1 + 2 * dl)
        ll = np.where(cat == 0, B.logcdf(tr(c_eval), a, b), np.where(cat == 2, B.logsf(tr(1 - c_eval), a, b),
                     B.logpdf(tr(y), a, b) - np.log1p(2 * dl)))
        return ll.reshape(4, -1, len(y))
    ll = np.where(cat == 0, lcdf, np.where(cat == 2, lsf, dens))  # Epoch's clipped beta, any clip
    return ll.reshape(4, -1, len(y))


res, ks = {}, {}
print(f"evaluation threshold c* = {c_eval}: {int((cat == 0).sum())} bottom, {int((cat == 2).sum())} top, {int((cat == 1).sum())} interior")
for arg in sys.argv[3:]:
    lab, fit = arg.split("=", 1)
    ll = pointwise(fit)
    loo = az.loo(az.from_dict({"posterior": {"dummy": np.zeros(ll.shape[:2])}, "log_likelihood": {"y": ll}}), pointwise=True)
    res[lab], ks[lab] = loo.elpd_i.values, loo.pareto_k.values
bad = np.zeros(len(y), bool)
for v in ks.values():
    bad |= v > 0.7
ref = list(res)[0]
print(f"{bad.sum()} points have Pareto k > 0.7 in at least one fit ({(bad & (cat != 1)).sum()} at the boundary); "
      "'reliable' columns drop them for every fit\n")
rows = []
for lab, p in res.items():
    dd = p - res[ref]
    se = lambda x: np.sqrt(len(x) * x.var()) if len(x) else 0.0
    r = dict(fit=lab, elpd=p.sum(), boundary=p[cat != 1].sum(), interior=p[cat == 1].sum(),
             diff=dd.sum(), diff_se=se(dd), diff_boundary=dd[cat != 1].sum(), diff_interior=dd[cat == 1].sum(),
             diff_reliable=dd[~bad].sum(), diff_reliable_se=se(dd[~bad]))
    rows.append(r)
t = pd.DataFrame(rows).set_index("fit")
print(f"elpd_loo and differences from {ref}:")
print(t.round(1).to_string())
if SAVE:
    pd.DataFrame({"Model": d.Model, "benchmark": d.benchmark, "performance": y, "cat": cat, **res}).to_csv(SAVE, index=False)
