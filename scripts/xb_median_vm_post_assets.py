"""Numbers for the clip post's last footnote: XB with median-centring and variance-matched capability noise
(bayes_eci.py --sbeta --median-centre --var-match; see median_vm.py), Nov 10 data.

Usage (repo root): python3 scripts/xb_median_vm_post_assets.py
Reads results/xb_median_vm_2026-11-10_0823/ (draws.npz, edi.csv, eci.csv; loo_pointwise_c0.001.csv from
loo_compare.py --save, labels epoch / xb / vm; fit_metrics.csv from fit_metrics.py, seed 1, labels epoch / xb / vm,
so the epoch and xb columns repeat the post's fit_metrics.csv) and results/sbeta_2026-11-10_0823/edi.csv, and writes
posts/eci-clip/xb_vm.json.
"""
import json
import numpy as np
import pandas as pd

FIT, XB = "results/xb_median_vm_2026-11-10_0823", "results/sbeta_2026-11-10_0823"
dr = np.load(f"{FIT}/draws.npz")
# The exceedance delta, and where the latent median reaches 100%: median-centring makes the curve the latent beta's
# median, so the median score -delta + (1 + 2 delta) mu reaches 1 when slope (ECI - difficulty) = log((1 + delta) / delta)
delta = dr["hyper_delta"].astype(float)
reach = np.log((1 + np.median(delta)) / np.median(delta)) / float(np.median(np.median(dr["slope"], 0)))
tau = dr["hyper_tau_cap"].astype(float) * dr["b"].astype(float)  # capability noise sd, raw scale -> ECI points

elpd = pd.read_csv(f"{FIT}/loo_pointwise_c0.001.csv")  # each score's log predictive density under each fit
at_boundary = elpd["cat"] != 1                          # scores of 0% or 100%
gain_xb, gain_epoch = elpd["vm"] - elpd["xb"], elpd["vm"] - elpd["epoch"]
se = lambda g: float(np.sqrt(len(g) * g.var(ddof=0)))
metrics = pd.read_csv(f"{FIT}/fit_metrics.csv").set_index("measure")
FITS = ("epoch", "xb", "vm")

out = dict(
    delta=float(np.median(delta)), delta_lo=float(np.quantile(delta, 0.05)), delta_hi=float(np.quantile(delta, 0.95)),
    reach_eci_median=float(reach),
    tau_eci=float(tau.mean()), tau_eci_lo=float(np.quantile(tau, 0.05)), tau_eci_hi=float(np.quantile(tau, 0.95)),
    sigma_median=float(pd.read_csv(f"{FIT}/edi.csv").sigma.median()),
    sigma_median_xb=float(pd.read_csv(f"{XB}/edi.csv").sigma.median()),
    loo_diff_xb=float(gain_xb.sum()), loo_se_xb=se(gain_xb),
    loo_boundary_xb=float(gain_xb[at_boundary].sum()), loo_interior_xb=float(gain_xb[~at_boundary].sum()),
    loo_diff_epoch=float(gain_epoch.sum()), loo_se_epoch=se(gain_epoch),
    **{f"bsrmse_{f}": float(metrics.loc["beta-scaled RMSE", f"{f}|PSIS-LOO"]) for f in FITS},
    fable_eci=float(pd.read_csv(f"{FIT}/eci.csv").set_index("Model").eci["Claude Fable 5.2"]),
    loo_measures={m: float(v) for m, v in metrics["vm|PSIS-LOO"].dropna().items()},  # the post's table of other measures
)
json.dump(out, open("posts/eci-clip/xb_vm.json", "w"), indent=1)
print(json.dumps(out, indent=1))
