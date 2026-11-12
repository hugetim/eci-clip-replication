"""Numbers for the clip post's footnote on the extended-support beta (XB) with delta tied to sigma_b (bayes_eci.py
--sbeta-shift), Nov 10 data. (The fit-side names, --sbeta and the results/sbeta* folders, predate the post's "XB".)

Usage (repo root): python3 scripts/xb_shift_post_assets.py
Reads results/sbeta_shift_2026-11-10_0823/ (draws.npz; loo_pointwise_c0.001.csv from loo_compare.py --save, labels
xb / shift; fit_metrics.csv from fit_metrics.py, seed 1, labels epoch / xb / shift) and writes
posts/eci-clip/xb_shift.json.
"""
import json
import numpy as np
import pandas as pd

FIT = "results/sbeta_shift_2026-11-10_0823"
dr = np.load(f"{FIT}/draws.npz")
gamma, delta, sigma_median = (dr[k].astype(float)[:, None] for k in ("hyper_gamma", "hyper_delta", "hyper_sigma_median"))
sigma = dr["sigma"].astype(float)  # draws x benchmarks
# each benchmark's delta_b = delta (sigma_b / sigma_median)^gamma, capped at 0.1 as in the fit: posterior mean
delta_b = pd.Series(np.minimum(delta * (sigma / sigma_median) ** gamma, 0.1).mean(0), index=list(dr["benchmarks"]))

elpd = pd.read_csv(f"{FIT}/loo_pointwise_c0.001.csv")  # each score's log predictive density under XB and the shifted XB
gain = elpd["shift"] - elpd["xb"]
at_boundary = elpd["cat"] != 1                          # scores of 0% or 100%
metrics = pd.read_csv(f"{FIT}/fit_metrics.csv").set_index("measure")
FITS = ("epoch", "xb", "shift")

out = dict(
    gamma=float(gamma.mean()), gamma_lo=float(np.quantile(gamma, 0.05)), gamma_hi=float(np.quantile(gamma, 0.95)),
    delta_b_min=float(delta_b.min()), delta_b_min_bench=delta_b.idxmin(),
    delta_b_max=float(delta_b.max()), delta_b_max_bench=delta_b.idxmax(),
    loo_diff=float(gain.sum()), loo_se=float(np.sqrt(len(gain) * gain.var(ddof=0))),
    loo_boundary=float(gain[at_boundary].sum()), loo_interior=float(gain[~at_boundary].sum()),
    **{f"bsrmse_{f}": float(metrics.loc["beta-scaled RMSE", f"{f}|PSIS-LOO"]) for f in FITS},
    **{f"ppp_var_{f}": float(metrics.loc["PPP variance", f"{f}|in sample"]) for f in FITS},
)
json.dump(out, open("posts/eci-clip/xb_shift.json", "w"), indent=1)
print(json.dumps(out, indent=1))
