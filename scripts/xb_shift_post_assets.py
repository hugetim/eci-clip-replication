"""Numbers for the clip post's bullet on the extended-support beta (XB) with delta tied to sigma_b (bayes_eci.py
--sbeta-shift), Nov 10 data. (The fit-side names, --sbeta and the results/sbeta* folders, predate the post's "XB".)

Usage (repo root): python3 scripts/xb_shift_post_assets.py
Reads results/sbeta_shift_2026-11-10_0823/ (draws.npz; loo_pointwise_c0.001.csv from loo_compare.py --save;
fit_metrics.csv from fit_metrics.py, seed 1) and writes posts/eci-clip/xb_shift.json.
"""
import json
import numpy as np
import pandas as pd

R = "results/sbeta_shift_2026-11-10_0823"
dr = np.load(f"{R}/draws.npz")
benches = list(dr["benchmarks"])
g, dl, sm = (dr[k].astype(float) for k in ("hyper_gamma", "hyper_delta", "hyper_sigma_median"))
sig = dr["sigma"].astype(float)
delta_b = pd.Series(np.minimum(dl[:, None] * (sig / sm[:, None]) ** g[:, None], 0.1).mean(0), index=benches)

lp = pd.read_csv(f"{R}/loo_pointwise_c0.001.csv")
dd = lp["shift"] - lp["sbeta"]
fm = pd.read_csv(f"{R}/fit_metrics.csv").set_index("measure")

NAME = {"epoch": "epoch", "sbeta": "xb", "shift": "shift"}  # input labels (from the fit runs) -> the post's keys
out = dict(
    gamma=float(g.mean()), gamma_lo=float(np.quantile(g, 0.05)), gamma_hi=float(np.quantile(g, 0.95)),
    delta_b_min=float(delta_b.min()), delta_b_min_bench=delta_b.idxmin(),
    delta_b_max=float(delta_b.max()), delta_b_max_bench=delta_b.idxmax(),
    loo_diff=float(dd.sum()), loo_se=float(np.sqrt(len(dd) * dd.var(ddof=0))),
    loo_boundary=float(dd[lp["cat"] != 1].sum()), loo_interior=float(dd[lp["cat"] == 1].sum()),
    **{f"bsrmse_{NAME[f]}": float(fm.loc["beta-scaled RMSE", f"{f}|PSIS-LOO"]) for f in ("epoch", "sbeta", "shift")},
    **{f"ppp_var_{NAME[f]}": float(fm.loc["PPP variance", f"{f}|in sample"]) for f in ("epoch", "sbeta", "shift")},
)
json.dump(out, open("posts/eci-clip/xb_shift.json", "w"), indent=1)
print(json.dumps(out, indent=1))
