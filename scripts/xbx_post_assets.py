"""Numbers for the clip post's footnote on XBX (Kosmidis and Zeileis's exponential mixture of XB), Nov 10 data:
the mean exceedance nu, GPT-6 Astra - GPT-6.1 Sol, Claude Fable 5.2's gap above the reasoning-era frontier trend
(scripts/trend.py: frontier chosen by the fit's own posterior means, the trend refit in each posterior draw with the
frontier held fixed, and the gap averaged over draws), and XBX's PSIS-LOO gain over XB.

Usage (repo root): python scripts/xbx_post_assets.py ECI_DIR
Reads results/xbx_2026-11-10_0823/ (draws.npz; loo_pointwise_c0.001.csv from scripts/loo_compare.py --save, labels
xb / xbx) and writes posts/eci-clip/xbx.json.
"""
import json
import sys
import numpy as np
import pandas as pd
import trend

ECI_DIR, FIT = sys.argv[1], "results/xbx_2026-11-10_0823"
dates = trend.release_dates(ECI_DIR)

z = np.load(f"{FIT}/draws.npz")
draws = pd.DataFrame(z["eci"].astype(float), columns=z["models"])
models = trend.frontier(draws.mean(), dates)
gap = trend.gap_above_trend(draws, trend.fit_line(draws, models, dates), dates)  # one per draw

elpd = pd.read_csv(f"{FIT}/loo_pointwise_c0.001.csv")  # each score's log predictive density under XB and XBX
gain = elpd["xbx"] - elpd["xb"]
out = dict(nu=float(np.median(z["hyper_nu"].astype(float))), fable_gap=float(gap.mean()),
           astra_minus_sol=float((draws["GPT-6 Astra"] - draws["GPT-6.1 Sol"]).mean()),
           loo_diff=float(gain.sum()), loo_se=float(np.sqrt(len(gain) * gain.var(ddof=0))))
json.dump(out, open("posts/eci-clip/xbx.json", "w"), indent=1)
print(json.dumps(out, indent=1))
