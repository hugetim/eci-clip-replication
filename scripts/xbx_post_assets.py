"""Numbers for the clip post's footnote on XBX (Kosmidis and Zeileis's exponential mixture of XB), Nov 10 data:
the mean exceedance nu, GPT-6 Astra - GPT-6.1 Sol, Claude Fable 5.2's gap above the reasoning-era frontier trend
(as in scripts/clip_trend_gap.py: frontier = reasoning models that were best on release by the fit's own posterior
means, o1-mini through Claude Fable 5.1; OLS; gap = Fable 5.2's ECI minus the trend at its release date, averaged
over the posterior draws with the frontier held fixed), and XBX's PSIS-LOO gain over XB.

Usage (repo root): python scripts/xbx_post_assets.py ECI_DIR
Reads results/xbx_2026-11-10_0823/ (draws.npz; loo_pointwise_c0.001.csv from scripts/loo_compare.py --save, labels
epoch / xb / xbx) and writes posts/eci-clip/xbx.json.
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ECI_DIR, R = Path(sys.argv[1]), "results/xbx_2026-11-10_0823"
dates = pd.read_csv(ECI_DIR / "eci_scores.csv", parse_dates=["date"]).set_index("Model").date
REASONING_EARLY = ["o1-preview", "o1-mini", "o1", "DeepSeek-R1", "Gemini 2.0 Flash Thinking (Jan 2025)", "o3-mini",
                   "Claude 3.7 Sonnet"]
reasoning = (dates > pd.Timestamp("2025-02-27")) | dates.index.isin(REASONING_EARLY)
START, THROUGH, FABLE = pd.Timestamp("2024-09-12"), pd.Timestamp("2026-09-01"), "Claude Fable 5.2"
yr = lambda d: np.asarray((pd.DatetimeIndex(d) - pd.Timestamp("2020-01-01")).days, float) / 365.25


def frontier(eci):
    pool = [m for m in eci.index if m in dates.index and reasoning[m] and START <= dates[m] <= THROUGH]
    d = pd.DataFrame({"eci": eci[pool], "date": dates[pool]}).sort_values(["date", "eci"], ascending=[True, False])
    return list(d[d.eci > d.eci.shift(1).cummax().fillna(-np.inf)].index)


z = np.load(f"{R}/draws.npz")
D = pd.DataFrame(z["eci"].astype(float), columns=z["models"])
fr = frontier(D.mean())
X = np.column_stack([np.ones(len(fr)), yr(dates[fr])])
beta = np.linalg.lstsq(X, D[fr].values.T, rcond=None)[0]               # 2 x draws
gap = D[FABLE].values - np.array([1.0, yr([dates[FABLE]])[0]]) @ beta
lp = pd.read_csv(f"{R}/loo_pointwise_c0.001.csv")
dd = lp["xbx"] - lp["xb"]
out = dict(nu=float(np.median(z["hyper_nu"].astype(float))), fable_gap=float(gap.mean()),
           astra_minus_sol=float((D["GPT-6 Astra"] - D["GPT-6.1 Sol"]).mean()),
           loo_diff=float(dd.sum()), loo_se=float(np.sqrt(len(dd) * dd.var(ddof=0))))
json.dump(out, open("posts/eci-clip/xbx.json", "w"), indent=1)
print(json.dumps(out, indent=1))
