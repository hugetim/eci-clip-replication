"""XB prior sensitivity, and XBX against XB (Nov 10, 2026 data). For each fit: the exceedance parameter (delta for XB,
nu for XBX), the top models' ECI, Astra - Sol, and Claude Fable 5.2's gap above the reasoning-era frontier trend
(the method of scripts/clip_trend_gap.py: frontier = reasoning models that were best on release by the fit's own
posterior means, o1-mini through Claude Fable 5.1; OLS; gap = Fable 5.2's ECI minus the trend at its release date,
90% interval over the posterior draws with the frontier held fixed).

Usage: python3 scripts/xb_sensitivity.py ECI_DIR OUT_CSV LABEL=FIT_DIR [LABEL=FIT_DIR ...]
  ECI_DIR: an archive's epoch_capabilities_index/ (for release dates). FIT_DIR: scripts/bayes_eci.py --sbeta
  (any --delta-prior) or --xbx fits.
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ECI_DIR, OUT, FITS = Path(sys.argv[1]), sys.argv[2], [a.split("=", 1) for a in sys.argv[3:]]
pub = pd.read_csv(ECI_DIR / "eci_scores.csv", parse_dates=["date"]).set_index("Model")
dates = pub.date
REASONING_EARLY = ["o1-preview", "o1-mini", "o1", "DeepSeek-R1", "Gemini 2.0 Flash Thinking (Jan 2025)", "o3-mini",
                   "Claude 3.7 Sonnet"]
reasoning = (dates > pd.Timestamp("2025-02-27")) | dates.index.isin(REASONING_EARLY)
START, THROUGH, FABLE = pd.Timestamp("2024-09-12"), pd.Timestamp("2026-09-01"), "Claude Fable 5.2"
yr = lambda d: np.asarray((pd.DatetimeIndex(d) - pd.Timestamp("2020-01-01")).days, float) / 365.25
TOP = [FABLE, "GPT-6 Astra", "GPT-6.1 Sol", "Claude Opus 5.5"]


def frontier(eci):
    pool = [m for m in eci.index if m in dates.index and reasoning[m] and START <= dates[m] <= THROUGH]
    d = pd.DataFrame({"eci": eci[pool], "date": dates[pool]}).sort_values(["date", "eci"], ascending=[True, False])
    return list(d[d.eci > d.eci.shift(1).cummax().fillna(-np.inf)].index)


rows = []
for label, fit in FITS:
    z = np.load(Path(fit) / "draws.npz"); diag = json.load(open(Path(fit) / "diagnostics.json"))
    D = pd.DataFrame(z["eci"].astype(float), columns=z["models"])
    par = "nu" if diag.get("xbx_nodes") else "delta"
    p = z["hyper_" + par].astype(float)
    fr = frontier(D.mean())
    X = np.column_stack([np.ones(len(fr)), yr(dates[fr])])
    beta = np.linalg.lstsq(X, D[fr].values.T, rcond=None)[0]               # 2 x draws
    x0 = np.array([1.0, yr([dates[FABLE]])[0]])
    gap = D[FABLE].values - x0 @ beta
    a_s = D["GPT-6 Astra"] - D["GPT-6.1 Sol"]
    rows.append(dict(fit=label, param=par, median=np.median(p), lo=np.quantile(p, 0.05), hi=np.quantile(p, 0.95),
                     passes=diag["passes_epoch_checks"], **{m: D[m].mean() for m in TOP},
                     astra_minus_sol=a_s.mean(), p_astra_ahead=(a_s > 0).mean(), slope=beta[1].mean(),
                     fable_gap=gap.mean(), fable_gap_lo=np.quantile(gap, 0.05), fable_gap_hi=np.quantile(gap, 0.95)))
out = pd.DataFrame(rows)
out.to_csv(OUT, index=False)
print(out.round(4).to_string(index=False))
