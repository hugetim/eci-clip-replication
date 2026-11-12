"""XB prior sensitivity, and XBX against XB (Nov 10, 2026 data). For each fit: the exceedance parameter (delta for XB,
nu for XBX), the top models' ECI, Astra - Sol, and Claude Fable 5.2's gap above the reasoning-era frontier trend
(scripts/trend.py: frontier chosen by the fit's own posterior means, the trend refit in each posterior draw with the
frontier held fixed; the gap's mean and 90% interval over draws).

Usage: python3 scripts/xb_sensitivity.py ECI_DIR OUT_CSV LABEL=FIT_DIR [LABEL=FIT_DIR ...]
  ECI_DIR: an archive's epoch_capabilities_index/ (for release dates). FIT_DIR: scripts/bayes_eci.py --sbeta
  (any --delta-prior) or --xbx fits.
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import trend
from trend import FABLE

ECI_DIR, OUT, FITS = sys.argv[1], sys.argv[2], [a.split("=", 1) for a in sys.argv[3:]]
dates = trend.release_dates(ECI_DIR)
ASTRA, SOL = "GPT-6 Astra", "GPT-6.1 Sol"
TOP = [FABLE, ASTRA, SOL, "Claude Opus 5.5"]

rows = []
for label, fit in FITS:
    z = np.load(Path(fit) / "draws.npz")
    diag = json.load(open(Path(fit) / "diagnostics.json"))
    draws = pd.DataFrame(z["eci"].astype(float), columns=z["models"])
    param = "nu" if diag.get("xbx_nodes") else "delta"
    exceedance = z["hyper_" + param].astype(float)
    line = trend.fit_line(draws, trend.frontier(draws.mean(), dates), dates)  # (intercept, slope), one per draw
    gap = trend.gap_above_trend(draws, line, dates)
    astra_lead = draws[ASTRA] - draws[SOL]
    rows.append(dict(fit=label, param=param, median=np.median(exceedance), lo=np.quantile(exceedance, 0.05),
                     hi=np.quantile(exceedance, 0.95), passes=diag["passes_epoch_checks"],
                     **{m: draws[m].mean() for m in TOP},
                     astra_minus_sol=astra_lead.mean(), p_astra_ahead=(astra_lead > 0).mean(), slope=line[1].mean(),
                     fable_gap=gap.mean(), fable_gap_lo=np.quantile(gap, 0.05), fable_gap_hi=np.quantile(gap, 0.95)))
out = pd.DataFrame(rows)
out.to_csv(OUT, index=False)
print(out.round(4).to_string(index=False))
