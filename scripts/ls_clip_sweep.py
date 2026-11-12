"""Epoch's original (pre-Bayesian, least-squares) ECI, refit on the models and benchmarks of the Nov 10, 2026 Bayesian
ECI at each clip used in the clip post: Claude Fable 5.2's gap above the reasoning-era frontier trend, and GPT-6 Astra
minus GPT-6.1 Sol.

Fit: fit.py from Epoch's benchmark-stitching repo (logistic curve, nonlinear least squares, ridge 0.1, Winogrande
anchored), on the archive's processed_data_for_eci.csv with scores clipped to [clip, 1 - clip] (the published
least-squares ECI used 0.001); rescaled so Claude 3.5 Sonnet = 130 and GPT-5 = 150. Point estimates only.
Trend: as in scripts/clip_trend_gap.py (OLS through reasoning models that were SOTA among reasoning models at release,
o1-mini through Claude Fable 5.1, frontier selected on each fit's own estimates); gap = ECI - trend prediction.

Usage (repo root, benchmark-stitching/ cloned there by setup.sh): python scripts/ls_clip_sweep.py ECI_DIR OUT_CSV
"""
import contextlib, io, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

ECI_DIR, OUT = Path(sys.argv[1]), sys.argv[2]
REPO = os.path.join(os.getcwd(), "benchmark-stitching")
CLIPS = [0.01, 0.001, 0.0001, 1e-05, 1e-06, 1e-08, 1e-10, 1e-12]
ANCHOR_LO, ANCHOR_HI = ("Claude 3.5 Sonnet", 130.0), ("GPT-5", 150.0)
FABLE, ASTRA, SOL = "Claude Fable 5.2", "GPT-6 Astra", "GPT-6.1 Sol"

sys.path.insert(0, REPO)
cwd = os.getcwd(); os.chdir(REPO)
try:
    with contextlib.redirect_stdout(io.StringIO()):
        from fit import fit_statistical_model
finally:
    os.chdir(cwd)

scores = pd.read_csv(ECI_DIR / "processed_data_for_eci.csv").assign(model=lambda d: d.Model)
dates = pd.read_csv(ECI_DIR / "eci_scores.csv", parse_dates=["date"]).set_index("Model").date
REASONING_EARLY = ["o1-preview", "o1-mini", "o1", "DeepSeek-R1", "Gemini 2.0 Flash Thinking (Jan 2025)",
                   "o3-mini", "Claude 3.7 Sonnet"]
reasoning = (dates > pd.Timestamp("2025-02-27")) | dates.index.isin(REASONING_EARLY)
THROUGH, START = pd.Timestamp("2026-09-01"), pd.Timestamp("2024-09-12")
yr = lambda d: (d - pd.Timestamp("2020-01-01")).days / 365.25 if isinstance(d, pd.Timestamp) else \
    (d - pd.Timestamp("2020-01-01")).dt.days.values / 365.25


def ls_eci(clip):
    """Least-squares ECI of every model, scores clipped to [clip, 1 - clip]."""
    df = scores.assign(performance=scores.performance.clip(clip, 1 - clip))
    with contextlib.redirect_stdout(io.StringIO()):
        _, cm, _ = fit_statistical_model(df, anchor_mode="benchmark", anchor_benchmark="Winogrande",
                                         anchor_difficulty=0, anchor_slope=1, regularization_strength=0.1,
                                         df_model=None)
    a = cm.set_index("model").estimated_capability
    lo, hi = a[ANCHOR_LO[0]], a[ANCHOR_HI[0]]
    return ANCHOR_LO[1] + (a - lo) * (ANCHOR_HI[1] - ANCHOR_LO[1]) / (hi - lo)


def fable_gap(e):
    pool = [m for m in e.index if m in dates.index and reasoning[m] and START <= dates[m] <= THROUGH]
    d = pd.DataFrame({"eci": e[pool], "date": dates[pool]}).sort_values(["date", "eci"], ascending=[True, False])
    fr = list(d[d.eci > d.eci.shift(1).cummax().fillna(-np.inf)].index)
    t = yr(dates[fr])
    beta = np.linalg.lstsq(np.column_stack([np.ones_like(t), t]), e[fr].values, rcond=None)[0]
    return e[FABLE] - np.array([1, yr(dates[FABLE])]) @ beta


rows = []
for c in CLIPS:
    e = ls_eci(c)
    rows.append({"clip": c, "fable_gap": fable_gap(e), "astra_minus_sol": e[ASTRA] - e[SOL]})
    print(rows[-1], flush=True)
out = pd.DataFrame(rows)
out.round(3).assign(clip=out["clip"]).to_csv(OUT, index=False)  # clip unrounded (1e-12 etc.)
