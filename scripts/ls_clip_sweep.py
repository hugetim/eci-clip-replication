"""Epoch's original (pre-Bayesian, least-squares) ECI, refit on the models and benchmarks of the Nov 10, 2026 Bayesian
ECI at each clip used in the clip post: Claude Fable 5.2's gap above the reasoning-era frontier trend, and GPT-6 Astra
minus GPT-6.1 Sol.

Fit: fit.py from Epoch's benchmark-stitching repo (logistic curve, nonlinear least squares, ridge 0.1, Winogrande
anchored), on the archive's processed_data_for_eci.csv with scores clipped to [clip, 1 - clip] (the published
least-squares ECI used 0.001); rescaled so Claude 3.5 Sonnet = 130 and GPT-5 = 150. Point estimates only.
Trend: scripts/trend.py, with the frontier selected on each fit's own estimates; gap = ECI - trend prediction.

Usage (repo root, benchmark-stitching/ cloned there by setup.sh): python scripts/ls_clip_sweep.py ECI_DIR OUT_CSV
"""
import contextlib
import io
import os
import sys
from pathlib import Path
import pandas as pd
import trend

ECI_DIR, OUT = Path(sys.argv[1]), sys.argv[2]
REPO = os.path.join(os.getcwd(), "benchmark-stitching")
CLIPS = [0.01, 0.001, 0.0001, 1e-05, 1e-06, 1e-08, 1e-10, 1e-12]
ANCHOR_LO, ANCHOR_HI = ("Claude 3.5 Sonnet", 130.0), ("GPT-5", 150.0)
ASTRA, SOL = "GPT-6 Astra", "GPT-6.1 Sol"

# Epoch's fit.py reads files relative to its own folder and prints on import
sys.path.insert(0, REPO)
cwd = os.getcwd()
os.chdir(REPO)
try:
    with contextlib.redirect_stdout(io.StringIO()):
        from fit import fit_statistical_model
finally:
    os.chdir(cwd)

scores = pd.read_csv(ECI_DIR / "processed_data_for_eci.csv").assign(model=lambda d: d.Model)
dates = trend.release_dates(ECI_DIR)


def ls_eci(clip):
    """Least-squares ECI of every model, scores clipped to [clip, 1 - clip]."""
    clipped = scores.assign(performance=scores.performance.clip(clip, 1 - clip))
    with contextlib.redirect_stdout(io.StringIO()):
        _, capabilities, _ = fit_statistical_model(clipped, anchor_mode="benchmark", anchor_benchmark="Winogrande",
                                                   anchor_difficulty=0, anchor_slope=1, regularization_strength=0.1,
                                                   df_model=None)
    raw = capabilities.set_index("model").estimated_capability
    (lo_model, lo_eci), (hi_model, hi_eci) = ANCHOR_LO, ANCHOR_HI
    return lo_eci + (raw - raw[lo_model]) * (hi_eci - lo_eci) / (raw[hi_model] - raw[lo_model])


rows = []
for clip in CLIPS:
    eci = ls_eci(clip)
    line = trend.fit_line(eci, trend.frontier(eci, dates), dates)
    rows.append({"clip": clip, "fable_gap": trend.gap_above_trend(eci, line, dates),
                 "astra_minus_sol": eci[ASTRA] - eci[SOL]})
    print(rows[-1], flush=True)
out = pd.DataFrame(rows)
out.round(3).assign(clip=out["clip"]).to_csv(OUT, index=False)  # clip unrounded (1e-12 etc.)
