"""How far Claude Fable 5.2 sits above the reasoning-era frontier trend, under Epoch's published ECI and under each
refit (clips, the extended-support beta XB). Replicates Epoch's Nov 9, 2026 tweet chart (the trend is defined in
scripts/trend.py); Epoch reports +16 ECI/yr (90% CI 14 to 18) and a 90% prediction interval ribbon. The "Epoch
published" row checks that this reproduces Epoch's chart.

Per source: the trend's slope; its prediction at Fable 5.2's release date; gap = ECI - prediction; gap_lo/gap_hi:
5th/95th percentiles of the gap over draws (Epoch's 500 bootstrap refits for the published ECI, posterior draws for
the refits), with the trend refit in each draw and frontier membership held fixed; pi_lo/pi_hi: the trend's 90%
prediction interval at that date, relative to the prediction (see prediction_band).
Usage: python scripts/clip_trend_gap.py ECI_DIR FIT_ROOT OUT_CSV [LINES_CSV]
  ECI_DIR: an archive's epoch_capabilities_index/ (eci_scores.csv, eci_bootstraps.json)
  FIT_ROOT: results/clip_refits_<stamp> (one folder per fit, each with draws.npz); siblings sbeta_<stamp> (XB) and
  xb_median_vm_<stamp> (XB+) are picked up if present.
  LINES_CSV (optional): each source's trend line and 90% prediction band on a 5-day grid (Mar 1 to Dec 20, 2026), as
  days from Fable 5.2's release and ECI relative to that source's own prediction there; the band's edges lo and hi
  are relative to the line (for the clip post's figure, posts/eci-clip/fable_gap_lines.csv).
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm
import trend
from trend import FABLE

ECI_DIR, FIT_ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
LINES = sys.argv[4] if len(sys.argv) > 4 else None
GRID = pd.date_range("2026-03-01", "2026-12-20", freq="5D")
ROW_ORDER = ["Epoch published", "clip_0.01", "base", "clip_0.0001", "clip_1e-05", "clip_1e-06", "clip_1e-08",
             "clip_1e-10", "clip_1e-12", "XB", "XB+"]

published = pd.read_csv(ECI_DIR / "eci_scores.csv", parse_dates=["date"]).set_index("Model")
dates = published.date


def prediction_band(draws, models, line, when):
    """The trend's 90% prediction interval at each date, as (lower, upper) distances from `line`: Epoch's ribbon.

    Each draw (a bootstrap refit or a posterior draw of every model's ECI) gives its own least-squares line through
    `models` and its own residual standard deviation around that line. A new frontier model's ECI is predicted, per
    draw, as that draw's line plus normal noise with that draw's standard deviation; the interval runs from the 5th to
    the 95th percentile of the resulting mixture over draws. So it reflects both the uncertainty in the models' ECIs
    and the scatter of frontier models around the trend. Epoch does not document its ribbon; on Epoch's published
    ECI and bootstraps, this matches the ribbon in its Nov 9, 2026 chart to about 0.1 ECI on both edges. (The
    classic OLS prediction interval from the point estimates alone is much narrower, about +-2.4 against -3.9/+4.2
    at Fable 5.2's release.)"""
    X = trend.design(dates[models])
    draw_lines = trend.fit_line(draws, models, dates)
    residuals = draws[models].values - (X @ draw_lines).T                      # draws x models
    residual_sd = np.sqrt((residuals ** 2).sum(1) / (len(models) - 2))        # one per draw
    offsets = trend.predict(draw_lines, when) - trend.predict(line, when)[:, None]  # dates x draws

    def quantile(p, offset):
        return brentq(lambda v: norm.cdf((v - offset) / residual_sd).mean() - p, -100, 100)

    return (np.array([quantile(0.05, offset) for offset in offsets]),
            np.array([quantile(0.95, offset) for offset in offsets]))


def analyse(source, eci, draws):
    """One source's row of OUT_CSV, its trend line on GRID (for LINES_CSV), and its frontier.
    eci: the source's ECI estimates (a Series by model); draws: its draws (a DataFrame, one column per model)."""
    models = trend.frontier(eci, dates)
    line = trend.fit_line(eci, models, dates)
    prediction = trend.predict(line, [dates[FABLE]])[0]
    gap_draws = trend.gap_above_trend(draws, trend.fit_line(draws, models, dates), dates)
    pi_lo, pi_hi = prediction_band(draws, models, line, [dates[FABLE]])
    row = {"source": source, "slope": line[1], "5.2_eci": eci[FABLE], "5.2_pred": prediction,
           "5.2_gap": eci[FABLE] - prediction,
           "5.2_gap_lo": np.percentile(gap_draws, 5), "5.2_gap_hi": np.percentile(gap_draws, 95),
           "5.2_pi_lo": pi_lo[0], "5.2_pi_hi": pi_hi[0]}
    band_lo, band_hi = prediction_band(draws, models, line, GRID)
    on_grid = pd.DataFrame({"source": source, "days": (GRID - dates[FABLE]).days,
                            "y": trend.predict(line, GRID) - prediction, "lo": band_lo, "hi": band_hi})
    return row, on_grid, models


def fit_draws(fit_dir):
    """A refit's posterior draws of every model's ECI."""
    z = np.load(fit_dir / "draws.npz")
    return pd.DataFrame(z["eci"], columns=z["models"])


# Epoch's published ECI with its bootstrap refits, then each refit with its posterior draws
boot = json.load(open(ECI_DIR / "eci_bootstraps.json"))["model"]
sources = [("Epoch published", published.eci, pd.DataFrame(np.array(boot["capability_samples"]), columns=boot["names"]))]
fit_dirs = [(p.name, p) for p in sorted(FIT_ROOT.iterdir()) if (p / "draws.npz").exists()]
xb_dir = FIT_ROOT.parent / FIT_ROOT.name.replace("clip_refits", "sbeta")
if (xb_dir / "draws.npz").exists():
    fit_dirs.append(("XB", xb_dir))
xbp_dir = FIT_ROOT.parent / FIT_ROOT.name.replace("clip_refits", "xb_median_vm")
if (xbp_dir / "draws.npz").exists():
    fit_dirs.append(("XB+", xbp_dir))
for name, fit_dir in fit_dirs:
    draws = fit_draws(fit_dir)
    sources.append((name, draws.mean(), draws))

rows, lines, frontiers = [], [], {}
for source, eci, draws in sources:
    row, on_grid, frontiers[source] = analyse(source, eci, draws)
    rows.append(row)
    lines.append(on_grid)
    if frontiers[source] != frontiers["Epoch published"]:
        ours, theirs = set(frontiers[source]), set(frontiers["Epoch published"])
        print(f"{source}: frontier differs from published: +{sorted(ours - theirs)} -{sorted(theirs - ours)}")

out = pd.DataFrame(rows).sort_values("source", key=lambda s: s.map(ROW_ORDER.index)).round(3)
out.to_csv(OUT, index=False)
if LINES:
    pd.concat(lines).round(4).to_csv(LINES, index=False)
print("frontier (published):", ", ".join(frontiers["Epoch published"]))
print(out.round(2).to_string(index=False))
