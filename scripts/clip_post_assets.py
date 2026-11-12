"""Data for the clip post (posts/eci-clip/index.qmd): the clip refits and XB. (Fable 5.2's trend gap is in
scripts/clip_trend_gap.py, XB with delta tied to sigma_b in xb_shift_post_assets.py, XBX in xbx_post_assets.py.)

Usage: python scripts/clip_post_assets.py ECI_DIR CLIP_REFITS_DIR XB_FIT OUT_DIR QUESTION_COUNTS_CSV [DATA_DATE]
  CLIP_REFITS_DIR: scripts/bayes_eci.py fits of Epoch's model at several clips: base (0.1%), clip_0.01, clip_0.0001, ...
  XB_FIT: the extended-support beta, XB (bayes_eci.py --sbeta; the fit-side names predate "XB"); the folder also holds
    loo_pointwise_c0.001.csv (scripts/loo_compare.py --save, labels epoch / xb) and fit_metrics.csv
    (scripts/fit_metrics.py --csv, labels epoch / xb).
  QUESTION_COUNTS_CSV: binom/benchmark_question_counts.csv (for FrontierMath Tier 4's question count).
Writes OUT_DIR/clip_table.csv, bench_sigma.csv, fit_metrics.csv, post_stats.json.
"""
import json
import shutil
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ECI_DIR, CLIP_REFITS, XB_FIT, OUT, QUESTION_COUNTS = (Path(x) for x in sys.argv[1:6])
DATA_DATE = sys.argv[6] if len(sys.argv) > 6 else ""  # recorded in post_stats.json
OUT.mkdir(parents=True, exist_ok=True)
scores = pd.read_csv(ECI_DIR / "processed_data_for_eci.csv")
ASTRA, SOL = "GPT-6 Astra", "GPT-6.1 Sol"
FM4 = "FrontierMath-Tier-4-v2-Private"
FRONTIER_ECI = 150  # "frontier models" below: at or above this under Epoch's 0.1% clip


def load(fit_dir):
    """A fit's posterior draws (bayes_eci.py's draws.npz): eci and sigma are draws x names."""
    return np.load(fit_dir / "draws.npz")


def posterior_mean(draws, key, names):
    """Posterior mean of every model's ECI (key "eci", names "models") or benchmark's sigma_b ("sigma", "benchmarks")."""
    return pd.Series(draws[key].mean(0), index=draws[names])


# The fits, in the tables' row order: Epoch's model at each clip (loosest first), then XB
clip_dirs = [d for d in CLIP_REFITS.iterdir() if (d / "draws.npz").exists()]
clip_of = {d: json.load(open(d / "diagnostics.json"))["clip"] for d in clip_dirs}
fits = [("clip", clip_of[d], d) for d in sorted(clip_dirs, key=lambda d: -clip_of[d])] + [("XB", np.nan, XB_FIT)]
base_eci = posterior_mean(load(CLIP_REFITS / "base"), "eci", "models")  # Epoch's 0.1% clip
frontier = base_eci.index[base_eci >= FRONTIER_ECI]

# 1. clip_table.csv: Astra vs 6.1 Sol, and how far frontier models' ECIs move from Epoch's 0.1% clip
# 2. bench_sigma.csv: each benchmark's fitted scatter sigma_b under each fit, with its count of 0% scores
table, sigma_b = [], {}
for kind, clip, fit_dir in fits:
    draws = load(fit_dir)
    models = list(draws["models"])
    astra_lead = (draws["eci"][:, models.index(ASTRA)].astype(float) - draws["eci"][:, models.index(SOL)].astype(float))
    moved = (posterior_mean(draws, "eci", "models") - base_eci)[frontier].abs()
    table.append(dict(model=kind, clip=clip, astra_minus_sol=astra_lead.mean(), p_astra_over_sol=np.mean(astra_lead > 0),
                      frontier_mean_abs_change=moved.mean(), frontier_max_abs_change=moved.max()))
    sigma_b["xb" if kind == "XB" else f"clip_{clip:g}"] = posterior_mean(draws, "sigma", "benchmarks")
table = pd.DataFrame(table)
table.to_csv(OUT / "clip_table.csv", index=False)
sigma_b = pd.DataFrame(sigma_b)
zeros = scores[scores.performance <= 0].groupby("benchmark").size()
sigma_b["zeros"] = zeros.reindex(sigma_b.index).fillna(0).astype(int)
sigma_b.rename_axis("benchmark").to_csv(OUT / "bench_sigma.csv")

# 3. XB's exceedance delta, and where its mean curve reaches 100%. The mean curve is -delta + (1 + 2 delta) mu with
# mu = sigmoid(slope (ECI - difficulty)), so it reaches 1 when slope (ECI - difficulty) = log((1 + delta) / delta).
xb_draws = load(XB_FIT)
delta = xb_draws["hyper_delta"].astype(float)
median_slope = float(np.median(np.median(xb_draws["slope"], 0)))  # the median benchmark's
reach = np.log((1 + np.median(delta)) / np.median(delta)) / median_slope  # ECI points above a benchmark's 50% point

# 4. Fit: PSIS-LOO on the common outcome space (c* = 0.1%; see loo_compare.py), and Epoch's reported measures
elpd = pd.read_csv(XB_FIT / "loo_pointwise_c0.001.csv")  # each score's log predictive density under each model
gain = elpd["xb"] - elpd["epoch"]
at_boundary = elpd.cat != 1                              # scores of 0% or 100%
shutil.copyfile(XB_FIT / "fit_metrics.csv", OUT / "fit_metrics.csv")

fm4 = scores[scores.benchmark == FM4].set_index("Model").performance
stats = dict(
    data_date=DATA_DATE, n_scores=len(scores), n_zero=int((scores.performance <= 0).sum()),
    fm4_questions=int(pd.read_csv(QUESTION_COUNTS).set_index("benchmark").n_used[FM4]),
    fm4_sol=float(fm4[SOL]), fm4_astra=float(fm4[ASTRA]),
    delta=float(np.median(delta)), delta_lo=float(np.quantile(delta, 0.05)), delta_hi=float(np.quantile(delta, 0.95)),
    reach_eci_median=float(reach),
    loo_diff=float(gain.sum()), loo_se=float(np.sqrt(len(gain) * gain.var())),
    loo_boundary=float(gain[at_boundary].sum()), loo_interior=float(gain[~at_boundary].sum()),
)
json.dump(stats, open(OUT / "post_stats.json", "w"), indent=1)
print(table.round(3).to_string())
print(json.dumps(stats, indent=1))
