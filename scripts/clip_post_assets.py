"""Data for the clip post (posts/eci-clip/index.qmd): the clip refits and XB. (Fable 5.2's trend gap is in
scripts/clip_trend_gap.py, XB with delta tied to sigma_b in xb_shift_post_assets.py, XBX in xbx_post_assets.py.)

Usage: python scripts/clip_post_assets.py ECI_DIR CLIP_REFITS_DIR XB_FIT OUT_DIR QUESTION_COUNTS_CSV [DATA_DATE]
  CLIP_REFITS_DIR: scripts/bayes_eci.py fits of Epoch's model at several clips: base (0.1%), clip_0.01, clip_0.0001, ...
  XB_FIT: the extended-support beta, XB (bayes_eci.py --sbeta; the fit-side names predate "XB"); the folder also holds
    loo_pointwise_c0.001.csv (scripts/loo_compare.py --save) and fit_metrics.csv (scripts/fit_metrics.py --csv).
  QUESTION_COUNTS_CSV: binom/benchmark_question_counts.csv (for FrontierMath Tier 4's question count).
Writes OUT_DIR/clip_table.csv, bench_sigma.csv, fit_metrics.csv, post_stats.json.
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

D, CR, SB, OUT, QC = (Path(x) for x in sys.argv[1:6])
DATA_DATE = sys.argv[6] if len(sys.argv) > 6 else ""  # recorded in post_stats.json
OUT.mkdir(parents=True, exist_ok=True)
d = pd.read_csv(D / "processed_data_for_eci.csv")
A, S, O = "GPT-6 Astra", "GPT-6.1 Sol", "Claude Opus 5.5"
FM4 = "FrontierMath-Tier-4-v2-Private"


def load(fit):
    return np.load(Path(fit) / "draws.npz"), json.load(open(Path(fit) / "diagnostics.json"))


# 1. Astra vs 6.1 Sol and 6.1 Sol vs Opus 5.5 by clip, and under XB
fits = [(json.load(open(CR / f / "diagnostics.json"))["clip"], CR / f) for f in sorted(p.name for p in CR.iterdir())
        if (CR / f / "draws.npz").exists()]
rows = [("clip", c, fit) for c, fit in sorted(fits, key=lambda t: -t[0])] + [("XB", np.nan, SB)]
tab = []
for kind, c, fit in rows:
    dr, diag = load(fit); m = list(dr["models"]); e = lambda x: dr["eci"][:, m.index(x)].astype(float)
    a_s, s_o = e(A) - e(S), e(S) - e(O)
    tab.append(dict(model=kind, clip=c, astra_minus_sol=a_s.mean(), p_astra_over_sol=np.mean(a_s > 0),
                    p_sol_over_opus=np.mean(s_o > 0)))
tab = pd.DataFrame(tab)
# how much other frontier models move with the clip: mean and max |change in ECI| vs Epoch's 0.1%, models at 150+
dr0, _ = load(CR / "base"); base = pd.Series(dr0["eci"].mean(0), index=dr0["models"])
mv_mean, mv_max = [], []
for kind, c, fit in rows:
    dr, _ = load(fit); e = pd.Series(dr["eci"].mean(0), index=dr["models"]).reindex(base.index)
    dlt = (e - base)[base >= 150].abs(); mv_mean.append(dlt.mean()); mv_max.append(dlt.max())
tab["frontier_mean_abs_change"], tab["frontier_max_abs_change"] = mv_mean, mv_max
tab.to_csv(OUT / "clip_table.csv", index=False)

# 2. Each benchmark's fitted scatter sigma_b by clip and under XB, with its count of 0% scores
sig = {}
for kind, c, fit in rows:
    dr, _ = load(fit)
    sig["xb" if kind == "XB" else f"clip_{c:g}"] = pd.Series(dr["sigma"].mean(0), index=dr["benchmarks"])
sig = pd.DataFrame(sig)
sig["zeros"] = d[d.performance <= 0].groupby("benchmark").size().reindex(sig.index).fillna(0).astype(int)
sig.rename_axis("benchmark").to_csv(OUT / "bench_sigma.csv")

# 3. XB's exceedance delta, and where its mean curve reaches 100%
dr, _ = load(SB); dl = dr["hyper_delta"].astype(float)
sl_med = float(np.median(np.median(dr["slope"], 0)))
reach = np.log((1 + np.median(dl)) / np.median(dl))  # index at which the mean curve reaches 1

# 4. Fit: PSIS-LOO on the common outcome space (c* = 0.1%), and Epoch's reported measures
lp = pd.read_csv(SB / "loo_pointwise_c0.001.csv"); dd = lp["sbeta"] - lp["epoch"]  # labels from the loo_compare.py run
# the fit_metrics.py run labelled XB "sbeta"; the post's columns say xb
head, body = open(SB / "fit_metrics.csv").read().split("\n", 1)
open(OUT / "fit_metrics.csv", "w").write(head.replace("sbeta|", "xb|") + "\n" + body)

fm = d[d.benchmark == FM4].set_index("Model").performance
stats = dict(
    data_date=DATA_DATE, n_scores=len(d), n_zero=int((d.performance <= 0).sum()),
    fm4_questions=int(pd.read_csv(QC).set_index("benchmark").n_used[FM4]), fm4_sol=float(fm[S]), fm4_astra=float(fm[A]),
    delta=float(np.median(dl)), delta_lo=float(np.quantile(dl, 0.05)), delta_hi=float(np.quantile(dl, 0.95)),
    reach_eci_median=float(reach / sl_med),
    loo_diff=float(dd.sum()), loo_se=float(np.sqrt(len(dd) * dd.var())),
    loo_boundary=float(dd[lp.cat != 1].sum()), loo_interior=float(dd[lp.cat == 1].sum()),
)
json.dump(stats, open(OUT / "post_stats.json", "w"), indent=1)
print(tab.round(3).to_string()); print(json.dumps(stats, indent=1))
