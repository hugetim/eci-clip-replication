"""How far Claude Fable 5.2 sits above the reasoning-era frontier trend, under Epoch's published ECI and under each
refit (clips, the extended-support beta XB). Replicates Epoch's Nov 9, 2026 tweet chart: OLS line through reasoning
models that were SOTA (among reasoning models) on release, from o1-mini (Sept 12, 2024) through Claude Fable 5.1
(Sept 1, 2026), frontier selected by each source's own point estimates; Epoch reports +16 ECI/yr (90% CI 14 to 18) and
a 90% prediction interval ribbon. The "Epoch published" row checks that this reproduces Epoch's chart.

Per source: slope; trend prediction at Fable 5.2's release date; gap = ECI - prediction; gap_lo/gap_hi: 5th/95th
percentiles of the gap over draws (Epoch's 500 bootstrap refits for the published ECI, posterior draws for the refits),
frontier membership held fixed.
Usage: python scripts/clip_trend_gap.py ECI_DIR FIT_ROOT OUT_CSV [LINES_CSV]
  ECI_DIR: an archive's epoch_capabilities_index/ (eci_scores.csv, eci_bootstraps.json)
  FIT_ROOT: results/clip_refits_<stamp> (one folder per fit, each with draws.npz); a sibling sbeta_<stamp> (XB) is
  picked up if present.
  LINES_CSV (optional): each source's trend line and classic OLS 90% prediction band half-width on a 5-day grid (Mar 1
  to Dec 20, 2026), as days from Fable 5.2's release and ECI relative to that source's own prediction there (for the
  clip post's figure, posts/eci-clip/fable_gap_lines.csv).
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ECI_DIR, FIT_ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
LINES = sys.argv[4] if len(sys.argv) > 4 else None
GRID = pd.date_range("2026-03-01", "2026-12-20", freq="5D")
lines = []
THROUGH, START = pd.Timestamp("2026-09-01"), pd.Timestamp("2024-09-12")
REASONING_EARLY = ["o1-preview", "o1-mini", "o1", "DeepSeek-R1", "Gemini 2.0 Flash Thinking (Jan 2025)",
                   "o3-mini", "Claude 3.7 Sonnet"]
GPT45_DATE = pd.Timestamp("2025-02-27")

pub = pd.read_csv(ECI_DIR / "eci_scores.csv", parse_dates=["date"]).set_index("Model")
dates = pub.date
reasoning = (dates > GPT45_DATE) | dates.index.isin(REASONING_EARLY)
yr = lambda d: (d - pd.Timestamp("2020-01-01")).days / 365.25 if isinstance(d, pd.Timestamp) else \
    (d - pd.Timestamp("2020-01-01")).dt.days.values / 365.25


def frontier(eci):
    pool = [m for m in eci.index if m in dates.index and reasoning[m] and START <= dates[m] <= THROUGH]
    d = pd.DataFrame({"eci": eci[pool], "date": dates[pool]}).sort_values(["date", "eci"], ascending=[True, False])
    return list(d[d.eci > d.eci.shift(1).cummax().fillna(-np.inf)].index)


def analyse(name, eci, draws):
    fr = frontier(eci)
    t, y = yr(dates[fr]), eci[fr].values
    X = np.column_stack([np.ones_like(t), t])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    n = len(fr); s = np.sqrt(((y - X @ beta) ** 2).sum() / (n - 2)); XtXi = np.linalg.inv(X.T @ X)
    tq = stats.t.ppf(0.95, n - 2)
    Td = draws[fr].values  # draws x frontier
    B = np.linalg.lstsq(X, Td.T, rcond=None)[0]  # 2 x draws
    row = {"source": name, "slope": beta[1]}
    tf = yr(dates["Claude Fable 5.2"]); tg = (GRID - pd.Timestamp("2020-01-01")).days.values / 365.25
    Xg = np.column_stack([np.ones_like(tg), tg])
    lines.append(pd.DataFrame({"source": name, "days": (GRID - dates["Claude Fable 5.2"]).days,
                               "y": Xg @ beta - (beta[0] + beta[1] * tf),
                               "half": tq * s * np.sqrt(1 + np.einsum("ij,jk,ik->i", Xg, XtXi, Xg))}))
    m = "Claude Fable 5.2"
    x0 = np.array([1, tf]); pred = x0 @ beta
    g = draws[m].values - x0 @ B
    row.update({"5.2_eci": eci[m], "5.2_pred": pred, "5.2_gap": eci[m] - pred,
                "5.2_gap_lo": np.percentile(g, 5), "5.2_gap_hi": np.percentile(g, 95)})
    return row, fr


rows = []
bs = json.load(open(ECI_DIR / "eci_bootstraps.json"))
E = pd.DataFrame(np.array(bs["model"]["capability_samples"]), columns=bs["model"]["names"])
r, fr_pub = analyse("Epoch published", pub.eci, E); rows.append(r)
fits = [(p.name, p) for p in sorted(FIT_ROOT.iterdir()) if (p / "draws.npz").exists()]
sb = FIT_ROOT.parent / FIT_ROOT.name.replace("clip_refits", "sbeta")
if (sb / "draws.npz").exists():
    fits.append(("XB", sb))
for name, p in fits:
    z = np.load(p / "draws.npz")
    D = pd.DataFrame(z["eci"], columns=z["models"])
    r, fr = analyse(name, D.mean(), D); rows.append(r)
    if fr != fr_pub:
        print(f"{name}: frontier differs from published: +{sorted(set(fr) - set(fr_pub))} -{sorted(set(fr_pub) - set(fr))}")
out = pd.DataFrame(rows)
order = {"Epoch published": 0, "clip_0.01": 1, "base": 2, "clip_0.0001": 3, "clip_1e-05": 4,
         "clip_1e-06": 5, "clip_1e-08": 6, "clip_1e-10": 7, "clip_1e-12": 8, "XB": 9}
out = out.sort_values("source", key=lambda s: s.map(order)).round(3)
out.to_csv(OUT, index=False)
if LINES:
    pd.concat(lines).round(4).to_csv(LINES, index=False)
print("frontier (published):", ", ".join(fr_pub))
print(out.round(2).to_string(index=False))
