"""The reasoning-era frontier trend of Epoch's Nov 9, 2026 chart, and a model's gap above it. Shared by the scripts
that report Claude Fable 5.2's gap (clip_trend_gap.py, ls_clip_sweep.py, xbx_post_assets.py, xb_sensitivity.py).

The trend is an ordinary least-squares line of ECI on release date through the frontier: the reasoning models that
had the highest ECI of any reasoning model yet when released, from o1-mini (Sept 12, 2024) through Claude Fable 5.1
(Sept 1, 2026). Each fit picks its own frontier from its own ECI estimates.
"""
from pathlib import Path
import numpy as np
import pandas as pd

FABLE = "Claude Fable 5.2"
START, THROUGH = pd.Timestamp("2024-09-12"), pd.Timestamp("2026-09-01")  # o1-mini's and Claude Fable 5.1's releases
# Reasoning models: everything released after GPT-4.5, plus these earlier ones
GPT45_DATE = pd.Timestamp("2025-02-27")
REASONING_EARLY = ["o1-preview", "o1-mini", "o1", "DeepSeek-R1", "Gemini 2.0 Flash Thinking (Jan 2025)", "o3-mini",
                   "Claude 3.7 Sonnet"]
YEAR_ZERO = pd.Timestamp("2020-01-01")


def release_dates(eci_dir):
    """Each model's release date, from an Epoch archive's epoch_capabilities_index/eci_scores.csv."""
    return pd.read_csv(Path(eci_dir) / "eci_scores.csv", parse_dates=["date"]).set_index("Model").date


def years(when):
    """Dates as years since Jan 1, 2020 (the trend's time variable)."""
    return np.asarray((pd.DatetimeIndex(when) - YEAR_ZERO).days, float) / 365.25


def design(when):
    """The line's regressors at the given dates: a column of ones and the date in years."""
    t = years(when)
    return np.column_stack([np.ones_like(t), t])


def frontier(eci, dates):
    """The trend's models: reasoning models released from START through THROUGH that beat every earlier reasoning
    model in that window on ECI (a Series by model), in release order."""
    is_reasoning = (dates > GPT45_DATE) | dates.index.isin(REASONING_EARLY)
    pool = [m for m in eci.index if m in dates.index and is_reasoning[m] and START <= dates[m] <= THROUGH]
    d = pd.DataFrame({"eci": eci[pool], "date": dates[pool]}).sort_values(["date", "eci"], ascending=[True, False])
    best_before = d.eci.shift(1).cummax().fillna(-np.inf)
    return list(d[d.eci > best_before].index)


def fit_line(eci, models, dates):
    """Least-squares line of ECI on release date through `models`: (intercept, slope per year).
    eci is one fit's estimates (a Series by model) or its draws (a DataFrame, one column per model); for draws, the
    intercept and slope are arrays with one value per draw."""
    return np.linalg.lstsq(design(dates[models]), eci[models].values.T, rcond=None)[0]


def predict(line, when):
    """The line's ECI at the given dates (dates x draws when the line was fit to draws)."""
    return design(when) @ line


def gap_above_trend(eci, line, dates, model=FABLE):
    """A model's ECI minus the line's prediction at its release date: one number, or one per draw."""
    return np.asarray(eci[model]) - predict(line, [dates[model]])[0]
