# Replication code for "ECI clip" post

This repo holds the code, data and fit summaries behind the blog post on how the Epoch Capabilities Index (ECI) handles scores of exactly 0% and 100%: <https://hugetim.github.io/posts/eci-clip/>. Every number and chart in the post is read from the files in [`posts/eci-clip/`](posts/eci-clip/). This repo recomputes those files from Epoch AI's data.

The post's own source (`index.qmd`, which turns these files into text and figures) can be viewed on the post page using the **Code** button.

## What's here

| Path | Contents |
|---|---|
| `scripts/`, `viz/` | The scripts that produce the post's files, trimmed from my working repo to what the post uses |
| `data/benchmark_data_2026-11-10_0823.zip` | Epoch AI's benchmark data as downloaded Nov 10, 2026, 08:23 CST, from <https://epoch.ai/data/benchmark_data.zip> |
| `binom/benchmark_question_counts.csv` | Question counts per benchmark, with sources (the post uses FrontierMath Tier 4's) |
| `results/` | Each fit's summaries: `eci.csv`, `edi.csv`, `hyper.json`, `diagnostics.json` (sampler settings, convergence, and the SHA-256 of the fit input), plus the model comparison outputs |
| `posts/eci-clip/` | The files the post reads, as published |
| `environment.yml`, `setup.sh`, `run_all.sh` | Environment, setup and the full pipeline |

The posterior draws (`draws.npz` for each of the 15 fits, about 140 MB in all) are too large to keep in Git. They are attached to this repo's [v1 release](https://github.com/hugetim/eci-clip-replication/releases/tag/v1).

## Running it

You need Linux (or WSL) and conda. The pipeline was run on x86-64 Ubuntu 24.04 (WSL) with Python 3.12.

```bash
conda env create -f environment.yml
conda activate eci-clip
bash setup.sh --draws     # clones Epoch's code at the pinned commits, unpacks the data, downloads the draws
bash run_all.sh           # recomputes posts/eci-clip/* from the draws (about 15 minutes)
```

`run_all.sh` ends by comparing the regenerated files with the published ones (`scripts/check_against_published.py`). Run from the released draws, everything should match up to the last stored digit (relative tolerance 10⁻⁶, absolute 0.0015): those last digits vary by platform.

To refit everything from scratch instead, run `bash setup.sh` without `--draws`, then `FITS=1 bash run_all.sh`.

- **Runtime:** most of the 15 MCMC fits take about 5–10 minutes each on a laptop with 2 cores per fit. The XBX fit takes about 40 minutes.
- **Matching:** a refit reproduces the published files up to Monte Carlo error, not digit for digit. NUTS draws depend on the platform and the BLAS build.
- **Seeds:** the seeds are those used for the post. The 0.1% and 10⁻¹⁰ fits use seed 20261105 because the default seed's chains narrowly missed Epoch's convergence checks.

## Where each post file comes from

| Post file | Script | From |
|---|---|---|
| `clip_table.csv`, `bench_sigma.csv`, `fit_metrics.csv`, `post_stats.json` | `scripts/clip_post_assets.py` | Epoch's model refit at each clip (`results/clip_refits_*`); XB (`results/sbeta_*`) with its `loo_compare.py` and `fit_metrics.py` outputs |
| `fable_gap.csv`, `fable_gap_lines.csv` | `scripts/clip_trend_gap.py` | the same fits, plus Epoch's published ECI and its 500 bootstrap draws |
| `xb_shift.json` | `scripts/xb_shift_post_assets.py` | XB with δ tied to σ_b (`results/sbeta_shift_*`) |
| `xbx.json` | `scripts/xbx_post_assets.py` | XBX (`results/xbx_*`) and its `loo_compare.py` output |
| (the XB footnote's prior sensitivity: not a post file) | `scripts/xb_sensitivity.py` | XB refit under three other priors on δ (`results/xb_prior_*`); writes `results/xb_prior_*/prior_sensitivity.csv` |
| `ls_clip_table.csv` | `scripts/ls_clip_sweep.py` | Epoch's original least-squares ECI (benchmark-stitching `fit.py`) at each clip |
| `astra_sol_panel.html`, `astra_sol_panel_curves.csv`, `astra_sol_scores.csv` | `scripts/eci_weights_beta.py`, then `viz/pair_panel.py` | Epoch's published ECI, difficulties and slopes; σ_b from a default-settings fit (`results/bayes_default_*`) |

The fits use Epoch's own model code. `scripts/bayes_eci.py` builds the model with `eci.bayesian` from [epoch-research/eci-public](https://github.com/epoch-research/eci-public) at commit `e04929a`, then adds the alternative likelihoods (XB and XBX), and samples it with nutpie. The least-squares fits use `fit.py` from [epoch-research/benchmark-stitching](https://github.com/epoch-research/benchmark-stitching) at commit `f87d3cc`.

Throughout the code and results, XB is called "sbeta" ("stretched beta"), because the scripts predate the post's naming.

## Credits and license

Data: Epoch AI, "Capabilities & benchmarking", <https://epoch.ai/benchmarks>, used under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The zip is unmodified. Epoch's ECI model code (`eci-public`, `benchmark-stitching`) is MIT-licensed and is fetched by `setup.sh`, not included here.

The code in this repo is released under the MIT license (see `LICENSE`).
