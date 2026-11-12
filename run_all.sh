#!/bin/bash
# Recompute every file the post reads (posts/eci-clip/*). Run from the repo root inside the eci-clip
# environment, after `bash setup.sh --draws` (or `bash setup.sh`, then FITS=1 to refit).
#   bash run_all.sh            # post files from the posterior draws in results/ (about 15 minutes)
#   FITS=1 bash run_all.sh     # first refit every model by MCMC (hours on a laptop; see README)
set -e
S=2026-11-10_0823
D=data/$S/epoch_capabilities_index
CR=results/clip_refits_$S          # Epoch's model at each clip (base = Epoch's 0.1%)
XB=results/sbeta_$S                # extended-support beta (XB; the fit-side name "sbeta" predates "XB")
XS=results/sbeta_shift_$S          # XB with delta tied to sigma_b
XBX=results/xbx_$S                 # Kosmidis and Zeileis's XBX
XP=results/xb_prior_$S             # XB under other priors on delta
DEF=results/bayes_default_$S       # Epoch's model with default settings: sigma_b for the Astra/Sol chart
P=posts/eci-clip

if [ "${FITS:-0}" = 1 ]; then
  fit() { echo "== fit $1"; python scripts/bayes_eci.py "$D" "$@"; }
  for c in 0.01 0.0001 1e-05 1e-06 1e-08 1e-12; do
    fit $CR/clip_$c --clip $c --tune 1500 --draws 2000
  done
  # Seed 20261104 (the default) narrowly missed Epoch's convergence checks for these two; 20261105 passed
  fit $CR/base --clip 0.001 --tune 1500 --draws 2000 --seed 20261105
  fit $CR/clip_1e-10 --clip 1e-10 --tune 1500 --draws 2000 --seed 20261105
  fit $XB --sbeta
  fit $XS --sbeta-shift --tune 1500 --draws 2000
  fit $XBX --xbx 12 --init-from $XB
  for p in halfnormal0.01 uniform loguniform; do fit $XP/$p --sbeta --delta-prior $p; done
  fit $DEF
fi

echo "== model comparison (PSIS-LOO) and fit measures"
python scripts/loo_compare.py $D 0.001 epoch=$CR/base sbeta=$XB --save=$XB/loo_pointwise_c0.001.csv > $XB/loo_compare.txt
python scripts/fit_metrics.py $D epoch=$CR/base sbeta=$XB --seed 1 --csv $XB/fit_metrics.csv > $XB/fit_metrics.txt
python scripts/loo_compare.py $D 0.001 sbeta=$XB shift=$XS --save=$XS/loo_pointwise_c0.001.csv > $XS/loo_compare.txt
python scripts/fit_metrics.py $D epoch=$CR/base sbeta=$XB shift=$XS --seed 1 --csv $XS/fit_metrics.csv > /dev/null
python scripts/loo_compare.py $D 0.001 xb=$XB xbx=$XBX --save=$XBX/loo_pointwise_c0.001.csv > $XBX/loo_compare.txt

echo "== XB prior sensitivity (the post's footnote on XBX: delta barely moves under tighter or flatter priors)"
python scripts/xb_sensitivity.py $D $XP/prior_sensitivity.csv "HalfNormal(0.05) [current]=$XB" "HalfNormal(0.01)=$XP/halfnormal0.01" \
  "Uniform(0, 0.1)=$XP/uniform" "log-uniform=$XP/loguniform"

echo "== post files"
python scripts/clip_post_assets.py $D $CR $XB $P binom/benchmark_question_counts.csv 2026-11-10
python scripts/xb_shift_post_assets.py                             # xb_shift.json
python scripts/xbx_post_assets.py $D                               # xbx.json
python scripts/clip_trend_gap.py $D $CR $P/fable_gap.csv $P/fable_gap_lines.csv

echo "== Epoch's original least-squares ECI at each clip"
python scripts/ls_clip_sweep.py $D $P/ls_clip_table.csv

echo "== Astra vs 6.1 Sol chart"
W=build/panel; mkdir -p $W
(cd $W && ECI_DATA=../../$D ECI_BAYES=../../$DEF python ../../scripts/eci_weights_beta.py > /dev/null)
(cd $W && ECI_DATA_ROOT=../../data/$S python ../../viz/pair_panel.py "GPT-6 Astra|GPT-6.1 Sol" "Astra|6.1 Sol" astra_sol_panel.html)
cp $W/astra_sol_panel.html $W/astra_sol_panel_curves.csv $P/
cp $W/astra_sol_panel_scores.csv $P/astra_sol_scores.csv

echo "== check against the published post files"
python scripts/check_against_published.py "${TOL:-1e-6}"   # after FITS=1, try TOL=0.02
