#!/bin/bash
# Fetch Epoch AI's code at the commits used, unpack Epoch's data, and (optionally) the posterior draws.
# Run from the repo root inside the eci-clip environment:  bash setup.sh [--draws]
set -e
clone() {  # url dir commit
  [ -d "$2" ] || git clone -q "$1" "$2"
  git -C "$2" checkout -q "$3"
  echo "$2 at $(git -C "$2" rev-parse --short HEAD)"
}
# Epoch's Bayesian ECI (model code used by scripts/bayes_eci.py)
clone https://github.com/epoch-research/eci-public.git eci-public e04929ab751531e558b1a61e86f436487c710ef8
pip install -q -e eci-public --no-deps
# Epoch's original least-squares ECI (fit.py, used by scripts/ls_clip_sweep.py)
clone https://github.com/epoch-research/benchmark-stitching.git benchmark-stitching f87d3cc8e8a8f8471be475ebce58fb9a73b4e756
# Epoch's benchmark data as of Nov 10, 2026 (https://epoch.ai/data/benchmark_data.zip, downloaded 08:23 CST)
D=data/2026-11-10_0823
if [ ! -d "$D" ]; then
  mkdir -p "$D" && unzip -q data/benchmark_data_2026-11-10_0823.zip -d "$D"
  cp data/ARCHIVE_INFO_2026-11-10_0823.json "$D/ARCHIVE_INFO.json"
fi
echo "data in $D"
if [ "$1" = "--draws" ]; then
  # Posterior draws of every fit (about 150 MB), from this repo's GitHub release
  curl -sSfL -o release/draws_2026-11-10_0823.tar \
    https://github.com/hugetim/eci-clip-replication/releases/download/v2/draws_2026-11-10_0823.tar
  tar -xf release/draws_2026-11-10_0823.tar
  echo "draws unpacked into results/"
fi
