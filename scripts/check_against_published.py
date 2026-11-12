"""Compare the regenerated post files (posts/eci-clip/) with the published versions (this repo's committed copies).
Numbers are compared with a relative tolerance (default 1e-6) plus an absolute 0.0015, because the last digits depend on the
platform and library builds (files rounded to 3 decimals can flip in the last one). Refits by MCMC differ by Monte
Carlo error instead: use a looser relative tolerance, e.g. 0.02.

Usage (repo root): python scripts/check_against_published.py [REL_TOL]
"""
import io, json, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd

TOL = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-6
POST = Path("posts/eci-clip")


def published(path):
    return subprocess.run(["git", "show", f"HEAD:{path.as_posix()}"], capture_output=True, text=True, check=True).stdout


def close(a, b):
    return np.isclose(a, b, rtol=TOL, atol=max(0.0015, TOL), equal_nan=True)


def flat(x, pre=""):
    if isinstance(x, dict):
        return {k2: v2 for k, v in x.items() for k2, v2 in flat(v, f"{pre}{k}.").items()}
    if isinstance(x, list):
        return {k2: v2 for i, v in enumerate(x) for k2, v2 in flat(v, f"{pre}{i}.").items()}
    return {pre[:-1]: x}


bad = 0
for f in sorted(POST.iterdir()):
    if f.suffix == ".csv":
        a, b = pd.read_csv(io.StringIO(published(f))), pd.read_csv(f)
        if list(a.columns) != list(b.columns) or a.shape != b.shape:
            msg = f"shape or columns differ: {a.shape} vs {b.shape}"
        else:
            num = a.select_dtypes("number").columns
            off = [c for c in num if not close(a[c].values, b[c].values).all()]
            off += [c for c in a.columns if c not in num and not (a[c].astype(str) == b[c].astype(str)).all()]
            msg = f"columns differ: {off}" if off else ""
    elif f.suffix == ".json":
        a, b = flat(json.loads(published(f))), flat(json.load(open(f)))
        off = [k for k in a.keys() | b.keys() if k not in a or k not in b or (
            close(a[k], b[k]) == False if isinstance(a[k], (int, float)) and isinstance(b[k], (int, float))  # noqa: E712
            else a[k] != b[k])]
        msg = f"keys differ: {sorted(off)}" if off else ""
    elif f.suffix == ".html":
        msg = "" if published(f) == f.read_text(encoding="utf-8") else "differs (see git diff)"
    else:
        continue
    bad += bool(msg)
    print(f"{'DIFF' if msg else 'ok  '} {f.name} {msg}")
print(f"\n{bad} file(s) differ beyond rel. tolerance {TOL:g}" if bad else f"\nAll post files match (rel. tolerance {TOL:g})")
sys.exit(1 if bad else 0)
