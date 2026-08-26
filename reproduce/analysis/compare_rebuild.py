"""
Compare a rebuilt result file against the shipped one, leaf by leaf.

Run from the repository root, after writing a rebuild. The README gives the
output name each rebuild script uses:

    python reproduce/analysis/decomposition_table.py --out decomposition_by_venue.rebuilt.json
    python reproduce/analysis/compare_rebuild.py --shipped
        decomposition_by_venue.json --rebuilt decomposition_by_venue.rebuilt.json

Every other check here asks whether a number is internally consistent or
whether the prose matches the data. None of them asks the plainest question:
does the shipped file still equal what the code produces? A shipped point
estimate could be edited by hand, or left behind by a change to the panels and
no other check here would notice.

Point estimates must agree to a tight tolerance. Percentile bootstrap bounds are allowed to drift, because the shipped files were
produced on a different random stream and reproducing that stream is not a goal.
The tolerance is 0.05 bp, above the drift the pooled rebuilds show and below the
width of the narrowest interval it guards. The per-coin file gets 0.25, because
its cells rest on the fewest months and its own docstring quotes drift up to
0.213 bp. Pass --bound-tol to override either. A flag that
flips is reported rather than tolerated, since the flags are what a reader acts
on.

Only keys present in both files are compared, and the count of shipped keys
with no rebuilt counterpart is printed alongside, because that is the part of a
file this check cannot speak for.

Exit status is non-zero when the two files share no leaves at all, which means
the wrong pair was passed. It is also non-zero when a point estimate moves or a
flag flips, unless that key is passed as a documented exception. The
Hyperliquid HYPE knife
edge is the only one in this repository: its interval reaches to within 0.0022
bp of zero and the two shipped files land on opposite sides, which the result
file records.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPRODUCE = HERE.parent
POINT_TOL = 1e-6
BOUND_TOL = 0.05
BOUND_TOL_BY_FILE = {"per_coin_intervals.json": 0.25}
BOUND_HINTS = ("ci95", "_ci", "bootstrap", "interval", "bound", "_sd")


def flatten(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = obj
    return out


def is_bound(key):
    return any(h in key for h in BOUND_HINTS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shipped", required=True,
                    help="file name inside reproduce/, or a path")
    ap.add_argument("--rebuilt", required=True)
    ap.add_argument("--known", nargs="*", default=[],
                    help="keys whose disagreement is documented, so a run can "
                         "pass while still reporting them")
    ap.add_argument("--bound-tol", type=float, default=None,
                    help="allowed drift on interval bounds, in basis points")
    ap.add_argument("--out")
    a = ap.parse_args()
    sp = Path(a.shipped)
    if not sp.exists():
        sp = REPRODUCE / a.shipped
    shipped = flatten(json.loads(sp.read_text()))
    rebuilt = flatten(json.loads(Path(a.rebuilt).read_text()))

    bound_tol = (a.bound_tol if a.bound_tol is not None
                 else BOUND_TOL_BY_FILE.get(sp.name, BOUND_TOL))
    known = set(a.known)
    moved, flipped, drifted, compared = [], [], 0, 0
    for key, sv in shipped.items():
        if key not in rebuilt or key.startswith("limitations"):
            continue
        rv = rebuilt[key]
        if isinstance(sv, bool) or isinstance(rv, bool) or sv is None or rv is None:
            compared += 1
            if sv != rv:
                flipped.append((key, sv, rv))
            continue
        if isinstance(sv, str) or isinstance(rv, str):
            compared += 1
            if sv != rv:
                moved.append((key, sv, rv))
            continue
        try:
            diff = abs(float(sv) - float(rv))
        except (TypeError, ValueError):
            continue
        compared += 1
        if is_bound(key):
            if diff > bound_tol:
                moved.append((key, sv, rv))
            elif diff > POINT_TOL:
                drifted += 1
        elif diff > POINT_TOL:
            moved.append((key, sv, rv))

    only_shipped = sum(1 for k in shipped
                       if k not in rebuilt and not k.startswith("limitations"))
    only_rebuilt = sorted(k for k in rebuilt if k not in shipped)
    print(f"  compared {compared} shared leaves of {sp.name}; "
          f"{only_shipped} shipped leaves have no rebuilt counterpart")
    if only_rebuilt:
        print(f"  {len(only_rebuilt)} rebuilt leaves are absent from the shipped "
              f"file and are not compared, first: {only_rebuilt[0]}")
    print(f"  {len(moved)} moved beyond tolerance, {len(flipped)} flags "
          f"flipped, {drifted} interval bounds drifted within tolerance")
    for k, s, r in moved[:10]:
        print(f"    MOVED {k}: shipped {s!r} rebuilt {r!r}")
    for k, s, r in flipped[:10]:
        print(f"    FLIPPED {k}: shipped {s!r} rebuilt {r!r}")
    if a.out:
        Path(a.out).write_text(json.dumps(
            {"file": sp.name, "n_compared": compared,
             "n_moved": len(moved), "n_flipped": len(flipped),
             "n_drifted": drifted, "n_shipped_only": only_shipped,
             "n_rebuilt_only": len(only_rebuilt),
             "rebuilt_only_keys": only_rebuilt,
             "moved": [{"key": k, "shipped": s, "rebuilt": r}
                       for k, s, r in moved],
             "flipped": [{"key": k, "shipped": s, "rebuilt": r}
                         for k, s, r in flipped]}, indent=2, default=float))
    if compared == 0:
        print("    no shared leaves; the two files describe different results")
        return 1
    unexpected = ([m for m in moved if m[0] not in known]
                  + [f for f in flipped if f[0] not in known])
    if known:
        print(f"  {len(moved) + len(flipped) - len(unexpected)} of those are "
              f"documented exceptions")
    return 1 if unexpected else 0


if __name__ == "__main__":
    raise SystemExit(main())
