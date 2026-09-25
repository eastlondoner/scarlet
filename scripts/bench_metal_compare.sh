#!/usr/bin/env bash
# Fails when a Metal timing is more than 10% worse than the baseline committed
# for this machine: the ratchet for scripts/bench_metal.sh's timings
# (docs/metal-design.md, "Tests that ratchet").
#
#   scripts/bench_metal_compare.sh                    # target/bench_metal.txt
#   scripts/bench_metal_compare.sh RESULTS [BASELINE]
#
# The baseline is scripts/baselines/bench_metal-<machine>.txt, where <machine>
# is this Mac's CPU as `sysctl machdep.cpu.brand_string` names it, lowercased
# with dashes: apple-m1-max. To make one for a new machine, or to lower one
# after a change made things faster:
#
#   scripts/bench_metal.sh --out scripts/baselines/bench_metal-<machine>.txt
set -euo pipefail

cd "$(dirname "$0")/.."

RESULTS="${1:-target/bench_metal.txt}"
if [[ $# -ge 2 ]]; then
  BASELINE="$2"
else
  if [[ "$(uname)" != Darwin ]]; then
    echo "bench_metal_compare: Metal runs only on macOS" >&2
    exit 2
  fi
  MACHINE="$(sysctl -n machdep.cpu.brand_string | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '-' | sed 's/-$//')"
  BASELINE="scripts/baselines/bench_metal-$MACHINE.txt"
fi

[[ -f "$RESULTS" ]] || { echo "bench_metal_compare: no results at $RESULTS; run scripts/bench_metal.sh" >&2; exit 2; }
[[ -f "$BASELINE" ]] || { echo "bench_metal_compare: no baseline for this machine at $BASELINE" >&2; exit 2; }

printf '%-12s %10s %12s %12s %9s\n' op bytes before_ns after_ns delta
# awk exits with 1 when a timing is worse, 2 when a baseline line is missing
# from the results, and 3 for both.
rc=0
awk -v limit=10 '
  NR == FNR { was[$1 " " $2] = $3; next }
  {
    key = $1 " " $2
    seen[key] = 1
    if (!(key in was)) { printf "%-12s %10s %12s %12s %9s\n", $1, $2, "-", $3, "new"; next }
    pct = 100 * ($3 - was[key]) / was[key]
    printf "%-12s %10s %12s %12s %+8.1f%%\n", $1, $2, was[key], $3, pct
    if (pct > limit) worse = 1
  }
  END {
    for (key in was) if (!(key in seen)) { print "missing from the results: " key; missing = 2 }
    exit worse + missing
  }
' "$BASELINE" "$RESULTS" || rc=$?

case $rc in
  0) ;;
  1) echo; echo "bench_metal_compare: slower than $BASELINE by more than 10%" >&2 ;;
  2) echo; echo "bench_metal_compare: $RESULTS lacks a timing $BASELINE has" >&2 ;;
  3) echo; echo "bench_metal_compare: slower than $BASELINE by more than 10%, and lacking a timing it has" >&2 ;;
  *) echo "bench_metal_compare: awk failed ($rc)" >&2 ;;
esac
# 2 is kept for a missing file, above.
[[ $rc -eq 0 ]] || exit 1
