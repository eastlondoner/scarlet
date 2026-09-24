#!/usr/bin/env bash
# Metal buffer timings (docs/metal-design.md, "Tests that ratchet"): runs
# examples/bench_metal.scrl, keeps the best of N runs for each operation, and
# writes each as the cost of one call.
#
#   scripts/bench_metal.sh                  # measure, write target/bench_metal.txt
#   scripts/bench_metal.sh --out FILE       # write FILE instead
#   scripts/bench_metal.sh --reps 9         # best of 9 rather than 5
#
# Each line written is `op bytes ns_per_call`. scripts/bench_metal_compare.sh
# checks a result against the baseline committed for the machine. Timings are
# noisy and belong to one machine, so this runs on demand, not in cargo test.
#
# Reports the *best* of N runs: the minimum is the least noise-contaminated
# estimate of the work done, and the thing that regresses when it gets slower.
set -euo pipefail

cd "$(dirname "$0")/.."

REPS=5
OUT=target/bench_metal.txt

while [[ $# -gt 0 ]]; do
  case "$1" in
    --reps)    REPS="$2"; shift 2 ;;
    --out)     OUT="$2"; shift 2 ;;
    --help|-h) sed -n '2,15p' "$0"; exit 0 ;;
    *)         echo "bench_metal: unknown argument $1" >&2; exit 2 ;;
  esac
done

cargo build --release >/dev/null 2>&1

RAW=target/bench_metal.raw
: >"$RAW"
for _ in $(seq "$REPS"); do
  if ! ./target/release/scarlet run examples/bench_metal.scrl >>"$RAW"; then
    echo "bench_metal: examples/bench_metal.scrl did not run" >&2
    exit 1
  fi
done

# Every line must be `op bytes calls ms`: an `Err(..)` or a `failed` means the
# program measured nothing.
if awk 'NF != 4 || $3 !~ /^[0-9]+$/ || $4 !~ /^[0-9]+$/ { bad = 1 } END { exit !bad }' "$RAW"; then
  echo "bench_metal: examples/bench_metal.scrl measured nothing:" >&2
  cat "$RAW" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT")"
awk '{
  key = $1 " " $2
  ns = int($4 * 1000000 / $3)
  if (!(key in best) || ns < best[key]) best[key] = ns
  if (!(key in order)) { order[key] = n++; keys[n - 1] = key }
} END {
  for (i = 0; i < n; i++) print keys[i], best[keys[i]]
}' "$RAW" >"$OUT"
cat "$OUT"
echo "wrote $OUT" >&2
