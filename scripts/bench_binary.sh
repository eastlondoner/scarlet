#!/usr/bin/env bash
# Building binaries in bulk: `binary.concat` of 1 MiB in 4,096 parts, and a
# 256 x 256 x 128 volume of 16-bit block states built from `repeat` and
# `concat` a column at a time (examples/bench_binary.scrl, which times each
# inside the run and prints microseconds per build).
#
#   scripts/bench_binary.sh             # best of 5 runs, one line per build
#   scripts/bench_binary.sh --reps 9
#
# Measured on the development Mac (Apple M1 Max), release build:
#
#   concat_4096x256       master 6b66992: 27.8 s (the fold over `append`
#                         copies everything so far at every part)
#                         now: about 85 us
#   volume_256x256x128    master: no `repeat`; the same volume with runs
#                         folded from `append` took 16.0 s for 1/16 of it
#                         (4,096 columns), and concat's n^2 puts the whole
#                         past an hour, so it was not run
#                         now: about 235 ms
set -euo pipefail

cd "$(dirname "$0")/.."

REPS=5
while [[ $# -gt 0 ]]; do
  case "$1" in
    --reps)    REPS="$2"; shift 2 ;;
    --help|-h) sed -n '2,20p' "$0"; exit 0 ;;
    *)         echo "bench_binary: unknown argument $1" >&2; exit 2 ;;
  esac
done

cargo build --release >/dev/null 2>&1
SCARLET="${CARGO_TARGET_DIR:-target}/release/scarlet"

# Each run prints `<name> <us> us <bytes> bytes`; keep each name's least.
for _ in $(seq "$REPS"); do
  "$SCARLET" run examples/bench_binary.scrl
done | awk '
  { if (!($1 in best) || $2 < best[$1]) { best[$1] = $2; size[$1] = $4 }
    if (!($1 in seen)) { seen[$1] = 1; order[++n] = $1 } }
  END { for (i = 1; i <= n; i++) printf "bench_best_us %s %s (%s bytes)\n", order[i], best[order[i]], size[order[i]] }'
