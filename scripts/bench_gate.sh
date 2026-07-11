#!/bin/sh
# scripts/bench_gate.sh — hard regression gate for benches/transcribe.rs.
#
# WHY THIS EXISTS
#   Criterion 0.8's own `--baseline` comparison prints a red
#   "Performance has regressed." line to the terminal, but that is display
#   only: `cargo bench` always exits 0 regardless of the comparison result
#   (verified by reading compare_to_threshold() / ComparisonResult in
#   ~/.cargo/registry/src/*/criterion-0.8.2/src/report.rs — the classification
#   feeds report text, never process::exit). So criterion's CLI output alone
#   cannot fail a CI job. This script closes that gap by parsing criterion's
#   on-disk JSON estimates directly and exiting non-zero on a real regression.
#
# WORKFLOW
#   1. Save a baseline (e.g. on master, or before a perf-sensitive change):
#        cargo bench --bench transcribe -- --save-baseline main
#   2. Make your change, then re-run comparing against it (this does NOT
#      overwrite the saved baseline; it only refreshes the "new" results):
#        cargo bench --bench transcribe -- --baseline main
#   3. Gate on the result:
#        scripts/bench_gate.sh main 10
#
# WHAT IT READS
#   For every benchmark, criterion writes:
#     target/criterion/<id>/new/estimates.json     (the run just performed)
#     target/criterion/<id>/<baseline>/estimates.json  (the saved baseline)
#   `<id>` is a group/function[/value] directory path
#   (see BenchmarkId::as_directory_name in criterion's source), so this script
#   does not hardcode benchmark names or directory depth — it discovers every
#   "*/new/estimates.json" under the criterion output dir and, for each one,
#   looks for a sibling "<baseline>/estimates.json" to compare against.
#
# EXIT CODES
#   0  every compared benchmark's mean is within THRESHOLD_PCT of baseline
#   1  at least one benchmark's mean regressed by more than THRESHOLD_PCT
#   2  usage / environment error (missing jq, no criterion output, etc.)

set -eu

BASELINE="${1:-main}"
THRESHOLD_PCT="${2:-10}"
CRITERION_DIR="${3:-target/criterion}"

if ! command -v jq >/dev/null 2>&1; then
    echo "bench_gate: jq is required but was not found on PATH" >&2
    exit 2
fi

if [ ! -d "$CRITERION_DIR" ]; then
    echo "bench_gate: criterion output dir '$CRITERION_DIR' not found; run 'cargo bench --bench transcribe -- --baseline $BASELINE' first" >&2
    exit 2
fi

new_files=$(find "$CRITERION_DIR" -type f -path '*/new/estimates.json' | sort)

if [ -z "$new_files" ]; then
    echo "bench_gate: no */new/estimates.json files found under $CRITERION_DIR" >&2
    echo "bench_gate: run 'cargo bench --bench transcribe -- --baseline $BASELINE' first" >&2
    exit 2
fi

pairs_found=0
regressions=0
report=""

# Reading `new_files` from a here-doc (rather than piping `find` straight
# into the loop) keeps the loop in the current shell, so pairs_found /
# regressions / report survive past the loop instead of being lost in a
# subshell — a well-known POSIX `while read` pitfall when the input is a pipe.
while IFS= read -r new_file; do
    [ -n "$new_file" ] || continue

    bench_dir=$(dirname "$new_file")     # .../<id>/new
    id_dir=$(dirname "$bench_dir")       # .../<id>
    base_file="$id_dir/$BASELINE/estimates.json"

    [ -f "$base_file" ] || continue

    bench_name=$(printf '%s\n' "$id_dir" | sed "s|^$CRITERION_DIR/||")

    new_mean=$(jq -r '.mean.point_estimate' "$new_file")
    base_mean=$(jq -r '.mean.point_estimate' "$base_file")

    case "$new_mean" in ''|*[!0-9.eE+-]*)
        echo "bench_gate: could not read numeric mean from $new_file (got '$new_mean')" >&2
        exit 2
        ;;
    esac
    case "$base_mean" in ''|*[!0-9.eE+-]*)
        echo "bench_gate: could not read numeric mean from $base_file (got '$base_mean')" >&2
        exit 2
        ;;
    esac

    pairs_found=$((pairs_found + 1))

    pct=$(awk -v n="$new_mean" -v b="$base_mean" 'BEGIN { printf "%.4f", (n - b) / b * 100.0 }')
    over=$(awk -v p="$pct" -v t="$THRESHOLD_PCT" 'BEGIN { print (p > t) ? "1" : "0" }')

    printf 'bench_gate: %-55s new=%14.1fns base=%14.1fns delta=%+8.2f%%\n' \
        "$bench_name" "$new_mean" "$base_mean" "$pct"

    if [ "$over" = "1" ]; then
        regressions=$((regressions + 1))
        report="${report}  - ${bench_name}: ${pct}% slower than baseline '${BASELINE}' (threshold ${THRESHOLD_PCT}%)
"
    fi
done <<BENCH_GATE_EOF
$new_files
BENCH_GATE_EOF

if [ "$pairs_found" -eq 0 ]; then
    echo "bench_gate: no benchmark had both a 'new' result and a '$BASELINE' baseline to compare" >&2
    echo "bench_gate: did you run 'cargo bench --bench transcribe -- --save-baseline $BASELINE' first?" >&2
    exit 2
fi

echo "bench_gate: compared $pairs_found benchmark(s) against baseline '$BASELINE' (threshold ${THRESHOLD_PCT}%)"

if [ "$regressions" -gt 0 ]; then
    printf 'bench_gate: FAIL - %d benchmark(s) regressed:\n' "$regressions"
    printf '%s' "$report"
    exit 1
fi

echo "bench_gate: PASS - no benchmark regressed beyond ${THRESHOLD_PCT}%"
exit 0
