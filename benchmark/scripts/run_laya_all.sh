#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  run_laya_all.sh \
    --study-root PATH \
    [--backend mlx|torch] \
    [--checkpoint typed-decisions|base|multilingual] \
    [extra run_laya_benchmark.py arguments]
USAGE
}

STUDY_ROOT=""
BACKEND="mlx"
CHECKPOINT="typed-decisions"
EXTRA_ARGS_FILE="$(mktemp -t laya-extra-args.XXXXXX)"
trap 'rm -f "$EXTRA_ARGS_FILE"' EXIT

while [ "$#" -gt 0 ]; do
  case "$1" in
    --study-root) [ "$#" -ge 2 ] || { echo "Missing value for --study-root" >&2; exit 2; }; STUDY_ROOT="$2"; shift 2 ;;
    --backend) [ "$#" -ge 2 ] || { echo "Missing value for --backend" >&2; exit 2; }; BACKEND="$2"; shift 2 ;;
    --checkpoint) [ "$#" -ge 2 ] || { echo "Missing value for --checkpoint" >&2; exit 2; }; CHECKPOINT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) printf '%s\n' "$1" >> "$EXTRA_ARGS_FILE"; shift ;;
  esac
done

[ -n "$STUDY_ROOT" ] || { usage >&2; exit 2; }
if [ -d "$STUDY_ROOT/jev-system-one-study/benchmark" ] && [ ! -d "$STUDY_ROOT/benchmark" ]; then
  STUDY_ROOT="$STUDY_ROOT/jev-system-one-study"
fi

RUNNER="$STUDY_ROOT/benchmark/scripts/run_laya_benchmark.py"
COMPARE="$STUDY_ROOT/benchmark/scripts/compare_jev_laya.py"
[ -f "$RUNNER" ] || { echo "Missing runner: $RUNNER" >&2; exit 2; }
[ -f "$COMPARE" ] || { echo "Missing comparison script: $COMPARE" >&2; exit 2; }

echo "Study root : $STUDY_ROOT"
echo "Backend    : $BACKEND"
echo "Checkpoint : $CHECKPOINT"
echo

run_benchmark() {
  set --
  if [ -s "$EXTRA_ARGS_FILE" ]; then
    while IFS= read -r arg; do set -- "$@" "$arg"; done < "$EXTRA_ARGS_FILE"
  fi
  python "$RUNNER" --study-root "$STUDY_ROOT" --backend "$BACKEND" --checkpoint "$CHECKPOINT" "$@"
}

echo "=== Running Laya benchmark ==="
run_benchmark

echo
echo "=== Building Jev vs Laya comparison ==="
python "$COMPARE" --study-root "$STUDY_ROOT" --backend "$BACKEND" --checkpoint "$CHECKPOINT"

SAFE_CHECKPOINT="$(printf '%s' "$CHECKPOINT" | sed 's/[^[:alnum:]]/-/g')"
REPORT="$STUDY_ROOT/comparisons/laya_${SAFE_CHECKPOINT}_${BACKEND}/JEV_VS_LAYA_REPORT.md"

echo
echo "=== Complete ==="
echo "Comparison report: $REPORT"
