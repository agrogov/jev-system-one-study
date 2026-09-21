#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  run_semif_all.sh --study-root PATH [extra run_semif_benchmark.py arguments]

Runs the five Jev benchmark suites against SemIf (Qwen3.5-4B, MLX) and then builds the
Jev-vs-SemIf comparison. Activate an environment created from
benchmark/requirements-semif-mlx.txt first (for example benchmark/.venv-semif).
USAGE
}

STUDY_ROOT=""
EXTRA_ARGS=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --study-root) [ "$#" -ge 2 ] || { echo "Missing value for --study-root" >&2; exit 2; }; STUDY_ROOT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

[ -n "$STUDY_ROOT" ] || { usage >&2; exit 2; }
if [ -d "$STUDY_ROOT/jev-system-one-study/benchmark" ] && [ ! -d "$STUDY_ROOT/benchmark" ]; then
  STUDY_ROOT="$STUDY_ROOT/jev-system-one-study"
fi

RUNNER="$STUDY_ROOT/benchmark/scripts/run_semif_benchmark.py"
COMPARE="$STUDY_ROOT/benchmark/scripts/compare_jev_semif.py"
[ -f "$RUNNER" ] || { echo "Missing runner: $RUNNER" >&2; exit 2; }
[ -f "$COMPARE" ] || { echo "Missing comparison script: $COMPARE" >&2; exit 2; }

echo "Study root : $STUDY_ROOT"
echo
echo "=== Running SemIf benchmark ==="
python "$RUNNER" --study-root "$STUDY_ROOT" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

echo
echo "=== Building Jev vs SemIf comparison ==="
python "$COMPARE" --study-root "$STUDY_ROOT"

echo
echo "=== Complete ==="
echo "Comparison report: $STUDY_ROOT/comparisons/semif_qwen3-5-4b_mlx/JEV_VS_SEMIF_REPORT.md"
