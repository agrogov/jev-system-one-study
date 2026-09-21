#!/usr/bin/env bash
set -euo pipefail

MODE="${1:---print}"

commands=(
"jevbench --config config/architecture.yaml run --profile core --concurrency 4"
"jevbench --config config/architecture.yaml run --profile core --concurrency 1 --only parallel_question_scaling --only choice_cardinality_scaling --only context_length_position"
"jevbench --config config/architecture.yaml run --profile full --concurrency 1 --only parallel_question_scaling --only choice_cardinality_scaling --only context_length_position"
"jevbench --config config/calibration.yaml run --profile calibration_full --concurrency 4"
"jevbench --config config/semantic.yaml run --profile semantic_full --concurrency 4"
)

case "$MODE" in
  --print)
    printf '%s\n' "${commands[@]}"
    ;;
  --run)
    if [[ -z "${TYPESAFE_API_KEY:-}" ]]; then
      echo "TYPESAFE_API_KEY is not set" >&2
      exit 2
    fi
    for cmd in "${commands[@]}"; do
      echo ">>> $cmd"
      eval "$cmd"
    done
    ;;
  *)
    echo "usage: $0 [--print|--run]" >&2
    exit 2
    ;;
esac
