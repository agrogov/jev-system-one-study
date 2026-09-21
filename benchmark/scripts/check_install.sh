#!/usr/bin/env bash
set -euo pipefail

python --version
python - <<'PY'
import h2
import httpx
import matplotlib
import numpy
import pandas
import yaml
import jevbench
print("jevbench import: OK")
print("httpx:", httpx.__version__)
print("h2:", h2.__version__)
print("numpy:", numpy.__version__)
print("pandas:", pandas.__version__)
print("matplotlib:", matplotlib.__version__)
print("PyYAML:", yaml.__version__)
PY

jevbench --config config/architecture.yaml generate --profile core --list >/tmp/jevbench-architecture.txt
jevbench --config config/calibration.yaml generate --profile calibration_full --list >/tmp/jevbench-calibration.txt
jevbench --config config/semantic.yaml generate --profile semantic_full --list >/tmp/jevbench-semantic.txt

grep -q 'profile=core cases=303' /tmp/jevbench-architecture.txt
grep -q 'profile=calibration_full cases=5810' /tmp/jevbench-calibration.txt
grep -q 'profile=semantic_full cases=1110' /tmp/jevbench-semantic.txt

echo "Suite generation checks: OK"
echo "Installation check: PASS"
