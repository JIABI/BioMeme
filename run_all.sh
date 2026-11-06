#!/usr/bin/env bash
set -euo pipefail
pkg_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_exec="${BIOMEME_PYTHON:-python3}"
result_dir="${1:-$pkg_root/results}"
"$python_exec" -c 'import sys; assert sys.version_info[:2] == (3,12), "Use Python 3.12; tested patch is 3.12.14."'
"$python_exec" -m venv "$pkg_root/.venv"
"$pkg_root/.venv/bin/python" -m pip install --disable-pip-version-check -r "$pkg_root/requirements.txt" -r "$pkg_root/requirements-test.txt"
if [[ -n "${BIOMEME_DATA_ARCHIVE:-}" ]]; then
    "$pkg_root/.venv/bin/python" "$pkg_root/scripts/fetch_data.py" --archive "$BIOMEME_DATA_ARCHIVE"
else
    "$pkg_root/.venv/bin/python" "$pkg_root/scripts/fetch_data.py"
fi
"$pkg_root/.venv/bin/python" "$pkg_root/run_demo.py" --output "$result_dir/demo"
"$pkg_root/.venv/bin/python" "$pkg_root/reproduce_stats.py" --output "$result_dir/statistics"
BIOMEME_DEMO_RESULTS="$result_dir/demo" BIOMEME_STATS_RESULTS="$result_dir/statistics" "$pkg_root/.venv/bin/python" -m pytest -q "$pkg_root/tests"
