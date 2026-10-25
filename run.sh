#!/usr/bin/env bash
set -euo pipefail
python run.py --audio-dir "${1:-./customer_audio}" --output-dir "${2:-./sample_outputs}" --language "${3:-en-IN}"
