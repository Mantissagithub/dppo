#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

write_env_value() {
  local key="$1"
  local value="$2"
  python - "$ROOT/.env" "$key" "$value" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
key = sys.argv[2]
value = sys.argv[3]
lines = []
if path.exists():
    lines = path.read_text(encoding="utf-8").splitlines()
updated = False
for idx, line in enumerate(lines):
    if line.startswith(f"{key}="):
        lines[idx] = f"{key}={value}"
        updated = True
        break
if not updated:
    lines.append(f"{key}={value}")
path.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
}

prompt_hf_credentials() {
  local current_username=""
  local current_token=""
  if [[ -f .env ]]; then
    current_username="$(grep '^HF_USERNAME=' .env | cut -d= -f2- || true)"
    current_token="$(grep '^HF_TOKEN=' .env | cut -d= -f2- || true)"
  fi

  if [[ -z "$current_username" ]]; then
    read -r -p "hf_username: " current_username
    write_env_value HF_USERNAME "$current_username"
  else
    echo "using HF_USERNAME from .env: $current_username"
  fi

  if [[ -z "$current_token" ]]; then
    read -r -s -p "hf_token: " current_token
    echo
    write_env_value HF_TOKEN "$current_token"
  else
    echo "using HF_TOKEN from .env"
  fi
}

run_timed() {
  local name="$1"
  shift
  set +e
  timeout --signal=INT --kill-after=30s "$@"
  local status=$?
  set -e
  if [[ "$status" -ne 0 && "$status" -ne 124 ]]; then
    echo "$name failed with status $status"
    return "$status"
  fi
  if [[ "$status" -eq 124 ]]; then
    echo "$name hit timeout and moved on"
  fi
}

prompt_hf_credentials

python scripts/prepare_gsm8k.py --config configs/base.yaml

python scripts/train_ppo.py --config configs/base.yaml --config configs/ppo.yaml --config configs/smoke.yaml --output-dir outputs/smoke_ppo --run-name smoke-ppo
python scripts/train_grpo.py --config configs/base.yaml --config configs/grpo.yaml --config configs/smoke.yaml --output-dir outputs/smoke_grpo --run-name smoke-grpo
python scripts/train_dppo.py --config configs/base.yaml --config configs/dppo_topk.yaml --config configs/smoke.yaml --output-dir outputs/smoke_dppo_topk --run-name smoke-dppo-topk

run_timed ppo 80m python scripts/train_ppo.py --config configs/base.yaml --config configs/ppo.yaml --output-dir outputs/ppo --run-name ppo
run_timed grpo 80m python scripts/train_grpo.py --config configs/base.yaml --config configs/grpo.yaml --output-dir outputs/grpo --run-name grpo
run_timed dppo 80m python scripts/train_dppo.py --config configs/base.yaml --config configs/dppo_topk.yaml --output-dir outputs/dppo_topk --run-name dppo-topk

python scripts/plot_results.py --outputs-root outputs --save-path outputs/summary.png
