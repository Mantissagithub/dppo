#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUNS=("$@")
if [[ ${#RUNS[@]} -eq 0 ]]; then
  RUNS=("ppo" "grpo" "dppo_topk" "dppo_full")
fi

LOG_DIR="$ROOT/outputs/pipeline_logs"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
PIPELINE_LOG="$LOG_DIR/pipeline_${STAMP}.log"

log() {
  echo "[$(date +%H:%M:%S)] $*" | tee -a "$PIPELINE_LOG"
}

run_and_log() {
  log "run: $*"
  "$@" 2>&1 | tee -a "$PIPELINE_LOG"
}

get_env_value() {
  local key="$1"
  python - "$ROOT/.env" "$key" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
key = sys.argv[2]
if not path.exists():
    raise SystemExit(0)
for raw_line in path.read_text(encoding="utf-8").splitlines():
    line = raw_line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    lhs, rhs = line.split("=", 1)
    if lhs.strip() == key:
        print(rhs.strip())
        break
PY
}

have_hf_creds() {
  [[ -f .env ]] && grep -q '^HF_USERNAME=' .env && grep -q '^HF_TOKEN=' .env
}

git_push_with_env_token() {
  local github_token=""
  local auth_header=""
  github_token="$(get_env_value GITHUB_TOKEN)"
  if [[ -z "$github_token" ]]; then
    git push
    return
  fi
  auth_header="$(python - "$github_token" <<'PY'
import base64
import sys

token = sys.argv[1]
payload = f"x-access-token:{token}".encode("utf-8")
print("AUTHORIZATION: basic " + base64.b64encode(payload).decode("ascii"))
PY
)"
  git -c "http.https://github.com/.extraheader=${auth_header}" push
}

clear_run_outputs() {
  case "$1" in
    ppo) rm -rf outputs/ppo outputs/ppo_eval ;;
    grpo) rm -rf outputs/grpo outputs/grpo_eval ;;
    dppo_topk) rm -rf outputs/dppo_topk outputs/dppo_topk_eval ;;
    dppo_full) rm -rf outputs/dppo_full outputs/dppo_full_eval ;;
    *) return 1 ;;
  esac
}

train_run() {
  case "$1" in
    ppo)
      run_and_log python scripts/train_ppo.py --config configs/base.yaml --config configs/ppo.yaml --output-dir outputs/ppo --run-name ppo
      ;;
    grpo)
      run_and_log python scripts/train_grpo.py --config configs/base.yaml --config configs/grpo.yaml --output-dir outputs/grpo --run-name grpo
      ;;
    dppo_topk)
      run_and_log python scripts/train_dppo.py --config configs/base.yaml --config configs/dppo_topk.yaml --output-dir outputs/dppo_topk --run-name dppo-topk
      ;;
    dppo_full)
      run_and_log python scripts/train_dppo.py --config configs/base.yaml --config configs/dppo_full.yaml --output-dir outputs/dppo_full --run-name dppo-full
      ;;
    *)
      return 1
      ;;
  esac
}

eval_run() {
  case "$1" in
    ppo)
      run_and_log python scripts/eval_model.py --config configs/base.yaml --model-path outputs/ppo/final_model --output-dir outputs/ppo_eval
      ;;
    grpo)
      run_and_log python scripts/eval_model.py --config configs/base.yaml --model-path outputs/grpo/final_model --output-dir outputs/grpo_eval
      ;;
    dppo_topk)
      run_and_log python scripts/eval_model.py --config configs/base.yaml --model-path outputs/dppo_topk/final_model --output-dir outputs/dppo_topk_eval
      ;;
    dppo_full)
      run_and_log python scripts/eval_model.py --config configs/base.yaml --model-path outputs/dppo_full/final_model --output-dir outputs/dppo_full_eval
      ;;
    *)
      return 1
      ;;
  esac
}

publish_run() {
  case "$1" in
    ppo)
      run_and_log python scripts/publish_model_card.py --config configs/base.yaml --run-name ppo --training-log outputs/ppo/training.log
      ;;
    grpo)
      run_and_log python scripts/publish_model_card.py --config configs/base.yaml --run-name grpo --training-log outputs/grpo/training.log
      ;;
    dppo_topk)
      run_and_log python scripts/publish_model_card.py --config configs/base.yaml --run-name dppo-topk --training-log outputs/dppo_topk/training.log
      ;;
    dppo_full)
      run_and_log python scripts/publish_model_card.py --config configs/base.yaml --run-name dppo-full --training-log outputs/dppo_full/training.log
      ;;
    *)
      return 1
      ;;
  esac
}

log "pipeline log: $PIPELINE_LOG"
run_and_log python scripts/prepare_gsm8k.py --config configs/base.yaml

for run_name in "${RUNS[@]}"; do
  log "reset outputs for $run_name"
  clear_run_outputs "$run_name"
  log "train $run_name"
  train_run "$run_name"
  log "eval $run_name"
  eval_run "$run_name"
done

log "clean outputs"
run_and_log python scripts/clean_outputs.py --outputs-root outputs --runs "${RUNS[@]}"

log "plot outputs"
run_and_log python scripts/plot_results.py --outputs-root outputs --save-dir outputs/plots --runs "${RUNS[@]}"

log "attempt git push"
if git_push_with_env_token 2>&1 | tee -a "$PIPELINE_LOG"; then
  log "git push succeeded"
else
  log "git push failed, moving to next step"
fi

if have_hf_creds; then
  for run_name in "${RUNS[@]}"; do
    log "publish hf model card for $run_name"
    publish_run "$run_name"
  done
else
  log "hf credentials missing in .env, skipping hf model-card publish"
fi

log "done"
