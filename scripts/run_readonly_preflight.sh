#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="/home/cunyuliu/rna_junction_preorganization_v1_1_20260801"
ARTIFACT_ROOT="/mnt/cunyuliu/rna_junction_preorganization_v1_1_20260801"
RUN_ID="preflight_$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="$CODE_ROOT/logs"
LOG_PATH="$LOG_DIR/${RUN_ID}.log"

mkdir -p "$LOG_DIR"
umask 077
exec > >(tee "$LOG_PATH") 2>&1

echo "RUN_ID=$RUN_ID"
echo "COMMAND=read-only remote preflight"
echo "HOST=$(hostname)"
echo "USER=$(id -un)"
echo "TIME=$(date -Is)"
echo "CODE_ROOT=$CODE_ROOT"
echo "ARTIFACT_ROOT=$ARTIFACT_ROOT"

echo "--- contract hash ---"
if [ -f "$CODE_ROOT/contract/1.1.docx" ]; then
  sha256sum "$CODE_ROOT/contract/1.1.docx"
else
  echo "CONTRACT_SOURCE=NOT_PRESENT_ON_REMOTE"
  echo "CONTRACT_EXPECTED_SHA256=218dec34037487fae14c50eef2aeb28b79292fe428bd4917a9da1f36687aa0e9"
  echo "STATUS=BLOCKED_CONTRACT_SOURCE_EGRESS_AUTHORIZATION_REQUIRED"
fi

echo "--- project git ---"
git -C "$CODE_ROOT" status --short --branch

echo "--- filesystem ---"
df -h /home /mnt

echo "--- process metadata (no process termination) ---"
ps -u "$(id -un)" -o pid,ppid,stat,etime,pcpu,pmem,comm --sort=-pcpu | head -n 80

echo "--- gpu metadata (no allocation) ---"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader || true
else
  echo "CUDA_PROBE=UNAVAILABLE"
fi

echo "--- artifact root metadata ---"
find "$ARTIFACT_ROOT" -maxdepth 1 -mindepth 1 -printf '%f\n' 2>/dev/null | sort || true

echo "--- terminal status ---"
if [ -f "$CODE_ROOT/contract/1.1.docx" ]; then
  echo "STATUS=READ_ONLY_PREFLIGHT_COMPLETE"
else
  echo "STATUS=READ_ONLY_PREFLIGHT_COMPLETE_WITH_CONTRACT_SOURCE_BLOCKED"
fi
