#!/usr/bin/env bash
set -u -o pipefail

if [ "$#" -ne 3 ]; then
  echo "usage: $0 ENV_PREFIX LOG_PATH STATUS_PATH" >&2
  exit 64
fi

prefix=$1
log=$2
status=$3
mkdir -p "$(dirname "$log")" "$(dirname "$status")"

{
  printf 'started_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'prefix=%s\n' "$prefix"
  printf 'torch_spec=torch==2.9.0\n'
  printf 'torch_index=https://download.pytorch.org/whl/cu126\n'
  "$prefix/bin/python" -m pip install --no-cache-dir torch==2.9.0 --index-url https://download.pytorch.org/whl/cu126
  torch_rc=$?
  printf 'torch_rc=%s\n' "$torch_rc" > "$status"
  if [ "$torch_rc" -eq 0 ]; then
    "$prefix/bin/python" -m pip install --no-cache-dir biopython
    biopython_rc=$?
    printf 'biopython_rc=%s\n' "$biopython_rc" >> "$status"
  else
    printf 'biopython_rc=SKIPPED\n' >> "$status"
  fi
  printf 'finished_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$status"
  exit "$torch_rc"
} > >(tee "$log") 2>&1
