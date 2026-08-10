#!/bin/bash
set -e
RUN=/home/cunyuliu/rna_junction_r0_20260809T105504Z
export PATH=/home/cunyuliu/miniconda3/envs/rna_junction_preorganization_v1_1/bin:$PATH
export PYTHONPATH=$RUN
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
rm -rf /tmp/fzl_shards && mkdir -p /tmp/fzl_shards
N=16
cd $RUN
for i in $(seq 0 $((N-1))); do
  nohup python -u audit/r1_frozen_lm_parallel_worker.py audit/provenance/r1_frozen_lm_cfg.json $i $N /tmp/fzl_shards > /tmp/fzl_worker_${i}.log 2>&1 &
  echo "worker $i pid $!"
done
echo "ALL_LAUNCHED N=$N"
