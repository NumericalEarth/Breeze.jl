#!/bin/bash
# sbatch validation/float32/run_l1a.sh RUNS   (decaying turbulence from the NCAR 32³ L1b field, Float64 then Float32)
#SBATCH -p gpua100largex4
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=60G
#SBATCH -t 240
#SBATCH -J f32-L1a
#SBATCH -o /home/greg_aeolus_earth/float32-env/jobs/l1a-%j.out
set -e
J=~/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia
HERE=/home/greg_aeolus_earth/Breeze-float32/validation/float32
export JULIA_NUM_THREADS=8 SNAPSHOTS=0 OUTPUT_DIR=$HERE/output/l1a
nvidia-smi -L
for F in ${FLOATS:-Float64 Float32}; do
  FLOAT=$F $J --startup-file=no --project=$HERE $HERE/resolution_runs.jl decay \
    /home/greg_aeolus_earth/ncar-dns/runs/L1b/n32_seed1/Zk10.out.ncf ${1:-64:centered4,128:centered4} 0.4261658353450919 $HERE/ncar_dump_times_20.txt
done
echo "=== done $(date)"
