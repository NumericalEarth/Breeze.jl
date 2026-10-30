#!/bin/bash
# sbatch validation/float32/run_l3.sh RUNS STOP_TIME   (env: FLOAT, LABEL, OUTPUT_DIR, SELF, ...)
#SBATCH -p gpua100largex4
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=60G
#SBATCH -t 240
#SBATCH -J f32-L3
#SBATCH -o /home/greg_aeolus_earth/float32-env/jobs/l3-%j.out
set -e
J=~/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia
HERE=/home/greg_aeolus_earth/Breeze-float32/validation/float32
export JULIA_NUM_THREADS=8
nvidia-smi -L
echo "commit $(git -C $HERE rev-parse HEAD) FLOATS=$FLOATS LABEL=$LABEL"
for F in ${FLOATS:-Float64 Float32}; do FLOAT=$F LABEL=${LABEL}_$F $J --startup-file=no --project=$HERE $HERE/condensation.jl /home/greg_aeolus_earth/ncar-dns/runs/L3 $1 ${2:-10}; done
echo "=== done $(date)"
