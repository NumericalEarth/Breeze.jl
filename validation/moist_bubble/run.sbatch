#!/bin/bash
#SBATCH --partition=gpua100,gpuprodflex,gpuprod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/greg_aeolus_earth/rce-energy/moist-bubble-work/logs/%x-%j.log
# Usage: sbatch --time=... --job-name=moist-bubble-X run.sbatch "<ENV=VALUE ...>" ["<ENV=VALUE ...>" ...]  (run in sequence)
set -u
ROOT=$HOME/rce-energy/Breeze-fix-moist-bubble
export RESULTS_DIR=$HOME/rce-energy/moist-bubble-work/results
export JULIA_NUM_THREADS=4
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "COMMIT $(git -C $ROOT rev-parse --short HEAD) HOST $(hostname)"
cd $ROOT
julia +1.11.9 --startup-file=no --project=examples -e 'using Pkg; Pkg.instantiate(); using Breeze, Oceananigans, CUDA, CloudMicrophysics; CUDA.functional() && println(CUDA.name(CUDA.device()))' 2>&1 | tail -5
for envs in "$@"; do
    echo "=== $envs"
    env $envs julia +1.11.9 --startup-file=no --project=examples validation/moist_bubble/bubble.jl 2>&1
done
echo done
