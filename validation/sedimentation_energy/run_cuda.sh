#!/usr/bin/env bash
#SBATCH --job-name=pr959-energy
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1

set -euo pipefail
output_directory=${1:?Specify an output directory}
julia_command=${JULIA:-julia}
mkdir -p "$output_directory"
export JULIA_NUM_THREADS=2
date -u
nvidia-smi
for precision in 64 32; do
    for suite in isolated coupled; do
        "$julia_command" --project=validation/sedimentation_energy \
            validation/sedimentation_energy/run_campaign.jl cuda "$precision" \
            "$output_directory/cuda-$precision-$suite.toml" "$suite"
    done
done
