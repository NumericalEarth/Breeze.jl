#!/usr/bin/env bash
set -euo pipefail

export PATH="/usr/local/cuda-13.0/bin:${PATH}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
env_dir="${RADIATIVE_HEATING_GPU_ENV_DIR:-${repo_root}/benchmarking/results/gpu_environment_h100}"
result_dir="${RADIATIVE_HEATING_RCEMIP_DIR:-${repo_root}/benchmarking/results/rcemip_h100_32x32x64}"
nsight_dir="${result_dir}/nsight"
nsys_report="${nsight_dir}/radiative_heating_rcemip_nsys.nsys-rep"
ncu_report="${nsight_dir}/radiative_heating_rcemip_ncu.ncu-rep"

if [[ -n "${RADIATIVE_HEATING_SLURM_JOB_ID:-}" && -z "${SLURM_STEP_ID:-}" ]]; then
  exec srun --jobid="${RADIATIVE_HEATING_SLURM_JOB_ID}" \
    --export=ALL,RADIATIVE_HEATING_GPU_ENV_DIR="${env_dir}",RADIATIVE_HEATING_RCEMIP_DIR="${result_dir}" \
    "$0"
fi

if [[ "${RADIATIVE_HEATING_USE_SRUN:-false}" == "true" && -z "${SLURM_JOB_ID:-}" ]]; then
  exec srun \
    --partition="${SLURM_PARTITION:-gpu-prod}" \
    --gres="${SLURM_GRES:-gpu:h100:1}" \
    --export=ALL,RADIATIVE_HEATING_GPU_ENV_DIR="${env_dir}",RADIATIVE_HEATING_RCEMIP_DIR="${result_dir}" \
    "$0"
fi

cd "${repo_root}"
julia --project=benchmarking benchmarking/radiative_heating_gpu_environment_check.jl

mkdir -p "${nsight_dir}"

export RADIATIVE_HEATING_BACKEND="${RADIATIVE_HEATING_BACKEND:-H100}"
export RADIATIVE_HEATING_GAS_MODEL_SOURCE="${RADIATIVE_HEATING_GAS_MODEL_SOURCE:-validated_ecCKD}"
export RADIATIVE_HEATING_BENCHMARK_STATUS="${RADIATIVE_HEATING_BENCHMARK_STATUS:-final_4x_evidence}"
export RADIATIVE_HEATING_NX="${RADIATIVE_HEATING_NX:-32}"
export RADIATIVE_HEATING_NY="${RADIATIVE_HEATING_NY:-32}"
export RADIATIVE_HEATING_NZ="${RADIATIVE_HEATING_NZ:-64}"
export RADIATIVE_HEATING_SAMPLES="${RADIATIVE_HEATING_SAMPLES:-5}"
export RADIATIVE_HEATING_RCEMIP_DIR="${result_dir}"
export RADIATIVE_HEATING_NSYS_REPORT="${nsys_report}"
export RADIATIVE_HEATING_NCU_REPORT="${ncu_report}"

julia --project=benchmarking benchmarking/radiative_heating_h100_support_preflight.jl

nsys profile \
  --force-overwrite=true \
  --output="${nsight_dir}/radiative_heating_rcemip_nsys" \
  julia --project=benchmarking benchmarking/radiative_heating_rcemip_benchmark.jl

ncu \
  --force-overwrite \
  --export="${ncu_report}" \
  julia --project=benchmarking benchmarking/radiative_heating_rcemip_benchmark.jl
