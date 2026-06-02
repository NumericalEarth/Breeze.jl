#!/bin/bash
#
# Multi-node, multi-GPU production run of the YuDidlake2019 tropical-cyclone
# rainband case (examples/distributed_tropical_cyclone.jl).
#
# Perlmutter has 4× A100 per GPU node, so NGPUS must be a multiple of 4 for
# whole-node allocations (and a divisor of Nx for the x-partition).
#
# Calibration (recommended FIRST — short timing run to estimate wall time):
#   NGPUS=20 DX=100 NZ=75 BENCHMARK_STEPS=10 sbatch examples/distributed_tropical_cyclone.sh
#
# Production spinup (24 h, fields to disk every 1 h):
#   NGPUS=20 DX=100 NZ=75 STOP_TIME=24 OUTPUT_INTERVAL=1 QOS=regular TIME=24:00:00 \
#     sbatch examples/distributed_tropical_cyclone.sh
#
#SBATCH --job-name=tc-rainband
#SBATCH --account=m5176_g
#SBATCH --constraint=gpu
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --output=examples/distributed_tropical_cyclone-%j.out
#SBATCH --error=examples/distributed_tropical_cyclone-%j.err

set -u
PROJECT_DIR="${PROJECT_DIR:-/global/u1/g/glwagner/Breeze.jl}"
cd "${PROJECT_DIR}"

module load julia/1.12.1

JULIA="${JULIA:-julia}"

## Julia 1.12.1 on Perlmutter ships libssl/libcrypto 3.x needing OPENSSL_3.3.0
## symbols newer than the compute-node system libs. Put Julia's private lib dir
## first so OpenSSL_jll initializes correctly.
JULIA_PREFIX=$(dirname "$(dirname "$(readlink -f "$(command -v "${JULIA}")")")")
export LD_LIBRARY_PATH="${JULIA_PREFIX}/lib/julia:${LD_LIBRARY_PATH:-}"

## Cray MPICH GPU-aware MPI + GTL preload (required for CUDA buffers in MPI calls).
export MPICH_GPU_SUPPORT_ENABLED=1
export LD_PRELOAD="${CRAY_MPICH_ROOTDIR:-/opt/cray/pe/mpich/9.0.1}/gtl/lib/libmpi_gtl_cuda.so${LD_PRELOAD:+:${LD_PRELOAD}}"

## CUDA.jl defaults to cudaMallocAsync; Cray MPICH's cuIpcGetMemHandle rejects
## pool allocations. Disable the pool to avoid CUDA_ERROR_INVALID_VALUE in MPI.
export JULIA_CUDA_MEMORY_POOL=none

## NCCL AWS-OFI plugin for CXI/Slingshot GPUDirect RDMA. WITHOUT THIS, NCCL silently
## falls back to 10 Gbps TCP sockets (RDMA disabled) and inter-node runs ~3.3× slower.
## 2.18.3 is the newest libnccl-net.so that loads with NCCL_jll 2.28.3 (AWS Libfabric v6).
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-hsn}"
PLUGIN="${PLUGIN:-2.18.3}"
if [ -n "${PLUGIN}" ] && [ "${NCCL:-1}" != "0" ]; then
    export LD_LIBRARY_PATH="/global/common/software/nersc9/nccl/${PLUGIN}/lib:${LD_LIBRARY_PATH}"
    export NCCL_NET_GDR_LEVEL=PHB
    export NCCL_CROSS_NIC=1
    export FI_CXI_DISABLE_HOST_REGISTER=1
    export FI_MR_CACHE_MONITOR=userfaultfd
    export FI_CXI_DEFAULT_CQ_SIZE=131072
fi

NGPUS="${NGPUS:-60}"
FLOAT_TYPE="${FLOAT_TYPE:-Float32}"
DX="${DX:-100}"
VERTICAL="${VERTICAL:-gate}"      # 'gate' (181-level stretched) or 'uniform'
NZ="${NZ:-75}"                    # uniform mode only
STOP_TIME="${STOP_TIME:-24}"
OUTPUT_INTERVAL="${OUTPUT_INTERVAL:-1}"
BENCHMARK_STEPS="${BENCHMARK_STEPS:-0}"
HEATING="${HEATING:-0}"          # 1 = rainband heating on, 0 = spinup
NCCL="${NCCL:-1}"                # 1 = NCCL communicator, 0 = MPI
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-6}"  # checkpoint cadence, sim-hours
RESTART="${RESTART:-0}"          # 1 = pick up from latest checkpoint
DT="${DT:-0.4}"                  # fixed time step, s (wizard is buggy on distributed face fields)
SUBSTEPS="${SUBSTEPS:-0}"        # acoustic substeps; 0 = compute host-side from Δx (avoids minimum_xspacing)
DIAGNOSE="${DIAGNOSE:-0}"        # 1 = print per-rank velocity/momentum extrema after set!
OUTPUT_DIR="${OUTPUT_DIR:-}"     # override output/checkpoint dir (default: $SCRATCH/tc_distributed)

echo "=========================================="
echo "Distributed TC rainband -- Perlmutter"
echo "=========================================="
echo "Job ID:          ${SLURM_JOB_ID}"
echo "Nodes:           ${SLURM_JOB_NUM_NODES} (${SLURM_NODELIST})"
echo "GPUs:            ${NGPUS}"
echo "Float type:      ${FLOAT_TYPE}"
echo "Δx:              ${DX} m    Vertical: ${VERTICAL} (Nz=${NZ} if uniform)"
echo "Stop time:       ${STOP_TIME} h    Output every: ${OUTPUT_INTERVAL} h"
echo "Benchmark steps: ${BENCHMARK_STEPS}  (0 = full production run)"
echo "Heating:         $([ "${HEATING}" = "1" ] && echo on || echo off)"
echo "Backend:         $([ "${NCCL}" = "0" ] && echo MPI || echo NCCL)"
echo "julia:           $(${JULIA} --version)"
echo "=========================================="

srun --ntasks="${NGPUS}" --gpu-bind=none \
    "${JULIA}" --project=examples examples/distributed_tropical_cyclone.jl \
        --float-type "${FLOAT_TYPE}" \
        --dx "${DX}" \
        --vertical "${VERTICAL}" \
        --nz "${NZ}" \
        --stop-time "${STOP_TIME}" \
        --output-interval "${OUTPUT_INTERVAL}" \
        --benchmark-steps "${BENCHMARK_STEPS}" \
        --checkpoint-interval "${CHECKPOINT_INTERVAL}" \
        --dt "${DT}" \
        --substeps "${SUBSTEPS}" \
        ${OUTPUT_DIR:+--output-dir "${OUTPUT_DIR}"} \
        $([ "${DIAGNOSE}" = "1" ] && echo "--diagnose") \
        $([ "${HEATING}" = "1" ] && echo "--heating") \
        $([ "${RESTART}" = "1" ] && echo "--restart") \
        $([ "${NCCL}" = "0" ] && echo "--no-nccl")

rc=$?
echo "=========================================="
echo "Run finished with exit code: ${rc}"
echo "=========================================="
exit ${rc}
