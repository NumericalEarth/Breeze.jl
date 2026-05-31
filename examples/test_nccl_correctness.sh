#!/bin/bash
#
# NCCL correctness test (reductions + halo fill + set!) vs analytic truth.
# Runs the test twice in one job: NCCL backend, then MPI backend, for comparison.
#
#   NGPUS=4 sbatch --qos=debug --time=00:20:00 --nodes=1 examples/test_nccl_correctness.sh
#   NGPUS=8 sbatch --qos=debug --time=00:20:00 --nodes=2 examples/test_nccl_correctness.sh
#
#SBATCH --job-name=nccl-test
#SBATCH --account=m5176_g
#SBATCH --constraint=gpu
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --output=examples/test_nccl_correctness-%j.out
#SBATCH --error=examples/test_nccl_correctness-%j.err

set -u
PROJECT_DIR="${PROJECT_DIR:-/global/u1/g/glwagner/Breeze.jl}"
cd "${PROJECT_DIR}"
module load julia/1.12.1
JULIA="${JULIA:-julia}"

JULIA_PREFIX=$(dirname "$(dirname "$(readlink -f "$(command -v "${JULIA}")")")")
export LD_LIBRARY_PATH="${JULIA_PREFIX}/lib/julia:${LD_LIBRARY_PATH:-}"
export MPICH_GPU_SUPPORT_ENABLED=1
export LD_PRELOAD="${CRAY_MPICH_ROOTDIR:-/opt/cray/pe/mpich/9.0.1}/gtl/lib/libmpi_gtl_cuda.so${LD_PRELOAD:+:${LD_PRELOAD}}"
export JULIA_CUDA_MEMORY_POOL=none

## --- Inter-node NCCL networking + debug ---
## Slingshot NIC for NCCL bootstrap + socket transport (prefix-matches hsn0..3).
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-hsn}"

## NCCL_DEBUG_LEVEL=INFO|WARN turns on NCCL's own diagnostics (bypasses the
## NCCL.jl `err` masking — shows transport selection and where it stalls).
if [ -n "${NCCL_DEBUG_LEVEL:-}" ]; then
    export NCCL_DEBUG="${NCCL_DEBUG_LEVEL}"
    export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET,ENV,BOOTSTRAP}"
fi

## Optional AWS-OFI plugin: PLUGIN=<dir under /global/common/software/nersc9/nccl>
## (only 2.15.5 / 2.17.1-cuda11 / 2.18.3 ship libnccl-net.so). Empty ⇒ NCCL's
## built-in socket transport over hsn (works without a plugin, just slower).
if [ -n "${PLUGIN:-}" ]; then
    export LD_LIBRARY_PATH="/global/common/software/nersc9/nccl/${PLUGIN}/lib:${LD_LIBRARY_PATH}"
    export NCCL_NET_GDR_LEVEL=PHB
    export NCCL_CROSS_NIC=1
    export FI_CXI_DISABLE_HOST_REGISTER=1
    export FI_MR_CACHE_MONITOR=userfaultfd
    export FI_CXI_DEFAULT_CQ_SIZE=131072
fi

NGPUS="${NGPUS:-4}"
BACKEND="${BACKEND:-both}"   # 'nccl', 'mpi', or 'both'
echo "=== NCCL correctness test: ${NGPUS} GPUs, ${SLURM_JOB_NUM_NODES} node(s), OFI=${OFI:-0}, BACKEND=${BACKEND} ==="

if [ "${BACKEND}" = "nccl" ] || [ "${BACKEND}" = "both" ]; then
    echo ">>> Backend: NCCL"
    srun --ntasks="${NGPUS}" --gpu-bind=none \
        "${JULIA}" --project=examples examples/test_nccl_correctness.jl
fi

if [ "${BACKEND}" = "mpi" ] || [ "${BACKEND}" = "both" ]; then
    echo ">>> Backend: MPI"
    srun --ntasks="${NGPUS}" --gpu-bind=none \
        "${JULIA}" --project=examples examples/test_nccl_correctness.jl --no-nccl
fi

echo "=== done ==="
