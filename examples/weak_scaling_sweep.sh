#!/bin/bash
#
# Weak-scaling sweep for the distributed split-explicit supercell:
# fixed work PER GPU, increasing GPU count 1 → 2 → 4 → 8. With 4 GPUs/node,
# 1/2/4 are intra-node and 8 crosses to a second node, so the sweep isolates
# the inter-node communication cost. One 2-node allocation, sequential srun steps.
#
#   sbatch --qos=debug --time=00:30:00 --nodes=2 examples/weak_scaling_sweep.sh
#
#SBATCH --job-name=weak-scaling
#SBATCH --account=m5176_g
#SBATCH --constraint=gpu
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --output=examples/weak_scaling_sweep-%j.out
#SBATCH --error=examples/weak_scaling_sweep-%j.err

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
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-hsn}"

## NCCL AWS-OFI plugin for CXI/Slingshot RDMA (otherwise NCCL falls back to 10 Gbps
## TCP sockets inter-node). PLUGIN=2.18.3 is the newest libnccl-net.so that loads
## with NCCL_jll 2.28.3 (AWS Libfabric v6, GPUDirect RDMA enabled).
if [ -n "${PLUGIN:-}" ]; then
    export LD_LIBRARY_PATH="/global/common/software/nersc9/nccl/${PLUGIN}/lib:${LD_LIBRARY_PATH}"
    export NCCL_NET_GDR_LEVEL=PHB
    export NCCL_CROSS_NIC=1
    export FI_CXI_DISABLE_HOST_REGISTER=1
    export FI_MR_CACHE_MONITOR=userfaultfd
    export FI_CXI_DEFAULT_CQ_SIZE=131072
fi

## Fixed per-GPU problem size (weak scaling) and backend.
NX_PER_GPU="${NX_PER_GPU:-200}"
NY="${NY:-200}"
NZ="${NZ:-100}"
WARMUP="${WARMUP:-5}"
PROFILE="${PROFILE:-10}"
BACKENDFLAG=""
[ "${NCCL:-1}" = "0" ] && BACKENDFLAG="--no-nccl"

echo "=== Weak-scaling sweep: ${NX_PER_GPU}x${NY}x${NZ} per GPU, backend=$([ "${NCCL:-1}" = 0 ] && echo MPI || echo NCCL) ==="

for N in ${GPUS:-1 2 4 8}; do
    echo ""
    echo ">>>>>> GPUs = ${N} <<<<<<"
    srun --ntasks="${N}" --ntasks-per-node=4 --gpu-bind=none \
        "${JULIA}" --project=examples examples/profile_supercell.jl \
            --nx-per-gpu "${NX_PER_GPU}" --ny "${NY}" --nz "${NZ}" \
            --warmup-steps "${WARMUP}" --profile-steps "${PROFILE}" \
            ${BACKENDFLAG} \
        || echo "GPUs=${N} FAILED (rc=$?)"
done

echo "=== sweep done ==="
