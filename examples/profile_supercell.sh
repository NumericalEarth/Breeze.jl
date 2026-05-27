#!/bin/bash
#
# Nsight Systems profiling for the distributed supercell benchmark.
# One nsys process per MPI rank; each writes its own .nsys-rep.
#
# Usage:
#   NGPUS=2 sbatch examples/profile_supercell.sh
#   NGPUS=4 NX_PER_GPU=200 PROFILE_STEPS=5 sbatch examples/profile_supercell.sh
#
# Open the resulting reports in the Nsight Systems GUI:
#   nsys-ui examples/nsys_reports/<JOB>/supercell_rank0.nsys-rep
#
#SBATCH --job-name=supercell-nsys
#SBATCH --account=m5176_g
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --output=examples/profile_supercell-%j.out
#SBATCH --error=examples/profile_supercell-%j.err

set -u
PROJECT_DIR="${PROJECT_DIR:-/global/u1/g/glwagner/Breeze.jl}"
cd "${PROJECT_DIR}"

module load julia/1.12.1
module load cudatoolkit  # ships nsys

JULIA="${JULIA:-julia}"
NSYS="${NSYS:-nsys}"

## Julia 1.12.1 on Perlmutter ships libssl.so.3 / libcrypto.so.3 that need
## OPENSSL_3.3.0 symbols; the compute-node system libcrypto.so.3 is older.
## Put Julia's private lib dir first so OpenSSL_jll initializes correctly.
JULIA_PREFIX=$(dirname "$(dirname "$(readlink -f "$(command -v "${JULIA}")")")")
export LD_LIBRARY_PATH="${JULIA_PREFIX}/lib/julia:${LD_LIBRARY_PATH:-}"

## Cray MPICH GPU-Aware MPI. Without this, CUDA buffer registration fails
## inside MPI_Isend. The GTL library must also be preloaded; the explicit
## LD_PRELOAD below covers the case where MPIPreferences's preloads_env_switch
## isn't honored by the loader at the time MPI.jl initializes.
export MPICH_GPU_SUPPORT_ENABLED=1
export LD_PRELOAD="${CRAY_MPICH_ROOTDIR:-/opt/cray/pe/mpich/9.0.1}/gtl/lib/libmpi_gtl_cuda.so${LD_PRELOAD:+:${LD_PRELOAD}}"

## CUDA.jl defaults to cudaMallocAsync (memory pool). Cray MPICH's
## cuIpcGetMemHandle does NOT accept pool allocations — IPC handles can only
## be exported for plain cudaMalloc memory. Disabling the pool fixes the
## "CUDA_ERROR_INVALID_VALUE" / "MPIDI_CRAY_Common_lmt_export_mem" failures.
export JULIA_CUDA_MEMORY_POOL=none

NGPUS="${NGPUS:-2}"
FLOAT_TYPE="${FLOAT_TYPE:-Float32}"
NX_PER_GPU="${NX_PER_GPU:-128}"
NY="${NY:-128}"
NZ="${NZ:-40}"
WARMUP_STEPS="${WARMUP_STEPS:-5}"
PROFILE_STEPS="${PROFILE_STEPS:-10}"

## nsys post-processes the captured stream into a .nsys-rep at job end.
## On Perlmutter, finalizing on the home filesystem fails with "Unknown error 524"
## (Lustre/GPFS interaction). Write to $SCRATCH, then copy the .nsys-rep back here.
REPORT_DIR="examples/nsys_reports/${SLURM_JOB_ID}"
SCRATCH_REPORT_DIR="${SCRATCH:-/tmp}/nsys_reports/${SLURM_JOB_ID}"
mkdir -p "${REPORT_DIR}" "${SCRATCH_REPORT_DIR}"

echo "=========================================="
echo "Supercell nsys profile -- Perlmutter"
echo "=========================================="
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Node:          ${SLURM_NODELIST}"
echo "GPUs:          ${NGPUS}"
echo "Float type:    ${FLOAT_TYPE}"
echo "Grid/GPU:      ${NX_PER_GPU} x ${NY} x ${NZ}"
echo "Warmup steps:  ${WARMUP_STEPS}"
echo "Profile steps: ${PROFILE_STEPS}"
echo "Reports:       ${PROJECT_DIR}/${REPORT_DIR}"
echo "nsys:          $(${NSYS} --version | head -1)"
echo "julia:         $(${JULIA} --version)"
echo "=========================================="

## nsys flags:
##   --capture-range=cudaProfilerApi --capture-range-end=stop
##       only collect inside the CUDA.@profile block (skips JIT/setup).
##   --trace=cuda,nvtx,mpi
##       capture CUDA API/kernels, NVTX ranges (from NVTX.@range), and MPI calls.
##   --mpi-impl=mpich
##       Perlmutter uses Cray MPICH, ABI-compatible with MPICH.
##   --cuda-graph-trace=node
##       fine-grained tracing inside CUDA graphs (Oceananigans/CUDA.jl may use them).
##   -o ...%q{SLURM_PROCID}
##       expand to a unique filename per rank.

srun --ntasks="${NGPUS}" --gpus="${NGPUS}" --gpu-bind=none \
    "${NSYS}" profile \
        --output="${SCRATCH_REPORT_DIR}/supercell_rank%q{SLURM_PROCID}" \
        --trace=cuda,nvtx,mpi \
        --mpi-impl=mpich \
        --cuda-graph-trace=node \
        --capture-range=cudaProfilerApi \
        --capture-range-end=stop \
        --force-overwrite=true \
        --sample=none \
        --cpuctxsw=none \
    "${JULIA}" --project=examples examples/profile_supercell.jl \
        --float-type "${FLOAT_TYPE}" \
        --nx-per-gpu "${NX_PER_GPU}" \
        --ny "${NY}" \
        --nz "${NZ}" \
        --warmup-steps "${WARMUP_STEPS}" \
        --profile-steps "${PROFILE_STEPS}"

rc=$?

## Copy .nsys-rep files into the repo. If only .qdstrm is present (finalize failed
## on the scratch filesystem too), convert it here using QdstrmImporter.
shopt -s nullglob
for f in "${SCRATCH_REPORT_DIR}"/*.nsys-rep; do
    cp "$f" "${REPORT_DIR}/"
done
for f in "${SCRATCH_REPORT_DIR}"/*.qdstrm; do
    base=$(basename "$f" .qdstrm)
    target="${REPORT_DIR}/${base}.nsys-rep"
    if [ ! -f "${target}" ]; then
        echo "Converting ${f} -> ${target}"
        /opt/nvidia/hpc_sdk/Linux_x86_64/25.5/profilers/Nsight_Systems/host-linux-x64/QdstrmImporter \
            --input-file="$f" --output-file="${target}" || cp "$f" "${REPORT_DIR}/"
    fi
done
shopt -u nullglob

echo "=========================================="
echo "nsys profile finished with exit code: ${rc}"
echo "Reports:"
ls -lh "${REPORT_DIR}" || true
echo "=========================================="
exit ${rc}
