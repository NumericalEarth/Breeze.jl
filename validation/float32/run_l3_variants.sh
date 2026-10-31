#!/bin/bash
# sbatch validation/float32/run_l3_variants.sh RUNS STOP_TIME "VARIANTS"
# VARIANTS: space-separated names among f64 f32naive f32 f32mixed (see REPORT); all runs sequential on one GPU.
#SBATCH -p gpua100largex4
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=60G
#SBATCH -t 720
#SBATCH -J f32-L3v
#SBATCH -o /home/greg_aeolus_earth/float32-env/jobs/l3v-%j.out
set -e
J=~/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia
RUN=/home/greg_aeolus_earth/Breeze-float32-run
NAIVE=/home/greg_aeolus_earth/Breeze-float32-naive
export JULIA_NUM_THREADS=8 OUTPUT_DIR=${OUTPUT_DIR:-/home/greg_aeolus_earth/Breeze-float32/validation/float32/output/l3}
nvidia-smi -L
echo "run tree $(git -C $RUN rev-parse HEAD); naive tree $(git -C $NAIVE rev-parse HEAD)"
for V in ${3:-f64 f32naive f32 f32mixed}; do
  export FLUCTUATION_STORAGE=0
  case $V in
    f64)      T=$RUN;   F=Float64; TF="" ;;
    f32naive) T=$NAIVE; F=Float32; TF="" ;;
    f32)      T=$RUN;   F=Float32; TF="" ;;
    f32mixed) T=$RUN;   F=Float32; TF=Float64 ;;
    f32fluct) T=$RUN;   F=Float32; TF=Float64; export FLUCTUATION_STORAGE=1 ;;
  esac
  echo "=== variant $V: tree $T FLOAT=$F THERMO_FLOAT=$TF $(date)"
  P=$T/validation/float32
  $J --startup-file=no --project=$P -e 'using Pkg; Pkg.instantiate()' 2>&1 | tail -2
  FLOAT=$F THERMO_FLOAT=$TF LABEL=_$V $J --startup-file=no --project=$P $P/condensation.jl \
      /home/greg_aeolus_earth/ncar-dns/runs/L3 $1 ${2:-10} 2>&1 | grep -v "^\s*$" | grep -v "CUDA library\|profiler\|LD_LIBRARY\|file an issue\|initialization.jl\|^│ *$"
done
echo "=== done $(date)"
