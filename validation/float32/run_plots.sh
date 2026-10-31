#!/bin/bash
# sbatch validation/float32/run_plots.sh   (all figures of FLOAT32_DNS.md)
#SBATCH -p gpua100largex4
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=60G
#SBATCH -t 60
#SBATCH -J f32-plots
#SBATCH -o /home/greg_aeolus_earth/float32-env/jobs/plots-%j.out
set -e
J=~/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia
HERE=${RUN_TREE:-/home/greg_aeolus_earth/Breeze-float32-run2}/validation/float32
OUT=/home/greg_aeolus_earth/Breeze-float32/validation/float32/output
FIG=/home/greg_aeolus_earth/lagrangian-microphysics-references/ice_scope/figures
run() { $J --startup-file=no --project=$HERE "$@" 2>&1 | grep -v "^\s*$"; }
run $HERE/plot_l1a.jl $OUT/l1a $FIG/float32_l1a
run $HERE/plot_l3.jl $OUT/l3 $FIG/float32_l3_M64 64 "f64 f32naive f32 f32rn f32sr f32fluct f32mixed" 1,2,3
run $HERE/plot_l3.jl $OUT/l3_128 $FIG/float32_l3_M128 128 "f64 f32 f32mixed" 1
run $HERE/plot_cost.jl $OUT/l1a/timing.csv $OUT/profile/timing.csv $FIG/float32_cost.png
echo "=== done $(date)"
