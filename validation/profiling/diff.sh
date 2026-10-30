#!/bin/bash
#SBATCH -p gpua100largex4 --gres=gpu:1 -c 6 --mem=100G -t 120 -J droplet-ad
#SBATCH -o /home/greg_aeolus_earth/droplet-tasks-env/jobs/diff-%j.out
J=~/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia
cd ~/droplet-tasks-env
D=~/Breeze-droplet-tasks/validation/profiling/droplet_differentiability.jl
echo "=== verbose"; ENZYME_VERBOSE=1 ERRLEN=6000 $J --startup-file=no --project=. $D k 2>&1 | grep -v "^\s*$" | cut -c1-400 | head -150
echo "=== loose"; ENZYME_LOOSE=1 $J --startup-file=no --project=. $D kbc 2>&1 | grep -v "^\s*$\|^I0000\|^WARNING: All" | cut -c1-900 | tail -80
