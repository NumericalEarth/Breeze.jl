#!/bin/bash
#SBATCH -p gpua100largex4 --gres=gpu:1 -c 6 --mem=100G -t 60 -J droplet-prof
#SBATCH -o /home/greg_aeolus_earth/droplet-tasks-env/jobs/prof-%j.out
J=~/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia
cd ~/droplet-tasks-env
S=~/Breeze-droplet-tasks/validation/profiling/droplet_cost.jl
echo "=== profile unsorted"; $J --startup-file=no --project=. $S "${1:-128:0,128:10000000}" profile 2>&1 | grep -E "RESULT|profile written"
echo "=== profile sorted"; SORT=1 $J --startup-file=no --project=. $S "${1:-128:10000000}" profile 2>&1 | grep -E "RESULT|profile written"
