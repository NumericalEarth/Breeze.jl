#!/bin/bash
#SBATCH -p gpua100largex4 --gres=gpu:1 -c 6 --mem=100G -t 150 -J droplet-cost
#SBATCH -o /home/greg_aeolus_earth/droplet-tasks-env/jobs/cost2-%j.out
J=~/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia
cd ~/droplet-tasks-env
S=~/Breeze-droplet-tasks/validation/profiling/droplet_cost.jl
$J --startup-file=no --project=. -e 'using Pkg; Pkg.precompile()' 2>&1 | tail -2
C="64:0,64:-1,64:0,64:6000,64:1000000,64:10000000,128:-1,128:0,128:6000,128:1000000,128:10000000,256:-1,256:0,256:6000,256:1000000,256:10000000"
echo "=== unsorted"; $J --startup-file=no --project=. $S "$C" 2>&1 | grep RESULT
echo "=== sorted"; SORT=1 $J --startup-file=no --project=. $S "$C" 2>&1 | grep RESULT
echo "=== profile unsorted"; $J --startup-file=no --project=. $S "128:0,128:10000000" profile 2>&1 | grep -v Warning | grep -v "^│\|^└" | cut -c1-250
echo "=== profile sorted"; SORT=1 $J --startup-file=no --project=. $S "128:0,128:10000000" profile 2>&1 | grep -v "^│\|^└" | cut -c1-250
