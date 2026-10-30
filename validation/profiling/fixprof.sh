#!/bin/bash
#SBATCH -p gpua100largex4 --gres=gpu:1 -c 6 --mem=100G -t 120 -J droplet-fix
#SBATCH -o /home/greg_aeolus_earth/droplet-tasks-env/jobs/fixprof-%j.out
J=~/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia
S=~/Breeze-droplet-tasks/validation/profiling/droplet_cost.jl
C="32:0,32:6000,64:0,64:6000,64:1000000,128:0,128:1000000,128:10000000"
for env in droplet-tasks-env droplet-profile-env; do
 cd ~/$env; $J --startup-file=no --project=. -e 'using Pkg; Pkg.precompile()' 2>&1 | tail -1
 for self in 0 1; do echo "=== $env SELF=$self"; SELF=$self $J --startup-file=no --project=. $S "$C" 2>&1 | grep -E "RESULT|ERROR" ; done
done
echo "=== fix profile"; cd ~/droplet-profile-env; LABEL=_fix56a5 $J --startup-file=no --project=. $S "128:10000000" profile 2>&1 | grep -E "RESULT|profile"
