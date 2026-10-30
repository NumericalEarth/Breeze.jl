#!/bin/bash
#SBATCH -p gpua100 --gres=gpu:1 -c 6 --mem=80G -t 180 -J toy-spawn
#SBATCH -o /home/greg_aeolus_earth/droplet-tasks-env/jobs/toy-%j.out
J=~/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia
cd ~/droplet-tasks-env
H=~/Breeze-droplet-tasks/validation/differentiability
for P in ${PARTS:-r w k g W}; do
    echo "######## parts $P"
    $H/watch.sh $J --startup-file=no --project=. $H/toy_particle_spawning.jl $P 2>&1 | grep -v "^\s*$\|^I0000\|^WARNING: All\|^E0000\|^│\|^┌\|^└" | cut -c1-400
done
