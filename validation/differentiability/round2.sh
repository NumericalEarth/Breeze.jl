#!/bin/bash
#SBATCH -p gpua100 --gres=gpu:1 -c 6 --mem=80G -t 240 -J diff-round2
#SBATCH -o /home/greg_aeolus_earth/droplet-tasks-env/jobs/round2-%j.out
J=~/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia
cd ~/droplet-tasks-env
H=~/Breeze-droplet-tasks/validation/differentiability
F="^\s*$\|^I0000\|^WARNING: All\|^E0000\|^W0000\|^│\|^┌\|^└"
export MAXGB=25 MAXSEC=900
echo "######## toy V (while reproducers)"
$H/watch.sh $J --startup-file=no --project=. $H/toy_particle_spawning.jl V 2>&1 | grep -v "$F" | cut -c1-400
echo "######## toy r with unique sort keys"
SORTKEYS=true $H/watch.sh $J --startup-file=no --project=. $H/toy_particle_spawning.jl r 2>&1 | grep -v "$F" | cut -c1-400
echo "######## toy w (traced for, runtime trip count)"
$H/watch.sh $J --startup-file=no --project=. $H/toy_particle_spawning.jl w 2>&1 | grep -v "$F" | cut -c1-400 | grep -v "^ *%\|^ *stablehlo\|^ *}\|^ *func\|^ *\^"
export MAXGB=60 MAXSEC=1800
for level in condense update advect step0 step; do
    echo "######## droplets $level"
    $H/watch.sh $J --startup-file=no --project=. $H/droplets_reactant.jl $level 100 2>&1 | grep -v "$F" | cut -c1-500 | tail -40
done
