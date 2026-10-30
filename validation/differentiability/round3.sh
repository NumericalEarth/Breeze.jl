#!/bin/bash
#SBATCH -p gpua100 --gres=gpu:1 -c 6 --mem=80G -t 240 -J diff-round3
#SBATCH -o /home/greg_aeolus_earth/droplet-tasks-env/jobs/round3-%j.out
J=~/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia
cd ~/droplet-tasks-env
H=~/Breeze-droplet-tasks/validation/differentiability
F="^\s*$\|^I0000\|^WARNING: All\|^E0000\|^W0000\|^│\|^┌\|^└\|^This may\|^If you\|^ensure\|^or .LD\|^In any other\|Warning: CUDA\|initialization.jl"
export MAXGB=40 MAXSEC=1200
export DYNAMICS=${DYNAMICS:-boussinesq}
for spec in ${SPECS:-"build:100" "rate:100" "growth:100"}; do
    level=${spec%%:*}; nd=${spec##*:}
    echo "######## droplets $level Nd=$nd FIXED=${FIXED:-0}"
    $H/watch.sh $J --startup-file=no --project=. $H/droplets_reactant.jl $level $nd 2>&1 | grep -v "$F" | cut -c1-500 | tail -40
done
