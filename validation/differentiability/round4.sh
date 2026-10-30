#!/bin/bash
#SBATCH -p gpua100 --gres=gpu:1 -c 6 --mem=80G -t 120 -J diff-round4
#SBATCH -o /home/greg_aeolus_earth/droplet-tasks-env/jobs/round4-%j.out
J=~/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia
cd ~/droplet-tasks-env
H=~/Breeze-droplet-tasks/validation/differentiability
F="^\s*$\|^I0000\|^WARNING: All\|^E0000\|^W0000\|^│\|^┌\|^└\|^This may\|^If you\|^ensure\|^or .LD\|^In any other\|Warning: CUDA\|initialization.jl"
export MAXGB=30 MAXSEC=900
for v in ${VARIANTS:-compressible_ppb boussinesq_ppb boussinesq_ppp anelastic_ppp}; do
    echo "######## build $v"
    $H/watch.sh $J --startup-file=no --project=. $H/build_bisect.jl $v 2>&1 | grep -v "$F" | cut -c1-500 | tail -30
done
