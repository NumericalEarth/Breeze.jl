#!/bin/bash
#SBATCH -p gpua100 --gres=gpu:1 -c 6 --mem=80G -t 120 -J diff-round5
#SBATCH -o /home/greg_aeolus_earth/droplet-tasks-env/jobs/round5-%j.out
J=~/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia
cd ~/droplet-tasks-env
H=~/Breeze-droplet-tasks/validation/differentiability
F="^\s*$\|^I0000\|^WARNING: All\|^E0000\|^W0000\|^│\|^┌\|^└\|^This may\|^If you\|^ensure\|^or .LD\|^In any other\|Warning: CUDA\|initialization.jl"
export MAXGB=30 MAXSEC=900
echo "######## build boussinesq_ppp with profile peek"
PEEK_AT=150 $H/watch.sh $J --startup-file=no --project=. -t 2 $H/build_bisect.jl boussinesq_ppp 2>&1 | grep -v "$F" | cut -c1-300 > $H/../../../droplet-tasks-env/jobs/peek-$SLURM_JOB_ID.txt
grep -v "^\s*[0-9]* *[0-9]* *@ *\(Base\|Core\)" ~/droplet-tasks-env/jobs/peek-$SLURM_JOB_ID.txt | head -400
echo "######## build anelastic_ppp"
$H/watch.sh $J --startup-file=no --project=. $H/build_bisect.jl anelastic_ppp 2>&1 | grep -v "$F" | cut -c1-500 | tail -30
