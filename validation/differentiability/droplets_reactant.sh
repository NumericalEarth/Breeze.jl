#!/bin/bash
#SBATCH -p gpua100 --gres=gpu:1 -c 6 --mem=80G -t 240 -J droplet-reactant
#SBATCH -o /home/greg_aeolus_earth/droplet-tasks-env/jobs/droplets-reactant-%j.out
# One process per level; a watchdog kills a level above 60 GB resident memory or after 30 minutes
# and reports the peak memory over time.
J=~/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia
cd ~/droplet-tasks-env
D=~/Breeze-droplet-tasks/validation/differentiability/droplets_reactant.jl
LEVELS=${LEVELS:-"condense update advect step0 step"}
for level in $LEVELS; do
    echo "######## level $level Nd=${ND:-100} RAISE=${RAISE:-false}"
    log=$(mktemp -p ~/droplet-tasks-env/jobs)
    $J --startup-file=no --project=. $D $level ${ND:-100} > $log 2>&1 &
    pid=$!
    start=$SECONDS; peak=0
    while kill -0 $pid 2>/dev/null; do
        rss=$(ps -o rss= -p $pid 2>/dev/null | tr -d ' '); rss=${rss:-0}
        [ $rss -gt $peak ] && peak=$rss
        if [ $rss -gt 60000000 ] || [ $((SECONDS - start)) -gt 1800 ]; then
            echo "WATCHDOG: killing $level at $((SECONDS - start)) s, RSS $((rss / 1000000)) GB"; kill -9 $pid
        fi
        if [ $(( (SECONDS - start) % 60 )) -eq 0 ]; then echo "  t=$((SECONDS - start)) s RSS=$((rss / 1000)) MB"; fi
        sleep 1
    done
    grep -v "^\s*$\|^I0000\|^WARNING: All\|^E0000\|^W0000" $log | cut -c1-500 | tail -60
    echo "level $level: wall $((SECONDS - start)) s, peak RSS $((peak / 1000)) MB"
    rm -f $log
done
