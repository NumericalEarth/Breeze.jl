#!/bin/bash
# Run a command, printing its resident memory every minute; kill it above $MAXGB GB (default 60) or $MAXSEC s.
MAXKB=$(( ${MAXGB:-60} * 1000000 )); MAXSEC=${MAXSEC:-1800}
"$@" &
pid=$!; start=$SECONDS; peak=0
while kill -0 $pid 2>/dev/null; do
    rss=0
    for p in $pid $(pgrep -P $pid); do r=$(ps -o rss= -p $p 2>/dev/null | tr -d ' '); rss=$((rss + ${r:-0})); done
    [ $rss -gt $peak ] && peak=$rss
    if [ $rss -gt $MAXKB ] || [ $((SECONDS - start)) -gt $MAXSEC ]; then
        echo "WATCHDOG: killed at $((SECONDS - start)) s, RSS $((rss / 1000)) MB"; pkill -9 -P $pid; kill -9 $pid
    fi
    [ -n "$PEEK_AT" ] && [ $((SECONDS - start)) -eq $PEEK_AT ] && kill -USR1 $pid && echo "[watch] sent SIGUSR1 (profile peek)"
    [ $(( (SECONDS - start) % 60 )) -eq 0 ] && echo "  [watch] t=$((SECONDS - start)) s RSS=$((rss / 1000)) MB"
    sleep 1
done
echo "[watch] done: wall $((SECONDS - start)) s, peak RSS $((peak / 1000)) MB"
