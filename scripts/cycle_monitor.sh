#!/usr/bin/env bash
# cycle_monitor.sh — watch an NNGPT iterative-pipeline run and push an ntfy.sh
# notification each time a new cycle completes.
#
# The iterative pipeline (ab/gpt/iterative_finetune.py) writes ONE directory per
# cycle under the curation output dir:  <output_dir>/cycle_<N>/ , and marks a
# cycle complete by writing  <output_dir>/cycle_<N>/evaluation_results.json
# (produced by evaluate_cycle_models.py). We track the highest N whose
# evaluation_results.json exists, and notify on each advance. (The pipeline does
# NOT emit a single rolling cycle_results.json — that older assumption is gone.)
#
# Usage: cycle_monitor.sh <output_dir> <ntfy_topic> [models_per_cycle]
#   <output_dir>       dir containing cycle_<N>/ subdirs
#                      (in the pod: /a/mm/out/curation_output)
#   <ntfy_topic>       ntfy.sh topic name to publish to
#   [models_per_cycle] denominator for the success-rate line (default 25)
#
# Dependencies: bash, curl, python3 (nothing else).
set -u

DIR="$1"
TOPIC="$2"
PER_CYCLE="${3:-25}"
LAST_FILE="/tmp/cycle_monitor_last_$TOPIC"
STALL_LIMIT=2400          # iterations (x60s) with no new cycle => finished/stalled (40h)

# latest_cycle: print the highest N for which cycle_<N>/evaluation_results.json
# exists, or nothing (exit 1) if no cycle has completed yet.
latest_cycle() {
    python3 - "$DIR" <<'PY'
import glob, os, re, sys
d = sys.argv[1]
best = -1
for p in glob.glob(os.path.join(d, "cycle_*", "evaluation_results.json")):
    m = re.search(r"cycle_(\d+)", p)
    if m:
        best = max(best, int(m.group(1)))
if best < 0:
    sys.exit(1)
print(best)
PY
}

# format_msg <N>: render the ntfy message for cycle N from its
# evaluation_results.json (best accuracy + how many models trained successfully).
format_msg() {
    python3 - "$DIR" "$1" "$PER_CYCLE" <<'PY'
import json, os, sys
d, n, per = sys.argv[1], sys.argv[2], int(sys.argv[3])
f = os.path.join(d, "cycle_%s" % n, "evaluation_results.json")
try:
    data = json.load(open(f))
except Exception:
    sys.exit(1)
models = data.get("models", []) or []
ok = [m for m in models if m.get("success") and "accuracy" in m]
if ok:
    best = max(m["accuracy"] for m in ok)
    print("Cycle %s done: best=%.4f, trained=%d/%d (%.0f%%)"
          % (n, best, len(ok), per, len(ok) / per * 100))
else:
    print("Cycle %s DEAD: 0 models evaluated" % n)
PY
}

# startup notification
curl -s -d "Monitor started, watching $DIR" "ntfy.sh/$TOPIC" >/dev/null

# resume last-seen cycle if we were restarted; otherwise seed to the cycle
# currently on disk so we only notify on the NEXT completed cycle (avoids
# re-notifying an already-reported one). If nothing has completed yet, start at
# -1 so cycle 0/1 will be caught.
last=""
[ -f "$LAST_FILE" ] && last=$(cat "$LAST_FILE" 2>/dev/null)
if [ -z "$last" ]; then
    cur=$(latest_cycle)
    if [ -n "$cur" ]; then last="$cur"; else last=-1; fi
    echo "$last" > "$LAST_FILE"
fi
stall=0

while true; do
    cur=$(latest_cycle)

    if [ -n "$cur" ] && [ "$cur" -gt "$last" ]; then
        # notify for every cycle from last+1..cur (covers >1 completing per poll)
        n=$((last + 1))
        while [ "$n" -le "$cur" ]; do
            msg=$(format_msg "$n")
            [ -n "$msg" ] && curl -s -d "$msg" "ntfy.sh/$TOPIC" >/dev/null
            n=$((n + 1))
        done
        last="$cur"
        echo "$last" > "$LAST_FILE"
        stall=0
    else
        stall=$((stall + 1))
        if [ "$stall" -ge "$STALL_LIMIT" ]; then
            curl -s -d "Run finished or stalled after cycle $last" "ntfy.sh/$TOPIC" >/dev/null
            exit 0
        fi
    fi

    sleep 60
done
