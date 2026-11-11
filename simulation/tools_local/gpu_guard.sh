#!/usr/bin/env bash
# GPU etiquette wrapper (shared RTX 5090):
#   1. wait until no foreign compute process (type C / C+G, other session) has been on the GPU for IDLE_S seconds;
#   2. run the command in its own session; poll every POLL_S seconds;
#   3. if a foreign compute process appears, kill our whole session (yield), log it, and go back to step 1.
# The wrapped pipelines skip outputs that already exist, so a yielded run resumes where it stopped.
# usage: tools_local/gpu_guard.sh <guard.log> <command ...>
IDLE_S=${IDLE_S:-600}
POLL_S=${POLL_S:-15}
GLOG=$1; shift
log() { echo "[gpu_guard $(date '+%F %T')] $*" >> "$GLOG"; }

compute_pids() {   # PIDs of compute processes from the nvidia-smi process table
  nvidia-smi 2>/dev/null | awk '/Processes:/{f=1} f && /^\|/ && ($6=="C" || $6=="C+G") {print $5}'
}
foreign_present() {   # $1 = our session id (empty: every compute process is foreign)
  local p sid
  for p in $(compute_pids); do
    sid=$(ps -o sid= -p "$p" 2>/dev/null | tr -d ' ')
    if [ -z "$1" ] || [ "$sid" != "$1" ]; then
      echo "$p $(ps -o user=,args= -p "$p" 2>/dev/null | cut -c1-100)"
      return 0
    fi
  done
  return 1
}

while true; do
  idle=0
  log "waiting for ${IDLE_S}s of GPU idleness"
  while [ "$idle" -lt "$IDLE_S" ]; do
    if f=$(foreign_present ""); then
      [ "$idle" -gt 0 ] && log "busy again: $f"
      idle=0
    else
      idle=$((idle + POLL_S))
    fi
    sleep "$POLL_S"
  done
  log "GPU idle for ${IDLE_S}s -> starting: $*"
  setsid "$@" &
  PID=$!
  yielded=0
  while kill -0 "$PID" 2>/dev/null; do
    if f=$(foreign_present "$PID"); then
      log "foreign GPU job detected ($f) -> yielding, killing session $PID"
      kill -TERM -- "-$PID" 2>/dev/null; pkill -TERM -s "$PID" 2>/dev/null
      sleep 10
      kill -KILL -- "-$PID" 2>/dev/null; pkill -KILL -s "$PID" 2>/dev/null
      yielded=1
      break
    fi
    sleep "$POLL_S"
  done
  wait "$PID" 2>/dev/null
  rc=$?
  if [ "$yielded" -eq 0 ]; then
    log "command finished (exit $rc)"
    exit "$rc"
  fi
done
