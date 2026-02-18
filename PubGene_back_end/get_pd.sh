#!/usr/bin/env bash
set -euo pipefail
umask 002 

trap 'echo "[get_pd.sh] failed at line $LINENO: $BASH_COMMAND" >&2' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CODE_DIR="$SCRIPT_DIR/jt1995"
PY="/home/jt1995/miniconda3/envs/ai_summary/bin/python"




LOG_DIR="/mnt/hc-storage/groups/cbf/tony/share/ai_summary/jt1995/logs"
LOG="$LOG_DIR/ai_summary_$(hostname).log"
mkdir -p "$LOG_DIR" >/dev/null 2>&1 || true
{
  echo "==== $(date) ===="
  echo "whoami: $(whoami)"
  echo "pwd: $(pwd)"
  echo "args: pmid=${1-?} gene=${2-?} db=${3-?}"
  echo "HOME: ${HOME:-"(unset)"}"
  echo "PY: $PY"
  ls -l "$PY" || true
  echo "CODE_DIR: $CODE_DIR"
  ls -ld "$CODE_DIR" || true
  ls -l "$CODE_DIR/.env" || true
} >> "$LOG" 2>&1


# log python version & test execution
echo "Trying: $PY -V" >> "$LOG"
"$PY" -V >> "$LOG" 2>&1 || echo "Python version check failed" >> "$LOG"

echo "MountInfo:" >> "$LOG"
grep -E 'ai_summary|cbf_share' /proc/mounts >> "$LOG" 2>&1 || true



cd "$CODE_DIR"

OUT="$("$PY" get_pd_cli.py --gene_id "$2" --host_db "$3" --pmid "$1" 2>>"$LOG" || true)"
if [ -z "${OUT}" ]; then
  OUT='{"summary":"<div class=\"ai-error\">Backend error running PD generator. See logs.</div>","prompt":"error"}'
fi

if command -v jq >/dev/null 2>&1; then
  OUT="$(printf '%s' "$OUT" | jq -c '.prompt="ok"')"
else
  OUT="$(printf '%s' "$OUT" | sed -E 's/"prompt"[[:space:]]*:[[:space:]]*"[^"]*"/"prompt":"ok"/')"
fi

printf '%s' "$OUT"
exit 0
