#!/bin/bash
# Invoked by launchd every weekday at 17:30 IST. Activates the venv and runs
# the engine; all stdout/stderr also goes to logs/launchd_run.log so a failed
# unattended run leaves a trail even if launchd's own logs get rotated away.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGFILE="$PROJECT_DIR/logs/launchd_run.log"

mkdir -p "$PROJECT_DIR/logs"
{
  echo "======================================"
  echo "Run started: $(date)"
} >> "$LOGFILE"

cd "$PROJECT_DIR"
source .venv/bin/activate

python run_engine.py --mode full >> "$LOGFILE" 2>&1

echo "Run finished: $(date)" >> "$LOGFILE"
