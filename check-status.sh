#!/bin/bash
# Check SLURM job status and recent logs
# Usage: bash check-status.sh [job_id]

SERVER="mewa8567@olympus.dsv.su.se"
REMOTE="~/my_project/"
JOB_ID="${1:-}"

ssh "${SERVER}" bash <<EOF
echo "================================================"
echo "Queue status:"
echo "================================================"
squeue -u \$USER

if [ -n "${JOB_ID}" ]; then
    echo ""
    echo "================================================"
    echo "Job ${JOB_ID} detail:"
    echo "================================================"
    scontrol show job ${JOB_ID} 2>/dev/null || echo "(job not found in queue — may have finished)"
fi

echo ""
echo "================================================"
echo "Recent log files:"
echo "================================================"
ls -lt ${REMOTE}logs/ 2>/dev/null | head -10

echo ""
echo "================================================"
echo "Latest stdout log (last 50 lines):"
echo "================================================"
LATEST=\$(ls -t ${REMOTE}logs/slurm_*.out 2>/dev/null | head -1)
if [ -n "\$LATEST" ]; then
    echo "File: \$LATEST"
    tail -50 "\$LATEST"
else
    echo "(no log files yet)"
fi

echo ""
echo "================================================"
echo "Latest stderr log (last 20 lines):"
echo "================================================"
LATEST_ERR=\$(ls -t ${REMOTE}logs/slurm_*.err 2>/dev/null | head -1)
if [ -n "\$LATEST_ERR" ]; then
    echo "File: \$LATEST_ERR"
    tail -20 "\$LATEST_ERR"
else
    echo "(no error logs)"
fi

echo ""
echo "================================================"
echo "Results files:"
echo "================================================"
ls -lh ${REMOTE}results/ 2>/dev/null || echo "(no results yet)"
EOF
