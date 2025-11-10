#!/bin/bash
# Monitor experiment progress

echo "=============================================="
echo "EXPERIMENT PROGRESS MONITOR"
echo "=============================================="
echo ""

echo "Background Processes:"
echo "---------------------"
ps aux | grep "python experiments" | grep -v grep | awk '{print $2, $11, $12, $13}'
echo ""

echo "Results Files Created:"
echo "---------------------"
ls -lh results/*/  2>/dev/null | grep "\.json$" | awk '{print $9, $5}'
echo ""

echo "Latest Log Entries:"
echo "---------------------"

if [ -f "results/scalability_output.log" ]; then
    echo "Scalability Test (last 5 lines):"
    tail -5 results/scalability_output.log
    echo ""
fi

if [ -f "results/operation_output.log" ]; then
    echo "Operation Comparison (last 5 lines):"
    tail -5 results/operation_output.log
    echo ""
fi

echo "=============================================="
echo "Check complete. Re-run to update."
echo "=============================================="
