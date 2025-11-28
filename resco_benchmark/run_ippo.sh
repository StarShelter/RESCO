#!/bin/bash

echo "Starting all 8 IPPO experiments..."

# echo ""
# echo "=== [1/8] Running: 4x4 Grid ==="
# python main.py @grid4x4 @IPPO libsumo:False save_console_log:False gui:False

# echo ""
# echo "=== [2/8] Running: 4x4 Avenues ==="
# python main.py @arterial4x4 @IPPO libsumo:False save_console_log:False gui:False

echo ""
echo "=== [3/8] Running: Cologne Single Signal ==="
python main.py @cologne1 @IPPO libsumo:False save_console_log:False gui:False

echo ""
echo "=== [4/8] Running: Cologne Corridor ==="
python main.py @cologne3 @IPPO libsumo:False save_console_log:False gui:False

echo ""
echo "=== [5/8] Running: Cologne Region ==="
python main.py @cologne8 @IPPO libsumo:False save_console_log:False gui:False

echo ""
echo "=== [6/8] Running: Ingolstadt Single Signal ==="
python main.py @ingolstadt1 @IPPO libsumo:False save_console_log:False gui:False

echo ""
echo "=== [7/8] Running: Ingolstadt Corridor ==="
python main.py @ingolstadt7 @IPPO libsumo:False save_console_log:False gui:False

echo ""
echo "=== [8/8] Running: Ingolstadt Region ==="
python main.py @ingolstadt21 @IPPO libsumo:False save_console_log:False gui:False

echo ""
echo "=== All 8 experiments finished. ==="