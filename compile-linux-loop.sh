#!/bin/bash
while [ 1 -gt 0 ]
do
    gcc -O3 -Wfatal-errors -lm -O3 -o ./main main.c && time ./main
    read -p "Press enter to recompile and run"
done