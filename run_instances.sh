#!/bin/bash

python3 train.py --mode "$simple" &
pid1=$!

python3 train.py --mode "per" &
pid2=$!

python3 train.py --mode "ucb" &
pid3=$!

python3 train.py --mode "delta_dep" &
pid4=$!

wait $pid1 $pid2 $pid3 $pid4

echo "All runs are done, check convergence plots"