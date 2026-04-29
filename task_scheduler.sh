#!/bin/bash
cd Masters_Thesis
uv run prepare_input.py
cd ..

NS=(1000)
TOPOLOGIES=("full")
for d in 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5; do
    for mu in 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5; do
        for N in "${NS[@]}"; do
            for topology in "${TOPOLOGIES[@]}"; do
                sbatch task.sh "$d" "$mu" "$N" "$topology"
            done
        done
    done
done