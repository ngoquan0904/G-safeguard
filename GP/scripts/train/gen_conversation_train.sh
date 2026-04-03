#!/bin/bash
set -e

echo "Starting parallel Gradient Escalation attack data generation (TRAIN)..."
echo "PID: $$"

# Run all jobs in parallel with different configurations
# Configurations: 5 sparsity levels × 4 attacker counts × 2 attack types = 40 configurations

SPARSITIES=(0.2 0.4 0.6 0.8 1.0)
ATTACKERS=(1 2 3 4)
ATTACK_TYPES=(escalation)

JOB_COUNT=0

for sparsity in "${SPARSITIES[@]}"; do
    for num_attackers in "${ATTACKERS[@]}"; do
        for attack_type in "${ATTACK_TYPES[@]}"; do
            echo "Launching job: sparsity=$sparsity, attackers=$num_attackers, type=$attack_type"
            python3 gen_graph.py \
                --num_nodes 8 \
                --sparsity "$sparsity" \
                --num_graphs 20 \
                --num_attackers "$num_attackers" \
                --samples 40 \
                --model_type llm-medium-v5 \
                --phase train \
                --attack_type "$attack_type" &
            
            ((JOB_COUNT++))
        done
    done
done

echo "Launched $JOB_COUNT parallel jobs"
echo "Waiting for all data generation jobs to complete..."
wait

echo "All training data generation completed successfully!"
