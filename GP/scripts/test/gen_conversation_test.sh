#!/bin/bash
set -e

echo "Starting parallel Gradient Escalation attack data generation (TEST)..."
echo "PID: $$"

# Test configurations: subset of training for faster evaluation
# 7 key configurations covering different sparsity and attacker scenarios

TEST_CONFIGS=(
    "0.2:1:escalation"
    "0.4:1:escalation"
    "0.6:1:escalation"
    "0.8:2:escalation"
    "1.0:2:escalation"
    "0.4:1:constant"
    "0.6:1:constant"
)

JOB_COUNT=0

for config in "${TEST_CONFIGS[@]}"; do
    IFS=':' read -r sparsity num_attackers attack_type <<< "$config"
    
    echo "Launching test job: sparsity=$sparsity, attackers=$num_attackers, type=$attack_type"
    python3 gen_graph.py \
        --num_nodes 8 \
        --sparsity "$sparsity" \
        --num_graphs 10 \
        --num_attackers "$num_attackers" \
        --samples 20 \
        --model_type qwen3.5-plus \
        --phase test \
        --attack_type "$attack_type" &
    
    ((JOB_COUNT++))
done

echo "Launched $JOB_COUNT parallel test jobs"
echo "Waiting for all data generation jobs to complete..."
wait

echo "All test data generation completed successfully!"
