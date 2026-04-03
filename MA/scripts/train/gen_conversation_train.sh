#!/bin/bash
set -e

echo "Starting parallel graph generation..."
echo "PID: $$"

# Run all jobs in parallel
python3 gen_graph.py --num_nodes 8 --sparsity 0.2 --num_graphs 20 --num_attackers 1 --samples 40 --model_type qwen3.5-plus --phase train &
python3 gen_graph.py --num_nodes 8 --sparsity 0.4 --num_graphs 20 --num_attackers 1 --samples 40 --model_type qwen3.5-plus --phase train &
python3 gen_graph.py --num_nodes 8 --sparsity 0.6 --num_graphs 20 --num_attackers 1 --samples 40 --model_type qwen3.5-plus --phase train &
python3 gen_graph.py --num_nodes 8 --sparsity 0.8 --num_graphs 20 --num_attackers 1 --samples 40 --model_type qwen3.5-plus --phase train &
python3 gen_graph.py --num_nodes 8 --sparsity 1.0 --num_graphs 20 --num_attackers 1 --samples 40 --model_type qwen3.5-plus --phase train &
python3 gen_graph.py --num_nodes 8 --sparsity 0.2 --num_graphs 20 --num_attackers 2 --samples 40 --model_type qwen3.5-plus --phase train &
python3 gen_graph.py --num_nodes 8 --sparsity 0.4 --num_graphs 20 --num_attackers 2 --samples 40 --model_type qwen3.5-plus --phase train &
python3 gen_graph.py --num_nodes 8 --sparsity 0.6 --num_graphs 20 --num_attackers 2 --samples 40 --model_type qwen3.5-plus --phase train &
python3 gen_graph.py --num_nodes 8 --sparsity 0.8 --num_graphs 20 --num_attackers 2 --samples 40 --model_type qwen3.5-plus --phase train &
python3 gen_graph.py --num_nodes 8 --sparsity 1.0 --num_graphs 20 --num_attackers 2 --samples 40 --model_type qwen3.5-plus --phase train &
python3 gen_graph.py --num_nodes 8 --sparsity 0.2 --num_graphs 20 --num_attackers 3 --samples 40 --model_type qwen3.5-plus --phase train &
python3 gen_graph.py --num_nodes 8 --sparsity 0.4 --num_graphs 20 --num_attackers 3 --samples 40 --model_type qwen3.5-plus --phase train &
python3 gen_graph.py --num_nodes 8 --sparsity 0.6 --num_graphs 20 --num_attackers 3 --samples 40 --model_type qwen3.5-plus --phase train &
python3 gen_graph.py --num_nodes 8 --sparsity 0.8 --num_graphs 20 --num_attackers 3 --samples 40 --model_type qwen3.5-plus --phase train &
python3 gen_graph.py --num_nodes 8 --sparsity 1.0 --num_graphs 20 --num_attackers 3 --samples 40 --model_type qwen3.5-plus --phase train &
python3 gen_graph.py --num_nodes 8 --sparsity 0.2 --num_graphs 20 --num_attackers 4 --samples 40 --model_type qwen3.5-plus --phase train &
python3 gen_graph.py --num_nodes 8 --sparsity 0.4 --num_graphs 20 --num_attackers 4 --samples 40 --model_type qwen3.5-plus --phase train &
python3 gen_graph.py --num_nodes 8 --sparsity 0.6 --num_graphs 20 --num_attackers 4 --samples 40 --model_type qwen3.5-plus --phase train &
python3 gen_graph.py --num_nodes 8 --sparsity 0.8 --num_graphs 20 --num_attackers 4 --samples 40 --model_type qwen3.5-plus --phase train &
python3 gen_graph.py --num_nodes 8 --sparsity 1.0 --num_graphs 20 --num_attackers 4 --samples 40 --model_type qwen3.5-plus --phase train &

# Wait for all background jobs to complete
wait
echo "All jobs completed!"