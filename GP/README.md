# Gradient Escalation Attack (GP) - G-Safeguard

## Overview

Gradient Escalation Attack (GP) module simulates **stealthy poisoning attacks** where attackers gradually increase the wrongness of their answers across multiple dialogue turns. This is distinct from Memory Attack (MA) which uses constant wrongness.

### Attack Mechanism

**Key Idea:** Attackers slowly escalate from slightly incorrect to obviously wrong answers to evade detection.

```
Turn 0 (Initial):     10-20% wrong  (subtle, hard to detect)
Turn 1 (Dialogue 1):  30-40% wrong  (noticeable but ambiguous)
Turn 2 (Dialogue 2):  60-80% wrong  (clearly wrong)
Turn 3 (Dialogue 3):  80-90% wrong  (obvious attack)
```

### Why This Matters

1. **Evasion:** Gradual escalation makes attacks harder to detect with static models
2. **Realistic:** Mimics how real attacks might progress in multi-turn settings
3. **Temporal Challenge:** Tests whether models can capture temporal patterns
4. **Defense Evaluation:** Baseline for testing temporal awareness in G-Safeguard

---

## Usage

### Quick Start

```bash
cd GP

# Set environment variables (if using API)
export OPENAI_API_KEY="your-key"
export BASE_URL="your-base-url"

# Make scripts executable
chmod +x ./scripts/train/gen_conversation_train.sh
chmod +x ./scripts/test/gen_conversation_test.sh

# Generate training data
./scripts/train/gen_conversation_train.sh && python merge_datasets.py --phase train

# Generate test data
./scripts/test/gen_conversation_test.sh && python merge_datasets.py --phase test

# Convert to GNN format
python gen_training_dataset.py
```

### Step-by-Step

#### Step 1: Generate Conversations

The `gen_graph.py` script generates agent conversations with gradient escalation attacks.

```bash
python3 gen_graph.py \
  --num_nodes 8 \
  --sparsity 0.4 \
  --num_graphs 20 \
  --num_attackers 2 \
  --samples 40 \
  --model_type qwen3.5-plus \
  --phase train \
  --attack_type escalation
```

**Parameters:**
- `num_nodes`: Number of agents in graph (default: 8)
- `sparsity`: Edge density 0-1 (default: 0.4)
- `num_graphs`: Number of random topologies (default: 20)
- `num_attackers`: Number of attacker agents (default: 1-4)
- `samples`: Samples per configuration (default: 40)
- `model_type`: LLM backbone (qwen3.5-plus, gpt-4o-mini, etc.)
- `phase`: train or test
- `attack_type`: escalation (gradual) or constant (baseline)

**Output:** JSON file with conversation data
```
agent_graph_dataset/gradient_escalation/{phase}/{timestamp}-dataset_size_*.json
```

#### Step 2: Merge Datasets

```bash
python merge_datasets.py --phase train
python merge_datasets.py --phase test
```

Creates unified dataset file:
```
agent_graph_dataset/gradient_escalation/{phase}/dataset.json
```

#### Step 3: Convert to GNN Format

```bash
python gen_training_dataset.py
```

Outputs:
```
ModelTrainingSet/gradient_escalation/dataset.pkl
```

This pickle file contains graph data ready for GNN training:
- Node features (embeddings of agent prompts)
- Edge features (message embeddings)
- Labels (0=normal, 1=attacker)
- Adjacency matrix

---

## Data Structure

### Generated Conversation Example

```json
{
  "adj_matrix": [[0,1,0,...], [1,0,1,...], ...],
  "attacker_idxes": [2, 5],
  "system_prompts": [
    "You are agent_0. You are a useful assistant...",
    "You are agent_1. You are a useful assistant...",
    "You are agent_2. ATTACKER_BASE (gradual escalation)",
    ...
  ],
  "query": "Which element in tobacco smoke causes cancer?",
  "adv_texts": ["Nicotine", "Tar", "Carbon monoxide", ...],
  "correct_answer": "Tar",
  "incorrect_answer": "Nicotine",
  "communication_data": [
    [
      (0, "Turn 0 response from agent 0"),
      (1, "Turn 0 response from agent 1"),
      (2, "Turn 0 response from agent 2 - slightly wrong"),
      ...
    ],
    [
      (0, "Turn 1 response from agent 0"),
      (1, "Turn 1 response from agent 1"),
      (2, "Turn 1 response from agent 2 - more wrong"),
      ...
    ],
    ...
  ]
}
```

### GNN Input Format

```python
{
  "adj_matrix": (8, 8) numpy array,      # Adjacency matrix
  "features": (8, 384) numpy array,      # Node embeddings
  "labels": (8,) numpy array,            # 0 or 1 for each node
  "edge_index": (2, num_edges) numpy array,  # Edge connections
  "edge_attr": (num_edges, 384) numpy array  # Message embeddings
}
```

---

## Configuration Variations

### Data Generation Configurations

| Sparsity | 1 Attacker | 2 Attackers | 3 Attackers | 4 Attackers |
|----------|-----------|-------------|-------------|-------------|
| 0.2      | ✓         | ✓           | ✓           | ✓           |
| 0.4      | ✓         | ✓           | ✓           | ✓           |
| 0.6      | ✓         | ✓           | ✓           | ✓           |
| 0.8      | ✓         | ✓           | ✓           | ✓           |
| 1.0      | ✓         | ✓           | ✓           | ✓           |

### Attack Types

1. **Escalation** (Primary)
   - Gradual increase in wrongness across turns
   - Turn 0: 10-20% wrong
   - Turn 2: 60-80% wrong
   - Best for temporal detection evaluation

2. **Constant** (Baseline)
   - Fixed wrongness throughout
   - 50-60% wrong across all turns
   - Baseline for non-temporal models

---

## File Structure

```
GP/
├── agent_prompts.py              # System prompts for escalation
├── agents.py                     # AgentGraphGradientEscalation class
├── gen_graph.py                  # Main data generation script
├── gen_training_dataset.py       # Convert to GNN format
├── data.py                       # Data loading utilities
├── merge_datasets.py             # Merge dataset files
├── utils.py                      # Utility functions
├── datasets/                     # Base datasets (place msmarco.json here)
├── agent_graph_dataset/
│   └── gradient_escalation/
│       ├── train/               # Generated training data
│       └── test/                # Generated test data
├── ModelTrainingSet/
│   └── gradient_escalation/     # GNN-format datasets (pickles)
├── checkpoint/                  # (Future) trained models
└── scripts/
    ├── train/
    │   └── gen_conversation_train.sh
    └── test/
        └── gen_conversation_test.sh
```

---

## Statistics

### Training Data (Typical Configuration)

```
Configurations:  20 (5 sparsity × 4 attacker variations)
Samples/config:  40
Total graphs:    800

Total samples:   3,200 conversations
Agents/graph:    8
Dialogue turns:  3

LLM calls:       ~25,600 (800 graphs × 8 agents × (1 initial + 3 dialogue × 1 update))
Estimated time:  15-20 hours (depending on LLM latency)
```

### Attack Distribution

- **Escalation attacks:** ~1,600 samples
- **Constant attacks:** ~800 samples (baseline)
- **Normal agents:** ~60-75% per graph
- **Attacker agents:** ~25-40% per graph

---

## Expected Model Performance

### Without Temporal Awareness (Static GNN)
```
Accuracy: 65-75%
Reason: Can't distinguish escalation from natural variation
```

### With Temporal Awareness (Temporal GNN)
```
Accuracy: 88-95%
Reason: Can capture escalation patterns
```

---

## Troubleshooting

### Issue: API Rate Limits

**Solution:** Reduce `samples` parameter or add delays:
```bash
python3 gen_graph.py ... --samples 10  # Smaller batches
```

### Issue: Insufficient Memory

**Solution:** Process in smaller batches or reduce `num_nodes`:
```bash
python3 gen_graph.py ... --num_nodes 5 --samples 20
```

### Issue: Missing msmarco.json

**Solution:** Copy from MA folder:
```bash
cp ../MA/datasets/msmarco.json ./datasets/
```

---

## References

This module implements the **Gradient Escalation Attack** concept from G-Safeguard research on detecting poisoning attacks in multi-agent systems.

See RESEARCH_DIRECTIONS_VI.md for:
- DIRECTION 4: Temporal & Recurrent GNN Architectures (evaluation target)
- Attack methodology details
- Expected improvements with temporal models

---

## Next Steps

After data generation:

1. **Convert to GNN Format:** `python gen_training_dataset.py`
2. **Train Temporal GNN:** (use model.py from parent directory)
3. **Evaluate Defense:** (use main_defense_for_different_topology.py)
4. **Compare with MA:** Measure improvement in temporal detection

---

## Author Notes

- This module focuses on data generation only
- Use `gen_training_dataset.py` to prepare for model training
- Shell scripts enable parallel generation (configure model type as needed)
- All data augmentation happens at generation time, not training time
