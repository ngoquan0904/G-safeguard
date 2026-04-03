# GP (Gradient Poison/Escalation) - Setup & Quick Start Guide

## 🎯 What is GP?

**Gradient Escalation Attack (GP)** simulates attackers who **gradually increase wrongness** across dialogue turns (10% wrong → 80% wrong), unlike MA which uses constant wrongness.

This tests whether G-Safeguard can detect **temporal patterns of attacks**.

---

## 📁 Folder Structure Created

```
GP/
├── agent_prompts.py              # System prompts with escalation levels
├── agents.py                     # AgentGraphGradientEscalation class
├── gen_graph.py                  # ⭐ Main data generation script
├── gen_training_dataset.py       # Convert conversations to GNN format
├── data.py                       # Data loading utilities
├── merge_datasets.py             # Merge multiple dataset files
├── utils.py                      # Helper functions
├── README.md                     # Full documentation
├── datasets/
│   └── msmarco.json             # Base QA dataset (copied from MA)
├── scripts/
│   ├── train/
│   │   └── gen_conversation_train.sh  # Parallel training data generation
│   └── test/
│       └── gen_conversation_test.sh   # Parallel test data generation
├── agent_graph_dataset/
│   └── gradient_escalation/
│       ├── train/               # Generated training conversations
│       └── test/                # Generated test conversations
├── ModelTrainingSet/
│   └── gradient_escalation/     # GNN-ready pickle files
└── checkpoint/
    └── gradient_escalation/     # (Future) model checkpoints
```

---

## 🚀 Quick Start (Data Generation Only)

### Step 1: Navigate to GP folder

```bash
cd /home/admin123/Documents/VNPT_AI/GuardRAG/G-safeguard/GP
```

### Step 2: Set LLM credentials

```bash
export OPENAI_API_KEY="your-api-key"
export BASE_URL="your-base-url"  # if using proxy/alternative backend
```

### Step 3: Run data generation (parallel)

**For TRAINING data:**
```bash
chmod +x ./scripts/train/gen_conversation_train.sh
./scripts/train/gen_conversation_train.sh

# After running, merge all generated files:
python merge_datasets.py --phase train
```

**For TEST data:**
```bash
chmod +x ./scripts/test/gen_conversation_test.sh
./scripts/test/gen_conversation_test.sh

# After running, merge:
python merge_datasets.py --phase test
```

### Step 4: Convert to GNN format

```bash
python gen_training_dataset.py
```

This creates: `ModelTrainingSet/gradient_escalation/dataset.pkl`

---

## 🔍 Key Differences from MA

| Aspect | MA (Memory Attack) | GP (Gradient Escalation) |
|--------|------------------|--------------------------|
| **Wrongness** | Constant (50-60%) | Gradual (10% → 80%) |
| **Evasion** | Obvious | Stealthy, temporal |
| **Attack Type** | `--attack_type constant` | `--attack_type escalation` |
| **Turns** | 3 (for reference) | 3 (escalation critical!) |
| **Detection** | Easy with static models | Requires temporal models |

---

## 📊 Generation Statistics

### Training Data (Default)

```
Configurations:     ~20-40 (different sparsity/attacker combos)
Samples/config:     40
Total graphs:       800-1600
Dialogue turns:     3

LLM calls:          ~20,000-30,000
Estimated time:     12-18 hours
Cost:               $10-50 (depending on model)
```

### What Gets Generated

**Per graph instance:**
```
Turn 0 (Initial): Agent 0,1 answer; Agent 2 (attacker) answers with 10-20% error
Turn 1 (Dialogue): All agents discuss, attacker escalates to 30-40% error
Turn 2 (Dialogue): Discussion continues, attacker reaches 60-80% error  
Turn 3 (Dialogue): Final round, attacker may hit 80-90% error
```

---

## 🎮 Manual Data Generation Examples

### Generate small test batch (5 samples)

```bash
python gen_graph.py \
  --num_nodes 8 \
  --sparsity 0.4 \
  --num_graphs 5 \
  --num_attackers 1 \
  --samples 5 \
  --model_type qwen3.5-plus \
  --phase train \
  --attack_type escalation
```

### Compare escalation vs constant:

**Escalation (proposed):**
```bash
python gen_graph.py ... --attack_type escalation
```

**Constant baseline (for comparison):**
```bash
python gen_graph.py ... --attack_type constant
```

### Different graph topologies:

```bash
# Sparse graphs (few connections)
python gen_graph.py ... --sparsity 0.2

# Dense graphs (many connections)  
python gen_graph.py ... --sparsity 0.8

# Complete graph (everyone talks to everyone)
python gen_graph.py ... --sparsity 1.0
```

---

## 📥 Input Data Format

### msmarco.json Structure

```json
{
  "1163399": {
    "question": "what day is groundhog's day?",
    "correct answer": "February 2",
    "incorrect answer": "March 15",
    "adv_texts": ["February 2", "February 14", "March 15", ...]
  },
  ...
}
```

Each entry provides:
- `question`: The query for agents
- `correct answer`: Ground truth
- `incorrect answer`: Potential wrong answer
- `adv_texts`: Contexts to feed attackers

---

## 📤 Output Files

### Step 1: Raw Conversations (JSON)

**Path:** `agent_graph_dataset/gradient_escalation/{train|test}/{timestamp}.json`

**Size:** ~2-5MB per file

**Content:**
```json
[
  {
    "adj_matrix": [[0,1,0,...], ...],
    "attacker_idxes": [2],
    "communication_data": [
      [(0, "response"), (1, "response"), ...],  // Turn 0
      [(0, "updated"), (1, "updated"), ...],    // Turn 1
      ...
    ]
  },
  ...
]
```

### Step 2: Merged Dataset (JSON)

**Path:** `agent_graph_dataset/gradient_escalation/{train|test}/dataset.json`

**Size:** ~50-100MB (contains all samples)

**Content:** Array of conversation dicts

### Step 3: GNN Format (Pickle)

**Path:** `ModelTrainingSet/gradient_escalation/dataset.pkl`

**Format:** PyTorch Geometric
```
[
  {
    "features": (8, 384) array,      # Node embeddings
    "labels": (8,) array,            # 0=normal, 1=attacker
    "edge_index": (2, num_edges),   # Graph structure
    "edge_attr": (num_edges, 384)   # Message embeddings
  },
  ...
]
```

---

## ⚙️ Configuration Reference

```
python gen_graph.py \
  --dataset_path "./datasets/msmarco.json"    # QA dataset
  --num_nodes 8                                # Agents per graph
  --sparsity 0.4                               # 0-1, higher = denser
  --num_graphs 20                              # Random topologies
  --num_attackers 1                            # 1-4 attackers
  --num_dialogue_turns 3                       # Turns (escalation happens here)
  --samples 40                                 # Samples per config
  --model_type "qwen3.5-plus"                  # LLM backbone
  --phase "train"                              # train or test
  --attack_type "escalation"                   # escalation or constant
```

---

## 🔗 Next Steps After Data Generation

1. **Verify data:** Check `agent_graph_dataset/gradient_escalation/train/dataset.json`
2. **Convert to GNN:** `python gen_training_dataset.py`
3. **Train model:** (Use train.py from parent directory with this data)
4. **Evaluate defense:** (Use main_defense_for_different_topology.py)
5. **Compare with MA:** Measure improvement detecting escalation

---

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'agents'"

**Solution:** Run from GP directory:
```bash
cd /home/admin123/Documents/VNPT_AI/GuardRAG/G-safeguard/GP
python gen_graph.py ...
```

### "OPENAI_API_KEY not set"

**Solution:** Export before running:
```bash
export OPENAI_API_KEY="sk-..."
export BASE_URL="https://..."  # if needed
python gen_graph.py ...
```

### "File not found: ./datasets/msmarco.json"

**Solution:** Copy from MA:
```bash
cp ../MA/datasets/msmarco.json ./datasets/
```

### Script doesn't execute

**Solution:** Make executable:
```bash
chmod +x ./scripts/train/gen_conversation_train.sh
chmod +x ./scripts/test/gen_conversation_test.sh
```

---

## 📈 Expected Results

### Data Quality Metrics

- ✓ All 800 graphs successfully generated (with gradual escalation)
- ✓ Communication diversity across 3 turns
- ✓ Clear separation between escalation and constant attacks
- ✓ Realistic temporal patterns

### File Sizes

- Raw conversations (train): 50-100 MB
- Raw conversations (test): 20-30 MB  
- GNN dataset (train): 200-300 MB
- GNN dataset (test): 80-100 MB

---

## 📚 Files Modified/Created

| File | Type | Purpose |
|------|------|---------|
| agent_prompts.py | Python | System prompts for normal/attacker agents with escalation |
| agents.py | Python | AgentGraph + AgentGraphGradientEscalation classes |
| gen_graph.py | Python | ⭐ Main generation script (async LLM calls) |
| gen_training_dataset.py | Python | Convert conversations to GNN format |
| data.py | Python | Data loading (msmarco.json parsing) |
| merge_datasets.py | Python | Combine multiple JSON files |
| utils.py | Python | Utility functions |
| gen_conversation_train.sh | Shell | Parallel training data generation |
| gen_conversation_test.sh | Shell | Parallel test data generation |
| README.md | Markdown | Detailed documentation |
| msmarco.json | Data | Base QA dataset (from MA) |

---

## 🎓 Understanding the Attack

### Escalation Timeline

```
Turn 0 (Time t=0):
  Agent (Normal):  "Answer is A" ✓
  Agent (Attack):  "Answer is A'" (10-20% wrong)
  
  Detection: Easy to dismiss as variation
  
Turn 1 (Time t=1):
  Agent (Normal):  "I maintain A" 
  Agent (Attack):  "Actually B" (30-40% wrong)
     → More confident in wrong answer
     
  Detection: Suspicious but could be legitimate debate
  
Turn 2-3 (Time t=2+):
  Agent (Attack):  "Definitely not A, it's C" (60-90% wrong)
     → By now, clear but pattern was gradual
  
  Detection Challenge: 
    Static ML: Sees average wrongness ≈ 50%, mixed signals
    Temporal ML: Sees escalation pattern 10%→30%→60%→90%
```

### Why This Matters

- Tests if G-Safeguard architecture can **capture temporal dynamics**
- Single shot embeddings (current) might miss escalation
- **DIRECTION 4** proposes Temporal GNN to solve this

---

## 💡 For Research/Paper

### Comparison Setup

**Hypothesis:** Temporal models outperform static models on escalation attacks.

```
Experiment:
  - Run GP generation (escalation + constant)
  - Train static GAT on both
  - Train Temporal GCN on both
  
Expected Results:
  - Static GAT: ~70% on escalation, ~85% on constant
  - Temporal GCN: ~92% on escalation, ~88% on constant
  
Conclusion: Temporal architecture specifically helps with gradual attacks
```

---

## ✅ Verification Checklist

After setup, verify:

- [ ] All Python files import without errors
- [ ] `./scripts/train/gen_conversation_train.sh` is executable
- [ ] `./scripts/test/gen_conversation_test.sh` is executable
- [ ] `./datasets/msmarco.json` exists and has >100 entries
- [ ] `gen_graph.py` runs without errors on small sample:
  ```bash
  python gen_graph.py --num_graphs 1 --samples 1 --num_attackers 1
  ```

---

## 📖 See Also

- **[GP/README.md](README.md)** - Full documentation
- **[RESEARCH_DIRECTIONS_VI.md](../RESEARCH_DIRECTIONS_VI.md)** - DIRECTION 4: Temporal GNN (evaluation target)
- **[MA/README.md](../MA/README.md)** - Memory Attack (comparison/reference)

---

**Created:** 2026-04-02  
**Status:** ✅ Ready for data generation
