import os
import json
import pickle
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def gen_model_training_set(language_dataset, embedding_model, save_path):
    """Convert conversation dataset into GNN-ready dataset"""
    dataset = []
    
    for meta_data in tqdm(language_dataset, desc="Generate training data"):
        adj_matrix = meta_data["adj_matrix"]
        attacker_idxes = meta_data["attacker_idxes"]
        system_prompts = meta_data["system_prompts"]
        communication_data = meta_data["communication_data"]

        adj_matrix_np = np.array(adj_matrix)
        labels = np.array([1 if i in attacker_idxes else 0 for i in range(len(adj_matrix))])
        
        # Embed system prompts (node features)
        system_prompts_embedding = []
        for i in range(len(system_prompts)):
            # Handle special marker for attacker base prompts
            prompt_text = system_prompts[i].replace("ATTACKER_BASE", "attacker agent")
            system_prompts_embedding.append(embedding_model.encode(prompt_text))
        system_prompts_embedding = np.array(system_prompts_embedding)

        # Extract edge structure
        edge_index = adj_matrix_np.nonzero()
        edge_index = np.array(edge_index)
        
        # Embed communications (edge attributes)
        communication_embeddings = [[] for _ in range(len(adj_matrix))]
        for turn_idx, turn_data in enumerate(communication_data):
            turn_embeddings = [None] * len(adj_matrix)
            for agent_idx, message_text in turn_data:
                message_embedding = embedding_model.encode(message_text)
                turn_embeddings[agent_idx] = message_embedding
            
            for agent_idx in range(len(turn_embeddings)):
                communication_embeddings[agent_idx].append(turn_embeddings[agent_idx])
        
        communication_embeddings = np.array(communication_embeddings)
        edge_attr = np.array(communication_embeddings[edge_index[1]], copy=True)
        
        data = {}
        data["adj_matrix"] = adj_matrix_np
        data["features"] = system_prompts_embedding
        data["labels"] = labels
        data["edge_index"] = edge_index
        data["edge_attr"] = edge_attr
        
        dataset.append(data)
    
    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert gradient escalation dataset to GNN format")
    parser.add_argument("--dataset", type=str, default="gradient_escalation",
                        help="Dataset type")
    parser.add_argument("--embedding_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Embedding model")
    
    args = parser.parse_args()
    
    # Set up paths
    data_dir = f"./agent_graph_dataset/{args.dataset}/train/dataset.json"
    save_dir = f"./ModelTrainingSet/{args.dataset}"
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    save_path = os.path.join(save_dir, "dataset.pkl")
    
    # Load embedding model
    print(f"Loading embedding model: {args.embedding_model}")
    embedding_model = SentenceTransformer(args.embedding_model)
    
    # Load language dataset
    with open(data_dir, 'r') as file:
        language_dataset = json.load(file)
    
    print(f"Converting {len(language_dataset)} samples to GNN format...")
    gen_model_training_set(language_dataset, embedding_model, save_path)
    
    # Verify
    with open(save_path, "rb") as f:
        loaded_dataset = pickle.load(f)
    
    print(f"Successfully created GNN dataset with {len(loaded_dataset)} samples")
    if len(loaded_dataset) > 0:
        sample = loaded_dataset[0]
        print(f"Sample structure:")
        print(f"  - Node features shape: {sample['features'].shape}")
        print(f"  - Labels: {sample['labels']}")
        print(f"  - Edge index shape: {sample['edge_index'].shape}")
        print(f"  - Edge attributes shape: {sample['edge_attr'].shape}")
