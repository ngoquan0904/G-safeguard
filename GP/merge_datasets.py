import os
import json
import argparse
from tqdm import tqdm


def merge_datasets(phase="train"):
    """Merge all generated dataset files into a single file"""
    dataset_dir = f"./agent_graph_dataset/gradient_escalation/{phase}"
    
    if not os.path.exists(dataset_dir):
        print(f"Directory not found: {dataset_dir}")
        return
    
    all_data = []
    
    # Find all JSON files in the directory
    json_files = [f for f in os.listdir(dataset_dir) if f.endswith(".json")]
    json_files.sort()
    
    print(f"Found {len(json_files)} JSON files in {dataset_dir}")
    
    for json_file in tqdm(json_files, desc=f"Merging {phase} datasets"):
        filepath = os.path.join(dataset_dir, json_file)
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_data.extend(data)
                else:
                    all_data.append(data)
        except Exception as e:
            print(f"Error reading {json_file}: {e}")
    
    # Save merged dataset
    output_file = os.path.join(dataset_dir, "dataset.json")
    with open(output_file, "w") as f:
        json.dump(all_data, f, indent=None)
    
    print(f"Merged dataset saved to {output_file}")
    print(f"Total samples: {len(all_data)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge generated datasets")
    parser.add_argument("--phase", type=str, default="train",
                        choices=["train", "test"],
                        help="Phase to merge (train or test)")
    
    args = parser.parse_args()
    merge_datasets(args.phase)
