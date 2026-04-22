import os
import json
import argparse
import numpy as np
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description="Generate item popularity categories (Cold/Mid/Hot).")
    
    # Dataset & Paths
    parser.add_argument('--dataset', type=str, default="movies", help='Dataset name (e.g., movies, books, games)')
    parser.add_argument('--data_root', type=str, default='./dataset', help='Data root directory')
    parser.add_argument('--output_root', type=str, default='./stats', help='Output directory for stats')
    
    # Stratification Configuration
    # Paper Strategy: Cold (Bottom 60%), Mid (Next 20%), Hot (Top 20%)
    parser.add_argument('--quantile_cold', type=float, default=0.60, help='Quantile threshold for Cold items (default: 0.60)')
    parser.add_argument('--quantile_hot', type=float, default=0.80, help='Quantile threshold for Hot items (default: 0.80)')
    
    return parser.parse_args()

def count_item_interactions(inter_path):
    """
    Count interaction frequency for each item from the .train.inter file.
    """
    print(f"Reading interactions from: {inter_path}")
    item_count = defaultdict(int)
    
    try:
        with open(inter_path, 'r', encoding='utf-8') as f:
            header = next(f) # Skip header
            for line in f:
                line = line.strip()
                if not line: continue
                
                parts = line.split('\t')
                if len(parts) < 2: continue
                
                # Assuming format: user_id \t item_id
                # Note: These IDs should match the ones used in training (either tokens or remapped IDs)
                # If using preprocess_split.py, these are tokens.
                item_id = parts[1]
                item_count[item_id] += 1
                
        print(f"   Total items with interactions: {len(item_count)}")
        return item_count
    except FileNotFoundError:
        print(f"Error: File not found {inter_path}")
        return {}

def categorize_items(item_count, q_cold, q_hot):
    """
    Categorize items based on interaction frequency quantiles.
    """
    counts = list(item_count.values())
    
    # Calculate thresholds
    thres_cold = np.quantile(counts, q_cold)
    thres_hot = np.quantile(counts, q_hot)
    
    print(f"\nStratification Thresholds:")
    print(f"   Cold < {q_cold*100}% ({thres_cold:.2f} interactions)")
    print(f"   Hot  > {q_hot*100}% ({thres_hot:.2f} interactions)")
    
    category_map = {}
    stats = defaultdict(int)
    
    for item_id, count in item_count.items():
        if count <= thres_cold:
            cat = "cold"
        elif count <= thres_hot:
            cat = "mid"
        else:
            cat = "hot"
            
        category_map[item_id] = cat
        stats[cat] += 1
        
    return category_map, stats

def save_json(data, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved category map to: {output_path}")

def main():
    args = parse_args()
    
    # Paths
    train_inter_path = os.path.join(args.data_root, args.dataset, f"{args.dataset}.inter")
    output_json_path = os.path.join(args.output_root, args.dataset, f"{args.dataset}_pop_category.json")
    
    # 1. Count Interactions
    item_count = count_item_interactions(train_inter_path)
    if not item_count:
        return

    # 2. Stratify
    category_map, stats = categorize_items(item_count, args.quantile_cold, args.quantile_hot)
    
    # 3. Print Stats
    total = sum(stats.values())
    print("\nGroup Statistics:")
    print(f"   Cold (Tail): {stats['cold']} ({stats['cold']/total*100:.1f}%)")
    print(f"   Mid        : {stats['mid']}  ({stats['mid']/total*100:.1f}%)")
    print(f"   Hot (Head) : {stats['hot']}  ({stats['hot']/total*100:.1f}%)")
    
    # 4. Save
    save_json(category_map, output_json_path)

if __name__ == "__main__":
    main()