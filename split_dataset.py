import os
import csv
import argparse
import sys
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed

def parse_args():
    parser = argparse.ArgumentParser(description="Split and export dataset using RecBole logic for consistency.")
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., games, books, movies)')
    parser.add_argument('--data_path', type=str, default='./dataset', help='Path to raw data directory')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for split files (default: data_path/dataset)')
    parser.add_argument('--seed', type=int, default=2020, help='Random seed for reproducibility (Must match the seed used in LightGCN)')
    return parser.parse_args()

def export_split_data(args):
    # Determine output directory
    output_dir = args.output_dir if args.output_dir else os.path.join(args.data_path, args.dataset)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Processing Dataset: {args.dataset}")
    print(f"Data Path: {args.data_path}")
    print(f"Output Directory: {output_dir}")
    print(f"Seed: {args.seed}")

    # 1. RecBole Configuration
    config_dict = {
        'model': 'LightGCN',
        'data_path': args.data_path,
        'dataset': args.dataset,
        'load_col': {'inter': ['user_id', 'item_id']},
        'USER_ID_FIELD': 'user_id',
        'ITEM_ID_FIELD': 'item_id',
        # Enable remap_id to use RecBole's internal splitting logic
        'remap_id': True, 
        # Standard 8:1:1 split (Train/Valid/Test)
        'eval_args': {
            'split': {'RS': [0.8, 0.1, 0.1]},
            'group_by': 'user',
            'order': 'RO', # Random Ordering
            'mode': 'full'
        },
        'seed': args.seed,
        'state': 'INFO',
        'reproducibility': True
    }

    config = Config(config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])

    # 2. Load and split dataset
    print("1. Loading and splitting dataset via RecBole...")
    dataset = create_dataset(config)
    # The data_preparation function returns DataLoaders for train, valid, and test
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # 3. Get ID mappings (Internal ID -> Original Token)
    # RecBole maps IDs to 1, 2, 3... internally. We need to restore original tokens for portability.
    user_id2token = dataset.field2id_token['user_id']
    item_id2token = dataset.field2id_token['item_id']

    # 4. Define export function
    def save_dataset_split(dataloader, filename):
        print(f"2. Exporting {filename} ...")
        filepath = os.path.join(output_dir, filename)
        
        # Access the underlying dataset object from the dataloader
        dataset_part = dataloader.dataset
        
        # Retrieve internal mapped IDs
        uids = dataset_part.inter_feat['user_id'].numpy()
        iids = dataset_part.inter_feat['item_id'].numpy()
        
        count = 0
        with open(filepath, 'w', encoding='utf-8', newline='') as f:
            # Write header (Standard RecBole format)
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['user_id:token', 'item_id:token'])
            
            for u_internal, i_internal in zip(uids, iids):
                try:
                    # Restore original Token
                    # Note: field2id_token is an array where index corresponds to the internal ID
                    u_token = user_id2token[u_internal]
                    i_token = item_id2token[i_internal]
                    
                    # Handle bytes type if necessary (RecBole sometimes stores tokens as bytes)
                    if isinstance(u_token, bytes): u_token = u_token.decode('utf-8')
                    if isinstance(i_token, bytes): i_token = i_token.decode('utf-8')
                    
                    writer.writerow([u_token, i_token])
                    count += 1
                except IndexError:
                    continue
                    
        print(f"   Saved {count} interactions to {filepath}")

    # 5. Execute export
    save_dataset_split(train_data, f"{args.dataset}.train.inter")
    save_dataset_split(valid_data, f"{args.dataset}.valid.inter")
    save_dataset_split(test_data,  f"{args.dataset}.test.inter")
    
    print("\nProcessing complete.")
    print("Files ready for independent loading.")

if __name__ == '__main__':
    args = parse_args()
    export_split_data(args)