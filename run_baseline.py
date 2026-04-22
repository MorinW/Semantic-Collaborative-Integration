import argparse
import os
import numpy as np
import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed, get_model, get_trainer

def parse_args():
    parser = argparse.ArgumentParser(description="Train Collaborative Baselines and Export Embeddings.")
    
    # Dataset & Model
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., movies, books, games)')
    parser.add_argument('--model', type=str, default='LightGCN', choices=['LightGCN', 'NCL', 'SimGCL'], help='Model name')
    parser.add_argument('--data_path', type=str, default='./dataset', help='Path to dataset')
    parser.add_argument('--output_root', type=str, default='./saved', help='Directory to save model checkpoints')
    
    # Training Hyperparameters
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_batch_size', type=int, default=2048, help='Training batch size')
    parser.add_argument('--embedding_size', type=int, default=64, help='Embedding dimension')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of GCN layers')
    
    # System & Evaluation
    parser.add_argument('--gpu_id', type=str, default='0', help='GPU ID to use')
    parser.add_argument('--seed', type=int, default=2020, help='Random seed for reproducibility')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    
    return parser.parse_args()

def export_embeddings(model, dataset, args, config):
    """
    Extracts user/item embeddings from the model, maps them back to original token IDs,
    and saves them as .npy files.
    """
    print(f"\n>>> Exporting {args.model} Embeddings...")
    
    model.eval()
    with torch.no_grad():
        # LightGCN/NCL/SimGCL in RecBole return all embeddings in forward()
        user_all_embeddings, item_all_embeddings = model.forward()
    
    user_all_embeddings = user_all_embeddings.cpu().numpy()
    item_all_embeddings = item_all_embeddings.cpu().numpy()
    
    n_int_users = user_all_embeddings.shape[0]
    n_int_items = item_all_embeddings.shape[0]
    dim = user_all_embeddings.shape[1]
    
    # --- Mapping Logic: Internal ID -> Token -> Original Integer ID ---
    # Retrieve field names
    uid_field = config['USER_ID_FIELD']
    iid_field = config['ITEM_ID_FIELD']
    
    # Get ID to Token mapping lists
    # RecBole stores these as lists where index=internal_id, value=token
    user_id2token = dataset.field2id_token[uid_field]
    item_id2token = dataset.field2id_token[iid_field]
    
    def get_max_original_id(token_list):
        max_id = 0
        for token in token_list:
            if isinstance(token, bytes): token = token.decode('utf-8')
            if token == '[PAD]': continue # Skip padding token
            try:
                oid = int(token)
                if oid > max_id: max_id = oid
            except ValueError:
                continue
        return max_id

    # Determine size of output arrays
    max_uid = get_max_original_id(user_id2token)
    max_iid = get_max_original_id(item_id2token)
    
    print(f"   Max Original User ID: {max_uid}, Max Original Item ID: {max_iid}")
    
    # Create zero-initialized arrays aligned to original IDs
    final_user_emb = np.zeros((max_uid + 1, dim), dtype=np.float32)
    final_item_emb = np.zeros((max_iid + 1, dim), dtype=np.float32)
    
    # Fill User Embeddings
    count_u = 0
    for internal_id in range(n_int_users):
        token = user_id2token[internal_id]
        if isinstance(token, bytes): token = token.decode('utf-8')
        try:
            original_id = int(token)
            if 0 <= original_id <= max_uid:
                final_user_emb[original_id] = user_all_embeddings[internal_id]
                count_u += 1
        except ValueError:
            continue

    # Fill Item Embeddings
    count_i = 0
    for internal_id in range(n_int_items):
        token = item_id2token[internal_id]
        if isinstance(token, bytes): token = token.decode('utf-8')
        try:
            original_id = int(token)
            if 0 <= original_id <= max_iid:
                final_item_emb[original_id] = item_all_embeddings[internal_id]
                count_i += 1
        except ValueError:
            continue
            
    # --- Saving ---
    # Define paths: dataset/{dataset}/{dataset}_user_emb_{model}.npy
    save_dir = os.path.join(args.data_path, args.dataset)
    os.makedirs(save_dir, exist_ok=True)
    
    suffix = args.model.lower() # lightgcn, ncl, etc.
    user_path = os.path.join(save_dir, f"{args.dataset}_user_embeddings-{suffix}.npy")
    item_path = os.path.join(save_dir, f"{args.dataset}_item_embeddings-{suffix}.npy")
    
    np.save(user_path, final_user_emb)
    np.save(item_path, final_item_emb)
    
    print(f"   Saved User Embeddings ({count_u} mapped): {user_path}")
    print(f"   Saved Item Embeddings ({count_i} mapped): {item_path}")

def main():
    args = parse_args()
    
    # 1. Configuration
    config_dict = {
        # Environment
        'gpu_id': args.gpu_id,
        'seed': args.seed,
        'reproducibility': True,
        'checkpoint_dir': args.output_root,
        
        # Data
        'data_path': args.data_path,
        'load_col': {'inter': ['user_id', 'item_id']}, 
        
        # Model
        'embedding_size': args.embedding_size,
        'n_layers': args.n_layers,
        
        # Training
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'train_batch_size': args.train_batch_size,
        'learner': 'adam',
        'eval_step': args.eval_step,
        'stopping_step': args.patience,
        
        # Evaluation (Consistent with other experiments)
        'eval_args': {
            'split': {'RS': [0.8, 0.1, 0.1]},
            'group_by': 'user',
            'order': 'RO',
            'mode': 'full'
        },
        'metrics': ['Recall', 'NDCG', 'MRR'],
        'topk': [10, 20],
        'valid_metric': 'NDCG@20',
        'eval_batch_size': 4096 * 100,
    }

    print(f">>> Starting Training: {args.model} on {args.dataset}")
    
    # 2. Initialization
    config = Config(model=args.model, dataset=args.dataset, config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])
    
    # 3. Data Loading
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    
    # 4. Model Construction
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    
    # 5. Trainer
    trainer = get_trainer(config['model'], config['model'])(config, model)
    
    # 6. Training Loop
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, show_progress=True)
    
    # 7. Final Evaluation
    test_result = trainer.evaluate(test_data)
    print(f"\nTraining Finished. Test Result: {test_result}")
    
    # 8. Export Embeddings (The Auto-Export Step)
    export_embeddings(model, dataset, args, config)

if __name__ == '__main__':
    main()