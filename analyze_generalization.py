import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from tqdm import tqdm
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze Alignment Generalization (Inductive Test)")
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--data_root', type=str, default='./dataset', help='Data root directory')
    
    # Suffixes
    parser.add_argument('--sem_suffix', type=str, default='proj64', help='Semantic embedding suffix (Source)')
    parser.add_argument('--col_suffix', type=str, default='pco64', help='Collaborative embedding suffix (Target)')
    
    # Experiment
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Ratio of items used for training')
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=2024)
    
    return parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= Data Loading =================
def load_data(args):
    print(f">>> Loading Embeddings for {args.dataset}...")
    sem_path = os.path.join(args.data_root, args.dataset, f"{args.dataset}_item_embeddings-{args.sem_suffix}.npy")
    col_path = os.path.join(args.data_root, args.dataset, f"{args.dataset}_item_embeddings-{args.col_suffix}.npy")
    user_path = os.path.join(args.data_root, args.dataset, f"{args.dataset}_user_embeddings-{args.col_suffix}.npy")
    
    sem_item = np.load(sem_path).astype(np.float32)
    col_item = np.load(col_path).astype(np.float32)
    col_user = np.load(user_path).astype(np.float32)
    
    num_items = sem_item.shape[0]
    indices = np.arange(num_items)
    
    # Random split
    random.seed(args.seed)
    np.random.seed(args.seed)
    np.random.shuffle(indices)
    
    split = int(num_items * args.train_ratio)
    train_idx = indices[:split]
    test_idx = indices[split:]
    
    print(f"    Total Items: {num_items}")
    print(f"    [Inductive Split] Train Items: {len(train_idx)} | Test Items (Unseen): {len(test_idx)}")
    
    sem_item_t = torch.from_numpy(sem_item).to(DEVICE)
    col_item_t = torch.from_numpy(col_item).to(DEVICE)
    col_user_t = torch.from_numpy(col_user).to(DEVICE)
    
    return sem_item_t, col_item_t, col_user_t, train_idx, test_idx

def load_interactions(args, suffix='train'):
    path = os.path.join(args.data_root, args.dataset, f"{args.dataset}.{suffix}.inter")
    print(f">>> Loading Interactions from {path}...")
    user_inter = {}
    try:
        with open(path, 'r') as f:
            next(f)
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    try:
                        u, i = int(parts[0]), int(parts[1])
                        if u not in user_inter: user_inter[u] = []
                        user_inter[u].append(i)
                    except: pass
    except FileNotFoundError:
        print(f"Warning: {path} not found.")
    return user_inter

# ================= Models =================
class LinearMapper(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Linear(input_dim, output_dim)
    def forward(self, x): return self.net(x)

class MLPMapper(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=2048, layers=2):
        super().__init__()
        modules = []
        in_d = input_dim
        for _ in range(layers - 1):
            modules.append(nn.Linear(in_d, hidden_dim))
            modules.append(nn.ReLU())
            in_d = hidden_dim
        modules.append(nn.Linear(in_d, output_dim))
        self.net = nn.Sequential(*modules)
    def forward(self, x): return self.net(x)

# ================= Training =================
def train_mapping(model, src_emb, tgt_emb, train_idx, args):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    train_idx_t = torch.from_numpy(train_idx).to(DEVICE)
    num_train = len(train_idx)
    
    for epoch in tqdm(range(args.epochs), desc="Training", leave=False):
        perm = torch.randperm(num_train, device=DEVICE)
        idx_perm = train_idx_t[perm]
        
        for i in range(0, num_train, args.batch_size):
            batch_idx = idx_perm[i : i+args.batch_size]
            x = src_emb[batch_idx]
            y = tgt_emb[batch_idx]
            
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
    return model

# ================= Evaluation =================
def eval_reconstruction(model, src_emb, tgt_emb, test_idx):
    """Evaluate vector reconstruction quality on unseen items"""
    model.eval()
    with torch.no_grad():
        x = src_emb[test_idx]
        y_true = tgt_emb[test_idx]
        y_pred = model(x)
        
        y_true_np = y_true.cpu().numpy()
        y_pred_np = y_pred.cpu().numpy()
        
        r2 = r2_score(y_true_np, y_pred_np)
        cos = F.cosine_similarity(y_pred, y_true, dim=1).mean().item()
        
    return r2, cos

def eval_geometry(model, src_emb, tgt_emb, test_idx, k_neighbor=20):
    """Evaluate neighborhood preservation on unseen items"""
    model.eval()
    with torch.no_grad():
        src_test = src_emb[test_idx]
        tgt_test = tgt_emb[test_idx]
        mapped_test = model(src_test)      
        
        mapped_norm = F.normalize(mapped_test, p=2, dim=1)
        target_norm = F.normalize(tgt_test, p=2, dim=1)
        
        num_test = len(test_idx)
        # Sample anchors for Rank Correlation efficiency
        anchor_perm = torch.randperm(num_test)[:min(1000, num_test)]
        anchors_tgt = target_norm[anchor_perm]
        anchors_map = mapped_norm[anchor_perm]
        
        batch_size = 500
        total_jaccard = 0.0
        total_spearman = 0.0
        count = 0
        
        for i in range(0, num_test, batch_size):
            batch_tgt = target_norm[i : i+batch_size]
            batch_map = mapped_norm[i : i+batch_size]
            bs = batch_tgt.shape[0]
            
            # Neighborhood Jaccard
            sim_tgt = torch.matmul(batch_tgt, target_norm.t())
            _, topk_tgt = torch.topk(sim_tgt, k=k_neighbor+1, dim=1)
            sim_map = torch.matmul(batch_map, mapped_norm.t())
            _, topk_map = torch.topk(sim_map, k=k_neighbor+1, dim=1)
            
            topk_tgt = topk_tgt[:, 1:].cpu().numpy()
            topk_map = topk_map[:, 1:].cpu().numpy()
            
            for k in range(bs):
                set_t = set(topk_tgt[k])
                set_m = set(topk_map[k])
                if len(set_t | set_m) > 0:
                    total_jaccard += len(set_t & set_m) / len(set_t | set_m)
            
            # Rank Correlation
            dist_tgt = torch.matmul(batch_tgt, anchors_tgt.t())
            dist_map = torch.matmul(batch_map, anchors_map.t())
            d_tgt_np = dist_tgt.cpu().numpy()
            d_map_np = dist_map.cpu().numpy()
            
            for k in range(bs):
                corr, _ = spearmanr(d_tgt_np[k], d_map_np[k])
                if not np.isnan(corr): total_spearman += corr
            count += bs
            
    return total_jaccard / count, total_spearman / count

def eval_inductive_recommendation(model, src_item_emb, col_user_emb, col_item_emb, 
                                  train_inter, test_inter, test_item_indices, eval_users):
    """
    Inductive Recommendation Evaluation:
    Can the model correctly rank UNSEEN items for users?
    
    - Candidate Space: All items (but historical training items are masked).
    - Ground Truth: Only interactions with UNSEEN items (test_item_indices).
    """
    model.eval()
    test_item_set = set(test_item_indices)
    topk = 20
    
    with torch.no_grad():
        # 1. Map all items (Training items learned, Test items inferred/generalized)
        mapped_item_emb = model(src_item_emb)
        
        recall_cf_sum = 0.0
        recall_ps_sum = 0.0
        valid_count = 0
        
        for i in range(0, len(eval_users), 1024):
            batch_users = eval_users[i : i+1024]
            batch_users_t = torch.tensor(batch_users, dtype=torch.long, device=DEVICE)
            batch_u_emb = col_user_emb[batch_users_t]
            
            # Scores over ALL items
            scores_cf = torch.matmul(batch_u_emb, col_item_emb.t())
            scores_ps = torch.matmul(batch_u_emb, mapped_item_emb.t())
            
            # Mask history
            for idx, u in enumerate(batch_users):
                if u in train_inter:
                    mask_items = train_inter[u]
                    scores_cf[idx, mask_items] = float('-inf')
                    scores_ps[idx, mask_items] = float('-inf')
            
            _, topk_cf = torch.topk(scores_cf, k=topk, dim=1)
            _, topk_ps = torch.topk(scores_ps, k=topk, dim=1)
            
            topk_cf = topk_cf.cpu().numpy()
            topk_ps = topk_ps.cpu().numpy()
            
            for idx, u in enumerate(batch_users):
                raw_truth = set(test_inter.get(u, []))
                # Only consider Unseen items in truth
                truth_inductive = raw_truth & test_item_set
                
                if not truth_inductive: continue
                
                rec_cf = set(topk_cf[idx])
                rec_ps = set(topk_ps[idx])
                
                recall_cf_sum += len(rec_cf & truth_inductive) / len(truth_inductive)
                recall_ps_sum += len(rec_ps & truth_inductive) / len(truth_inductive)
                valid_count += 1
                
    return recall_cf_sum / valid_count, recall_ps_sum / valid_count

# ================= Main =================
def main():
    args = parse_args()
    
    # Set Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print(f"\n{'='*80}\nAlignment Generalization Experiment (Inductive Test)\n{'='*80}")
    
    # Load
    sem_emb, col_emb, col_user_emb, train_idx, test_idx = load_data(args)
    train_inter = load_interactions(args, 'train')
    test_inter = load_interactions(args, 'test')
    
    # Pre-select users who actually interacted with Unseen Items
    test_item_set = set(test_idx)
    eval_users = []
    for u, items in test_inter.items():
        if len(set(items) & test_item_set) > 0:
            eval_users.append(u)
    print(f"    Evaluation Users (interacted with unseen items): {len(eval_users)}")
    
    in_dim = sem_emb.shape[1]
    out_dim = col_emb.shape[1]
    
    models = [
        ("Linear Map",        lambda: LinearMapper(in_dim, out_dim)),
        ("MLP-1 (Small)",     lambda: MLPMapper(in_dim, out_dim, hidden_dim=512, layers=2)),
        ("MLP-2 (Medium)",    lambda: MLPMapper(in_dim, out_dim, hidden_dim=2048, layers=3)),
        ("MLP-3 (Deep)",      lambda: MLPMapper(in_dim, out_dim, hidden_dim=2048, layers=4)),
    ]
    
    print(f"\n{'Model':<15} | {'R2':<6} | {'Cos':<6} | {'GeoJac':<6} | {'RecCF':<7} | {'RecPs':<7} | {'Drop%':<6}")
    print("-" * 80)
    
    for name, factory in models:
        model = factory().to(DEVICE)
        
        # Train on Train Items
        model = train_mapping(model, sem_emb, col_emb, train_idx, args)
        
        # Eval on Unseen Test Items
        r2, cos = eval_reconstruction(model, sem_emb, col_emb, test_idx)
        geo_jac, _ = eval_geometry(model, sem_emb, col_emb, test_idx)
        rec_cf, rec_ps = eval_inductive_recommendation(
            model, sem_emb, col_user_emb, col_emb, 
            train_inter, test_inter, test_idx, eval_users
        )
        
        drop = (rec_cf - rec_ps) / rec_cf * 100 if rec_cf > 0 else 0
        print(f"{name:<15} | {r2:<6.3f} | {cos:<6.3f} | {geo_jac:<6.3f} | {rec_cf:<7.4f} | {rec_ps:<7.4f} | {drop:<6.1f}")

    print("=" * 80)
    print("Note: 'RecPs' is the recall of the mapped semantic model on unseen items.")
    print("      'Drop%' indicates performance loss compared to Oracle CF.")

if __name__ == "__main__":
    main()