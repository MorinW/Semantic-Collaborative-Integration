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
    parser = argparse.ArgumentParser(description="Analyze Alignment Limits (Claim 2)")
    parser.add_argument('--dataset', type=str, default='movies')
    parser.add_argument('--data_root', type=str, default='./dataset')
    
    # Suffixes to identify Source (Semantic) and Target (Collaborative)
    parser.add_argument('--src_suffix', type=str, default='proj64', help='Source embedding suffix (Semantic)')
    parser.add_argument('--tgt_suffix', type=str, default='lightgcn', help='Target embedding suffix (Collaborative)')
    
    # Training Params for the Probe
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=1024)
    
    return parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_embeddings(args):
    data_dir = os.path.join(args.data_root, args.dataset)
    
    # Source: Semantic (e.g., movies_item_embeddings-proj64.npy)
    src_path = os.path.join(data_dir, f"{args.dataset}_item_embeddings-{args.src_suffix}.npy")
    # Target: Collaborative (e.g., movies_item_embeddings-lightgcn.npy)
    tgt_path = os.path.join(data_dir, f"{args.dataset}_item_embeddings-{args.tgt_suffix}.npy")
    # Target User (for Recall Eval)
    tgt_user_path = os.path.join(data_dir, f"{args.dataset}_user_embeddings-{args.tgt_suffix}.npy")
    
    print(f">>> Loading Embeddings...")
    print(f"    Source: {os.path.basename(src_path)}")
    print(f"    Target: {os.path.basename(tgt_path)}")
    
    try:
        src_item = np.load(src_path).astype(np.float32)
        tgt_item = np.load(tgt_path).astype(np.float32)
        tgt_user = np.load(tgt_user_path).astype(np.float32)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        exit(1)
        
    assert src_item.shape[0] == tgt_item.shape[0], "Item count mismatch between Source and Target!"
    
    # Split Train/Test Items (80/20)
    num_items = src_item.shape[0]
    indices = np.arange(num_items)
    np.random.seed(2024)
    np.random.shuffle(indices)
    
    split = int(0.8 * num_items)
    train_idx = indices[:split]
    test_idx = indices[split:]
    
    print(f"    Total Items: {num_items} | Train: {len(train_idx)} | Test: {len(test_idx)}")
    
    return (
        torch.from_numpy(src_item).to(DEVICE),
        torch.from_numpy(tgt_item).to(DEVICE),
        torch.from_numpy(tgt_user).to(DEVICE),
        train_idx, test_idx
    )

def load_interactions(args):
    # Load Test Interactions for Recall evaluation
    path = os.path.join(args.data_root, args.dataset, f"{args.dataset}.test.inter")
    user_inter = {}
    print(f">>> Loading Test Interactions: {path}")
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
    return user_inter

class MLPMapper(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=512, layers=1):
        super().__init__()
        if layers == 0: # Linear
            self.net = nn.Linear(input_dim, output_dim)
        else:
            modules = []
            in_d = input_dim
            for _ in range(layers):
                modules.append(nn.Linear(in_d, hidden_dim))
                modules.append(nn.ReLU())
                in_d = hidden_dim
            modules.append(nn.Linear(in_d, output_dim))
            self.net = nn.Sequential(*modules)
            
    def forward(self, x):
        return self.net(x)

def train_probe(model, src, tgt, train_idx, args):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss() # Align geometry directly
    
    train_src = src[train_idx]
    train_tgt = tgt[train_idx]
    num_train = len(train_idx)
    
    for epoch in tqdm(range(args.epochs), desc="Training Probe", leave=False):
        perm = torch.randperm(num_train, device=DEVICE)
        
        for i in range(0, num_train, args.batch_size):
            batch_idx = perm[i : i+args.batch_size]
            x = train_src[batch_idx]
            y = train_tgt[batch_idx]
            
            pred = model(x)
            loss = criterion(pred, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    return model

def eval_reconstruction(model, src, tgt, test_idx):
    model.eval()
    with torch.no_grad():
        x = src[test_idx]
        y_true = tgt[test_idx]
        y_pred = model(x)
        
        # 1. Cosine Similarity
        cos = F.cosine_similarity(y_pred, y_true).mean().item()
        
        # 2. R2 Score (Coefficient of Determination)
        y_true_np = y_true.cpu().numpy()
        y_pred_np = y_pred.cpu().numpy()
        r2 = r2_score(y_true_np, y_pred_np)
        
    return r2, cos

def eval_recommendation_overlap(model, src_item, tgt_user, tgt_item, test_inter):
    """
    Check if the mapped items preserve recommendation ranking.
    Compare: TopK(User @ TargetItem) vs TopK(User @ MappedItem)
    """
    model.eval()
    with torch.no_grad():
        mapped_items = model(src_item) # All items mapped
        
        # Sample users for speed
        test_users = list(test_inter.keys())
        if len(test_users) > 1000:
            test_users = random.sample(test_users, 1000)
            
        users_t = torch.tensor(test_users, dtype=torch.long, device=DEVICE)
        u_emb = tgt_user[users_t]
        
        # Original CF Scores
        scores_cf = u_emb @ tgt_item.t()
        _, topk_cf = torch.topk(scores_cf, k=20, dim=1)
        
        # Mapped Semantic Scores (Pseudo-CF)
        scores_map = u_emb @ mapped_items.t()
        _, topk_map = torch.topk(scores_map, k=20, dim=1)
        
        # Jaccard
        topk_cf = topk_cf.cpu().numpy()
        topk_map = topk_map.cpu().numpy()
        
        jaccard_sum = 0
        for i in range(len(test_users)):
            s1 = set(topk_cf[i])
            s2 = set(topk_map[i])
            jaccard_sum += len(s1 & s2) / len(s1 | s2)
            
    return jaccard_sum / len(test_users)

def main():
    args = parse_args()
    print(f"🚀 Analyzing Alignment: {args.src_suffix} -> {args.tgt_suffix}")
    
    # 1. Load Embeddings
    src_emb, tgt_emb, tgt_user_emb, train_idx, test_idx = load_embeddings(args)
    test_inter = load_interactions(args)
    
    in_dim = src_emb.shape[1]
    out_dim = tgt_emb.shape[1]
    
    # 2. Define Probes
    probes = [
        ("Linear", 0, 0),
        ("MLP-1 (Small)", 1, 512),
        ("MLP-2 (Large)", 2, 2048)
    ]
    
    print("\n" + "="*65)
    print(f"{'Model':<15} | {'R2 (Test)':<10} | {'Cos (Test)':<10} | {'Rec Jaccard':<10}")
    print("-" * 65)
    
    for name, layers, hidden in probes:
        # Train
        model = MLPMapper(in_dim, out_dim, hidden, layers).to(DEVICE)
        train_probe(model, src_emb, tgt_emb, train_idx, args)
        
        # Evaluate
        r2, cos = eval_reconstruction(model, src_emb, tgt_emb, test_idx)
        rec_jac = eval_recommendation_overlap(model, src_emb, tgt_user_emb, tgt_emb, test_inter)
        
        print(f"{name:<15} | {r2:<10.4f} | {cos:<10.4f} | {rec_jac:<10.4f}")
    
    print("=" * 65)
    print("💡 Key Takeaway: Low R2/Jaccard implies structural mismatch (Claim 2).")

if __name__ == "__main__":
    main()