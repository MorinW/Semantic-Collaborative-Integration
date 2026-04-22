import os
import time
import argparse
import json
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ================= Argument Parsing =================
def parse_args():
    parser = argparse.ArgumentParser(description="Train Naive Fusion Model (Norm-Concat-Norm + Hard Negatives)")
    
    # Dataset & Paths
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., movies, books, games)')
    parser.add_argument('--data_root', type=str, default='./dataset', help='Data root directory')
    parser.add_argument('--output_root', type=str, default='./saved', help='Model save directory')
    parser.add_argument('--log_root', type=str, default='./stats', help='Log directory')
    
    # Dimensions & Architecture
    parser.add_argument('--raw_dim', type=int, default=1024, help='Input dimension of raw semantic embeddings (BGE-M3)')
    parser.add_argument('--final_dim', type=int, default=128, help='Final fusion embedding dimension')
    parser.add_argument('--gcn_layers', type=int, default=2, help='Number of LightGCN layers')
    parser.add_argument('--collab_init', type=str, default='random', choices=['random', 'mlp'], help='Initialization mode for collaborative view')
    
    # Training Hyperparameters
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2048, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--temperature', type=float, default=0.15, help='InfoNCE temperature (lower = harder)')
    parser.add_argument('--sem_dropout', type=float, default=0.0, help='Dropout for semantic branch')
    
    # Hard Negative Mining Strategy
    parser.add_argument('--hard_neg_factor', type=int, default=2, help='Candidate sampling factor for hard negative mining')
    parser.add_argument('--neg_per_pos', type=int, default=256, help='Number of negative samples per positive')
    
    # System
    parser.add_argument('--seed', type=int, default=2020, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    
    return parser.parse_args()

# ================= Utils =================
def ensure_dir(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d): 
        os.makedirs(d, exist_ok=True)

class Logger:
    def __init__(self, filename):
        self.filename = filename
        ensure_dir(filename)
    
    def log(self, msg):
        print(msg)
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        with open(self.filename, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {msg}\n")

# ================= Data Loading =================
def load_data(args, device):
    print(">>> Loading Data...")
    dataset_path = os.path.join(args.data_root, args.dataset)
    item_emb_path = os.path.join(dataset_path, f"{args.dataset}_item_embeddings.npy")
    
    if not os.path.exists(item_emb_path): 
        raise FileNotFoundError(f"Item embeddings not found at {item_emb_path}")
        
    raw_item_embs = np.load(item_emb_path).astype(np.float32)
    # Normalize raw input embeddings
    norms = np.linalg.norm(raw_item_embs, axis=1, keepdims=True)
    raw_item_embs = raw_item_embs / (norms + 1e-10)
    
    def read_inter(suffix):
        d = {}; path = os.path.join(dataset_path, f"{args.dataset}.{suffix}.inter")
        if not os.path.exists(path): return d
        with open(path,'r',encoding='utf-8') as f:
            next(f) # skip header
            for line in f:
                p=line.strip().split('\t')
                if len(p)<2: continue
                try:
                    u,i=int(p[0]),int(p[1])
                    d.setdefault(u,[]).append(i)
                except ValueError: continue
        return d
    
    train, valid, test = read_inter("train"), read_inter("valid"), read_inter("test")
    
    # Determine ID space
    all_users = set(train.keys()) | set(valid.keys()) | set(test.keys())
    n_users = (max(all_users)+1) if all_users else 0
    n_items = raw_item_embs.shape[0]
    
    # Build Graph for LightGCN
    train_u, train_i = [], []
    for u, items in train.items():
        for i in items:
            if i < n_items: train_u.append(u); train_i.append(i)
    
    rows = np.concatenate([train_u, np.array(train_i)+n_users])
    cols = np.concatenate([np.array(train_i)+n_users, train_u])
    data = np.ones(len(rows))
    
    adj = sp.coo_matrix((data, (rows, cols)), shape=(n_users+n_items, n_users+n_items))
    
    # Normalize Adjacency Matrix
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -0.5, where=rowsum>0)
    d_inv[rowsum==0]=0
    d_mat = sp.diags(d_inv)
    norm_adj = d_mat.dot(adj).dot(d_mat).tocoo()
    
    # Convert to Sparse Tensor
    adj_tensor = torch.sparse_coo_tensor(
        torch.from_numpy(np.vstack((norm_adj.row, norm_adj.col))),
        torch.from_numpy(norm_adj.data).float(), norm_adj.shape).coalesce()
        
    # Pre-compute User Semantic Base (Mean Pooling)
    user_sem = np.zeros((n_users, raw_item_embs.shape[1]), dtype=np.float32)
    for u, items in train.items():
        v = [i for i in items if i < n_items]
        if v: 
            m = np.mean(raw_item_embs[v],0)
            user_sem[u] = m / (np.linalg.norm(m)+1e-10)
            
    return raw_item_embs, user_sem, adj_tensor, train, valid, test, n_users, n_items

class UPDataset(Dataset):
    def __init__(self, train):
        self.s = []
        for u, items in train.items():
            for i in items: self.s.append((u,i))
    def __len__(self): return len(self.s)
    def __getitem__(self, i): return self.s[i]

# ================= Model: Naive Fusion =================
class DualBranchModel(nn.Module):
    def __init__(self, args, n_users, n_items, raw_item_embs, user_sem_base, adj_graph):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.adj_graph = adj_graph
        self.branch_dim = args.final_dim // 2
        self.gcn_layers = args.gcn_layers
        self.temperature = args.temperature
        
        # Buffers
        self.register_buffer("raw_item_embs", torch.from_numpy(raw_item_embs))
        self.register_buffer("user_sem_base", torch.from_numpy(user_sem_base))
        in_dim = self.raw_item_embs.shape[1]

        # 1. Semantic Branch: Linear Projection
        self.sem_mlp = nn.Sequential(
            nn.Linear(in_dim, self.branch_dim),
            nn.Dropout(args.sem_dropout)
        )
        
        # 2. Collaborative Branch: LightGCN
        if args.collab_init == 'random':
            self.collab_item_emb = nn.Embedding(n_items, self.branch_dim)
            nn.init.xavier_normal_(self.collab_item_emb.weight)
        else:
            # Optional: Init collaborative view from content (not used in main results)
            self.collab_mlp = nn.Sequential(
                nn.Linear(in_dim, 2 * in_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(2*in_dim, self.branch_dim)
            )
        self.collab_user_emb = nn.Embedding(n_users, self.branch_dim)
        nn.init.xavier_normal_(self.collab_user_emb.weight)

    def get_semantic_view(self):
        # Project and Normalize
        u_sem = self.sem_mlp(self.user_sem_base) 
        i_sem = self.sem_mlp(self.raw_item_embs)
        return F.normalize(u_sem, dim=-1), F.normalize(i_sem, dim=-1)

    def get_collaborative_view(self):
        u_0 = self.collab_user_emb.weight
        if hasattr(self, 'collab_item_emb'): 
            i_0 = self.collab_item_emb.weight
        else: 
            i_0 = self.collab_mlp(self.raw_item_embs)
        
        ego = torch.cat([u_0, i_0], 0)
        embs = [ego]
        
        # LightGCN Propagation
        for k in range(self.gcn_layers):
            # Disable AMP for sparse mm to avoid instability
            with torch.cuda.amp.autocast(enabled=False):
                ego = ego.float()
                ego = torch.sparse.mm(self.adj_graph, ego)
            embs.append(ego)
            
        final = torch.mean(torch.stack(embs, 1), 1)
        u_final, i_final = torch.split(final, [self.n_users, self.n_items])
        return F.normalize(u_final, dim=-1), F.normalize(i_final, dim=-1)

    def forward_embedding(self):
        """
        Norm-Concat-Norm Fusion Strategy
        """
        u_sem, i_sem = self.get_semantic_view()
        u_collab, i_collab = self.get_collaborative_view()
        
        # Concat (Implicit Union)
        u_all = torch.cat([u_collab, u_sem], dim=1)
        i_all = torch.cat([i_collab, i_sem], dim=1)
        
        # Final Normalization
        return F.normalize(u_all, dim=-1), F.normalize(i_all, dim=-1)

    def forward_loss_hard(self, uids, pos_iids, args):
        """
        End-to-End Training with Dynamic Hard Negative Mining
        """
        # 1. Compute current fused embeddings
        all_u, all_i = self.forward_embedding()
        
        batch_u = all_u[uids]       # [B, D]
        batch_pos = all_i[pos_iids] # [B, D]
        batch_size = uids.size(0)

        # 2. Hard Negative Mining
        # Only mine negatives if we are not in simple random mode
        with torch.no_grad():
            num_cand = args.neg_per_pos * args.hard_neg_factor
            cand_ids = torch.randint(0, self.n_items, (batch_size, num_cand), device=uids.device)
            
            # Mask positives
            pos_mask = (cand_ids == pos_iids.unsqueeze(1))
            cand_ids[pos_mask] = (cand_ids[pos_mask] + 1) % self.n_items
            
            # Compute scores for candidates
            cand_embs = all_i[cand_ids] # [B, Cand, D]
            cand_scores = torch.bmm(cand_embs, batch_u.unsqueeze(2)).squeeze(2)
            
            # Select Top-K hardest negatives
            _, top_indices = torch.topk(cand_scores, k=args.neg_per_pos, dim=1)
            hard_neg_ids = torch.gather(cand_ids, 1, top_indices) # [B, NEG]

        # 3. InfoNCE Loss with Hard Negatives
        flat_negs = hard_neg_ids.view(-1)
        batch_neg = all_i[flat_negs].view(batch_size, args.neg_per_pos, -1)

        # Positive Logits
        pos_logit = torch.sum(batch_u * batch_pos, dim=-1, keepdim=True) / self.temperature
        
        # Negative Logits
        neg_logit = torch.bmm(batch_neg, batch_u.unsqueeze(2)).squeeze(2) / self.temperature
        
        logits = torch.cat([pos_logit, neg_logit], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=uids.device)
        
        loss = F.cross_entropy(logits, labels)
        return loss

# ================= Evaluation Helper =================
def build_train_pos_index(train_set, n_users):
    lengths = np.zeros(n_users, dtype=np.int64); total = 0
    for u, items in train_set.items(): l = len(items); lengths[u] = l; total += l
    offsets = np.zeros(n_users + 1, dtype=np.int64); np.cumsum(lengths, out=offsets[1:])
    items_concat = np.empty(total, dtype=np.int64); cursor = 0
    for u in range(n_users):
        items = train_set.get(u, [])
        if items: items_concat[cursor:cursor+len(items)] = np.array(items, dtype=np.int64); cursor += len(items)
    return offsets, items_concat

def mask_train_positives(scores, batch_uids, offsets, items_concat, chunk_start, chunk_end):
    device = scores.device; bu = torch.as_tensor(batch_uids, dtype=torch.long, device="cpu")
    row_ids, col_ids = [], []
    for r, u in enumerate(bu.tolist()):
        if u >= len(offsets) - 1: continue
        s, e = offsets[u], offsets[u+1]
        if e <= s: continue
        items = items_concat[s:e]
        m = items[(items >= chunk_start) & (items < chunk_end)]
        if m.size == 0: continue
        row_ids.append(np.full(m.shape[0], r, dtype=np.int64)); col_ids.append(m - chunk_start)
    if row_ids:
        row_ids = torch.from_numpy(np.concatenate(row_ids)).to(device, dtype=torch.long)
        col_ids = torch.from_numpy(np.concatenate(col_ids)).to(device, dtype=torch.long)
        scores[row_ids, col_ids] = -float("inf")

@torch.no_grad()
def evaluate(model, train_set, target_set, offsets, items_concat, n_items, device, Ks=[10, 20]):
    model.eval()
    uids = list(target_set.keys())
    if not uids: return {}
    
    all_u, all_i = model.forward_embedding() 
    all_i_t = all_i.t().contiguous()
    
    recall_records, ndcg_records = {k: [] for k in Ks}, {k: [] for k in Ks}
    
    EVAL_BATCH = 1024
    ITEM_CHUNK = 20000
    
    for us in range(0, len(uids), EVAL_BATCH):
        batch_uids = uids[us:us + EVAL_BATCH]
        bu = torch.tensor(batch_uids, dtype=torch.long, device=device)
        uvec = all_u[bu]
        top_scores = torch.full((uvec.size(0), max(Ks)), -float("inf"), device=device)
        top_items = torch.full((uvec.size(0), max(Ks)), -1, device=device, dtype=torch.long)

        for cs in range(0, n_items, ITEM_CHUNK):
            ce = min(cs + ITEM_CHUNK, n_items)
            ivec_chunk = all_i_t[:, cs:ce]
            scores = uvec @ ivec_chunk
            mask_train_positives(scores, batch_uids, offsets, items_concat, cs, ce)
            
            chunk_k = min(max(Ks), scores.size(1))
            c_scores, c_idx = torch.topk(scores, k=chunk_k, dim=1)
            c_items = c_idx + cs
            merged_scores = torch.cat([top_scores, c_scores], dim=1)
            merged_items = torch.cat([top_items, c_items], dim=1)
            new_scores, new_pos = torch.topk(merged_scores, k=max(Ks), dim=1)
            new_items = torch.gather(merged_items, 1, new_pos)
            top_scores, top_items = new_scores, new_items

        top_items_cpu = top_items.cpu().numpy()
        for r, u in enumerate(batch_uids):
            gt = set(target_set.get(u, []))
            if not gt: continue
            rec_items = top_items_cpu[r, :].tolist()
            
            for k in Ks:
                hits = len(set(rec_items[:k]) & gt)
                recall_records[k].append(hits / len(gt))
                
                dcg = sum([1.0 / np.log2(i + 2) for i, item in enumerate(rec_items[:k]) if item in gt])
                idcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(gt), k))])
                ndcg_records[k].append(dcg / idcg if idcg > 0 else 0.0)

    res = {k: float(np.mean(v)) if v else 0.0 for k, v in recall_records.items()}
    res.update({f"nDCG@{k}": float(np.mean(v)) if v else 0.0 for k, v in ndcg_records.items()})
    return res

# ================= Main =================
def main():
    args = parse_args()
    
    # Init Logging
    log_file = os.path.join(args.log_root, f"{args.dataset}_fusion.log")
    logger = Logger(log_file)
    logger.log(f"Args: {vars(args)}")
    
    # Seed
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Data
    raw_embs, user_base, adj_tensor, train, valid, test, n_users, n_items = load_data(args, device)
    offsets, items_concat = build_train_pos_index(train, n_users)
    
    # Initialize Model
    model = DualBranchModel(args, n_users, n_items, raw_embs, user_base, adj_tensor.to(device)).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    
    dataset = UPDataset(train)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, 
                        num_workers=args.num_workers, pin_memory=True, drop_last=True)
    
    best_val_score = 0.0
    patience_cnt = 0
    save_path = os.path.join(args.output_root, f"{args.dataset}_fusion_model.pth")
    
    logger.log("\n>>> Start Training...")
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0
        
        with tqdm(loader, desc=f"Ep {epoch}", ncols=100) as pbar:
            for u, pos in pbar:
                u = u.to(device, non_blocking=True)
                pos = pos.to(device, non_blocking=True)
                
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=True):
                    loss = model.forward_loss_hard(u, pos, args)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                total_loss += loss.item()
                steps += 1
                pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
        
        avg_loss = total_loss / steps
        logger.log(f"Epoch {epoch}: Loss={avg_loss:.4f}")
        
        if epoch % 5 == 0:
            metrics = evaluate(model, train, valid, offsets, items_concat, n_items, device)
            score = metrics.get(20, 0) # Recall@20 as selection metric
            logger.log(f"  Valid: {metrics}")
            
            if score > best_val_score:
                best_val_score = score
                patience_cnt = 0
                torch.save(model.state_dict(), save_path)
                logger.log("  [New Best Saved]")
            else:
                patience_cnt += 1
                if patience_cnt >= args.patience:
                    logger.log(">>> Early Stopping")
                    break

    # Final Test
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))
        
    logger.log(">>> Testing on Test Set...")
    test_metrics = evaluate(model, train, test, offsets, items_concat, n_items, device)
    logger.log(f"Test Results: {test_metrics}")

    logger.log("\n>>> Exporting Fusion Embeddings for Analysis...")
    model.eval()
    with torch.no_grad():
        u_all, i_all = model.forward_embedding()
        u_all = u_all.cpu().numpy().astype(np.float32)
        i_all = i_all.cpu().numpy().astype(np.float32)

    save_dir = os.path.join(args.data_root, args.dataset)
    os.makedirs(save_dir, exist_ok=True)
    save_path_u = os.path.join(save_dir, f"{args.dataset}_user_embeddings-fusion.npy")
    save_path_i = os.path.join(save_dir, f"{args.dataset}_item_embeddings-fusion.npy")

    # 保存文件
    np.save(save_path_u, u_all)
    np.save(save_path_i, i_all)

    logger.log(f"Saved User Embeddings to: {save_path_u}")
    logger.log(f"Saved Item Embeddings to: {save_path_i}")

if __name__ == "__main__":
    main()