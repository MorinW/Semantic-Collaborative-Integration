import os
import argparse
import numpy as np
from tqdm import tqdm
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ================= Argument Parsing =================
def parse_args():
    parser = argparse.ArgumentParser(description="Train AlphaRec Baseline (frozen semantics + projection + GCN).")
    
    # Dataset & Paths
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., movies, books, games)')
    parser.add_argument('--data_root', type=str, default='./dataset', help='Data root directory')
    parser.add_argument('--output_root', type=str, default='./saved', help='Model save directory')
    
    # Model Hyperparameters
    parser.add_argument('--hidden_size', type=int, default=64, help='Dimension of projected embeddings')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of LightGCN layers')
    parser.add_argument('--mlp_multiplier', type=float, default=0.5, help='Hidden size multiplier for MLP projection')
    parser.add_argument('--tau', type=float, default=0.1, help='InfoNCE temperature')
    
    # Training Hyperparameters
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2048, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--neg_per_pos', type=int, default=64, help='Negative samples per positive (64 is usually sufficient for speed)')
    parser.add_argument('--train_use_gcn', action='store_true', help='If Set, apply GCN propagation during training (Slower but closer to original paper). Default: False (MLP only during train, GCN during eval)')
    
    # System
    parser.add_argument('--seed', type=int, default=2020, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    
    return parser.parse_args()

# ================= Data Loading =================
def load_data(args):
    print(">>> Loading Data...")
    dataset_path = os.path.join(args.data_root, args.dataset)
    item_emb_path = os.path.join(dataset_path, f"{args.dataset}_item_embeddings.npy")
    
    if not os.path.exists(item_emb_path):
        raise FileNotFoundError(f"Item embeddings not found: {item_emb_path}")

    # 1. Load Item Embeddings
    print("1. Loading raw item embeddings...")
    raw_item_embs = np.load(item_emb_path).astype(np.float32)  # [n_items, d]
    norms = np.linalg.norm(raw_item_embs, axis=1, keepdims=True)
    item_embeddings = raw_item_embs / (norms + 1e-10)

    # 2. Load Interactions
    def read_inter(suffix):
        data = {}
        path = os.path.join(dataset_path, f"{args.dataset}.{suffix}.inter")
        if not os.path.exists(path):
            print(f"Warning: {path} not found")
            return data
        with open(path, "r", encoding="utf-8") as f:
            next(f) # skip header
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2: continue
                try:
                    u, i = int(parts[0]), int(parts[1])
                    data.setdefault(u, []).append(i)
                except ValueError: continue
        return data

    print("2. Reading interaction data...")
    train_set = read_inter("train")
    valid_set = read_inter("valid")
    test_set  = read_inter("test")

    all_users = set(train_set.keys()) | set(valid_set.keys()) | set(test_set.keys())
    n_users = (max(all_users) + 1) if all_users else 0
    n_items = len(item_embeddings)
    dim = item_embeddings.shape[1]
    print(f"   Users: {n_users}, Items: {n_items}, Dim: {dim}")

    # 3. Compute User Semantics (Mean Pooling on Train)
    print("3. Computing initial user semantic vectors (Mean Pooling on train)...")
    user_sem = np.zeros((n_users, dim), dtype=np.float32)
    for u, items in tqdm(train_set.items(), desc="Init Users"):
        valid_items = [i for i in items if 0 <= i < n_items]
        if not valid_items: continue
        vecs = item_embeddings[valid_items]
        mean_vec = np.mean(vecs, axis=0)
        norm = np.linalg.norm(mean_vec)
        if norm > 1e-10:
            user_sem[u] = mean_vec / norm

    return item_embeddings, user_sem, train_set, valid_set, test_set

# ================= Dataset =================
class UPDataset(Dataset):
    def __init__(self, train_set):
        self.samples = []
        for u, items in train_set.items():
            for i in items:
                self.samples.append((u, i))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# ================= Graph Construction =================
def build_normalized_graph(n_users, n_items, train_set, device):
    """
    Builds the symmetric normalized adjacency matrix for LightGCN.
    A_hat = D^{-1/2} A D^{-1/2}
    """
    print("4. Building LightGCN-style normalized graph...")
    num_nodes = n_users + n_items
    rows = []
    cols = []

    for u, items in train_set.items():
        for i in items:
            rows.append(u)
            cols.append(n_users + i)
            rows.append(n_users + i)
            cols.append(u)

    if not rows:
        raise ValueError("Training set is empty.")

    rows = np.array(rows, dtype=np.int64)
    cols = np.array(cols, dtype=np.int64)
    
    # Calculate degrees
    deg = np.zeros(num_nodes, dtype=np.float32)
    np.add.at(deg, rows, 1.0) # Faster than looping

    deg_inv_sqrt = 1.0 / np.sqrt(deg + 1e-8)
    norm_data = deg_inv_sqrt[rows] * deg_inv_sqrt[cols]

    indices = torch.from_numpy(np.vstack([rows, cols]))
    values = torch.from_numpy(norm_data)
    
    adj = torch.sparse_coo_tensor(indices, values, torch.Size([num_nodes, num_nodes]))
    adj = adj.coalesce().to(device=device)
    print(f"   Graph built. nnz={adj._nnz()}")
    return adj

# ================= Model: AlphaRec =================
class AlphaRecLocal(nn.Module):
    def __init__(self, user_sem_np, item_sem_np, graph,
                 hidden_size=64, n_layers=2, tau=0.1,
                 mlp_multiplier=0.5, train_use_gcn=False, device="cpu"):
        super().__init__()
        self.device = device
        self.tau = tau
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.train_use_gcn = train_use_gcn

        # Frozen initial semantic embeddings
        self.init_user_cf_embeds = torch.tensor(user_sem_np, dtype=torch.float32, device=device)
        self.init_item_cf_embeds = torch.tensor(item_sem_np, dtype=torch.float32, device=device)

        self.n_users = self.init_user_cf_embeds.shape[0]
        self.n_items = self.init_item_cf_embeds.shape[0]
        self.in_dim  = self.init_user_cf_embeds.shape[1]

        self.Graph = graph

        # MLP Projector (Semantics -> Latent CF Space)
        # Structure: In -> In*Multiplier -> Hidden
        hidden_mlp = int(mlp_multiplier * self.in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.in_dim, hidden_mlp),
            nn.LeakyReLU(),
            nn.Linear(hidden_mlp, hidden_size)
        )

    def _mlp_embeddings(self):
        """Apply MLP projection only."""
        users_cf_emb = self.mlp(self.init_user_cf_embeds)
        items_cf_emb = self.mlp(self.init_item_cf_embeds)
        return users_cf_emb, items_cf_emb

    def _lightgcn_propagate(self, users_cf_emb, items_cf_emb):
        """Apply LightGCN propagation on top of projected embeddings."""
        all_emb = torch.cat([users_cf_emb, items_cf_emb], dim=0)
        embs = [all_emb]
        
        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(self.Graph, all_emb)
            embs.append(all_emb)

        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.n_users, self.n_items], dim=0)
        return users, items

    def compute_train_embs(self):
        """
        Embeddings for training step.
        If train_use_gcn is False, we skip GCN during training for speed (strong optimization).
        """
        users_cf_emb, items_cf_emb = self._mlp_embeddings()
        if self.train_use_gcn:
            return self._lightgcn_propagate(users_cf_emb, items_cf_emb)
        else:
            return users_cf_emb, items_cf_emb

    def compute_infer_embs(self):
        """
        Embeddings for inference/eval. Always use GCN propagation.
        """
        users_cf_emb, items_cf_emb = self._mlp_embeddings()
        return self._lightgcn_propagate(users_cf_emb, items_cf_emb)

    def forward(self, users, pos_items, neg_items):
        """
        InfoNCE Loss
        """
        all_users, all_items = self.compute_train_embs()

        users_emb = F.normalize(all_users[users], dim=-1)
        pos_emb   = F.normalize(all_items[pos_items], dim=-1)
        neg_emb   = F.normalize(all_items[neg_items], dim=-1) # [B, NEG, d]

        # Dot products
        pos_scores = torch.sum(users_emb * pos_emb, dim=-1) # [B]
        # [B, 1, d] x [B, d, NEG] -> [B, 1, NEG]
        neg_scores = torch.bmm(users_emb.unsqueeze(1), neg_emb.transpose(1, 2)).squeeze(1) # [B, NEG]

        # InfoNCE
        logits = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1) # [B, 1+NEG]
        logits /= self.tau
        
        labels = torch.zeros(users.size(0), dtype=torch.long, device=users.device)
        return F.cross_entropy(logits, labels)

    @torch.no_grad()
    def export_all_embeddings(self, device):
        self.eval()
        all_users, all_items = self.compute_infer_embs()
        return F.normalize(all_users, dim=-1).to(device), F.normalize(all_items, dim=-1).to(device)

# ================= Evaluation Utils =================
def build_train_pos_index(train_set, n_users):
    # Same helper as in other scripts
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
    
    all_u, all_i = model.export_all_embeddings(device)
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
    
    if not os.path.exists(args.output_root):
        os.makedirs(args.output_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load Data
    item_sem, user_sem, train_set, valid_set, test_set = load_data(args)
    n_items = item_sem.shape[0]
    n_users = user_sem.shape[0]

    # 2. Build Graph & Index
    graph = build_normalized_graph(n_users, n_items, train_set, device=device)
    offsets, items_concat = build_train_pos_index(train_set, n_users)

    # 3. Model
    model = AlphaRecLocal(
        user_sem_np=user_sem,
        item_sem_np=item_sem,
        graph=graph,
        hidden_size=args.hidden_size,
        n_layers=args.n_layers,
        tau=args.tau,
        mlp_multiplier=args.mlp_multiplier,
        train_use_gcn=args.train_use_gcn,
        device=device,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 4. Training Loop
    dataset = UPDataset(train_set)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    best_val_score = 0.0
    patience_cnt = 0
    save_path = os.path.join(args.output_root, f"{args.dataset}_alpharec.pth")

    print("\n>>> Start Training AlphaRec...")
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        
        # Training loop
        for u, pos in tqdm(loader, desc=f"Ep {epoch}", ncols=100):
            u = u.to(device, non_blocking=True)
            pos = pos.to(device, non_blocking=True)

            # Negative Sampling
            negs = torch.randint(1, n_items, (u.size(0), args.neg_per_pos), device=device)
            # Collision check
            eq = (negs == pos.unsqueeze(1))
            if eq.any():
                negs = torch.where(eq, (negs + 1) % n_items, negs)
                negs = torch.where(negs == 0, torch.ones_like(negs), negs)

            optimizer.zero_grad(set_to_none=True)
            loss = model(u, pos, negs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch} Loss: {total_loss/len(loader):.4f}")

        # Validation (Every 5 epochs)
        if epoch % 5 == 0 or epoch == 1:
            metrics = evaluate(model, train_set, valid_set, offsets, items_concat, n_items, device)
            score = metrics.get(20, 0)
            print(f"  Valid: {metrics}")

            if score > best_val_score:
                best_val_score = score
                patience_cnt = 0
                torch.save(model.state_dict(), save_path)
                print("  [New Best Saved]")
            else:
                patience_cnt += 1
                if patience_cnt >= args.patience:
                    print(">>> Early Stopping")
                    break

    # Final Test
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))
    
    print("\n>>> Testing...")
    test_metrics = evaluate(model, train_set, test_set, offsets, items_concat, n_items, device)
    print(f"Test Results: {test_metrics}")

    print("\n>>> Exporting Embeddings for Analysis...")
    model.eval()
    with torch.no_grad():
        u_all, i_all = model.export_all_embeddings(device)
        u_all = u_all.cpu().numpy().astype(np.float32)
        i_all = i_all.cpu().numpy().astype(np.float32)

    save_path_u = os.path.join(args.output_root, f"{args.dataset}_user_embeddings-AlphaRec.npy")
    save_path_i = os.path.join(args.output_root, f"{args.dataset}_item_embeddings-AlphaRec.npy")
    
    np.save(save_path_u, u_all)
    np.save(save_path_i, i_all)

    print(f"Saved User Embeddings to: {save_path_u}")
    print(f"Saved Item Embeddings to: {save_path_i}")
if __name__ == "__main__":
    main()