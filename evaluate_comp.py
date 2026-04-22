import argparse
import numpy as np
import torch
import os
import math
from tqdm import tqdm
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze Complementarity between two models (View A vs View B).")
    
    # Dataset & Paths
    parser.add_argument('--dataset', type=str, default="movies", help='Dataset name (e.g., movies, books)')
    parser.add_argument('--data_root', type=str, default='./dataset', help='Data root directory')
    
    # Model A Configuration (e.g., Baseline/Collaborative)
    parser.add_argument('--suffix_a', type=str, default='pco64', 
                        help='Suffix for Model A embeddings (e.g., lightgcn). Reads: {dataset}_user_embeddings-{suffix_a}.npy')
    
    # Model B Configuration (e.g., Fusion/Semantic)
    parser.add_argument('--suffix_b', type=str, default='proj64', 
                        help='Suffix for Model B embeddings (e.g., fusion). Reads: {dataset}_user_embeddings-{suffix_b}.npy')
    
    # Analysis Config
    parser.add_argument('--topk', type=int, default=20, help='Top-K cutoff for analysis')
    parser.add_argument('--batch_size', type=int, default=4096, help='Inference batch size')
    
    return parser.parse_args()

# ================= 1. Data Loading =================

def load_inter_file(file_path):
    """Load interaction file into {user: {item1, item2...}}"""
    data = defaultdict(set)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            next(f) # Skip header
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 2: continue
                try:
                    u, i = int(parts[0]), int(parts[1])
                    data[u].add(i)
                except ValueError: continue
    except FileNotFoundError:
        print(f"Error: Interaction file not found: {file_path}")
        exit(1)
    return data

def load_embeddings(dataset_dir, dataset_name, suffix):
    """
    Load embeddings from: ./dataset/{DATASET}/{DATASET}_{type}_embeddings-{suffix}.npy
    """
    # 构建符合你要求的文件路径格式
    u_path = os.path.join(dataset_dir, f"{dataset_name}_user_embeddings-{suffix}.npy")
    i_path = os.path.join(dataset_dir, f"{dataset_name}_item_embeddings-{suffix}.npy")
    
    print(f"   Loading: {os.path.basename(u_path)} ...")
    try:
        u_emb = np.load(u_path).astype(np.float32)
        i_emb = np.load(i_path).astype(np.float32)
        return u_emb, i_emb
    except FileNotFoundError:
        print(f"❌ Error: Embeddings not found for suffix '{suffix}'. Check path: {u_path}")
        exit(1)

def get_stats(train_data):
    """Calculate Item Popularity (Log) and User Activity (Log) from Training Data."""
    item_counts = defaultdict(int)
    user_counts = defaultdict(int)
    
    for u, items in train_data.items():
        user_counts[u] = len(items)
        for i in items:
            item_counts[i] += 1
            
    # Convert to Log Scale: ln(count + 1)
    # Log scale is better for analyzing long-tail distributions
    item_pop = {i: np.log1p(c) for i, c in item_counts.items()}
    user_act = {u: np.log1p(c) for u, c in user_counts.items()}
    
    return item_pop, user_act

# ================= 2. Inference Engine =================

def generate_topk(u_emb, i_emb, test_users, mask_history, topk, batch_size, device):
    """
    Generate Top-K recommendations on-the-fly.
    Returns: {user_id: [item1, item2, ...]}
    """
    num_items = i_emb.shape[0]
    u_tensor = torch.from_numpy(u_emb).to(device)
    i_tensor = torch.from_numpy(i_emb).to(device)
    
    preds = {}
    
    with torch.no_grad():
        for i in tqdm(range(0, len(test_users), batch_size), desc="Inference", leave=False):
            batch_uids = test_users[i : i+batch_size]
            batch_u_emb = u_tensor[batch_uids]
            
            # Score
            scores = torch.matmul(batch_u_emb, i_tensor.t())
            
            # Mask Training History (CPU side for safety/simplicity)
            scores_np = scores.cpu().numpy()
            for idx, uid in enumerate(batch_uids):
                hist = mask_history.get(uid, set())
                # Only mask items that exist in current item set
                valid_hist = [x for x in hist if x < num_items]
                if valid_hist:
                    scores_np[idx, valid_hist] = -np.inf
            
            # Get Top-K
            for idx, uid in enumerate(batch_uids):
                # partition is faster than full sort
                unsorted = np.argpartition(scores_np[idx], -topk)[-topk:]
                # Sort the top k to get correct order (though for set overlap order doesn't matter, 
                # for NDCG/Recall it does. Here we just strictly need the set, but let's be precise)
                top_scores = scores_np[idx, unsorted]
                sorted_idx = np.argsort(-top_scores)
                final_list = unsorted[sorted_idx].tolist()
                preds[uid] = final_list
                
    return preds

# ================= 3. Complementarity Analysis =================

def safe_mean(l):
    return np.mean(l) if l else 0.0

def analyze_complementarity(preds_a, preds_b, ground_truth, item_pop, user_act, topk, name_a, name_b):
    print(f"\n📊 Analyzing Complementarity: {name_a} vs {name_b} (Top-{topk})")
    
    # Containers for stats
    jaccard_list = []
    
    # Hit Classification Counts
    users_both_hit = 0
    users_a_only = 0
    users_b_only = 0
    users_no_hit = 0
    
    # Activity Analysis (Log Activity of users in each group)
    act_both = []
    act_a_only = []
    act_b_only = []
    
    # Item Popularity Analysis (Log Popularity of HIT items)
    # We collect the popularity of items that were correctly recommended (Hits)
    pop_both_hits = [] 
    pop_a_only_hits = []
    pop_b_only_hits = []
    
    # Hit Counts
    hits_common_cnt = 0
    hits_a_unique_cnt = 0
    hits_b_unique_cnt = 0
    
    common_users = sorted(list(set(preds_a.keys()) & set(preds_b.keys())))
    
    for uid in common_users:
        truth = ground_truth.get(uid, set())
        if not truth: continue
        
        list_a = preds_a[uid]
        list_b = preds_b[uid]
        
        set_a = set(list_a)
        set_b = set(list_b)
        
        # 1. List Overlap (Jaccard)
        union = set_a | set_b
        inter = set_a & set_b
        jaccard = len(inter) / len(union) if union else 0
        jaccard_list.append(jaccard)
        
        # 2. Identify Hits
        hits_a = set_a & truth
        hits_b = set_b & truth
        
        has_hit_a = len(hits_a) > 0
        has_hit_b = len(hits_b) > 0
        
        u_act = user_act.get(uid, 0.0)
        
        # User Classification
        if has_hit_a and has_hit_b:
            users_both_hit += 1
            act_both.append(u_act)
        elif has_hit_a:
            users_a_only += 1
            act_a_only.append(u_act)
        elif has_hit_b:
            users_b_only += 1
            act_b_only.append(u_act)
        else:
            users_no_hit += 1
            
        # Item Hit Analysis
        # Which items were found by both vs only one?
        common_hits = hits_a & hits_b
        unique_a = hits_a - hits_b
        unique_b = hits_b - hits_a
        
        hits_common_cnt += len(common_hits)
        hits_a_unique_cnt += len(unique_a)
        hits_b_unique_cnt += len(unique_b)
        
        for iid in common_hits: pop_both_hits.append(item_pop.get(iid, 0))
        for iid in unique_a: pop_a_only_hits.append(item_pop.get(iid, 0))
        for iid in unique_b: pop_b_only_hits.append(item_pop.get(iid, 0))

    # ================= Reporting =================
    total_users = len(common_users)
    total_hits = hits_common_cnt + hits_a_unique_cnt + hits_b_unique_cnt
    
    print("\n" + "="*60)
    print(f"1. Recommendation Overlap (List Similarity)")
    print(f"   • Mean Jaccard Index: {safe_mean(jaccard_list):.4f}")
    print(f"   • Interpretation: Lower means lists are more different.")

    print("\n" + "="*60)
    print(f"2. User-Level Complementarity (Who gets 'saved'?)")
    print(f"   Total Users: {total_users}")
    print(f"   {'Group':<15} | {'Count':<8} | {'Ratio':<8} | {'Avg User Activity (Log)':<25}")
    print("-" * 65)
    print(f"   {'Both Hit':<15} | {users_both_hit:<8} | {users_both_hit/total_users:<8.1%} | {safe_mean(act_both):<25.2f}")
    print(f"   {name_a + ' Only':<15} | {users_a_only:<8} | {users_a_only/total_users:<8.1%} | {safe_mean(act_a_only):<25.2f}")
    print(f"   {name_b + ' Only':<15} | {users_b_only:<8} | {users_b_only/total_users:<8.1%} | {safe_mean(act_b_only):<25.2f}")
    print(f"   {'No Hit':<15} | {users_no_hit:<8} | {users_no_hit/total_users:<8.1%} | -")

    print("\n" + "="*60)
    print(f"3. Item-Level Complementarity (Unique Hits Characteristics)")
    print(f"   Total Correct Recommendations (Hits): {total_hits}")
    print(f"   {'Hit Type':<15} | {'Count':<8} | {'Avg Item Popularity (Log)':<25}")
    print("-" * 65)
    print(f"   {'Common Hits':<15} | {hits_common_cnt:<8} | {safe_mean(pop_both_hits):<25.2f}")
    print(f"   {name_a + ' Unique':<15} | {hits_a_unique_cnt:<8} | {safe_mean(pop_a_only_hits):<25.2f}")
    print(f"   {name_b + ' Unique':<15} | {hits_b_unique_cnt:<8} | {safe_mean(pop_b_only_hits):<25.2f}")

    print("\n" + "="*60)
    print("💡 Key Insights")
    
    # Insight Logic
    unique_hit_ratio = (hits_a_unique_cnt + hits_b_unique_cnt) / total_hits if total_hits else 0
    print(f"   • Complementary Hit Ratio: {unique_hit_ratio:.1%} of hits are found by only one model.")
    
    pop_a = safe_mean(pop_a_only_hits)
    pop_b = safe_mean(pop_b_only_hits)
    
    if pop_b < pop_a and abs(pop_a - pop_b) > 0.1:
        print(f"   • {name_b} tends to hit significantly COLDER items (Log Pop: {pop_b:.2f}) than {name_a} ({pop_a:.2f}).")
        print(f"     -> This confirms {name_b} explores the long-tail better.")
    elif pop_a < pop_b and abs(pop_a - pop_b) > 0.1:
        print(f"   • {name_a} tends to hit colder items ({pop_a:.2f}) than {name_b} ({pop_b:.2f}).")
    
    act_a = safe_mean(act_a_only)
    act_b = safe_mean(act_b_only)
    if act_b < act_a and abs(act_a - act_b) > 0.1:
        print(f"   • Users rescued by {name_b} are less active (Log Act: {act_b:.2f}) than those by {name_a}.")
        print(f"     -> {name_b} is more robust to user cold-start.")

# ================= 4. Main =================

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting Complementarity Analysis on {args.dataset}")
    print(f"   Model A (Baseline): ...{args.suffix_a}.npy")
    print(f"   Model B (Target):   ...{args.suffix_b}.npy")
    
    # 1. Define Paths
    data_dir = os.path.join(args.data_root, args.dataset)
    train_path = os.path.join(data_dir, f"{args.dataset}.train.inter")
    test_path = os.path.join(data_dir, f"{args.dataset}.test.inter")
    
    # 2. Load Data
    print("\n>>> Loading Interactions...")
    train_data = load_inter_file(train_path)
    test_data = load_inter_file(test_path)
    
    # 3. Get Stats (Popularity/Activity)
    print(">>> Calculating Statistics (Log Scale)...")
    item_pop, user_act = get_stats(train_data)
    
    # 4. Load Embeddings
    print("\n>>> Loading Embeddings...")
    ua, ia = load_embeddings(data_dir, args.dataset, args.suffix_a)
    ub, ib = load_embeddings(data_dir, args.dataset, args.suffix_b)
    
    # 5. Masking Set (Train history)
    mask_history = train_data # Reuse the dict
    test_users = sorted(list(test_data.keys()))
    # Filter users out of bound
    max_uid = min(ua.shape[0], ub.shape[0])
    test_users = [u for u in test_users if u < max_uid]
    
    # 6. Inference
    print(f"\n>>> Running Inference (Top-{args.topk})...")
    print(f"   Generating lists for Model A ({args.suffix_a})...")
    preds_a = generate_topk(ua, ia, test_users, mask_history, args.topk, args.batch_size, device)
    
    print(f"   Generating lists for Model B ({args.suffix_b})...")
    preds_b = generate_topk(ub, ib, test_users, mask_history, args.topk, args.batch_size, device)
    
    # 7. Analyze
    analyze_complementarity(
        preds_a, preds_b, test_data, 
        item_pop, user_act, args.topk, 
        name_a=args.suffix_a.upper(), 
        name_b=args.suffix_b.upper()
    )

if __name__ == "__main__":
    main()