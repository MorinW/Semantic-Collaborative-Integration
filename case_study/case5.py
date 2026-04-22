import numpy as np
import json
import os
import sys
import random
from sklearn.preprocessing import normalize
from collections import defaultdict, Counter

# ================= 1. Experiment Configuration =================
CURRENT_DATASET = "movies" 
BASE_DIR = "./dataset"
SAVED_DIR = "./saved"
TOP_K = 10  # We check if Top 10 hits

def get_paths(dataset):
    base_data = f"{BASE_DIR}/{dataset}"
    
    def find_file(filename):
        p1 = f"{base_data}/{filename}"
        if os.path.exists(p1): return p1
        p2 = f"{SAVED_DIR}/{filename}"
        if os.path.exists(p2): return p2
        return None 

    paths = {
        "train_inter": f"{base_data}/{dataset}.train.inter",
        "test_inter":  f"{base_data}/{dataset}.test.inter", 
        "map_file":    f"{base_data}/item_map.json",
        "emb_sem":     find_file(f"{dataset}_item_embeddings-proj64.npy"), 
        "emb_pco":     find_file(f"{dataset}_item_embeddings-pco64.npy"), 
        "meta_file":   f"{base_data}/metadata.jsonl" 
    }
    return paths

PATHS = get_paths(CURRENT_DATASET)

# ================= 2. Core Logic =================

def load_data():
    print(f"🔬 [Init] Hunting for COLD-START DIRECT HITS | Dataset: {CURRENT_DATASET}")
    
    # 1. Load Vectors
    p_sem = PATHS["emb_sem"]
    p_pco = PATHS["emb_pco"]
    if not (p_sem and os.path.exists(p_sem) and p_pco and os.path.exists(p_pco)):
        print("❌ Critical: Embedding files missing."); sys.exit(1)
        
    sem_embs = np.load(p_sem).astype(np.float32)
    cf_embs = np.load(p_pco).astype(np.float32)
    
    num_items = min(cf_embs.shape[0], sem_embs.shape[0])
    cf_embs = normalize(cf_embs[:num_items], norm='l2')
    sem_embs = normalize(sem_embs[:num_items], norm='l2')

    # 2. Load ID Mapping
    id2asin = {}
    if os.path.exists(PATHS["map_file"]):
        try:
            with open(PATHS["map_file"], 'r', encoding='utf-8') as f:
                item_map = json.load(f)
                for k, v in item_map.items():
                    id2asin[int(v)] = str(k)
        except: pass

    # 3. Load Metadata
    asin2info = {}
    if os.path.exists(PATHS["meta_file"]):
        with open(PATHS["meta_file"], 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    asin = obj.get('item_id')
                    if not asin: continue
                    title = obj.get('title', asin)
                    cate = "Unknown"
                    details = obj.get('details', {})
                    if details and 'Genre' in details:
                        cate = details['Genre']
                    else:
                        cats = obj.get('categories', [])
                        valid_cats = [c for c in cats if c not in ["Movies & TV", "Genre for Featured Categories"]]
                        if valid_cats: cate = valid_cats[-1]
                    
                    if len(title) > 45: title = title[:42] + "..."
                    asin2info[asin] = {"title": title, "cate": cate}
                except: continue
    
    id2info = {}
    for i in range(num_items):
        asin = id2asin.get(i)
        if asin and asin in asin2info:
            id2info[i] = asin2info[asin]
        else:
            id2info[i] = {"title": f"Item {i}", "cate": "Unknown"}

    # 4. Load Interactions & Counts
    print("   >>> Loading Interactions...")
    train_inter = defaultdict(list)
    test_inter = defaultdict(set)
    item_pop = Counter()
    
    with open(PATHS["train_inter"], 'r') as f:
        next(f)
        for line in f:
            try:
                parts = line.strip().split('\t')
                u, i = int(parts[0]), int(parts[1])
                if i < num_items: 
                    train_inter[u].append(i)
                    item_pop[i] += 1
            except: continue
            
    if os.path.exists(PATHS["test_inter"]):
        with open(PATHS["test_inter"], 'r') as f:
            next(f)
            for line in f:
                try:
                    parts = line.strip().split('\t')
                    u, i = int(parts[0]), int(parts[1])
                    if i < num_items: test_inter[u].add(i)
                except: continue
    
    return cf_embs, sem_embs, train_inter, test_inter, id2info, item_pop

def get_recommendations(user_history, item_embs, topk=TOP_K):
    if not user_history: return []
    user_vec = np.mean(item_embs[user_history], axis=0).reshape(1, -1)
    scores = np.dot(item_embs, user_vec.T).flatten()
    scores[user_history] = -np.inf
    return np.argsort(scores)[-topk:][::-1]

def format_item_str(iid, id2info, item_pop, is_hit=False):
    info = id2info.get(iid, {"title": f"Item {iid}", "cate": "Unknown"})
    count = item_pop.get(iid, 0)
    
    # Hit highlight
    prefix = "🎯HIT! " if is_hit else ""
    return f"{prefix}[Cnt:{count}] {info['title']}", info['cate']

def find_cold_start_hits(cf_embs, sem_embs, train, test, id2info, item_pop):
    print("\n🔎 Searching for 'COLD-START DIRECT HITS'...")
    print("   Strategy: Check users with few interactions first (Sorted by Length)")
    
    found_count = 0
    users = list(train.keys())
    
    # ================= Modification: Sort by history length =================
    # Prioritize users with the fewest interactions (Cold Start First)
    # We only care about users with 3 to 10 interactions (too few is random, too many isn't cold start)
    filtered_users = [u for u in users if 3 <= len(train[u]) <= 10]
    
    # Sort: From 3, sequentially backwards
    filtered_users.sort(key=lambda u: len(train[u]))
    
    print(f"   Target Users: {len(filtered_users)} (History len 3-10)")
    
    for u in filtered_users:
        history = train[u]
        gt_set = test[u]
        if not gt_set: continue
        
        # 2. Get Recommendations (Top 10)
        rec_cf = get_recommendations(history, cf_embs, topk=TOP_K)
        rec_sem = get_recommendations(history, sem_embs, topk=TOP_K)
        
        # 3. Core Filtering Logic:
        # A. Semantic MUST hit at least one GT
        sem_hits = [i for i in rec_sem if i in gt_set]
        
        # B. CF MUST NOT hit any (or hit very few)
        cf_hits = [i for i in rec_cf if i in gt_set]
        
        # Strict condition: Semantic hit, Collaborative missed
        if len(sem_hits) > 0 and len(cf_hits) == 0:
            
            print("\n" + "="*120)
            print(f"❄️ COLD-START GOLDEN CASE! User ID: {u} (History: {len(history)})")
            print("="*120)
            
            # --- History ---
            print(f"{'TYPE':<12} | {'INTERACTS':<12} | {'TITLE':<55} | {'CATEGORY':<25}")
            print("-" * 120)
            for i in history:
                t_str, c_str = format_item_str(i, id2info, item_pop)
                print(f"{'History':<12} | {t_str.split('] ')[0]+']':<12} | {t_str.split('] ')[1]:<55} | {c_str:<25}")
            
            # --- Ground Truth ---
            print("-" * 120)
            for i in gt_set:
                t_str, c_str = format_item_str(i, id2info, item_pop)
                print(f"{'GroundTruth':<12} | {t_str.split('] ')[0]+']':<12} | {t_str.split('] ')[1]:<55} | {c_str:<25}")

            # --- Comparison ---
            print("\n👇 MODEL COMPARISON (Top 5 Displayed) 👇")
            print(f"{'#':<4} | {'BASELINE (LightGCN) - Missed':<55} | {'OURS (Semantic) - DIRECT HIT!':<55}")
            print("-" * 120)
            
            display_k = 5
            for r in range(display_k):
                # CF
                ic = rec_cf[r]
                t_c, c_c = format_item_str(ic, id2info, item_pop, False) # CF known miss
                collab_text = f"{t_c} ({c_c})"
                if len(collab_text) > 52: collab_text = collab_text[:49] + "..."
                
                # Sem
                is_ = rec_sem[r]
                is_hit_s = is_ in gt_set
                t_s, c_s = format_item_str(is_, id2info, item_pop, is_hit_s)
                sem_text = f"{t_s} ({c_s})"
                if len(sem_text) > 52: sem_text = sem_text[:49] + "..."
                
                print(f"#{r+1:<3} | {collab_text:<55} | {sem_text:<55}")
            
            # If Hit is in 6-10, add a note
            if sem_hits[0] not in rec_sem[:display_k]:
                hit_item = sem_hits[0]
                rank = list(rec_sem).index(hit_item) + 1
                t_h, c_h = format_item_str(hit_item, id2info, item_pop, True)
                print("-" * 120)
                print(f"NOTE: Ours HIT at Rank #{rank}: {t_h} ({c_h})")

            print("="*120)
            
            found_count += 1
            if found_count >= 3: break 

if __name__ == "__main__":
    cf, sem, train, test, info, pops = load_data()
    find_cold_start_hits(cf, sem, train, test, info, pops)