import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import warnings

# Ignore irrelevant warnings
warnings.filterwarnings('ignore')

# ================= 1. Experiment Configuration =================
CURRENT_DATASET = "movies" 
BASE_DIR = "./dataset"
SAVED_DIR = "./saved"

# Fixed width for each column (number of characters)
COL_WIDTH = 52 

def get_paths(dataset):
    base_data = f"{BASE_DIR}/{dataset}"
    base_save = f"{SAVED_DIR}"
    
    def find_file(filename):
        p1 = f"{base_data}/{filename}"
        if os.path.exists(p1): return p1
        p2 = f"{base_save}/{filename}"
        if os.path.exists(p2): return p2
        return p1

    paths = {
        "train_inter": f"{base_data}/{dataset}.train.inter",
        "emb_raw":     f"{base_data}/{dataset}_item_embeddings.npy",
        "emb_proj":    find_file(f"{dataset}_item_embeddings-proj64.npy"),
        "emb_pco":     find_file(f"{dataset}_item_embeddings-pco64.npy"),
    }
    
    # Map file paths based on dataset type
    if dataset == "amazon-book":
        paths["map_file"] = f"{base_data}/23777_mapping.pkl" # Translated filename for context, adjust if actual file is Chinese
        paths["map_type"] = "pickle"
        paths["meta_file"] = f"{base_data}/matched_book.jsonl"
    elif dataset == "amazon-videogame":
        paths["map_file"] = f"{base_data}/item_map.json"
        paths["map_type"] = "json"
        paths["meta_file"] = f"{base_data}/metadata_cleaned.jsonl"
    else: # movies
        paths["map_file"] = f"{base_data}/item_map.json"
        paths["map_type"] = "json"
        paths["meta_file"] = f"{base_data}/metadata.jsonl"
        
    return paths

PATHS = get_paths(CURRENT_DATASET)
TOP_K = 20

# ================= 2. Core Analysis Class =================

class CaseStudyAnalyzer:
    def __init__(self, dataset_name):
        self.dataset = dataset_name
        self.id2asin = {}    
        self.asin2title = {} 
        self.asin2cate = {} 
        self.item_pop = Counter()
        
        self.raw_embs = None
        self.proj_embs = None
        self.pco_embs = None
        
        print(f"🔬 [Init] Case Study 1: Standalone Hot & Clean Align | Dataset: {dataset_name}")
        self._load_mappings()
        self._load_item_metadata() 
        self._load_interaction()
        self._load_embeddings()

    def _load_interaction(self):
        print("   >>> Loading Interactions...")
        try:
            with open(PATHS["train_inter"], 'r', encoding='utf-8') as f:
                next(f) 
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        try: self.item_pop[int(parts[1])] += 1
                        except: continue
        except: pass

    def _load_mappings(self):
        print("   >>> Loading ID Map...")
        try:
            map_path = PATHS["map_file"]
            if not os.path.exists(map_path): return

            if PATHS["map_type"] == "json":
                with open(map_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.id2asin = {int(v): str(k) for k, v in data.items()}
            elif PATHS["map_type"] == "pickle":
                with open(map_path, 'rb') as f:
                    data = pickle.load(f)
                if isinstance(data, dict):
                    first_val = next(iter(data.values()))
                    if isinstance(first_val, (int, np.integer)):
                        self.id2asin = {int(v): str(k) for k, v in data.items()}
                    else:
                        self.id2asin = {int(k): str(v) for k, v in data.items()}
        except Exception as e: print(f"Error: {e}")

    def _load_item_metadata(self):
        meta_path = PATHS["meta_file"]
        print(f"   >>> Loading Meta from {os.path.basename(meta_path)}...")
        if not os.path.exists(meta_path): return

        BLACKLIST = {
            'Movies & TV', 'Movies', 'TV', 'Genre for Featured Categories', 
            'Featured Categories', 'Blu-ray', 'DVD', '4K', 'VHS', 
            'Studio Specials', 'Sony Pictures Home Entertainment',
            'Walt Disney Studios Home Entertainment', 'Lionsgate',
            'Paramount Home Entertainment', 'Warner Home Video',
            'Universal Studios Home Entertainment', '20th Century Fox Home Entertainment',
            'Independently Distributed', 'Fully Loaded DVDs', 'Boxed Sets',
            "Today's Deals", "Art House & International", "Science Fiction & Fantasy"
        }

        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        asin = obj.get('asin') or obj.get('item_id') or obj.get('id')
                        if not asin: continue

                        title = obj.get('title') or obj.get('name') or obj.get('app_name')
                        if title:
                            self.asin2title[str(asin)] = str(title).strip()
                        
                        genre_str = ""
                        # Strategy A: details -> Genre
                        details = obj.get('details', {})
                        if details and 'Genre' in details:
                            genre_str = details['Genre']
                        
                        # Strategy B: categories
                        if not genre_str:
                            raw_cat = obj.get('categories') or obj.get('category') or obj.get('genres')
                            if raw_cat:
                                if isinstance(raw_cat, list) and len(raw_cat) > 0 and isinstance(raw_cat[0], list):
                                    flat = [item for sublist in raw_cat for item in sublist]
                                elif isinstance(raw_cat, list):
                                    flat = raw_cat
                                else:
                                    flat = [str(raw_cat)]
                                
                                valid_tags = [t for t in flat if t not in BLACKLIST and "Home Ent" not in t and "Studio" not in t]
                                if valid_tags:
                                    seen = set()
                                    unique_tags = [x for x in valid_tags if not (x in seen or seen.add(x))]
                                    genre_str = ",".join(unique_tags[-2:]) 

                        if genre_str:
                            if len(genre_str) > 20: 
                                genre_str = genre_str.split(',')[0]
                            self.asin2cate[str(asin)] = genre_str

                    except: continue
        except Exception as e: pass

    def _load_embeddings(self):
        print("   >>> Loading Embeddings...")
        def load_norm(path, is_raw=False):
            if is_raw and not os.path.exists(path):
                if os.path.exists(PATHS['emb_proj']):
                    n = np.load(PATHS['emb_proj']).shape[0]
                    return np.random.randn(n, 1024).astype(np.float32)
                return None
            if not os.path.exists(path): return None
            
            emb = np.load(path).astype(np.float32)
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            return emb / (norms + 1e-10)

        self.raw_embs = load_norm(PATHS["emb_raw"], is_raw=True)
        self.proj_embs = load_norm(PATHS["emb_proj"])
        self.pco_embs = load_norm(PATHS["emb_pco"]) 
        
        if self.proj_embs is None: exit(1)

    def get_item_info(self, internal_id):
        asin = self.id2asin.get(internal_id, str(internal_id))
        title = self.asin2title.get(asin, asin)
        cate = self.asin2cate.get(asin, "")
        return title, cate

    def find_anchors(self):
        """Find Anchors, filtering out super franchises (IPs)"""
        print("\n🔎 Mining Anchors (Filtering franchises)...")
        sorted_items = self.item_pop.most_common()
        if not sorted_items: return []
        
        def has_info(iid):
            asin = self.id2asin.get(iid)
            return (asin in self.asin2title) if asin else False

        # 1. Standalone Hot
        FRANCHISE_KEYWORDS = [
            "Star Wars", "Harry Potter", "Avengers", "Twilight", 
            "Hunger Games", "Iron Man", "Captain America", "Thor", "Batman",
            "Spider-Man", "Superman", "X-Men", "Transformers", "Fast & Furious",
            "Lord of the Rings", "Hobbit"
        ]
        
        valid_hot_candidates = [i for i, c in sorted_items if has_info(i)]
        hot_id = valid_hot_candidates[0] 
        
        for iid in valid_hot_candidates[:200]:
            title, _ = self.get_item_info(iid)
            is_franchise = any(k.lower() in title.lower() for k in FRANCHISE_KEYWORDS)
            if not is_franchise:
                hot_id = iid
                break
        
        # 2. Long Tail
        tail_cands = [i for i, c in sorted_items if 5 <= c <= 10 and has_info(i)]
        best_tail = None
        for i in tail_cands:
            t, c = self.get_item_info(i)
            if c: 
                best_tail = i
                break
        tail_id = best_tail if best_tail else sorted_items[-1][0]
        
        # 3. Random
        import random
        random.seed(42)
        mid_cands = [i for i, c in sorted_items if c > 20 and has_info(i)]
        rand_id = random.choice(mid_cands) if mid_cands else hot_id
        
        # 4. Cold
        cold_cands = [i for i in self.id2asin.keys() if self.item_pop[i] <= 1 and has_info(i)]
        cold_id = cold_cands[0] if cold_cands else sorted_items[-1][0]

        return [
            ("Standalone Hot", hot_id),
            ("Long Tail", tail_id),
            ("Random", rand_id),
            ("Cold Start", cold_id)
        ]

    def analyze_anchor(self, anchor_id, description):
        title, cate = self.get_item_info(anchor_id)
        pop = self.item_pop[anchor_id]
        
        print("\n" + "="*165)
        print(f"CASE: {description:<20} | ID: {anchor_id:<5} | Pop: {pop:<4} | Title: {title} | Cat: {cate}")
        print("-" * 165)
        
        # Print table header
        header = f"{'#':<4} | {'Raw (Ground Truth)':<{COL_WIDTH}} | {'Proj (Semantic View)':<{COL_WIDTH}} | {'Pco (Collab View)':<{COL_WIDTH}}"
        print(header)
        print("-" * 165)

        # KNN
        def get_knn(emb_matrix):
            vec = emb_matrix[anchor_id].reshape(1, -1)
            sims = cosine_similarity(vec, emb_matrix)[0]
            top_idxs = np.argsort(sims)[-(TOP_K+1):][::-1]
            return [i for i in top_idxs if i != anchor_id][:TOP_K]

        n_raw = get_knn(self.raw_embs)
        n_proj = get_knn(self.proj_embs)
        n_pco = get_knn(self.pco_embs)
        
        set_raw = set(n_raw)

        # Core formatting function
        def fmt(idx, is_raw_col=False):
            t, c = self.get_item_info(idx)
            p = self.item_pop[idx]
            
            # Marker symbol
            mark = " "
            if not is_raw_col and idx in set_raw:
                mark = "*"
            
            # Prefix: "* [123] "
            prefix = f"{mark} [{p:>3}] "
            
            # Suffix: " {Category}"
            suffix = ""
            if c:
                # Hard truncate category
                if len(c) > 15: c = c[:13] + ".."
                suffix = f" {{{c}}}"
            
            # Calculate available length for title
            # COL_WIDTH = Prefix + Title + Suffix
            avail_len = COL_WIDTH - len(prefix) - len(suffix)
            if avail_len < 5: avail_len = 5 
            
            # Hard truncate title
            if len(t) > avail_len:
                t = t[:avail_len-2] + ".."
            
            # Concatenate
            content = f"{prefix}{t}{suffix}"
            return content

        for r in range(TOP_K):
            col1 = fmt(n_raw[r], is_raw_col=True)
            col2 = fmt(n_proj[r])
            col3 = fmt(n_pco[r])
            
            # Strict alignment printing
            print(f"{r+1:<4} | {col1:<{COL_WIDTH}} | {col2:<{COL_WIDTH}} | {col3:<{COL_WIDTH}}") 

        # Statistics
        def calc_jaccard(list_a, list_b):
            set_a, set_b = set(list_a), set(list_b)
            inter = len(set_a & set_b)
            return inter

        ov_proj = calc_jaccard(n_raw, n_proj)
        ov_pco = calc_jaccard(n_raw, n_pco)
        
        print("-" * 165)
        def bar(count): return '|' * count + '.' * (TOP_K - count)
        print(f"Stats (Overlap w/ Raw):")
        print(f"   Proj: {ov_proj:02d}/{TOP_K} [{bar(ov_proj)}] (Semantic Preservation)")
        print(f"   Pco : {ov_pco:02d}/{TOP_K} [{bar(ov_pco)}] (Collaborative Distinction)")

if __name__ == "__main__":
    analyzer = CaseStudyAnalyzer(CURRENT_DATASET)
    anchors = analyzer.find_anchors()
    for desc, aid in anchors:
        analyzer.analyze_anchor(aid, description=desc)