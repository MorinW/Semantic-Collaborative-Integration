import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from collections import Counter
import os
import sys
import warnings

# Ignore irrelevant warnings
warnings.filterwarnings('ignore')

# ================= 1. Experiment Configuration =================
CURRENT_DATASET = "movies" 
BASE_DIR = "./dataset"
SAVED_DIR = "./saved"

# Plotting parameters
N_SAMPLES = 4000       # Total number of samples
TSNE_PERPLEXITY = 40   # Perplexity
RANDOM_STATE = 42      # Random seed

# 🟢 Stratified Sampling Ratio 🟢
# Force 20% of points from the hot head to ensure obvious "structure" in the visualization
HOT_RATIO = 0.2  # 800 hot items
TAIL_RATIO = 0.8 # 3200 tail items

def get_paths(dataset):
    base_data = f"{BASE_DIR}/{dataset}"
    base_save = f"{SAVED_DIR}"
    
    def find_file(filename):
        p1 = f"{base_data}/{filename}"
        if os.path.exists(p1): return p1
        p2 = f"{base_save}/{filename}"
        if os.path.exists(p2): return p2
        return None 

    paths = {
        "train_inter": f"{base_data}/{dataset}.train.inter",
        "emb_raw":     f"{base_data}/{dataset}_item_embeddings.npy",
        "emb_proj":    find_file(f"{dataset}_item_embeddings-proj64.npy"),
        "emb_pco":     find_file(f"{dataset}_item_embeddings-pco64.npy"),
    }
    return paths

PATHS = get_paths(CURRENT_DATASET)

# ================= 2. Core Logic =================

def load_and_process_data():
    print(f"🔬 [Init] t-SNE Final Visualizer | Dataset: {CURRENT_DATASET}")
    
    # Check paths
    p_raw = PATHS["emb_raw"]
    p_pco = PATHS["emb_pco"]
    p_inter = PATHS["train_inter"]

    if not (p_raw and os.path.exists(p_raw) and p_pco and os.path.exists(p_pco)):
        print("❌ Critical: Embedding files missing."); sys.exit(1)

    # Load
    print("   >>> Loading Embeddings...")
    sem_embs = np.load(p_raw).astype(np.float32)
    cf_embs = np.load(p_pco).astype(np.float32)
        
    # Align
    num_items = min(cf_embs.shape[0], sem_embs.shape[0])
    cf_embs = cf_embs[:num_items]
    sem_embs = sem_embs[:num_items]

    # Calculate popularity
    print("   >>> Calculating Popularity...")
    pop_counter = Counter()
    try:
        with open(p_inter, 'r', encoding='utf-8') as f:
            next(f) 
            for line in f:
                try:
                    parts = line.strip().split('\t')
                    if len(parts) < 2: parts = line.strip().split()
                    iid = int(parts[1])
                    if iid < num_items: pop_counter[iid] += 1
                except: continue
    except: pass

    # Raw popularity (for stratification) and Log popularity (for coloring)
    pop_raw = np.array([pop_counter.get(i, 1) for i in range(num_items)])
    pop_log = np.log10(pop_raw + 1)
    
    return cf_embs, sem_embs, pop_log, pop_raw

def stratified_sample(num_items, pop_raw_array):
    """
    Core function for stratified sampling:
    If not done, random sampling of 4000 points might only include 10 hot items,
    resulting in a t-SNE plot dominated by tail items, obscuring LightGCN's centripetal structure.
    """
    print(f"   >>> Stratified Sampling ({int(HOT_RATIO*100)}% Hot / {int(TAIL_RATIO*100)}% Tail)...")
    
    # 1. Indices sorted by popularity (descending)
    sorted_indices = np.argsort(pop_raw_array)[::-1]
    
    # 2. Define head pool (Top 20% items)
    cutoff = int(num_items * 0.2)
    head_pool = sorted_indices[:cutoff]
    tail_pool = sorted_indices[cutoff:]
    
    # 3. Calculate sample counts
    n_hot = int(N_SAMPLES * HOT_RATIO)
    n_tail = N_SAMPLES - n_hot
    
    # 4. Execute sampling
    np.random.seed(RANDOM_STATE)
    
    # Sample 20% from head pool
    if len(head_pool) >= n_hot:
        idx_hot = np.random.choice(head_pool, n_hot, replace=False)
    else:
        idx_hot = head_pool # Take all if not enough
        
    # Sample 80% from tail pool
    if len(tail_pool) >= n_tail:
        idx_tail = np.random.choice(tail_pool, n_tail, replace=False)
    else:
        idx_tail = tail_pool

    # 5. Merge and shuffle
    final_indices = np.concatenate([idx_hot, idx_tail])
    np.random.shuffle(final_indices)
    
    print(f"      - Sampled {len(idx_hot)} Hot items + {len(idx_tail)} Tail items.")
    return final_indices

def run_tsne_and_save_separate(cf_embs, sem_embs, pop_log, pop_raw):
    num_items = len(pop_log)
    
    # 🟢 Use stratified sampling instead of simple random choice
    indices = stratified_sample(num_items, pop_raw)
    
    # Normalization (L2) - Essential
    print("   >>> Normalizing Embeddings (L2)...")
    cf_samp = normalize(cf_embs[indices], norm='l2')
    sem_samp = normalize(sem_embs[indices], norm='l2')
    
    # Extract popularity for coloring
    pop_color = pop_log[indices]
    
    print(f"   >>> Running t-SNE (Perplexity={TSNE_PERPLEXITY})...")
    
    # --- 1. Process Collaborative View ---
    print("      - Calculating Collaborative Layout...")
    tsne_pco = TSNE(n_components=2, perplexity=TSNE_PERPLEXITY, random_state=RANDOM_STATE, init='pca')
    cf_2d = tsne_pco.fit_transform(cf_samp)
    
    # --- 2. Process Semantic View ---
    print("      - Calculating Semantic Layout...")
    tsne_sem = TSNE(n_components=2, perplexity=TSNE_PERPLEXITY, random_state=RANDOM_STATE, init='pca')
    sem_2d = tsne_sem.fit_transform(sem_samp)
    
    print("\n🎨 Saving Images (Reversed Colors)...")
    sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    
    # 🟢 Color config: Reversed colormap (Bright Yellow=Cold, Dark Purple=Hot)
    cmap_name = 'plasma_r' 
    
    # ================= Plot 1: Collaborative =================
    plt.figure(figsize=(10, 8), dpi=300)
    plt.scatter(cf_2d[:, 0], cf_2d[:, 1], c=pop_color, cmap=cmap_name, s=20, alpha=0.7, edgecolor='none')
    plt.title('(A) Collaborative View (LightGCN)\nPopularity-Driven Structure', fontsize=18, fontweight='bold', pad=15)
    plt.axis('off')
    plt.colorbar(label='Log10(Popularity)\n(Bright: Cold -> Dark: Hot)')
    
    file_pco = "tsne_collab_complex.png"
    plt.savefig(file_pco, bbox_inches='tight')
    print(f"   ✅ Saved: ./{file_pco}")
    plt.close()

    # ================= Plot 2: Semantic =================
    plt.figure(figsize=(10, 8), dpi=300)
    plt.scatter(sem_2d[:, 0], sem_2d[:, 1], c=pop_color, cmap=cmap_name, s=20, alpha=0.7, edgecolor='none')
    plt.title('(B) Semantic View (bge-m3)\nContent-Driven Structure', fontsize=18, fontweight='bold', pad=15)
    plt.axis('off')
    plt.colorbar(label='Log10(Popularity)\n(Bright: Cold -> Dark: Hot)')
    
    file_sem = "tsne_semantic_complex.png"
    plt.savefig(file_sem, bbox_inches='tight')
    print(f"   ✅ Saved: ./{file_sem}")
    plt.close()

if __name__ == "__main__":
    # Note: Returns 4 variables
    cf, sem, p_log, p_raw = load_and_process_data()
    run_tsne_and_save_separate(cf, sem, p_log, p_raw)