import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from collections import defaultdict, Counter
import os
import sys
import warnings
import random

# Ignore warnings
warnings.filterwarnings('ignore')

# ================= 1. Experiment Configuration =================
CURRENT_DATASET = "movies" 
BASE_DIR = "./dataset"
SAVED_DIR = "./saved"

# Plotting parameters
N_BACKGROUND = 1500    # Number of background points
TOP_K = 10             # Number of recommendations to show
TSNE_PERPLEXITY = 30
RANDOM_STATE = 42

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

# ================= 2. Core Algorithms =================

def get_recommendations(user_history, item_embs, topk=TOP_K):
    """Simple Mean Pooling recommendation logic"""
    if not user_history: return []
    # User vector = Average of history item vectors
    user_vec = np.mean(item_embs[user_history], axis=0).reshape(1, -1)
    # Calculate cosine similarity (Assume input is normalized, dot product equals cosine)
    scores = np.dot(item_embs, user_vec.T).flatten()
    # Exclude interacted items
    scores[user_history] = -np.inf
    return np.argsort(scores)[-topk:][::-1]

def load_data():
    print(f"🔬 [Init] User View Visualizer | Dataset: {CURRENT_DATASET}")
    
    # Use proj64 as semantic branch, pco64 as collaborative branch
    p_sem = PATHS["emb_proj"] if PATHS["emb_proj"] else PATHS["emb_raw"]
    p_pco = PATHS["emb_pco"]
    p_inter = PATHS["train_inter"]
    
    if not (p_sem and os.path.exists(p_sem) and p_pco and os.path.exists(p_pco)):
        print("❌ Critical: Embedding files missing."); sys.exit(1)

    print("   >>> Loading Embeddings...")
    sem_embs = np.load(p_sem).astype(np.float32)
    cf_embs = np.load(p_pco).astype(np.float32)
    
    num_items = min(cf_embs.shape[0], sem_embs.shape[0])
    # Normalization is key to good t-SNE performance
    cf_embs = normalize(cf_embs[:num_items], norm='l2')
    sem_embs = normalize(sem_embs[:num_items], norm='l2')
    
    print("   >>> Loading User Interactions...")
    user_inter = defaultdict(list)
    with open(p_inter, 'r', encoding='utf-8') as f:
        next(f) 
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2: parts = line.strip().split()
            uid, iid = int(parts[0]), int(parts[1])
            if iid < num_items:
                user_inter[uid].append(iid)
        
    return cf_embs, sem_embs, user_inter, num_items

# ================= 3. Data Preparation =================

def prepare_plot_data(target_uid, user_inter, num_items, cf_embs, sem_embs):
    history = user_inter[target_uid]
    
    # 1. Generate recommendation lists
    rec_sem = get_recommendations(history, sem_embs, topk=TOP_K)
    rec_cf = get_recommendations(history, cf_embs, topk=TOP_K)
    
    # 2. Determine all points to plot
    special_set = set(history) | set(rec_sem) | set(rec_cf)
    background_candidates = [i for i in range(num_items) if i not in special_set]
    n_bg = min(len(background_candidates), N_BACKGROUND)
    background_items = np.random.choice(background_candidates, n_bg, replace=False)
    
    # 3. Combine data and labels
    # Label legend: 0:Background, 1:History(Blue), 2:Semantic Rec(Red), 3:Collab Rec(Purple)
    final_indices = np.concatenate([background_items, history, rec_sem, rec_cf])
    labels = np.concatenate([
        np.zeros(len(background_items)),
        np.ones(len(history)),
        np.ones(len(rec_sem)) * 2,
        np.ones(len(rec_cf)) * 3
    ])
    
    return sem_embs[final_indices], cf_embs[final_indices], labels

# ================= 4. Plotting Logic =================

def run_tsne_and_plot(sem_samp, cf_samp, labels, uid):
    print(f"\n🚀 Running Dual-View t-SNE for User {uid}...")
    
    tsne = TSNE(n_components=2, perplexity=TSNE_PERPLEXITY, random_state=RANDOM_STATE, init='pca')
    sem_2d = tsne.fit_transform(sem_samp)
    cf_2d = tsne.fit_transform(cf_samp)
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 9), dpi=300)
    sns.set_style("white")

    def plot_view(ax, data_2d, title):
        # A. Background (Grey)
        mask = labels == 0
        ax.scatter(data_2d[mask, 0], data_2d[mask, 1], c='#cccccc', s=15, alpha=0.3, label='Background')
        
        # B. Collaborative Rec (Purple - Cross)
        mask = labels == 3
        ax.scatter(data_2d[mask, 0], data_2d[mask, 1], c='purple', marker='x', s=70, alpha=0.8, 
                   linewidth=1.5, label='Baseline (LightGCN)', zorder=2)

        # C. Semantic Rec (Red - Circle)
        mask = labels == 2
        ax.scatter(data_2d[mask, 0], data_2d[mask, 1], c='#d62728', s=70, alpha=0.9, 
                   edgecolors='white', label='Ours (Semantic)', zorder=3)
        
        # D. History Interactions (Blue - Big Circle) - Changed to blue as requested
        mask = labels == 1
        ax.scatter(data_2d[mask, 0], data_2d[mask, 1], c='#1f77b4', s=120, alpha=1.0, 
                   edgecolors='white', linewidth=1.5, label='User History', zorder=4)

        ax.set_title(title, fontsize=15, fontweight='bold')
        ax.axis('off')
        if "Semantic" in title:
            ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=1, edgecolor='black')

    plot_view(axes[0], sem_2d, f'(A) Semantic View\nUser {uid} Content Universe')
    plot_view(axes[1], cf_2d, f'(B) Collaborative View\nUser {uid} Behavioral Universe')
    
    plt.tight_layout()
    outfile = f"dual_view_user_{uid}.png"
    plt.savefig(outfile, bbox_inches='tight')
    print(f"✅ Visualization Saved: ./{outfile}")

if __name__ == "__main__":
    cf_all, sem_all, interactions, n_items = load_data()
    # Pick an active user to observe clustering trends
    sorted_users = sorted(interactions.items(), key=lambda x: len(x[1]), reverse=True)
    target_uid = random.choice(sorted_users[:50])[0] 
    
    sem_sub, cf_sub, labs = prepare_plot_data(target_uid, interactions, n_items, cf_all, sem_all)
    run_tsne_and_plot(sem_sub, cf_sub, labs, target_uid)