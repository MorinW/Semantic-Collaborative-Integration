import argparse
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed, get_trainer
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType

# ============================================================================
#                               Model Definition
# ============================================================================

class SimGCL(GeneralRecommender):
    """
    SimGCL: Simple Graph Contrastive Learning
    Core Idea: Based on LightGCN, adds random uniform noise to the embedding 
    at each layer to generate augmented views, and maximizes consistency 
    between views using InfoNCE Loss.
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(SimGCL, self).__init__(config, dataset)
        
        # --- 1. Load Basic Configuration ---
        self.embedding_size = config['embedding_size']
        self.n_layers = config['n_layers']
        self.reg_weight = config['reg_weight']
        
        # --- 2. Load SimGCL Specific Configuration ---
        # eps: Noise magnitude (default 0.1)
        self.eps = config['eps'] if 'eps' in config else 0.1
        # cl_rate: Contrastive Learning weight (lambda)
        self.cl_rate = config['cl_rate'] if 'cl_rate' in config else 0.1
        # tau: Temperature for InfoNCE
        self.tau = config['tau'] if 'tau' in config else 0.2
        
        # --- 3. Initialize Embeddings ---
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        
        # --- 4. Build Adjacency Matrix ---
        # Use the passed dataset (which should be train_data.dataset)
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.create_adj_matrix()
        
        # --- 5. Loss Components ---
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()
        
        # Initialize parameters
        self.apply(xavier_normal_initialization)
        
        # Evaluation cache
        self._cached_eval = False

    def create_adj_matrix(self):
        """Build Normalized Adjacency Matrix (Standard LightGCN Adjacency)"""
        user_np, item_np = self.interaction_matrix.row, self.interaction_matrix.col
        n_nodes = self.n_users + self.n_items
        rows = np.concatenate([user_np, item_np + self.n_users])
        cols = np.concatenate([item_np + self.n_users, user_np])
        data = np.ones(len(rows), dtype=np.float32)
        
        adj = sp.coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
        
        deg = np.array(adj.sum(1)).flatten()
        deg_inv_sqrt = np.power(deg, -0.5, where=deg>0)
        deg_inv_sqrt[deg == 0] = 0.0
        
        D_inv_sqrt = sp.diags(deg_inv_sqrt)
        adj_norm = D_inv_sqrt.dot(adj).dot(D_inv_sqrt).tocoo()
        
        indices = torch.LongTensor(np.vstack((adj_norm.row, adj_norm.col)))
        values = torch.FloatTensor(adj_norm.data)
        self.adj_matrix = torch.sparse_coo_tensor(indices, values, size=adj_norm.shape).coalesce().to(self.device)

    def forward(self, perturbed=False):
        """
        Forward Pass
        Args:
            perturbed (bool): Whether to add perturbation noise (for CL views)
        """
        all_embs = []
        # Layer 0 Embedding
        ego_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        all_embs.append(ego_embeddings)
        
        for k in range(self.n_layers):
            if perturbed:
                # === SimGCL Core: Add Random Noise ===
                random_noise = torch.rand_like(ego_embeddings).to(self.device)
                # Normalize noise and scale to eps
                random_noise = F.normalize(random_noise, dim=1) * self.eps
                ego_embeddings = ego_embeddings + random_noise
                # =====================================
                
            # Graph Convolution: E = A * E
            ego_embeddings = torch.sparse.mm(self.adj_matrix, ego_embeddings)
            all_embs.append(ego_embeddings)
            
        # Mean Aggregation
        final_embs = torch.stack(all_embs, dim=1).mean(dim=1)
        u_final, i_final = torch.split(final_embs, [self.n_users, self.n_items])
        
        return u_final, i_final

    def calculate_loss(self, interaction):
        # 1. Main Task Loss (Recommendation)
        # perturbed=False, clean LightGCN propagation
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]
        
        self._cached_eval = False
        
        user_e, item_e = self.forward(perturbed=False)
        u_e, pos_e, neg_e = user_e[user], item_e[pos_item], item_e[neg_item]
        
        # BPR Loss
        pos_scores = torch.mul(u_e, pos_e).sum(dim=1)
        neg_scores = torch.mul(u_e, neg_e).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)
        
        # Regularization Loss
        u_ego = self.user_embedding(user)
        pos_ego = self.item_embedding(pos_item)
        neg_ego = self.item_embedding(neg_item)
        reg_loss = self.reg_loss(u_ego, pos_ego, neg_ego)
        
        # 2. Contrastive Learning Loss
        # Generate two perturbed views
        user_e1, item_e1 = self.forward(perturbed=True) # View 1
        user_e2, item_e2 = self.forward(perturbed=True) # View 2
        
        # Calculate only for nodes in the batch
        u_e1, u_e2 = user_e1[user], user_e2[user]
        i_e1, i_e2 = item_e1[pos_item], item_e2[pos_item]
        
        # InfoNCE Loss
        cl_loss = self.calc_cl_loss(u_e1, u_e2) + self.calc_cl_loss(i_e1, i_e2)
        
        # 3. Total Loss
        return mf_loss + self.reg_weight * reg_loss + self.cl_rate * cl_loss

    def calc_cl_loss(self, view1, view2):
        """
        Calculate InfoNCE Loss
        Args:
            view1, view2: Embeddings of the same batch in two different views [Batch, Dim]
        """
        # Normalize (required for Cosine Similarity)
        view1 = F.normalize(view1, dim=1)
        view2 = F.normalize(view2, dim=1)
        
        # Similarity Matrix: [Batch, Batch]
        pos_score = (view1 * view2).sum(dim=1) / self.tau
        ttl_score = torch.matmul(view1, view2.t()) / self.tau
        
        # InfoNCE Formula: -log( exp(pos) / sum(exp(all)) )
        # Using log_sum_exp for numerical stability
        cl_loss = -pos_score + torch.logsumexp(ttl_score, dim=1)
        
        return cl_loss.mean()

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        with torch.no_grad():
            user_e, item_e = self.forward(perturbed=False)
            score = torch.mul(user_e[user], item_e[item]).sum(dim=1)
        return score

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if not self._cached_eval:
            with torch.no_grad():
                self.user_e, self.item_e = self.forward(perturbed=False)
            self._cached_eval = True
        u_embeddings = self.user_e[user]
        scores = torch.matmul(u_embeddings, self.item_e.t())
        return scores

# ============================================================================
#                               Pipeline Logic
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train SimGCL and Export Embeddings.")
    
    # Dataset & Paths
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., movies, books, games)')
    parser.add_argument('--data_path', type=str, default='./dataset', help='Path to dataset')
    parser.add_argument('--output_root', type=str, default='./saved', help='Directory to save model checkpoints')
    
    # Training Hyperparameters
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_batch_size', type=int, default=2048, help='Training batch size')
    parser.add_argument('--embedding_size', type=int, default=64, help='Embedding dimension')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of GCN layers')
    parser.add_argument('--reg_weight', type=float, default=1e-4, help='L2 Regularization weight')
    
    # SimGCL Specific Hyperparameters
    parser.add_argument('--eps', type=float, default=0.1, help='SimGCL noise epsilon')
    parser.add_argument('--tau', type=float, default=0.2, help='SimGCL temperature tau')
    parser.add_argument('--cl_rate', type=float, default=0.1, help='SimGCL contrastive loss weight')
    
    # System & Evaluation
    parser.add_argument('--gpu_id', type=str, default='0', help='GPU ID to use')
    parser.add_argument('--seed', type=int, default=2020, help='Random seed')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    
    return parser.parse_args()

def export_embeddings(model, dataset, args, config):
    """
    Extracts user/item embeddings from the model, maps them back to original token IDs,
    and saves them as .npy files.
    """
    print(f"\n>>> Exporting SimGCL Embeddings...")
    
    model.eval()
    with torch.no_grad():
        # Retrieve embeddings via forward (without perturbation)
        user_all_embeddings, item_all_embeddings = model.forward(perturbed=False)
    
    user_all_embeddings = user_all_embeddings.cpu().numpy()
    item_all_embeddings = item_all_embeddings.cpu().numpy()
    
    n_int_users = user_all_embeddings.shape[0]
    n_int_items = item_all_embeddings.shape[0]
    dim = user_all_embeddings.shape[1]
    
    # --- Mapping Logic: Internal ID -> Token -> Original Integer ID ---
    uid_field = config['USER_ID_FIELD']
    iid_field = config['ITEM_ID_FIELD']
    
    # Get ID to Token mapping lists
    user_id2token = dataset.field2id_token[uid_field]
    item_id2token = dataset.field2id_token[iid_field]
    
    def get_max_original_id(token_list):
        max_id = 0
        for token in token_list:
            if isinstance(token, bytes): token = token.decode('utf-8')
            if token == '[PAD]': continue 
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
    save_dir = os.path.join(args.data_path, args.dataset)
    os.makedirs(save_dir, exist_ok=True)
    
    user_path = os.path.join(save_dir, f"{args.dataset}_user_embeddings-simgcl.npy")
    item_path = os.path.join(save_dir, f"{args.dataset}_item_embeddings-simgcl.npy")
    
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
        'show_progress': True,
        
        # Data
        'data_path': args.data_path,
        'load_col': {'inter': ['user_id', 'item_id']}, 
        
        # Model Structure
        'embedding_size': args.embedding_size,
        'n_layers': args.n_layers,
        'reg_weight': args.reg_weight,
        
        # SimGCL Params
        'eps': args.eps,
        'tau': args.tau,
        'cl_rate': args.cl_rate,
        
        # Training
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'train_batch_size': args.train_batch_size,
        'learner': 'adam',
        'eval_step': 1,
        'stopping_step': args.patience,
        
        # Evaluation
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

    print(f">>> Starting Training: SimGCL on {args.dataset}")
    print(f"    Params: eps={args.eps}, tau={args.tau}, cl_rate={args.cl_rate}")
    
    # 2. Initialization
    # Note: We pass SimGCL class directly to Config, though strictly Config usually takes a string.
    # RecBole checks if 'model' is a class or string.
    config = Config(model=SimGCL, dataset=args.dataset, config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])
    
    # 3. Data Loading
    dataset = create_dataset(config)
    print(f"    Item Num: {dataset.item_num}")
    
    train_data, valid_data, test_data = data_preparation(config, dataset)
    
    # 4. Model Construction
    # We instantiate SimGCL manually to ensure it gets the training dataset for adjacency matrix construction
    model = SimGCL(config, train_data.dataset).to(config['device'])
    print(model)
    
    # 5. Trainer
    trainer = get_trainer(config['model'], config['model'])(config, model)
    
    # 6. Training Loop
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, show_progress=True)
    
    # 7. Final Evaluation
    print("\n>>> Evaluating Best Model...")
    test_result = trainer.evaluate(test_data)
    print(f"Test Result: {test_result}")
    
    # 8. Export Embeddings
    export_embeddings(model, dataset, args, config)

if __name__ == '__main__':
    main()