# Func.py - Data processing and graph construction utilities for SpaMCAR
import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import coo_matrix, block_diag
import math
import scanpy as sc
from sklearn.decomposition import PCA

##### Generate adjacency matrix via KNN
def generate_adj_mat(adata, include_self=False, n=6):
    from sklearn import metrics
    assert 'spatial' in adata.obsm, 'AnnData object should provided spatial information'

    dist = metrics.pairwise_distances(adata.obsm['spatial'])

    adj_mat = np.zeros((len(adata), len(adata)))
    for i in range(len(adata)):
        n_neighbors = np.argsort(dist[i, :])[:n+1]
        adj_mat[i, n_neighbors] = 1

    if not include_self:
        x, y = np.diag_indices_from(adj_mat)
        adj_mat[x, y] = 0

    adj_mat = adj_mat + adj_mat.T
    adj_mat = adj_mat > 0
    adj_mat = adj_mat.astype(np.int64)

    return adj_mat

def generate_adj_mat_1(adata, max_dist):
    from sklearn import metrics
    assert 'spatial' in adata.obsm, 'AnnData object should provided spatial information'

    dist = metrics.pairwise_distances(adata.obsm['spatial'], metric='euclidean')
    adj_mat = dist < max_dist
    adj_mat = adj_mat.astype(np.int64)
    return adj_mat

##### Normalize graph
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def preprocess_graph(adj):
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)

def graph_construction(adata, n=6, dmax=50, mode='KNN'):
    if mode == 'KNN':
        adj_m1 = generate_adj_mat(adata, include_self=False, n=n)
    else:
        adj_m1 = generate_adj_mat_1(adata, dmax)
    adj_m1 = sp.coo_matrix(adj_m1)

    adj_m1 = adj_m1 - sp.dia_matrix((adj_m1.diagonal()[np.newaxis, :], [0]), shape=adj_m1.shape)
    adj_m1.eliminate_zeros()

    adj_norm_m1 = preprocess_graph(adj_m1)
    adj_m1 = adj_m1 + sp.eye(adj_m1.shape[0])

    adj_m1 = adj_m1.tocoo()
    shape = adj_m1.shape
    values = adj_m1.data
    indices = np.stack([adj_m1.row, adj_m1.col])
    adj_label_m1 = torch.sparse_coo_tensor(indices, values, shape)

    norm_m1 = adj_m1.shape[0] * adj_m1.shape[0] / float((adj_m1.shape[0] * adj_m1.shape[0] - adj_m1.sum()) * 2)

    graph_dict = {
        "adj_norm": adj_norm_m1,
        "adj_label": adj_label_m1.coalesce(),
        "norm_value": norm_m1,
    }

    return graph_dict

def coo2csr(coo_matrix):
    coo_matrix = coo_matrix.coalesce()
    indices = coo_matrix.indices()
    values = coo_matrix.values()
    sparse_matrix = sp.coo_matrix((values.numpy(), indices.numpy()), shape=coo_matrix.size())
    csr_matrix = sparse_matrix.tocsr()
    return csr_matrix

def csr2coo(csr_matrix):
    coo_matrix = csr_matrix.tocoo()
    indices = torch.tensor([coo_matrix.row, coo_matrix.col])
    values = torch.tensor(coo_matrix.data)
    size = torch.Size(coo_matrix.shape)
    sparse_tensor = torch.sparse_coo_tensor(indices, values, size)
    return sparse_tensor

def combine_graph_dict_1(dict_1, dict_2):
    tmp_adj_norm = csr2coo(block_diag([coo2csr(dict_1['adj_norm']), coo2csr(dict_2['adj_norm'])]))
    tmp_adj_label = csr2coo(block_diag([coo2csr(dict_1['adj_label']), coo2csr(dict_2['adj_label'])]))
    graph_dict = {
        "adj_norm": tmp_adj_norm,
        "adj_label": tmp_adj_label,
        "norm_value": np.mean([dict_1['norm_value'], dict_2['norm_value']])
    }
    return graph_dict

def combine_graph_dict(dict_1, dict_2):
    tmp_adj_norm = torch.block_diag(dict_1['adj_norm'].to_dense(), dict_2['adj_norm'].to_dense())
    tmp_adj_norm = tmp_adj_norm.to_sparse()
    tmp_adj_label = torch.block_diag(dict_1['adj_label'].to_dense(), dict_2['adj_label'].to_dense())
    tmp_adj_label = tmp_adj_label.to_sparse()
    graph_dict = {
        "adj_norm": tmp_adj_norm,
        "adj_label": tmp_adj_label,
        "norm_value": np.mean([dict_1['norm_value'], dict_2['norm_value']])
    }
    return graph_dict

# ==================== Multi-Scale Feature Processing Module ====================
class MultiScaleFeatureProcessor:
    def __init__(self, token_dim=256, conv_dim=128, final_dim=200, config=None):
        self.token_dim = token_dim
        self.conv_dim = conv_dim
        self.final_dim = final_dim
        self.config = config
        print(f"Initializing convolution layer: in_channels={token_dim}, out_channels={conv_dim}")
        self.conv_layer = nn.Conv1d(in_channels=token_dim, out_channels=conv_dim,
                                  kernel_size=1, stride=1, padding=0)
        # Using PCA for dimension reduction instead of linear layer

    def spatial_position_encoding(self, spatial_coords, d_model):
        """Construct position encoding based on real spatial coordinates."""
        coords = np.asarray(spatial_coords, dtype=np.float32)
        if coords.ndim != 2:
            raise ValueError("spatial coords must be a 2D array")
        if coords.shape[1] < 2:
            raise ValueError("spatial coords must contain at least x and y")

        coords = coords[:, :2]
        coords = (coords - coords.mean(axis=0, keepdims=True)) / (coords.std(axis=0, keepdims=True) + 1e-8)
        coords_t = torch.from_numpy(coords)

        # Multi-frequency sin/cos encoding based on x/y coordinates
        n_freq = max(1, d_model // 4)
        if n_freq == 1:
            freq = torch.ones(1)
        else:
            freq = torch.exp(torch.linspace(0, -math.log(10000.0), n_freq))

        x = coords_t[:, 0:1]
        y = coords_t[:, 1:2]
        x_enc = torch.cat([torch.sin(x * freq), torch.cos(x * freq)], dim=1)
        y_enc = torch.cat([torch.sin(y * freq), torch.cos(y * freq)], dim=1)
        pos = torch.cat([x_enc, y_enc], dim=1)

        if pos.shape[1] < d_model:
            pad = torch.zeros(pos.shape[0], d_model - pos.shape[1])
            pos = torch.cat([pos, pad], dim=1)
        elif pos.shape[1] > d_model:
            pos = pos[:, :d_model]
        return pos

    def process_features(self, adata, use_pca=False):
        """
        Multi-scale feature processing:
        1. Raw features
        2. Convolution features
        3. Position encoding
        """
        print("Starting multi-scale feature processing...")

        if use_pca:
            n_samples, n_features = adata.X.shape
            max_components = min(n_samples, n_features)

            # Dynamically adjust final_dim
            final_dim = min(200, max_components)

            print(f"Using PCA as alternative, dimension set to {final_dim}...")

            X_raw = PCA(n_components=final_dim, random_state=self.config.get('seed', 3407)).fit_transform(adata.X)
            print(f"Using PCA features, shape: {X_raw.shape}")
        else:
            X_raw = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
            print(f"Using raw features, shape: {X_raw.shape}")

        # Convert to token sequences
        num_samples, num_features = X_raw.shape
        embedding_dim = self.token_dim

        remaining_features = num_features % embedding_dim
        if remaining_features != 0:
            padding_size = embedding_dim - remaining_features
            features_padded = np.concatenate([X_raw, np.zeros((num_samples, padding_size))], axis=1)
            print(f"Padded feature shape: {features_padded.shape}")
        else:
            features_padded = X_raw

        # Tokenize [batch_size, num_tokens, token_dim]
        num_tokens = features_padded.shape[1] // embedding_dim
        token_features = np.zeros((num_samples, num_tokens, embedding_dim))
        print(f"Number of tokens: {num_tokens}")

        for i in range(num_tokens):
            start_idx = i * embedding_dim
            end_idx = start_idx + embedding_dim
            token_features[:, i, :] = features_padded[:, start_idx:end_idx]

        # Convert to tensor
        token_tensor = torch.FloatTensor(token_features)
        print(f"Token tensor shape: {token_tensor.shape}")

        # Compute convolution features [batch_size, num_tokens, conv_dim]
        batch_size, seq_len, token_dim = token_tensor.shape
        conv_input = token_tensor.view(batch_size * seq_len, token_dim, 1)
        conv_features = self.conv_layer(conv_input)
        conv_features = conv_features.view(batch_size, seq_len, -1)
        print(f"Convolution feature shape: {conv_features.shape}")

        # Position encoding [batch_size, num_tokens, token_dim]
        if 'spatial' not in adata.obsm:
            raise ValueError("adata.obsm['spatial'] is required for spatial position encoding")
        position_encoding = self.spatial_position_encoding(adata.obsm['spatial'], token_dim)
        position_encoding = position_encoding.unsqueeze(1).expand(batch_size, seq_len, token_dim)
        print(f"Position encoding shape: {position_encoding.shape}")

        # Concatenate multi-scale features [batch_size, num_tokens, token_dim + conv_dim + token_dim]
        multi_scale_features = torch.cat([
            token_tensor,           # Raw token features
            conv_features,          # Convolution features
            position_encoding       # Position encoding
        ], dim=-1)
        print(f"Concatenated feature shape: {multi_scale_features.shape}")

        # Pool to get per-sample final features [batch_size, token_dim + conv_dim + token_dim]
        pooled_features = torch.max(multi_scale_features, dim=1)[0]
        print(f"Pooled feature shape: {pooled_features.shape}")

        # PCA dimension reduction to target dimension [batch_size, final_dim]
        print("Applying PCA for dimension reduction...")
        pooled_features_np = pooled_features.detach().numpy()

        pca = PCA(n_components=self.final_dim, random_state=self.config.get('seed', 3407))
        reduced_features = pca.fit_transform(pooled_features_np)

        explained_variance = np.sum(pca.explained_variance_ratio_)
        print(f"PCA explained variance ratio: {explained_variance:.4f}")
        print(f"Reduced feature shape: {reduced_features.shape}")

        return reduced_features

def enhanced_data_processing(adata, token_dim=64, conv_dim=128, final_dim=200,
                           use_pca=False, config=None):
    """
    Enhanced data processing pipeline with original filtering approach
    """
    print("=== Starting multi-scale feature processing ===")
    print("Original data shape:", adata.shape)

    try:
        # Data preprocessing
        print("Performing data preprocessing...")
        print("Creating counts layer...")
        if hasattr(adata.X, 'toarray'):
            # Sparse matrix -> dense matrix
            adata.layers['count'] = adata.X.toarray()
            print("Sparse matrix converted to dense matrix")
        else:
            # Already dense, copy directly
            adata.layers['count'] = adata.X.copy()
            print("Using existing dense matrix")

        # 2. Filter genes
        print("Filtering low-expression genes...")
        sc.pp.filter_genes(adata, min_cells=3)
        print(f"After min_cells filter: {adata.shape}")

        if adata.X.dtype.kind in 'iu':  # 'i' for int, 'u' for unsigned int
            import scipy.sparse as sp
            # Save original integer data
            adata.layers['raw_counts'] = adata.X.copy()
            # Convert to float
            adata.X = adata.X.astype(np.float32)

        # 3. Normalize
        sc.pp.normalize_per_cell(adata)
        sc.pp.log1p(adata)

        # 4. Select highly variable genes
        if config and 'top_genes' in config['data']:
            n_top_genes = config['data']['top_genes']
        else:
            n_top_genes = 3000  # Default value

        print(f"Selecting highly variable genes, count: {n_top_genes}")
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer='count', n_top_genes=n_top_genes)
        adata = adata[:, adata.var['highly_variable'] == True]
        print(f"After HVG selection: {adata.shape}")

        # 5. Scale
        sc.pp.scale(adata)

        # Generate multi-scale features
        print("Initializing MultiScaleFeatureProcessor...")
        processor = MultiScaleFeatureProcessor(
            token_dim=token_dim,
            conv_dim=conv_dim,
            final_dim=final_dim,
            config=config
        )

        print("Processing features...")
        multi_scale_features = processor.process_features(adata, use_pca=use_pca)

        print(f"Multi-scale features generated successfully, shape: {multi_scale_features.shape}")

        # Save to adata.obsm
        adata.obsm['multi_scale_features'] = multi_scale_features
        print("Multi-scale features saved to adata.obsm['multi_scale_features']")

    except Exception as e:
        print(f"Multi-scale feature processing failed: {e}")
        import traceback
        traceback.print_exc()

        # Fallback: use PCA
        print("Using PCA as fallback...")
        adata_X = PCA(n_components=final_dim, random_state=config.get('seed', 3407)).fit_transform(adata.X)
        adata.obsm['multi_scale_features'] = adata_X
        print(f"PCA feature shape: {adata_X.shape}")

    print("=== Multi-scale feature processing complete ===")
    return adata
