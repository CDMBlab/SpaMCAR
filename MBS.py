import warnings
warnings.filterwarnings("ignore")
import torch
import pandas as pd
import numpy as np
import scanpy as sc
import os
import yaml
import random
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['R_HOME'] = '/opt/conda/envs/stm/lib/R/'
os.environ['R_USER'] = '/opt/conda/envs/stm/lib/python3.11/site-packages/rpy2'
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from SpaMCAR.Func import enhanced_data_processing, graph_construction
from SpaMCAR.Utils import refine_label
from SpaMCAR.MGAC import Mgac
from SpaMCAR.Models import SMGA
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm

plt.rcParams['font.family'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

# Font size settings
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

print("Global font set to Times New Roman")

def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='latent', key_added_pred='mclust',
             random_seed=666):
    """Clustering using the mclust algorithm."""
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")
    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']
    
    res = rmclust(adata.obsm[used_obsm], num_cluster, modelNames)
    mclust_res = np.array(res[-2])
    
    adata.obs[key_added_pred] = mclust_res
    adata.obs[key_added_pred] = adata.obs[key_added_pred].astype('int')
    adata.obs[key_added_pred] = adata.obs[key_added_pred].astype('category')
    return adata

def evaluate_clustering(features, labels):
    """Evaluate clustering quality with multiple metrics."""
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    if n_clusters < 2:
        return {'n_clusters': n_clusters, 'silhouette': -1, 'davies_bouldin': np.inf, 
                'calinski_harabasz': -1, 'valid': False}
    
    try:
        sil_score = silhouette_score(features, labels)
        db_score = davies_bouldin_score(features, labels)
        ch_score = calinski_harabasz_score(features, labels)
        return {
            'n_clusters': n_clusters,
            'silhouette': sil_score,
            'davies_bouldin': db_score,
            'calinski_harabasz': ch_score,
            'valid': True
        }
    except:
        return {'n_clusters': n_clusters, 'silhouette': -1, 'davies_bouldin': np.inf, 
                'calinski_harabasz': -1, 'valid': False}

def find_optimal_clusters(features, min_clusters=2, max_clusters=30, method='combined'):
    """
    Automatically find optimal number of clusters.
    method: 'silhouette', 'db', 'ch', 'combined'
    """
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    results = []
    cluster_range = range(min_clusters, min(max_clusters + 1, features_scaled.shape[0] - 1))
    
    print(f"\nTesting cluster range: {min_clusters} to {min(max_clusters, features_scaled.shape[0]-2)}")
    
    for n_clust in cluster_range:
        try:
            # KMeansmclust
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_clust, random_state=666, n_init=10)
            labels = kmeans.fit_predict(features_scaled)
            
            metrics = evaluate_clustering(features_scaled, labels)
            if metrics['valid']:
                results.append(metrics)
                print(f"  n={n_clust:2d}: SC={metrics['silhouette']:.4f}, "
                      f"DB={metrics['davies_bouldin']:.4f}, "
                      f"CH={metrics['calinski_harabasz']:.2f}")
        except Exception as e:
            print(f"  n={n_clust:2d}: Failed - {str(e)[:50]}")
            continue
    
    if not results:
        print("Cannot compute clustering metrics")
        return 10, None
    
    results_df = pd.DataFrame(results)
    
    if method == 'silhouette':
        best_idx = results_df['silhouette'].idxmax()
    elif method == 'db':
        # Davies-Bouldin
        best_idx = results_df['davies_bouldin'].idxmin()
    elif method == 'ch':
        # Calinski-Harabasz
        best_idx = results_df['calinski_harabasz'].idxmax()
    else:  # combined score 
        results_df['silhouette_norm'] = (results_df['silhouette'] - results_df['silhouette'].min()) / \
                                        (results_df['silhouette'].max() - results_df['silhouette'].min())
        results_df['db_norm'] = (results_df['davies_bouldin'].max() - results_df['davies_bouldin']) / \
                                (results_df['davies_bouldin'].max() - results_df['davies_bouldin'].min())
        results_df['ch_norm'] = (results_df['calinski_harabasz'] - results_df['calinski_harabasz'].min()) / \
                                (results_df['calinski_harabasz'].max() - results_df['calinski_harabasz'].min())
        results_df['combined_score'] = (results_df['silhouette_norm'] + results_df['db_norm'] + results_df['ch_norm']) / 3
        best_idx = results_df['combined_score'].idxmax()
    
    best_n_clusters = results_df.loc[best_idx, 'n_clusters']
    
    return int(best_n_clusters), results_df

# MBS
proj_name = 'MBS'
result_dir = './result/MBS'
os.makedirs(result_dir, exist_ok=True)

config_path = './Config/MBS.yaml'
if os.path.exists(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
else:
    config = {
        'seed': 3407,
        'data': {
            'k_cutoff': 10,
            'n_top_genes': 3000
        },
        'model': {
            'epochs': 500,
            'lr': 0.001,
            'weight_decay': 0.0001
        }
    }

# MBS data loading
data_path = './data/MBS'
adata = sc.read_visium(data_path, count_file='V1_Mouse_Brain_Sagittal_Posterior_filtered_feature_bc_matrix.h5')
adata.var_names_make_unique()

print("MBS data loaded")
print(f"Data shape: {adata.shape}")
print("No ground truth labels - will determine optimal cluster count automatically")

seed = config.get('seed', 3407)

# Set random seeds
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
random.seed(seed)
sc.settings.seed = seed
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print(f"Random seed set to {seed}")

# Data processing
print("\n=== Data processing ===")
adata = enhanced_data_processing(
    adata,
    token_dim=250,
    conv_dim=128,
    final_dim=200,
    use_pca=False,
    config=config
)

# Build graph structure
print("\nBuilding graph structure...")
graph_dict = graph_construction(adata, config['data']['k_cutoff'])
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Use initial cluster count (later refined via latent features)
initial_clusters = 30
print(f"\nInitial cluster count: {initial_clusters} (will be optimized after training)")

# Train SpaMCAR model
print("\nStarting SpaMCAR training...")
net = Mgac(adata, graph_dict=graph_dict, num_clusters=initial_clusters, device=device, config=config)
net.trian()

# Process results
enc_rep, recon = net.process()
enc_rep = enc_rep.data.cpu().numpy()
recon = recon.data.cpu().numpy()
adata.obsm['latent'] = enc_rep
adata.obsm['recon'] = recon

print(f"\nModel training complete")
print(f"Latent feature shape: {adata.obsm['latent'].shape}")

# ============================================
# Fixed cluster count
# ============================================
print("\n" + "="*60)
print("Using fixed cluster count")
print("="*60)

best_n_clusters = 8
print(f"\nUsing fixed cluster count: {best_n_clusters}")

print(f"\n=== Final clustering with k={best_n_clusters} ===")
adata = mclust_R(adata, num_cluster=best_n_clusters, used_obsm='latent', key_added_pred='mclust')

# Evaluate final clustering
final_labels = adata.obs['mclust'].astype(int).values
from sklearn.preprocessing import StandardScaler
latent_scaled = StandardScaler().fit_transform(adata.obsm['latent'])
final_metrics = evaluate_clustering(latent_scaled, final_labels)

print(f"\nFinal clustering metrics (k={best_n_clusters}):")
print(f"  Silhouette Score: {final_metrics['silhouette']:.4f} (higher better, range [-1,1])")
print(f"  Davies-Bouldin Index: {final_metrics['davies_bouldin']:.4f} (lower better)")
print(f"  Calinski-Harabasz Index: {final_metrics['calinski_harabasz']:.2f} (higher better)")

adata.obs['domain'] = adata.obs['mclust'].copy()

sc.pp.neighbors(adata, use_rep='latent')
sc.tl.umap(adata)

plt.figure(figsize=(3, 3))
sc.pl.umap(
    adata,
    color="domain",
    title='SpaMCAR (SC=%.2f,DB=%.2f)' % (final_metrics['silhouette'], final_metrics['davies_bouldin']),
    show=False
)
plt.savefig("./result/MBS/MBS_UMAP.jpg", dpi=600, bbox_inches='tight')
plt.close()

plt.figure(figsize=(3, 3))
sc.pl.spatial(
    adata,
    color="domain",
    title='SpaMCAR (SC=%.2f,DB=%.2f)' % (final_metrics['silhouette'], final_metrics['davies_bouldin']),
    show=False
)
plt.savefig("./result/MBS/MBS_Spatial.jpg", dpi=600, bbox_inches='tight')
plt.close()

adata_path = './result/MBS/MBS_processed.h5ad'
adata.write_h5ad(adata_path)
print(f'Saved: {adata_path}')
