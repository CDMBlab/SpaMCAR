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
from sklearn.metrics import normalized_mutual_info_score
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['R_HOME'] = '/opt/conda/envs/stm/lib/R/'
os.environ['R_USER'] = '/opt/conda/envs/stm/lib/python3.11/site-packages/rpy2'
from sklearn.metrics import adjusted_rand_score as ari_score
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

def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='imputation', key_added_pred='impute_mclust',
             random_seed=666):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """

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

import SpaMCAR
proj_name = 'HBC'
num_clusters = 20
result_dir = './result/HBC'
os.makedirs(result_dir, exist_ok=True)
with open('./Config/HBC.yaml', 'r', encoding='utf-8') as f:
    config = yaml.load(f.read(), Loader=yaml.FullLoader)
adata = sc.read_visium('./data/V1_Breast_Cancer_Block_A_Section_1',
                                   count_file='V1_Breast_Cancer_Block_A_Section_1_filtered_feature_bc_matrix.h5')
adata.var_names_make_unique()
seed = config.get('seed', 3407)

# --------------------------------------------------------------
# Set random seeds for full reproducibility
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
random.seed(seed)
sc.settings.seed = seed
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print(f"Random seed set to {seed}")
##### Load layer_guess label, if have
truth_path = './data/V1_Breast_Cancer_Block_A_Section_1/metadata.tsv'
truth_labels = pd.read_csv(truth_path, sep='\t', header=0)
truth_labels.index = adata.obs_names
adata.obs['layer_guess'] = truth_labels['fine_annot_type']

print("Original data shape:", adata.shape)
print("=== Data loading complete ===")
print("Data shape:", adata.shape)
print("Available obsm keys:", list(adata.obsm.keys()))
adata = enhanced_data_processing(
    adata,
    token_dim=256,
    conv_dim=128,
    final_dim=200,
    use_pca=False,
    config=config
)

# Build graph structure
print("Building graph structure...")
graph_dict = graph_construction(adata, config['data']['k_cutoff'])
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
net = Mgac(adata, graph_dict=graph_dict, num_clusters=num_clusters, device=device, config=config)
print("Starting training...")
net.trian()
# Process results
enc_rep, recon = net.process()
enc_rep = enc_rep.data.cpu().numpy()
recon = recon.data.cpu().numpy()
adata.obsm['latent'] = enc_rep
adata.obsm['recon'] = recon
print("Multi-scale feature shape:", adata.obsm['multi_scale_features'].shape)
print("obsm keys:", list(adata.obsm.keys()))

# Check if multi-scale features were generated
if 'multi_scale_features' in adata.obsm:
    print("Multi-scale features successfully generated and stored")
else:
    print("Multi-scale features not found")
adata = mclust_R(adata, num_cluster=num_clusters, used_obsm='latent', key_added_pred='mclust')
adata = adata[~adata.obs['layer_guess'].isna() & ~adata.obs['mclust'].isna()].copy()
ARI = ari_score(adata.obs['layer_guess'], adata.obs['mclust'])
print("ARI:", ARI)
NMI = normalized_mutual_info_score(adata.obs['layer_guess'], adata.obs['mclust'])
print("NMI:", NMI)
print(adata.obs)
print(adata.obsm)
sc.pp.neighbors(adata, use_rep='latent')
sc.tl.umap(adata)

# Draw UMAP plot and save (600 DPI)
plt.figure(figsize=(3, 3))
sc.pl.umap(
    adata,
    color=["mclust", 'layer_guess'],
    title=['SpaMCAR (ARI=%.2f,NMI=%.2f)' % (ARI, NMI), 'Ground Truth'],
    show=False
)
plt.savefig("./result/HBC/HBC_UMAP.jpg", dpi=600, bbox_inches='tight')
plt.close()

plt.figure(figsize=(3, 3))
sc.pl.spatial(
    adata,
    color=["mclust", 'layer_guess'],
    title=['SpaMCAR (ARI=%.2f,NMI=%.2f)' % (ARI, NMI), 'Ground Truth'],
    show=False
)
plt.savefig("./result/HBC/HBC_Spatial.jpg", dpi=600, bbox_inches='tight')
plt.close()
adata_path = './result/HBC/HBC_processed.h5ad'
adata.write_h5ad(adata_path)
print(f'Saved: {adata_path}')

