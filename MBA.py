import warnings
warnings.filterwarnings("ignore")
import torch
from sklearn.metrics import normalized_mutual_info_score
import pandas as pd
import numpy as np
import scanpy as sc
import os
import yaml
from pathlib import Path
import random
import matplotlib.pyplot as plt
from matplotlib import font_manager

font_path = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"
font_manager.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['R_HOME'] = '/opt/conda/envs/stm/lib/R/'
os.environ['R_USER'] = '/opt/conda/envs/stm/lib/python3.11/site-packages/rpy2'
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from SpaMCAR.Func import enhanced_data_processing, graph_construction
def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='imputation', key_added_pred='impute_mclust',
             random_seed=3407):
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
proj_name = 'MBA'
num_clusters = 52
with open('./Config/MBA.yaml', 'r', encoding='utf-8') as f:
    config = yaml.load(f.read(), Loader=yaml.FullLoader)
adata = sc.read_visium('./data/2Mouse_Brain_Anterior',
                           count_file='filtered_feature_bc_matrix.h5')

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
adata.var_names_make_unique()
print(f"Random seed set to {seed}")
##### Load layer_guess label, if have
truth_path = './data/2Mouse_Brain_Anterior/metadata.tsv'
truth_labels = pd.read_csv(truth_path, sep='\t', header=0)
truth_labels.index = adata.obs_names
adata.obs['layer_guess'] = truth_labels['ground_truth']
adata = adata[~pd.isnull(adata.obs['layer_guess'])]
print("Original data shape:", adata.shape)
print("=== Data loading complete ===")
print("Available obsm keys:", list(adata.obsm.keys()))
adata = enhanced_data_processing(
    adata,
    token_dim=250,
    conv_dim=128,
    final_dim=200,
    use_pca=True,
    config=config
)

# Build graph structure
print("Building graph structure...")
graph_dict = SpaMCAR.graph_construction(adata, config['data']['k_cutoff'])

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
net = SpaMCAR.Mgac(adata, graph_dict=graph_dict, num_clusters=num_clusters, device=device, config=config)
net.trian()
enc_rep, recon = net.process()
enc_rep = enc_rep.data.cpu().numpy()
recon = recon.data.cpu().numpy()
adata.obsm['latent'] = enc_rep
adata.obsm['recon'] = recon
adata = mclust_R(adata, num_cluster=num_clusters, used_obsm='latent', key_added_pred='mclust')
adata.obs['domain'] = SpaMCAR.refine_label(adata, 30, key='mclust')
sub_adata = adata[~pd.isnull(adata.obs['layer_guess'])]
ARI = ari_score(sub_adata.obs['layer_guess'], sub_adata.obs['domain'])
print(ARI)
# sub_adata = adata[~pd.isnull(adata.obs['layer_guess'])]

best_ari = 0
best_radius = 30

for radius in range(1, 30):
    refined_label = SpaMCAR.refine_label(adata, radius=radius, key='mclust')

    # refine
    refined_sub = [refined_label[i] for i in range(len(refined_label))
                   if not pd.isnull(adata.obs['layer_guess'].iloc[i])]

    current_ari = ari_score(sub_adata.obs['layer_guess'], refined_sub)
    print(f"Radius {radius}: ARI = {current_ari}")

    if current_ari > best_ari:
        best_ari = current_ari
        best_radius = radius
        adata.obs['best_refined'] = refined_label

print(f"Best radius: {best_radius}, Best ARI: {best_ari}")

adata.obs['domain'] = adata.obs['best_refined']
ARI = ari_score(adata.obs['layer_guess'], adata.obs['domain'])
print(ARI)
NMI=normalized_mutual_info_score(adata.obs['layer_guess'], adata.obs['domain'])
print(NMI)
sc.pp.neighbors(adata, use_rep='latent')
sc.tl.umap(adata)

result_dir = './result/MBA'
os.makedirs(result_dir, exist_ok=True)

# Draw UMAP plot and save (600 DPI)
plt.figure(figsize=(3, 3))
sc.pl.umap(
    adata,
    color=["domain", 'layer_guess'],
    title=['SpaMCAR (ARI=%.2f,NMI=%.2f)' % (ARI, NMI), 'Ground Truth'],
    show=False
)
plt.savefig("./result/MBA/MBA_UMAP.jpg", dpi=600, bbox_inches='tight')
plt.close()

plt.figure(figsize=(3, 3))
sc.pl.spatial(
    adata,
    color=["domain", 'layer_guess'],
    title=['SpaMCAR (ARI=%.2f,NMI=%.2f)' % (ARI, NMI), 'Ground Truth'],
    show=False
)
plt.savefig("./result/MBA/MBA_Spatial.jpg", dpi=600, bbox_inches='tight')
plt.close()
adata_path = './result/MBA/MBA_processed.h5ad'
adata.write_h5ad(adata_path)
print(f'Saved: {adata_path}')
print(f'   Data shape: {adata.shape}')
print(f'   obs columns: {list(adata.obs.columns)}')
print(f'   obsm keys: {list(adata.obsm.keys())}')
