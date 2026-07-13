"""
SpaGRA — Spatial Graph Relation Analysis for Spatial Transcriptomics
Usage: python RUN_Train.py
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from pathlib import Path
from sklearn.metrics import adjusted_rand_score

from SpaGRA.train_model import train as SpaGRA_train
from SpaGRA.utils import refine_label


def mclust_R(adata, n_clusters, use_rep='SpaGRA', key_added='mclust', random_seed=0):
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")
    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    robjects.r['set.seed'](random_seed)
    res = robjects.r['Mclust'](adata.obsm[use_rep], n_clusters, 'EEE')
    adata.obs[key_added] = np.array(res[-2]).astype(int).astype(str).astype('category')
    return adata


def Cal_Spatial_Net(adata, rad_cutoff=150):
    """Build spatial neighbor network."""
    from scipy.spatial import distance_matrix
    coords = adata.obsm['spatial']
    dist = distance_matrix(coords, coords)
    adj = dist < rad_cutoff
    np.fill_diagonal(adj, 0)
    adata.uns['Spatial_Net'] = adj


def main():
    # Config
    sample = "151673"
    n_clusters = 7
    data_root = Path("../data/DLPFC") / sample
    count_file = f"{sample}_filtered_feature_bc_matrix.h5"

    # Load & preprocess
    adata = sc.read_visium(data_root, count_file=count_file)
    adata.var_names_make_unique()
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Build spatial network
    Cal_Spatial_Net(adata, rad_cutoff=150)

    # Train
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    adata = SpaGRA_train(adata, n_epochs=500, device=device, key_added='SpaGRA', random_seed=0)
    adata = mclust_R(adata, n_clusters, use_rep='SpaGRA')

    # Evaluate
    truth_path = data_root / f"{sample}_truth.txt"
    Ann_df = pd.read_csv(truth_path, sep='\t', header=None, index_col=0)
    Ann_df.columns = ['Ground Truth']
    adata.obs['Ground Truth'] = Ann_df.loc[adata.obs_names, 'Ground Truth']
    adata = adata[~pd.isnull(adata.obs['Ground Truth'])]
    ARI = adjusted_rand_score(adata.obs['Ground Truth'], adata.obs['mclust'])
    print(f"SpaGRA {sample}: ARI = {ARI:.4f}")


if __name__ == '__main__':
    main()
