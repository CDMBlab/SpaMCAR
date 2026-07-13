"""
GAAEST — Graph Autoencoder with Adversarial Edge Sampling for Spatial Transcriptomics
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

from GAAEST import GAAEST
from GAAEST.utils import Transfer_pytorch_Data


def mclust_R(adata, n_clusters, use_rep='latent', key_added='mclust', random_seed=0):
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")
    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    robjects.r['set.seed'](random_seed)
    res = robjects.r['Mclust'](adata.obsm[use_rep], n_clusters, 'EEE')
    adata.obs[key_added] = np.array(res[-2]).astype(int).astype(str).astype('category')
    return adata


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

    # Train
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = GAAEST(adata, device=device, epochs=600)
    adata = model.train()

    # Evaluate
    truth_path = data_root / f"{sample}_truth.txt"
    Ann_df = pd.read_csv(truth_path, sep='\t', header=None, index_col=0)
    Ann_df.columns = ['Ground Truth']
    adata.obs['Ground Truth'] = Ann_df.loc[adata.obs_names, 'Ground Truth']
    adata = adata[~pd.isnull(adata.obs['Ground Truth'])]
    ARI = adjusted_rand_score(adata.obs['Ground Truth'], adata.obs['mclust'])
    print(f"GAAEST {sample}: ARI = {ARI:.4f}")


if __name__ == '__main__':
    main()
