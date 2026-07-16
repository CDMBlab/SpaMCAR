# SpaMCAR: A Masked Autoencoder with Multiscale Feature Enhancement and Boundary-Aware Contrastive Learning for Spatial Transcriptomics Analysis

## Introduction

Spatial transcriptomics is an emerging technology that enables spatially resolved measurement of gene expression profiles while preserving tissue spatial organization, providing insights into tissue heterogeneity. A central task in spatial transcriptomics analysis is spatial domain identification, which aims to partition tissues into biologically meaningful spatial regions by jointly leveraging gene expression patterns and spatial neighborhood relationships. However, existing methods remain limited in their ability to exploit multiscale spatial information, preserve spatial topology, and provide reliable contrastive supervision in complex boundary regions. To address these challenges, we propose SpaMCAR, a masked autoencoder with multiscale feature enhancement and boundary-aware contrastive learning for spatial transcriptomics analysis. SpaMCAR integrates gene expression profiles, multi-frequency positional encodings, and local convolutional features to construct enhanced representations that capture molecular and spatial structural information. Subsequently, a masked dual-reconstruction strategy is designed to jointly recover gene expression features and spatial adjacency structures, thereby enforcing consistency between feature representations and graph topology. Furthermore, SpaMCAR proposes a boundary-aware contrastive learning strategy that dynamically constructs informative contrastive pairs by leveraging local spatial consistency and global cross-cluster similarity, while incorporating hard negative mining and adaptive boundary resampling to improve boundary delineation in ambiguous tissue regions. Extensive experiments on multiple public datasets demonstrate that SpaMCAR consistently improves spatial domain identification accuracy and spatial topology preservation, outperforming state-of-the-art methods and providing enhanced biological interpretability for downstream analyses.

## Datasets

| Platform | Dataset | Tissue / Section | Spots | Genes | Domains | Source / Reference |
|---|---|---|---:|---:|---:|---|
| 10x Visium | Human Breast Cancer (HBC) | — | 3,798 | 36,601 | 20 | [10x Genomics](https://support.10xgenomics.com/spatial-gene-expression/datasets/1.1.0/V1_Breast_Cancer_Block_A_Section_1) |
| 10x Visium | Human Melanoma (HM) | — | 293 | 16,148 | — | [ScribbleDom](https://github.com/1alnoman/ScribbleDom/tree/master/preprocessed_data/cancers/Melanoma) |
| 10x Visium | Mouse Brain Anterior | — | 2,695 | 32,285 | 52 | [10x Genomics](https://www.10xgenomics.com/datasets/mouse-brain-serial-section-1-sagittal-anterior-1-standard-1-1-0) |
| 10x Visium | Mouse Brain Posterior | — | 3,353 | 31,053 | 8 | [10x Genomics](https://www.10xgenomics.com/datasets/mouse-brain-serial-section-1-sagittal-posterior-1-standard-1-0-0) |
| 10x Visium | Human DLPFC | 151507 | 4,221 | 36,601 | 6–8 (per section) | [spatialLIBD](https://github.com/LieberInstitute/spatialLIBD) |
| 10x Visium | Human DLPFC | 151508 | 4,381 | 36,601 | 6–8 (per section) | [spatialLIBD](https://github.com/LieberInstitute/spatialLIBD) |
| 10x Visium | Human DLPFC | 151509 | 4,788 | 36,601 | 6–8 (per section) | [spatialLIBD](https://github.com/LieberInstitute/spatialLIBD) |
| 10x Visium | Human DLPFC | 151510 | 4,595 | 36,601 | 6–8 (per section) | [spatialLIBD](https://github.com/LieberInstitute/spatialLIBD) |
| 10x Visium | Human DLPFC | 151669 | 3,636 | 36,601 | 6–8 (per section) | [spatialLIBD](https://github.com/LieberInstitute/spatialLIBD) |
| 10x Visium | Human DLPFC | 151670 | 3,484 | 36,601 | 6–8 (per section) | [spatialLIBD](https://github.com/LieberInstitute/spatialLIBD) |
| 10x Visium | Human DLPFC | 151671 | 4,093 | 36,601 | 6–8 (per section) | [spatialLIBD](https://github.com/LieberInstitute/spatialLIBD) |
| 10x Visium | Human DLPFC | 151672 | 3,888 | 36,601 | 6–8 (per section) | [spatialLIBD](https://github.com/LieberInstitute/spatialLIBD) |
| 10x Visium | Human DLPFC | 151673 | 3,611 | 36,601 | 6–8 (per section) | [spatialLIBD](https://github.com/LieberInstitute/spatialLIBD) |
| 10x Visium | Human DLPFC | 151674 | 3,635 | 36,601 | 6–8 (per section) | [spatialLIBD](https://github.com/LieberInstitute/spatialLIBD) |
| 10x Visium | Human DLPFC | 151675 | 3,566 | 36,601 | 6–8 (per section) | [spatialLIBD](https://github.com/LieberInstitute/spatialLIBD) |
| 10x Visium | Human DLPFC | 151676 | 3,431 | 36,601 | 6–8 (per section) | [spatialLIBD](https://github.com/LieberInstitute/spatialLIBD) |
| MERFISH | Mouse Hypothalamus | Bregma -0.045 | 488 | 1,558 | 8 | [Dryad](https://datadryad.org/stash/dataset/doi:10.5061/dryad.8t8s248) |
| MERFISH | Mouse Hypothalamus | Bregma -0.095 | 557 | 155 | 8 | [Dryad](https://datadryad.org/stash/dataset/doi:10.5061/dryad.8t8s248) |
| MERFISH | Mouse Hypothalamus | Bregma -0.145 | 926 | 155 | 8 | [Dryad](https://datadryad.org/stash/dataset/doi:10.5061/dryad.8t8s248) |
| MERFISH | Mouse Hypothalamus | Bregma -0.195 | 803 | 155 | 8 | [Dryad](https://datadryad.org/stash/dataset/doi:10.5061/dryad.8t8s248) |
| MERFISH | Mouse Hypothalamus | Bregma -0.245 | 543 | 155 | 8 | [Dryad](https://datadryad.org/stash/dataset/doi:10.5061/dryad.8t8s248) |
| BaristaSeq | Mouse Primary Visual Area | — | 4,489 | 797 | 7 | [SpaceTx](https://spacetx.github.io/data.html) |
| Stereo-seq | Mouse Embryo | E9.5 | 5,913 | 25,568 | >10 | [STOmics](https://db.cngb.org/stomics/mosta/) |
| Stereo-seq | Mouse Embryo | E10.5 | 18,408 | 25,201 | >10 | [STOmics](https://db.cngb.org/stomics/mosta/) |
| STARmap | Mouse Visual Cortex | — | 1,207 | 1,020 | — | [Google Drive](https://drive.google.com/drive/folders/1I1nxheWlc2RXSdiv24dex3YRaEh780my?usp=sharing) |

## Setup

```bash
pip install -r requirement.txt
```

## Quick Start

```bash
# Human breast cancer
python HBC.py
```

Each script loads its corresponding config from `Config/`, runs the full SpaMCAR pipeline (preprocessing → multi-scale feature extraction → graph construction → training → clustering), and saves results to `./result/`.

## Project Structure

```
├── HBC.py          # HBC main training script
├── SpaMCAR/                  # Core model package
│   ├── __init__.py
│   ├── Func.py               # Data preprocessing & graph construction
│   ├── MGAC.py               # Training loop & model orchestration
│   ├── Models.py             # SMGA / FusedSMGA model architectures
│   └── Utils.py              # Clustering metrics & label refinement
├── Config/                   # Dataset-specific configuration
│   ├── HBC.yaml
└── result/                   # Output directory (created at runtime)
```

## License

This project is for research purposes. See the accompanying paper for details.
