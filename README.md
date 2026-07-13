# SpaMCAR: Spatially Guided Multi-scale Comparative Alignment and Dual-Reconstruction Framework

## Introduction

Spatial transcriptomics enables high-throughput gene expression profiling while preserving in situ spatial context, making it a powerful tool for characterizing tissue spatial heterogeneity. As a key task, spatial domain identification aims to partition functionally coherent regions based on expression patterns and spatial neighborhood relationships. However, existing methods are still limited by insufficient exploitation of multiscale spatial information, inadequate preservation of topological structure, and unreliable contrastive supervision in complex boundary regions. To address these challenges, we propose SpaMCAR, a spatially guided multi-scale comparative alignment and dual-reconstruction framework. SpaMCAR integrates spatially guided feature enhancement, dual reconstruction of gene expression and spatial topology, and a local–global contrastive learning module with boundary-aware resampling and hard negative mining, enabling effective modeling of multi-level spatial dependencies while preserving structural consistency. Specifically, SpaMCAR fuses gene expression profiles, multi-scale spatial encodings, and local convolutional features to construct a unified multiscale representation capturing both molecular and structural information. A dual reconstruction mechanism is then introduced to jointly recover masked gene expression and spatial adjacency relationships, enforcing consistency in both feature space and graph topology. Finally, a boundary-aware contrastive learning strategy dynamically constructs reliable contrastive pairs by integrating local spatial consistency and global cross-cluster similarity, while mining hard negatives and adaptively refining ambiguous boundary regions, thereby enhancing discrimination across heterogeneous tissue domains. Extensive experiments on multiple public datasets demonstrate that SpaMCAR consistently improves spatial domain clustering accuracy and tissue structure reconstruction, and outperforms state-of-the-art methods in downstream biological interpretability.

## Datasets

| # | Dataset | Script | Config | Platform | Spots | Genes | Domains | Source |
|---|---------|--------|--------|----------|-------|-------|---------|--------|
| 1 | Human Breast Cancer (HBC) | `breast_cancer.py` | `Config/HBC.yaml` | 10x Visium | 3,798 | 36,601 | 20 | [10x Genomics](https://support.10xgenomics.com/spatial-gene-expression/datasets/1.1.0/V1_Breast_Cancer_Block_A_Section_1) |
| 2 | Human DLPFC | — | — | Visium | 151,676 | — | — | [spatialLIBD](https://github.com/LieberInstitute/spatialLIBD) |
| 3 | Mouse Hypothalamus | — | — | MERFISH | 5,488–5,926 | — | 8 | [Dryad](https://datadryad.org/stash/dataset/doi:10.5061/dryad.8t8s248) |
| 4 | Mouse Primary Visual Area | — | — | BaristaSeq | 4,489 | 79 | 7 | [SpaceTx](https://spacetx.github.io/data.html) |
| 5 | Mouse Embryo | — | — | Stereo-seq | — | — | >10 | [STOmics](https://db.cngb.org/stomics/mosta/) |
| 6 | Mouse Visual Cortex | — | — | STARmap | — | — | — | [Google Drive](https://drive.google.com/drive/folders/1I1nxheWlc2RXSdiv24dex3YRaEh780my?usp=sharing) |
| 7 | Mouse Brain Anterior (MBA) | `mouse brain anterior.py` | `Config/MBA.yaml` | 10x Visium | — | — | 52 | [10x Genomics](https://www.10xgenomics.com/datasets/mouse-brain-serial-section-1-sagittal-anterior-1-standard-1-1-0) |
| 8 | Mouse Brain Sagittal (MBS) | `MBS.PY` | `Config/MBS.yaml` | 10x Visium | — | — | 8 | [10x Genomics](https://www.10xgenomics.com/datasets/mouse-brain-serial-section-1-sagittal-posterior-1-standard-1-0-0) |

Additional: Human Melanoma dataset — [ScribbleDom](https://github.com/1alnoman/ScribbleDom/tree/master/preprocessed_data/cancers/Melanoma)

## Setup

```bash
pip install -r requirement.txt
```

## Quick Start

```bash
# Human breast cancer
python HBC.py

# Mouse brain anterior
python "MBA.py"

# Mouse brain sagittal
python MBS.PY
```

Each script loads its corresponding config from `Config/`, runs the full SpaMCAR pipeline (preprocessing → multi-scale feature extraction → graph construction → training → clustering), and saves results to `./result/`.

## Project Structure

```
├── breast_cancer.py          # HBC main training script
├── mouse brain anterior.py   # MBA main training script
├── MBS.PY                    # MBS main training script
├── SpaMCAR/                  # Core model package
│   ├── __init__.py
│   ├── Func.py               # Data preprocessing & graph construction
│   ├── MGAC.py               # Training loop & model orchestration
│   ├── Models.py             # SMGA / FusedSMGA model architectures
│   └── Utils.py              # Clustering metrics & label refinement
├── Config/                   # Dataset-specific configuration
│   ├── HBC.yaml
│   ├── MBA.yaml
│   └── MBS.yaml
└── result/                   # Output directory (created at runtime)
```

## License

This project is for research purposes. See the accompanying paper for details.
