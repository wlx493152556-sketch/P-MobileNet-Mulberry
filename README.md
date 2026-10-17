# P-MobileNet-Mulberry

Official reproducible resources for the paper:

**Trustworthy Evaluation for Mulberry Leaf Disease Recognition: Dataset Auditing and an Ultra-Lightweight Model**

Lingxiao Weng, 2026

## Overview

This repository provides the complete code, cleaned dataset (via Zenodo), the dataset-audit candidate-generation
script, and the ultra-lightweight model (**P-MobileNet**) described in the
paper. It enables full reproduction of:

- Dataset audit (near-duplicate detection and removal; evaluation bias quantification)
- Model training & evaluation (ablation, comparison, CPU benchmarking)
- Online diagnostic prototype (Gradio web app)

## Repository Structure

```
P-MobileNet-Mulberry/
├── README.md
├── LICENSE                     # MIT (code and weights)
├── DATA_LICENSE                # Cleaned dataset (Zenodo): CC0 1.0
├── requirements.txt
├── data/
│   ├── README_data.md          # How to obtain the cleaned dataset
│   └── removed_image_index.csv     # List of 96 removed images with reasons
├── models/
│   ├── ablation.py             # Ablation study training & evaluation
│   ├── comparison.py           # Comparison with other lightweight models
│   └── cpu_benchmark.py        # Single-thread CPU inference benchmark
├── audit/
│   ├── generate_candidates.py      # three-signal candidate generation (SHA + pHash + DINOv2)
│   ├── embeddings_dinov2_vits14_orig1091.npz   # DINOv2 ViT-S/14 features (orig 1091 images)
│   ├── candidates_reviewed.csv     # manually reviewed candidate pairs (y/n)
│   └── README_audit.md
└── app/
    ├── my_gradio.py            # Online diagnostic prototype
    └── weights/
        └── best_t=3+PConv_fold1.pth   # Pre-trained P-MobileNet weights (MIT)
```

## Getting the Cleaned Dataset

The cleaned mulberry leaf dataset (995 images) is archived on Zenodo (CC0 1.0).

- Permanent DOI: **https://doi.org/10.5281/zenodo.21261479**

After downloading, extract the archive so that `data` contains
three sub-folders: `Disease Free leaves/`, `Leaf Rust/`, `Leaf spot/`.

The audit pipeline is applied to the **original** dataset released by Nahiduzzaman
et al. (2023), which is publicly available on Kaggle and is **not** redistributed
here. Please obtain it from the original source; see reference [1] of the paper.

Pre-trained weights are **not** included in the Zenodo archive; they are provided in this repository under app/weights/.

## Installation

```bash
git clone https://github.com/wlx493152556-sketch/P-MobileNet-Mulberry.git
cd P-MobileNet-Mulberry
pip install -r requirements.txt
```

Note: PyTorch may need a custom install depending on your CUDA version.
See pytorch.org for instructions.

## Usage

### 1. Reproducing Model Experiments

All experiments use 5-fold cross-validation on a fixed 80/20 train-test split,
as described in the paper.

- Ablation study (t=3/6, PConv, ECA):
  `python models/ablation.py`
- Comparison with other lightweight models (MobileNetV2, MobileNetV3-Small,
  ShuffleNetV2, GhostNet, FasterNet-T0, MobileViT-XXS, MobileViTv2, ResNet18):
  `python models/comparison.py`
- CPU inference benchmark (single-thread, BN-fused):
  `python models/cpu_benchmark.py`

All results are printed to the console and averaged over 5 folds.

### 2. Running the Dataset Audit

the candidate-generation pipeline identifies near-duplicate groups for manual verification:

```bash
python audit/generate_candidates.py
```

The embedding-based signals use a pre-computed feature cache
(audit/embeddings_dinov2_vits14_orig1091.npz, ~1.8 MB) extracted with
DINOv2 ViT-S/14 from the 1091 original images (resized to 256×256,
L2-normalised, float32). It is included so that the candidate list can be
reproduced exactly.

The threshold controls sensitivity; the paper used 14. The final list of 96
removed images is provided in `data/removed_image_index.csv`. For detailed
instructions, see `audit/README_audit.md`.

### 3. External Validation (Apple Dataset)
The external validation in Section 4.5 uses the same training/evaluation
logic on the three-class apple leaf dataset (Plant Pathology 2020). Because
the class folder names differ from the mulberry dataset, two places must
be edited in the training script:
1.Config.DATA_DIR → the apple image root;
2.the classes = [...] list → ['healthy', 'rust', 'scab'] (matching the actual folder names).
Then run the script as usual. The apple experiments in the paper were
executed with the same protocol; results are the ablation (Table 7) and
comparison (Table 8) reported in Section 4.5.

### 4. Launching the Web Demo

A Gradio prototype is provided for live mulberry leaf disease diagnosis:

```bash
python app/my_gradio.py
```

The pre-trained weight (app/weights/best_t=3+PConv_fold1.pth) is included
in this repository. Run the demo with python app/my_gradio.py and open
http://localhost:7860.

## License

- Code (`models/`, `audit/`, `app/`): MIT License
- Pre-trained weights (`app/weights/`): MIT License
- Cleaned dataset (hosted on Zenodo): CC0 1.0

## Citation

If you use these resources in your research, please cite our paper
(bibliographic details will be added upon publication):

To be updated after acceptance.

## Contact

For questions or issues, please open an issue on this repository or contact
the corresponding author.
