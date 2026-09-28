# PE-MobileNet-Mulberry

Official reproducible resources for the paper:
Trustworthy Evaluation for Mulberry Leaf Disease Recognition: Dataset Auditing and an Ultra-Lightweight Model

Lingxiao Weng, 2026

## Overview

This repository provides the complete code, cleaned dataset (via Zenodo), dataset
audit scripts, and the lightweight model (PE-MobileNet) described in the paper.
It enables full reproduction of:

- Dataset audit (near-duplicate detection and removal; evaluation bias quantification)
- Model training & evaluation (ablation, comparison, CPU benchmarking)
- Online diagnostic prototype (Gradio web app)

## Repository Structure

PE-MobileNet-Mulberry/
├── README.md
├── LICENSE                     # MIT (code)
├── DATA_LICENSE                # CC0 1.0 (data)
├── requirements.txt
├── data/
│   ├── README_data.md          # How to obtain the cleaned dataset
│   └── removed_samples.csv     # List of 87 removed images with reasons
├── models/
│   ├── ablation.py             # Ablation study training & evaluation
│   ├── comparison.py           # Comparison with other lightweight models
│   └── cpu_benchmark.py        # Single-thread CPU inference benchmark
├── audit/
│   ├── phash_audit.py          # pHash-based near-duplicate detection
│   └── README_audit.md         # Instructions for the audit pipeline
└── app/
    ├── my_gradio.py            # Online diagnostic prototype
    └── weights/                # Place pre-trained weights here

## Getting the Cleaned Dataset

The cleaned mulberry leaf dataset (1004 images) is archived on Zenodo (CC0 1.0).

- Permanent DOI: **https://doi.org/10.5281/zenodo.21261479**

After downloading, extract the archive so that `data/cleaned_dataset/` contains
three sub-folders: `Disease Free leaves/`, `Leaf Rust/`, `Leaf spot/`.
(Pre-trained weights for the demo are also included in this archive.)

## Installation

git clone https://github.com/<your-account>/PE-MobileNet-Mulberry.git
cd PE-MobileNet-Mulberry
pip install -r requirements.txt

Note: PyTorch may need a custom install depending on your CUDA version.
See pytorch.org for instructions.

## Usage

### 1. Reproducing Model Experiments

All experiments use 5-fold cross-validation on a fixed 80/20 train-test split,
as described in the paper.

- Ablation study (t=3/6, PConv, ECA, kernel sizes):
  `python models/ablation.py`
- Comparison with other lightweight models (MobileNetV3, ShuffleNetV2, etc.):
  `python models/comparison.py`
- CPU inference benchmark (single-thread, BN-fused):
  `python models/cpu_benchmark.py`

All results are printed to the console and averaged over 5 folds.

### 2. Running the Dataset Audit

The pHash pipeline identifies near-duplicate groups for manual verification:

python audit/phash_audit.py --image_dir /path/to/original/dataset --threshold 14

Adjust threshold to control sensitivity; the paper used 14.
For detailed instructions, see `audit/README_audit.md`.

### 3. Launching the Web Demo

A Gradio prototype is provided for live mulberry leaf disease diagnosis:

python app/my_gradio.py

Then open http://localhost:7860 in your browser. Place the pre-trained weight file
(e.g., `best_Proposed_(t=3+P+E)_fold4.pth`) inside `app/weights/` before running.

## License

- Code (`models/`, `audit/`, `app/`): MIT License
- Dataset (`data/`): CC0 1.0 Universal

## Citation

If you use these resources in your research, please cite our paper
(bibliographic details will be added upon publication):

To be updated after acceptance.

## Contact

For questions or issues, please open an issue on this repository or contact
the corresponding author.
