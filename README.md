# PE-MobileNet-Mulberry

Official reproducible resources for the paper:

> **Dual Auditing and Reproducible Baselines for Mulberry Disease Recognition:
> From Evaluation Bias to Edge Feasibility Validation**

*[Authors]*, 2025

---

## Overview

This repository provides the complete code, cleaned dataset (via Zenodo),
audit tools (pHash pipeline & DSRCT spreadsheet), and lightweight model
(PE‑MobileNet) described in the paper. It enables full reproduction of:

- **Dataset audit** (near‑duplicate removal, evaluation bias quantification)
- **Methodology audit** (discrete sample reverse consistency test on prior works)
- **Model training & evaluation** (ablation, comparison, CPU benchmarking)
- **Online diagnostic prototype** (Gradio web app)

---

## Repository Structure

```text
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
├── dsrct_tool/
│   ├── DSRCT_Tool.xlsx  # Excel tool for discrete sample reverse consistency test
│   └── README_DSRCT.md         # Quick start for the DSRCT tool
└── app/
    ├── my_gradio.py            # Online diagnostic prototype
    └── weights/                # Place pre-trained weights here
```

---

## Getting the Cleaned Dataset

The cleaned mulberry leaf dataset (1004 images, 512×512 px) is archived on **Zenodo** (CC0 1.0).

> **Review access**: [Click here to access the dataset (Zenodo Anonymous Link)](https://zenodo.org/records/21261479?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjE1ODg1YThkLTNmNmQtNGQ2Ny1hMjA5LWFkNzc3Y2NlNzc5MyIsImRhdGEiOnt9LCJyYW5kb20iOiJjYTcyOTAyOWI2NGQ5NTZjNzMyY2NmM2Y4N2IwMjU2MCJ9.coZnyyEeHTUu2tB23IT0Tj7VZEu1Rr0N3DnehcQWI7VmCJGkiqo70D-9A-FiuWdVgWqAtdbMCQ8_Bf2fNHvEHA)

> **Permanent DOI**: 10.5281/zenodo.21261479 *(will be activated upon publication)*

After downloading, extract the archive so that `data/cleaned_dataset/` contains
three sub‑folders: `Disease Free leaves/`, `Leaf Rust/`, `Leaf spot/`.
*(Note: Pre-trained weights for the demo are also included in this archive).*

---

## Installation

```bash
git clone <Anonymous-Repository-URL>
cd PE-MobileNet-Mulberry
pip install -r requirements.txt

```

*Note: PyTorch may need a custom install depending on your CUDA version.
See [pytorch.org](https://pytorch.org) for instructions.*

---

## Usage

### 1. Reproducing Model Experiments

All experiments use 5‑fold cross‑validation on a fixed 80/20 train‑test split,
as described in the paper.

* **Ablation study** (`t=3/6`, `PConv`, `ECA`, kernel sizes):
```bash
python models/ablation.py

```


* **Comparison with other lightweight models** (MobileNetV3, ShuffleNetV2, etc.):
```bash
python models/comparison.py

```


* **CPU inference benchmark** (single‑thread, BN‑fused):
```bash
python models/cpu_benchmark.py

```



All results are printed to the console and averaged over 5 folds.

### 2. Running the Dataset Audit

The pHash pipeline identifies near‑duplicate groups for manual verification:

```bash
python audit/phash_audit.py --image_dir /path/to/original/dataset --threshold 14

```

Adjust `threshold` to control sensitivity; the paper used `14`.
For detailed instructions, see `audit/README_audit.md`.

### 3. Using the DSRCT Tool

The Discrete Sample Reverse Consistency Test (DSRCT) spreadsheet
checks arithmetic self‑consistency of reported accuracy metrics.

* Open `dsrct_tool/DSRCT_Tool.xlsx`
* Fill in the four green cells (sample size, decimal places, model names, reported values)
* The tool instantly flags any inconsistency

See `dsrct_tool/README_DSRCT.md` for a quick guide.

### 4. Launching the Web Demo

A Gradio prototype is provided for live mulberry leaf disease diagnosis:

```bash
python app/my_gradio.py

```

Then open `http://localhost:7860` in your browser.
Place the pre‑trained weight file (e.g., `best_Proposed_(t=3+P+E)_fold4.pth`) inside
`app/weights/` before running.

---

## License

* **Code** (`models/`, `audit/`, `app/`, `dsrct_tool/`): [MIT License](https://www.google.com/search?q=LICENSE)
* **Dataset** (`data/`): [CC0 1.0 Universal](https://www.google.com/search?q=DATA_LICENSE)

---

## Citation

If you use these resources in your research, please cite our paper
(bibliographic details will be added upon publication):

> *To be updated after acceptance.*

---

## Contact

For questions or issues, please open an issue on this repository or contact the corresponding author.

```

```
