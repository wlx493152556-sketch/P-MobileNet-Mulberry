# Dataset Audit — Candidate Generation

Supplementary code for:

> *Trustworthy Evaluation for Mulberry Leaf Disease Recognition:
> Dataset Auditing and an Ultra-Lightweight Model*

Audit of the **Mulberry Leaf Disease** dataset (1091 images) for
duplicate / near-duplicate / invalid samples (Section 3 of the manuscript).

## Contents

| File | Description |
|------|-------------|
| `generate_candidates.py` | Generates candidate near-duplicate pairs and visual montages. |
| `candidates_reviewed.csv` | Candidate list after manual review; the `same` column records the human decision. |
| `README_audit.md` | This file. |

## Workflow

1. **Candidate generation** (`generate_candidates.py`) — three signals
   (SHA-256, pHash, self-supervised visual embeddings) produce one candidate
   list `candidates.csv` plus one montage per pair.
2. **Manual review** — each pair is judged from its montage; the decision is
   written to the `same` column, giving `candidates_reviewed.csv`.

Confirmed pairs are merged into duplicate groups (connected components), one
image is kept per group, and a final manual sweep catches any missed pairs.

## Requirements

```bash
pip install numpy pillow imagehash
```

## Inputs

Set at the top of `generate_candidates.py`:

```python
DATASET_PATH = "/openbayes/input/input0"   # <class>/ folders: 1091 images
CACHE_NPZ    = "embeddings_cache_v2.npz"   # pre-computed DINOv2 ViT-S/14 embeddings (L2-normalised)
```

If the embedding cache is absent, the script still runs but skips the
embedding-based signals (E3/E4).

## Usage

```bash
python generate_candidates.py
```

Outputs:

- `candidates.csv` — one row per candidate pair (within the same class), with
  columns `class, file_a, file_b, signal, confidence, phash, cos_sim, same`.
  Confidence 1–6, **smaller = more reliable**.
- `candidates_vis/<class>/` — one montage per pair, named in confidence order.

## Signals

| Signal | Rule | Conf. |
|--------|------|:-----:|
| `SHA` | byte-identical SHA-256 | 1 |
| `pHash=0` | Hamming distance = 0 | 2 |
| `pHash_1_7` | Hamming distance 1–7 | 3 |
| `pHash_8_13` | Hamming distance 8–13 | 4 |
| `E3_high` | cosine ≥ 0.99 | 4 |
| `E3` | 0.97 ≤ cosine < 0.99 | 5 |
| `E4` | 0.92 ≤ cosine < 0.97 and filename-contiguous / EXIF gap ≤ 20 s | 6 |

Only the most reliable signal is kept per pair.

## Result

Aggregating `candidates_reviewed.csv` reproduces Online Resource 1, Table S2:

| Signal | Pairs | Confirmed | Precision |
|--------|------:|----------:|----------:|
| SHA-256 | 17 | 17 | 100 % |
| pHash ≤ 7 | 2 | 2 | 100 % |
| pHash 8–13 | 22 | 12 | 54.5 % |
| Embedding, cos ≥ 0.97 | 53 | 40 | 75.5 % |
| Contiguous & cos ≥ 0.92 | 229 | 24 | 10.5 % |
| **Total** | **323** | **95** | **29.4 %** |

The low-precision signals (`pHash_8_13`, `E4`) are used only as candidates,
never as automatic decisions.

Final audit: **96 images removed (8.80 %)**; cleaned dataset = **995 images**.
See the manuscript (Table 1) and Online Resource 1 for details.

## Notes

- Images are enumerated with `sorted(glob(...))`; results are independent of
  file-system order.
- Stage 1 is deterministic given the same inputs; only the manual review is
  subjective, and `candidates_reviewed.csv` makes those decisions auditable.

## Licence

MIT. If you use this code, please cite the manuscript above.
