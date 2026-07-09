# pHash Image Similarity Auditor

A Python-based utility to scan a directory for duplicate or highly similar images using **Perceptual Hashing (pHash)**. It automatically groups visually similar images into clusters based on a customizable Hamming distance threshold and visualizes them side-by-side in a Jupyter/Kaggle Notebook environment.

## Features

- [cite_start]**Recursive Image Scanning**: Automatically scans the target directory and its subdirectories for common image formats (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`).
- [cite_start]**Perceptual Hashing (pHash)**: Uses `imagehash.phash` to generate robust hashes that are resilient to minor changes like resizing or compression.
- [cite_start]**Graph-Based Clustering**: Uses an optimized Disjoint/Connected Components DFS algorithm to cluster all images that match the similarity criteria[cite: 4, 5, 6].
- [cite_start]**Notebook Visualization**: Displays similar image groups side-by-side using `matplotlib`, highlighting the exact Hamming distance from a reference image for quick auditing[cite: 8, 9, 11].

## Prerequisites

Ensure you have the required dependencies installed:

```bash
pip install pillow imagehash matplotlib


Usage
1.Open your Python script or Notebook environment.
2.Import the function and specify your dataset path and preferred Hamming distance threshold.


from phash_audit import find_and_display_similar_images

# Path to your image dataset
dataset_path = "/path/to/your/image/dataset"

# Run the auditor 
# (Lower threshold = stricter matching; threshold=0 finds exact duplicates)
find_and_display_similar_images(dataset_path, threshold=14)

How It Works
1.Hash Generation: The script reads each image and computes its 64-bit perceptual hash.  
2.Distance Calculation: It compares the Hamming distance between all image pairs. If the distance is below the threshold, they are considered "similar".  
3.Clustering: Similar pairs are linked together into a graph structure, and Connected Components (DFS) are used to group them into final clusters.  
4.Result Presentation: For each group, the first image is picked as the Reference (Dist: 0). All other images in the group are displayed with their relative distance from this reference.  