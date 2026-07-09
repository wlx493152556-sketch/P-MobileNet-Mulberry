import os
import matplotlib.pyplot as plt
from PIL import Image
import imagehash
from collections import defaultdict

def find_and_display_similar_images(image_dir, threshold=6):
    """
    Find and visualize groups of images with Hamming distance below the threshold.
    Displays the Hamming distance of each image relative to the first image (reference) in the group.
    """
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    image_paths = []
    
    # 1. Collect all image paths
    for root, _, files in os.walk(image_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() in valid_extensions:
                image_paths.append(os.path.join(root, file))

    if not image_paths:
        print(f"No images found in {image_dir}. Please check the path.")
        return

    print(f"Scan complete. Found {len(image_paths)} images. Computing pHash...")
    
    # 2. Compute pHash for all images
    hashes = {}
    for path in image_paths:
        try:
            with Image.open(path) as img:
                hashes[path] = imagehash.phash(img)
        except Exception as e:
            print(f"Failed to read image {os.path.basename(path)}: {e}")

    # 3. Compute Hamming distances and build adjacency graph
    print(f"Hash computation complete. Searching for image pairs with Hamming distance < {threshold}...")
    adj_list = defaultdict(list)
    paths_list = list(hashes.keys())

    for i in range(len(paths_list)):
        for j in range(i + 1, len(paths_list)):
            path1, path2 = paths_list[i], paths_list[j]
            if hashes[path1] - hashes[path2] < threshold:
                adj_list[path1].append(path2)
                adj_list[path2].append(path1)

    # 4. Find connected components (clusters) using DFS
    visited = set()
    groups = []

    for node in paths_list:
        if node not in visited and node in adj_list:
            group = []
            stack = [node]
            while stack:
                curr = stack.pop()
                if curr not in visited:
                    visited.add(curr)
                    group.append(curr)
                    stack.extend(adj_list[curr])
            if len(group) > 1:
                groups.append(group)

    # 5. Visualize with distance annotations
    print(f"\nMatching complete! Found {len(groups)} similar image groups.")
    
    for idx, group in enumerate(groups):
        print(f"\n--- Group {idx + 1}, {len(group)} similar images ---")
        
        max_display = 8
        display_group = group[:max_display]
        if len(group) > max_display:
            print(f"(Group contains too many images, showing first {max_display} only)")

        fig, axes = plt.subplots(1, len(display_group), figsize=(4 * len(display_group), 4))
        if len(display_group) == 1:
             axes = [axes]

        # Use the first image in the group as the reference for distance calculation
        base_path = display_group[0]
        base_hash = hashes[base_path]

        for ax, img_path in zip(axes, display_group):
            try:
                img = Image.open(img_path)
                ax.imshow(img)
                
                # Calculate Hamming distance from the reference image
                dist = hashes[img_path] - base_hash
                filename = os.path.basename(img_path)
                
                # Set title with distance information
                if img_path == base_path:
                    title_text = f"{filename}\n(Reference / Dist: 0)"
                    ax.set_title(title_text, fontsize=10, pad=10, color='blue', fontweight='bold')
                else:
                    title_text = f"{filename}\n(Dist from ref: {dist})"
                    ax.set_title(title_text, fontsize=10, pad=10)
                
                ax.axis('off')
            except Exception as e:
                ax.set_title("Load failed", color='red')
                ax.axis('off')
                
        plt.tight_layout()
        plt.show()

# =============== Run Entry ===============
dataset_path = "/path/to/your/dataset"  # Change this to your image directory
find_and_display_similar_images(dataset_path, threshold=14)  # Adjust threshold as needed