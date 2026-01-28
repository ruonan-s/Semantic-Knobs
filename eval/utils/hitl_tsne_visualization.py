#!/usr/bin/env python3
"""
HITL Refinement t-SNE Visualization

Visualizes the image embeddings from HITL refinement rounds.
Same color for each round to show how compositions evolve.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import re

import numpy as np
import torch
import clip
from PIL import Image
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Configure matplotlib
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'figure.dpi': 150,
})


def load_clip_model(device: str = None):
    """Load CLIP ViT-L/14 model."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading CLIP ViT-L/14 on {device}...")
    model, preprocess = clip.load("ViT-L/14", device=device)
    model.eval()
    
    return model, preprocess, device


def discover_hitl_images(hitl_folder: Path) -> List[Dict]:
    """
    Discover all HITL images and parse round/image info.
    
    Returns:
        List of dicts with keys: path, round, image_idx
    """
    images = []
    pattern = re.compile(r'round_(\d+)_img_(\d+)\.png')
    
    for img_path in sorted(hitl_folder.glob('round_*_img_*.png')):
        match = pattern.match(img_path.name)
        if match:
            round_num = int(match.group(1))
            img_idx = int(match.group(2))
            images.append({
                'path': img_path,
                'round': round_num,
                'image_idx': img_idx,
            })
    
    print(f"Found {len(images)} HITL images")
    return images


def compute_embeddings(images: List[Dict], model, preprocess, device: str) -> np.ndarray:
    """Compute CLIP embeddings for all images."""
    embeddings = []
    
    for i, img_info in enumerate(images):
        image = Image.open(img_info['path']).convert('RGB')
        image_input = preprocess(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            features = model.encode_image(image_input)
            features = features / features.norm(dim=-1, keepdim=True)
        
        embeddings.append(features.cpu().numpy()[0])
    
    return np.array(embeddings, dtype=np.float32)


def run_tsne(embeddings: np.ndarray, perplexity: int = 5, n_components: int = 2) -> np.ndarray:
    """Run t-SNE on embeddings."""
    n_samples = len(embeddings)
    adjusted_perplexity = min(perplexity, max(2, n_samples // 4))
    
    print(f"Running t-SNE with perplexity={adjusted_perplexity}, n_components={n_components}...")
    
    tsne = TSNE(
        n_components=n_components,
        perplexity=adjusted_perplexity,
        max_iter=1000,
        random_state=42,
        init='pca' if n_components <= 3 else 'random',
        learning_rate='auto'
    )
    
    return tsne.fit_transform(embeddings)


def plot_hitl_tsne_3d(
    coords_3d: np.ndarray,
    images: List[Dict],
    output_path: Path,
    session_name: str = "HITL Refinement"
):
    """
    Plot 3D t-SNE visualization with color per round.
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    rounds = np.array([img['round'] for img in images])
    unique_rounds = sorted(set(rounds))
    n_rounds = len(unique_rounds)
    
    # Color palette for rounds
    cmap = plt.colormaps.get_cmap('viridis')
    round_colors = {r: cmap(i / max(n_rounds - 1, 1)) for i, r in enumerate(unique_rounds)}
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot each round
    for round_num in unique_rounds:
        mask = rounds == round_num
        ax.scatter(
            coords_3d[mask, 0],
            coords_3d[mask, 1],
            coords_3d[mask, 2],
            c=[round_colors[round_num]],
            s=120,
            alpha=0.9,
            label=f'Round {round_num}',
            edgecolors='white',
            linewidths=0.5
        )
        
        # Add image index labels
        round_coords = coords_3d[mask]
        img_indices = [img['image_idx'] for img in images if img['round'] == round_num]
        for i, (x, y, z) in enumerate(round_coords):
            ax.text(x, y, z + 0.5, str(img_indices[i]), fontsize=8, ha='center')
    
    ax.set_title(f'{session_name}\nImage Embeddings by Round (3D)', fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE Dim 1')
    ax.set_ylabel('t-SNE Dim 2')
    ax.set_zlabel('t-SNE Dim 3')
    
    # Legend
    ax.legend(loc='upper right', title='Round')
    
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    plt.close()


def plot_hitl_tsne(
    coords_2d: np.ndarray,
    images: List[Dict],
    output_path: Path,
    session_name: str = "HITL Refinement"
):
    """
    Plot 2D t-SNE visualization with color per round.
    """
    rounds = np.array([img['round'] for img in images])
    unique_rounds = sorted(set(rounds))
    n_rounds = len(unique_rounds)
    
    # Color palette for rounds
    cmap = plt.colormaps.get_cmap('viridis')
    round_colors = {r: cmap(i / max(n_rounds - 1, 1)) for i, r in enumerate(unique_rounds)}
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot each round
    for round_num in unique_rounds:
        mask = rounds == round_num
        ax.scatter(
            coords_2d[mask, 0],
            coords_2d[mask, 1],
            c=[round_colors[round_num]],
            s=100,
            alpha=0.8,
            label=f'Round {round_num}',
            edgecolors='white',
            linewidths=0.5
        )
        
        # Add image index labels
        for i, (x, y) in enumerate(coords_2d[mask]):
            img_idx = [img['image_idx'] for img in images if img['round'] == round_num][i]
            ax.annotate(
                str(img_idx),
                (x, y),
                textcoords="offset points",
                xytext=(0, 5),
                ha='center',
                fontsize=8,
                color='black'
            )
    
    ax.set_title(f'{session_name}\nImage Embeddings by Round', fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    
    # Legend
    ax.legend(loc='upper right', title='Round')
    
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    plt.close()


def main(hitl_folder: str):
    """Main function."""
    hitl_path = Path(hitl_folder)
    
    if not hitl_path.exists():
        print(f"Error: Folder not found: {hitl_path}")
        sys.exit(1)
    
    # Get session name from parent folder
    session_name = hitl_path.parent.name
    
    print(f"Session: {session_name}")
    print(f"HITL folder: {hitl_path}")
    
    # Discover images
    images = discover_hitl_images(hitl_path)
    
    if not images:
        print("No images found!")
        sys.exit(1)
    
    # Load CLIP
    model, preprocess, device = load_clip_model()
    
    # Compute embeddings
    print("Computing embeddings...")
    embeddings = compute_embeddings(images, model, preprocess, device)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Run t-SNE 2D
    coords_2d = run_tsne(embeddings, n_components=2)
    output_path_2d = hitl_path / 'tsne_by_round.png'
    plot_hitl_tsne(coords_2d, images, output_path_2d, session_name)
    
    # Run t-SNE 3D
    coords_3d = run_tsne(embeddings, n_components=3)
    output_path_3d = hitl_path / 'tsne_by_round_3d.png'
    plot_hitl_tsne_3d(coords_3d, images, output_path_3d, session_name)
    
    print("Done!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python hitl_tsne_visualization.py <hitl_folder>")
        print("Example: python hitl_tsne_visualization.py /path/to/session/hitl")
        sys.exit(1)
    
    main(sys.argv[1])
