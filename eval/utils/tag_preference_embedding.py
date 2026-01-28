#!/usr/bin/env python3
"""
Tag Preference Embedding Visualization

Visualizes tag preferences (positive, negative, neutral) in 2D or 3D embedding space
using CLIP text embeddings and TSNE dimensionality reduction.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import clip
from PIL import Image
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - required for 3D projection
import glob as glob_module

# Configure matplotlib for publication-quality figures
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
})

# ============== Constants ==============

# Default input path - modify this to run directly without command line arguments
DEFAULT_INPUT_PATH = "/home/nancy/Semantic-Knobs/eval/session_logs/eval_test_5_Calm_Home_Office_Sample_2026-01-27_19-20-14/impression/tag_preferences.json"

# Color palette for tag preference categories
PREFERENCE_COLORS = {
    'positive': '#6acc64',   # Green
    'negative': '#d65f5f',   # Red
    'neutral': '#a0a0a0',    # Gray
}

PREFERENCE_MARKERS = {
    'positive': 'o',   # Circle
    'negative': 'X',   # X
    'neutral': 's',    # Square
}

# Color and marker for image embeddings
IMAGE_COLORS = {
    'impression': '#9467bd',  # Purple
    'slider': '#17becf',      # Cyan
}

IMAGE_MARKER = 'D'  # Diamond for all images


# ============== CLIP Model Loading ==============

def load_clip_model(device: Optional[str] = None) -> Tuple:
    """Load CLIP ViT-L/14 model."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading CLIP ViT-L/14 on {device}...")
    model, preprocess = clip.load("ViT-L/14", device=device)
    model.eval()
    print("CLIP model loaded successfully")
    
    return model, preprocess, device


# ============== Text Embedding Computation ==============

def compute_text_embedding(
    text: str,
    model,
    device: str
) -> np.ndarray:
    """Compute CLIP embedding for a single text."""
    text_tokens = clip.tokenize([text]).to(device)
    
    with torch.no_grad():
        features = model.encode_text(text_tokens)
        # L2 normalize
        features = features / features.norm(dim=-1, keepdim=True)
    
    return features.cpu().numpy()[0].astype(np.float32)


def compute_all_text_embeddings(
    tags: List[str],
    model,
    device: str
) -> np.ndarray:
    """
    Compute embeddings for all tags.
    
    Returns:
        embeddings: (N, 768) array
    """
    n_tags = len(tags)
    print(f"\nComputing embeddings for {n_tags} tags...")
    
    embeddings = []
    
    for i, tag in enumerate(tags):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Processing {i + 1}/{n_tags}: '{tag}'")
        
        try:
            emb = compute_text_embedding(tag, model, device)
            embeddings.append(emb)
        except Exception as e:
            print(f"  Error processing '{tag}': {e}")
            # Use zero embedding as fallback
            embeddings.append(np.zeros(768, dtype=np.float32))
    
    embeddings = np.array(embeddings)
    print(f"  Computed {len(embeddings)} embeddings with shape {embeddings.shape}")
    
    return embeddings


# ============== Image Embedding Computation ==============

def compute_image_embedding(
    image_path: Path,
    model,
    preprocess,
    device: str
) -> np.ndarray:
    """Compute CLIP embedding for a single image."""
    image = Image.open(image_path).convert('RGB')
    image_input = preprocess(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        features = model.encode_image(image_input)
        # L2 normalize
        features = features / features.norm(dim=-1, keepdim=True)
    
    return features.cpu().numpy()[0].astype(np.float32)


def discover_images(session_dir: Path) -> Tuple[List[Path], List[str], List[Path], List[str]]:
    """
    Discover impression and slider images in a session directory.
    
    Args:
        session_dir: Path to the session directory (parent of 'impression' folder)
    
    Returns:
        impression_paths: List of paths to impression images
        impression_labels: List of labels for impression images
        slider_paths: List of paths to slider images
        slider_labels: List of labels for slider images
    """
    impression_paths = []
    impression_labels = []
    slider_paths = []
    slider_labels = []
    
    # Find impression images (impression_*_*.png)
    impression_dir = session_dir / 'impression'
    if impression_dir.exists():
        pattern = str(impression_dir / 'impression_*_*.png')
        for img_path in sorted(glob_module.glob(pattern)):
            impression_paths.append(Path(img_path))
            # Extract label from filename (e.g., "impression_0_0.png" -> "Impression 0")
            name = Path(img_path).stem
            parts = name.split('_')
            if len(parts) >= 2:
                impression_labels.append(f"Impression {parts[1]}")
            else:
                impression_labels.append(name)
    
    # Find slider images in Home_Office folder
    slider_dir = session_dir / 'slider' / 'Home_Office'
    if slider_dir.exists():
        for img_path in sorted(slider_dir.glob('*.png')):
            slider_paths.append(img_path)
            # Create readable label from filename
            name = img_path.stem
            if name.startswith('eval_alpha'):
                slider_labels.append('Ours')
            elif name == 'sd_baseline_prefs':
                slider_labels.append('SD Baseline Prefs')
            elif name == 'sd_baseline_tags':
                slider_labels.append('SD Baseline Tags')
            elif name == 'sd_baseline_text':
                slider_labels.append('SD Baseline Text')
            elif name == 'sd_style_transfer':
                slider_labels.append('SD Style Transfer')
            else:
                slider_labels.append(name)
    
    print(f"Discovered {len(impression_paths)} impression images")
    print(f"Discovered {len(slider_paths)} slider images")
    
    return impression_paths, impression_labels, slider_paths, slider_labels


def compute_all_image_embeddings(
    image_paths: List[Path],
    model,
    preprocess,
    device: str
) -> np.ndarray:
    """
    Compute embeddings for all images.
    
    Returns:
        embeddings: (N, 768) array
    """
    n_images = len(image_paths)
    if n_images == 0:
        return np.array([])
    
    print(f"\nComputing embeddings for {n_images} images...")
    
    embeddings = []
    
    for i, img_path in enumerate(image_paths):
        print(f"  Processing {i + 1}/{n_images}: '{img_path.name}'")
        
        try:
            emb = compute_image_embedding(img_path, model, preprocess, device)
            embeddings.append(emb)
        except Exception as e:
            print(f"  Error processing '{img_path}': {e}")
            embeddings.append(np.zeros(768, dtype=np.float32))
    
    embeddings = np.array(embeddings)
    print(f"  Computed {len(embeddings)} image embeddings")
    
    return embeddings


# ============== TSNE Computation ==============

def run_tsne(
    embeddings: np.ndarray,
    n_components: int = 2,
    perplexity: int = 30,
    max_iter: int = 1000,
    random_state: int = 42
) -> np.ndarray:
    """Run TSNE on embeddings.
    
    Args:
        embeddings: High-dimensional embeddings array
        n_components: Number of output dimensions (2 or 3)
        perplexity: TSNE perplexity parameter
        max_iter: Maximum iterations for optimization
        random_state: Random seed for reproducibility
    
    Returns:
        Reduced coordinates array of shape (N, n_components)
    """
    # Adjust perplexity for small datasets
    n_samples = len(embeddings)
    adjusted_perplexity = min(perplexity, max(5, n_samples // 4))
    
    print(f"  Running TSNE ({n_components}D) with perplexity={adjusted_perplexity} on {n_samples} samples...")
    
    tsne = TSNE(
        n_components=n_components,
        perplexity=adjusted_perplexity,
        max_iter=max_iter,
        random_state=random_state,
        init='pca',
        learning_rate='auto'
    )
    
    coords = tsne.fit_transform(embeddings)
    return coords


# ============== Visualization ==============

def plot_tag_preferences(
    coords_2d: np.ndarray,
    tags: List[str],
    categories: List[str],
    output_path: Path,
    title: str = "Tag Preference Embedding",
    show_labels: bool = True,
    impression_coords: Optional[np.ndarray] = None,
    impression_labels: Optional[List[str]] = None,
    slider_coords: Optional[np.ndarray] = None,
    slider_labels: Optional[List[str]] = None
):
    """
    Create scatter plot of tag preferences with category coloring.
    
    Args:
        coords_2d: (N, 2) array of 2D coordinates for tags
        tags: List of tag strings
        categories: List of category labels ('positive', 'negative', 'neutral')
        output_path: Path to save the figure
        title: Plot title
        show_labels: Whether to show tag labels on points
        impression_coords: (M, 2) array of 2D coordinates for impression images
        impression_labels: List of labels for impression images
        slider_coords: (K, 2) array of 2D coordinates for slider images
        slider_labels: List of labels for slider images
    """
    fig, ax = plt.subplots(figsize=(14, 11))
    
    # Plot each tag category
    for category in ['positive', 'negative', 'neutral']:
        mask = np.array([c == category for c in categories])
        if not mask.any():
            continue
        
        ax.scatter(
            coords_2d[mask, 0],
            coords_2d[mask, 1],
            c=PREFERENCE_COLORS[category],
            marker=PREFERENCE_MARKERS[category],
            s=100,
            alpha=0.7,
            label=f"Tag: {category.capitalize()}",
            edgecolors='white',
            linewidths=0.5
        )
    
    # Plot impression images
    if impression_coords is not None and len(impression_coords) > 0:
        ax.scatter(
            impression_coords[:, 0],
            impression_coords[:, 1],
            c=IMAGE_COLORS['impression'],
            marker=IMAGE_MARKER,
            s=200,
            alpha=0.9,
            label='Impression Images',
            edgecolors='black',
            linewidths=1.5
        )
        # Always show labels for impression images
        if impression_labels:
            for i, (x, y) in enumerate(impression_coords):
                ax.annotate(
                    impression_labels[i],
                    (x, y),
                    xytext=(8, 8),
                    textcoords='offset points',
                    fontsize=10,
                    fontweight='bold',
                    color=IMAGE_COLORS['impression'],
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
                )
    
    # Plot slider images
    if slider_coords is not None and len(slider_coords) > 0:
        ax.scatter(
            slider_coords[:, 0],
            slider_coords[:, 1],
            c=IMAGE_COLORS['slider'],
            marker=IMAGE_MARKER,
            s=200,
            alpha=0.9,
            label='Slider Images',
            edgecolors='black',
            linewidths=1.5
        )
        # Always show labels for slider images
        if slider_labels:
            for i, (x, y) in enumerate(slider_coords):
                ax.annotate(
                    slider_labels[i],
                    (x, y),
                    xytext=(8, -12),
                    textcoords='offset points',
                    fontsize=10,
                    fontweight='bold',
                    color=IMAGE_COLORS['slider'],
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
                )
    
    # Add tag labels if enabled
    if show_labels:
        for i, (x, y) in enumerate(coords_2d):
            ax.annotate(
                tags[i],
                (x, y),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                alpha=0.8,
                color=PREFERENCE_COLORS[categories[i]]
            )
    
    # Styling
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('TSNE Dimension 1', fontsize=12)
    ax.set_ylabel('TSNE Dimension 2', fontsize=12)
    
    # Legend - include image categories
    legend_handles = [
        Line2D([0], [0], marker=PREFERENCE_MARKERS[cat], color='w', 
               markerfacecolor=PREFERENCE_COLORS[cat], markersize=12, 
               label=f"Tag: {cat.capitalize()}")
        for cat in ['positive', 'negative', 'neutral']
    ]
    # Add image legend entries
    if impression_coords is not None and len(impression_coords) > 0:
        legend_handles.append(
            Line2D([0], [0], marker=IMAGE_MARKER, color='w',
                   markerfacecolor=IMAGE_COLORS['impression'], markersize=14,
                   markeredgecolor='black', markeredgewidth=1.5,
                   label='Impression Images')
        )
    if slider_coords is not None and len(slider_coords) > 0:
        legend_handles.append(
            Line2D([0], [0], marker=IMAGE_MARKER, color='w',
                   markerfacecolor=IMAGE_COLORS['slider'], markersize=14,
                   markeredgecolor='black', markeredgewidth=1.5,
                   label='Slider Images')
        )
    
    ax.legend(
        handles=legend_handles,
        loc='upper right',
        fontsize=10,
        title='Category',
        title_fontsize=11
    )
    
    # Grid and spines
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save figures
    png_path = output_path.with_suffix('.png')
    pdf_path = output_path.with_suffix('.pdf')
    
    plt.savefig(png_path, bbox_inches='tight', dpi=300)
    plt.savefig(pdf_path, bbox_inches='tight', dpi=300)
    
    print(f"\nSaved: {png_path}")
    print(f"Saved: {pdf_path}")
    
    plt.close()


def plot_tag_preferences_3d(
    coords_3d: np.ndarray,
    tags: List[str],
    categories: List[str],
    output_path: Path,
    title: str = "Tag Preference Embedding (3D)",
    show_labels: bool = True,
    impression_coords: Optional[np.ndarray] = None,
    impression_labels: Optional[List[str]] = None,
    slider_coords: Optional[np.ndarray] = None,
    slider_labels: Optional[List[str]] = None
):
    """
    Create 3D scatter plot of tag preferences with category coloring.
    
    Args:
        coords_3d: (N, 3) array of 3D coordinates for tags
        tags: List of tag strings
        categories: List of category labels ('positive', 'negative', 'neutral')
        output_path: Path to save the figure
        title: Plot title
        show_labels: Whether to show tag labels on points
        impression_coords: (M, 3) array of 3D coordinates for impression images
        impression_labels: List of labels for impression images
        slider_coords: (K, 3) array of 3D coordinates for slider images
        slider_labels: List of labels for slider images
    """
    fig = plt.figure(figsize=(14, 11))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot each tag category
    for category in ['positive', 'negative', 'neutral']:
        mask = np.array([c == category for c in categories])
        if not mask.any():
            continue
        
        ax.scatter(
            coords_3d[mask, 0],
            coords_3d[mask, 1],
            coords_3d[mask, 2],
            c=PREFERENCE_COLORS[category],
            marker=PREFERENCE_MARKERS[category],
            s=100,
            alpha=0.7,
            label=f"Tag: {category.capitalize()}",
            edgecolors='white',
            linewidths=0.5
        )
    
    # Plot impression images
    if impression_coords is not None and len(impression_coords) > 0:
        ax.scatter(
            impression_coords[:, 0],
            impression_coords[:, 1],
            impression_coords[:, 2],
            c=IMAGE_COLORS['impression'],
            marker=IMAGE_MARKER,
            s=200,
            alpha=0.9,
            label='Impression Images',
            edgecolors='black',
            linewidths=1.5
        )
        # Always show labels for impression images
        if impression_labels:
            for i, (x, y, z) in enumerate(impression_coords):
                ax.text(x, y, z, f"  {impression_labels[i]}", 
                       fontsize=10, fontweight='bold',
                       color=IMAGE_COLORS['impression'])
    
    # Plot slider images
    if slider_coords is not None and len(slider_coords) > 0:
        ax.scatter(
            slider_coords[:, 0],
            slider_coords[:, 1],
            slider_coords[:, 2],
            c=IMAGE_COLORS['slider'],
            marker=IMAGE_MARKER,
            s=200,
            alpha=0.9,
            label='Slider Images',
            edgecolors='black',
            linewidths=1.5
        )
        # Always show labels for slider images
        if slider_labels:
            for i, (x, y, z) in enumerate(slider_coords):
                ax.text(x, y, z, f"  {slider_labels[i]}",
                       fontsize=10, fontweight='bold',
                       color=IMAGE_COLORS['slider'])
    
    # Add tag labels if enabled
    if show_labels:
        for i, (x, y, z) in enumerate(coords_3d):
            ax.text(
                x, y, z,
                tags[i],
                fontsize=7,
                alpha=0.8,
                color=PREFERENCE_COLORS[categories[i]]
            )
    
    # Styling
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('TSNE Dimension 1', fontsize=11)
    ax.set_ylabel('TSNE Dimension 2', fontsize=11)
    ax.set_zlabel('TSNE Dimension 3', fontsize=11)
    
    # Legend - include image categories
    legend_handles = [
        Line2D([0], [0], marker=PREFERENCE_MARKERS[cat], color='w', 
               markerfacecolor=PREFERENCE_COLORS[cat], markersize=12, 
               label=f"Tag: {cat.capitalize()}")
        for cat in ['positive', 'negative', 'neutral']
    ]
    # Add image legend entries
    if impression_coords is not None and len(impression_coords) > 0:
        legend_handles.append(
            Line2D([0], [0], marker=IMAGE_MARKER, color='w',
                   markerfacecolor=IMAGE_COLORS['impression'], markersize=14,
                   markeredgecolor='black', markeredgewidth=1.5,
                   label='Impression Images')
        )
    if slider_coords is not None and len(slider_coords) > 0:
        legend_handles.append(
            Line2D([0], [0], marker=IMAGE_MARKER, color='w',
                   markerfacecolor=IMAGE_COLORS['slider'], markersize=14,
                   markeredgecolor='black', markeredgewidth=1.5,
                   label='Slider Images')
        )
    
    ax.legend(
        handles=legend_handles,
        loc='upper right',
        fontsize=10,
        title='Category',
        title_fontsize=11
    )
    
    # Adjust viewing angle for better visualization
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    
    # Save figures
    png_path = output_path.with_suffix('.png')
    pdf_path = output_path.with_suffix('.pdf')
    
    plt.savefig(png_path, bbox_inches='tight', dpi=300)
    plt.savefig(pdf_path, bbox_inches='tight', dpi=300)
    
    print(f"\nSaved: {png_path}")
    print(f"Saved: {pdf_path}")
    
    plt.close()


# ============== Data Loading ==============

def load_tag_preferences(json_path: Path) -> Tuple[List[str], List[str]]:
    """
    Load tag preferences from JSON file.
    
    Returns:
        tags: List of all tags
        categories: List of category labels for each tag
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    tags = []
    categories = []
    
    for category in ['positive', 'negative', 'neutral']:
        if category in data:
            for tag in data[category]:
                tags.append(tag)
                categories.append(category)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_tags = []
    unique_categories = []
    for tag, cat in zip(tags, categories):
        if tag.lower() not in seen:
            seen.add(tag.lower())
            unique_tags.append(tag)
            unique_categories.append(cat)
    
    print(f"Loaded {len(unique_tags)} unique tags:")
    print(f"  Positive: {sum(1 for c in unique_categories if c == 'positive')}")
    print(f"  Negative: {sum(1 for c in unique_categories if c == 'negative')}")
    print(f"  Neutral: {sum(1 for c in unique_categories if c == 'neutral')}")
    
    return unique_tags, unique_categories


# ============== Main ==============

def main():
    parser = argparse.ArgumentParser(
        description='Visualize tag preferences as 2D embeddings'
    )
    parser.add_argument(
        'input_json',
        type=str,
        nargs='?',
        default=None,
        help='Path to tag_preferences.json file (uses DEFAULT_INPUT_PATH if not provided)'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output path for visualization (default: same directory as input)'
    )
    parser.add_argument(
        '--no-labels',
        action='store_true',
        help='Disable tag labels on plot'
    )
    parser.add_argument(
        '--title',
        type=str,
        default=None,
        help='Custom title for the plot'
    )
    parser.add_argument(
        '--3d',
        dest='use_3d',
        action='store_true',
        help='Generate 3D visualization instead of 2D'
    )
    
    args = parser.parse_args()
    
    # Setup paths - use DEFAULT_INPUT_PATH if no argument provided
    if args.input_json:
        input_path = Path(args.input_json)
    else:
        # Resolve relative to workspace root (two levels up from this script)
        script_dir = Path(__file__).parent
        workspace_root = script_dir.parent.parent
        input_path = workspace_root / DEFAULT_INPUT_PATH
        print(f"Using default input path: {DEFAULT_INPUT_PATH}")
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / 'tag_preference_embedding'
    
    # Determine title
    if args.title:
        title = args.title
    else:
        # Extract session info from path if available
        parent_name = input_path.parent.parent.name if input_path.parent.name == 'impression' else input_path.parent.name
        title = f"Tag Preference Embedding\n{parent_name}"
    
    print("="*60)
    print("Tag Preference Embedding Visualization")
    print("="*60)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    
    # Determine session directory (parent of 'impression' folder)
    if input_path.parent.name == 'impression':
        session_dir = input_path.parent.parent
    else:
        session_dir = input_path.parent
    
    # Step 1: Load tag preferences
    print("\n" + "="*60)
    print("STEP 1: Loading tag preferences")
    print("="*60)
    
    tags, categories = load_tag_preferences(input_path)
    
    if len(tags) == 0:
        print("Error: No tags found in input file!")
        sys.exit(1)
    
    # Step 1b: Discover images
    print("\n" + "="*60)
    print("STEP 1b: Discovering images")
    print("="*60)
    
    impression_paths, impression_labels, slider_paths, slider_labels = discover_images(session_dir)
    
    # Step 2: Load CLIP and compute embeddings
    print("\n" + "="*60)
    print("STEP 2: Computing CLIP embeddings")
    print("="*60)
    
    model, preprocess, device = load_clip_model()
    
    # Compute text embeddings for tags
    print("\n--- Text embeddings (tags) ---")
    tag_embeddings = compute_all_text_embeddings(tags, model, device)
    
    # Compute image embeddings
    print("\n--- Image embeddings (impression) ---")
    impression_embeddings = compute_all_image_embeddings(impression_paths, model, preprocess, device)
    
    print("\n--- Image embeddings (slider) ---")
    slider_embeddings = compute_all_image_embeddings(slider_paths, model, preprocess, device)
    
    # Combine all embeddings for joint TSNE
    n_tags = len(tag_embeddings)
    n_impression = len(impression_embeddings) if len(impression_embeddings) > 0 else 0
    n_slider = len(slider_embeddings) if len(slider_embeddings) > 0 else 0
    
    embeddings_list = [tag_embeddings]
    if n_impression > 0:
        embeddings_list.append(impression_embeddings)
    if n_slider > 0:
        embeddings_list.append(slider_embeddings)
    
    all_embeddings = np.vstack(embeddings_list)
    print(f"\nTotal embeddings: {len(all_embeddings)} (tags: {n_tags}, impression: {n_impression}, slider: {n_slider})")
    
    # Step 3: Run TSNE
    print("\n" + "="*60)
    n_components = 3 if args.use_3d else 2
    print(f"STEP 3: Running TSNE ({n_components}D)")
    print("="*60)
    
    all_coords = run_tsne(all_embeddings, n_components=n_components, perplexity=15)
    
    # Split coordinates back
    tag_coords = all_coords[:n_tags]
    impression_coords = all_coords[n_tags:n_tags + n_impression] if n_impression > 0 else None
    slider_coords = all_coords[n_tags + n_impression:] if n_slider > 0 else None
    
    # Step 4: Create visualization
    print("\n" + "="*60)
    print(f"STEP 4: Creating {'3D' if args.use_3d else '2D'} visualization")
    print("="*60)
    
    if args.use_3d:
        plot_tag_preferences_3d(
            tag_coords,
            tags,
            categories,
            output_path,
            title=title,
            show_labels=not args.no_labels,
            impression_coords=impression_coords,
            impression_labels=impression_labels if n_impression > 0 else None,
            slider_coords=slider_coords,
            slider_labels=slider_labels if n_slider > 0 else None
        )
    else:
        plot_tag_preferences(
            tag_coords,
            tags,
            categories,
            output_path,
            title=title,
            show_labels=not args.no_labels,
            impression_coords=impression_coords,
            impression_labels=impression_labels if n_impression > 0 else None,
            slider_coords=slider_coords,
            slider_labels=slider_labels if n_slider > 0 else None
        )
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)


if __name__ == "__main__":
    main()
