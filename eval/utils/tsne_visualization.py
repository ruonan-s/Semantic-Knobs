#!/usr/bin/env python3
"""
TSNE Visualization for Semantic Knobs Evaluation

Generates a 4-panel TSNE visualization comparing image embeddings across
4 methods (LLM-text, LLM-text+tags, LLM-text+images, Ours) for 3 descriptors
(Calm, Cozy, Inviting) using CLIP ViT-L/14 embeddings.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import glob as glob_module

import numpy as np
import torch
import clip
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
import pandas as pd

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

DESCRIPTORS = ['Calm', 'Cozy', 'Inviting']
METHODS = ['LLM-text', 'LLM-text+tags', 'LLM-text+images', 'Ours']

# 8 locations (with proper naming conventions)
LOCATIONS = [
    'Bathroom', 'Bedroom', 'Café', 'Home Office', 
    'Kitchen', 'Livingroom', 'Reading Nook', 'Restaurant'
]

# Mapping from space-separated to underscore-separated location names
LOCATION_TO_UNDERSCORE = {
    'Bathroom': 'Bathroom',
    'Bedroom': 'Bedroom',
    'Café': 'Café',
    'Home Office': 'Home_Office',
    'Kitchen': 'Kitchen',
    'Livingroom': 'Livingroom',
    'Reading Nook': 'Reading_Nook',
    'Restaurant': 'Restaurant',
}

# Handle case-insensitive bedroom folder (some use lowercase)
LOCATION_ALIASES = {
    'bedroom': 'Bedroom',
}

# Color palette for methods
METHOD_COLORS = {
    'LLM-text': '#4878d0',        # Blue
    'LLM-text+tags': '#ee854a',   # Orange  
    'LLM-text+images': '#6acc64', # Green
    'Ours': '#d65f5f'             # Red
}

# Marker shapes for descriptors (Panel a)
DESCRIPTOR_MARKERS = {
    'Cozy': 'o',      # Circle
    'Calm': '^',      # Triangle
    'Inviting': 's'   # Square
}

# Expected participant counts per descriptor
EXPECTED_PARTICIPANTS = {
    'Calm': 3,
    'Cozy': 6,
    'Inviting': 6,
}


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


# ============== Image Discovery ==============

def discover_participant_folders(base_dir: Path, descriptor: str) -> List[Path]:
    """Find all participant evaluation folders for a descriptor."""
    pattern = f"eval_P*_{descriptor}_*"
    descriptor_dir = base_dir / descriptor
    
    folders = sorted([
        p for p in descriptor_dir.iterdir()
        if p.is_dir() and p.name.startswith('eval_P')
    ])
    
    return folders


def find_image_path(folder: Path, location: str, method: str, descriptor: str, base_dir: Path) -> Optional[Path]:
    """
    Find the image path for a given method and location.
    
    Args:
        folder: Participant folder (for participant-specific methods) or base_dir for LLM-text
        location: Location name (with spaces, e.g., "Home Office")
        method: One of 'LLM-text', 'LLM-text+tags', 'LLM-text+images', 'Ours'
        descriptor: Descriptor name (e.g., 'Calm')
        base_dir: Base directory (eval/Eval1_descriptors)
    
    Returns:
        Path to the image file, or None if not found
    """
    location_underscore = LOCATION_TO_UNDERSCORE.get(location, location.replace(' ', '_'))
    
    if method == 'LLM-text':
        # Baseline is in baseline_generic_{descriptor}/{Location}/{Descriptor}_{Location}.png
        baseline_dir = base_dir / descriptor / f"baseline_generic_{descriptor}"
        # Location folder uses spaces
        location_folder = baseline_dir / location
        # Filename uses underscores
        location_filename = location.replace(' ', '_')
        image_path = location_folder / f"{descriptor}_{location_filename}.png"
        
        if image_path.exists():
            return image_path
        
        # Try with Café variations
        if 'Café' in location or 'Cafe' in location:
            for variant in ['Café', 'Cafe', 'café', 'cafe']:
                variant_folder = baseline_dir / variant
                for fname_loc in ['Café', 'Cafe', 'café', 'cafe']:
                    variant_path = variant_folder / f"{descriptor}_{fname_loc}.png"
                    if variant_path.exists():
                        return variant_path
        
        return None
    
    else:
        # Participant-specific images in eval_P{nn}_.../{Location_underscore}/
        # Try the expected underscore version first
        location_folder = folder / location_underscore
        
        # Handle case variations (e.g., "bedroom" vs "Bedroom")
        if not location_folder.exists():
            # Try lowercase version
            for subdir in folder.iterdir():
                if subdir.is_dir() and subdir.name.lower() == location_underscore.lower():
                    location_folder = subdir
                    break
        
        if not location_folder.exists():
            return None
        
        if method == 'LLM-text+tags':
            image_path = location_folder / 'llm_baseline_tags.png'
        elif method == 'LLM-text+images':
            image_path = location_folder / 'llm_style_transfer.png'
        elif method == 'Ours':
            # Find eval_alpha_1.00_*.png
            pattern = str(location_folder / 'eval_alpha_1.00_*.png')
            matches = glob_module.glob(pattern)
            if matches:
                image_path = Path(matches[0])
            else:
                return None
        else:
            return None
        
        return image_path if image_path.exists() else None


def discover_all_images(base_dir: Path) -> List[Dict]:
    """
    Discover all images for all descriptors, participants, locations, and methods.
    
    Returns:
        List of dicts with keys: path, descriptor, participant, location, method
    """
    all_images = []
    
    for descriptor in DESCRIPTORS:
        print(f"\nDiscovering images for {descriptor}...")
        participant_folders = discover_participant_folders(base_dir, descriptor)
        
        print(f"  Found {len(participant_folders)} participant folders")
        
        for folder in participant_folders:
            # Extract participant ID from folder name (e.g., "eval_P01_Calm_..." -> "P01")
            parts = folder.name.split('_')
            participant_id = parts[1] if len(parts) > 1 else "Unknown"
            
            for location in LOCATIONS:
                for method in METHODS:
                    image_path = find_image_path(folder, location, method, descriptor, base_dir)
                    
                    if image_path:
                        all_images.append({
                            'path': image_path,
                            'descriptor': descriptor,
                            'participant': participant_id,
                            'location': location,
                            'method': method,
                        })
                    else:
                        print(f"    Missing: {participant_id}/{location}/{method}")
    
    return all_images


# ============== Embedding Computation ==============

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


def compute_all_embeddings(
    images: List[Dict],
    model,
    preprocess,
    device: str,
    batch_size: int = 32
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Compute embeddings for all images.
    
    Returns:
        embeddings: (N, 768) array
        metadata: List of dicts with descriptor, participant, location, method
    """
    n_images = len(images)
    print(f"\nComputing embeddings for {n_images} images...")
    
    embeddings = []
    metadata = []
    
    for i, img_info in enumerate(images):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Processing {i + 1}/{n_images}...")
        
        try:
            emb = compute_image_embedding(img_info['path'], model, preprocess, device)
            embeddings.append(emb)
            metadata.append({
                'descriptor': img_info['descriptor'],
                'participant': img_info['participant'],
                'location': img_info['location'],
                'method': img_info['method'],
            })
        except Exception as e:
            print(f"  Error processing {img_info['path']}: {e}")
    
    embeddings = np.array(embeddings)
    print(f"  Computed {len(embeddings)} embeddings with shape {embeddings.shape}")
    
    return embeddings, metadata


# ============== TSNE Computation ==============

def run_tsne(
    embeddings: np.ndarray,
    perplexity: int = 30,
    max_iter: int = 1000,
    random_state: int = 42
) -> np.ndarray:
    """Run TSNE on embeddings."""
    # Adjust perplexity for small datasets
    n_samples = len(embeddings)
    adjusted_perplexity = min(perplexity, max(5, n_samples // 4))
    
    print(f"  Running TSNE with perplexity={adjusted_perplexity} on {n_samples} samples...")
    
    tsne = TSNE(
        n_components=2,
        perplexity=adjusted_perplexity,
        max_iter=max_iter,
        random_state=random_state,
        init='pca',
        learning_rate='auto'
    )
    
    coords_2d = tsne.fit_transform(embeddings)
    return coords_2d


# ============== Visualization ==============

def plot_confidence_ellipse(ax, x: np.ndarray, y: np.ndarray, color: str, alpha: float = 0.2):
    """Draw 95% confidence ellipse."""
    if len(x) < 3:
        return
    
    try:
        cov = np.cov(x, y)
        lambda_, v = np.linalg.eig(cov)
        lambda_ = np.sqrt(np.abs(lambda_))
        
        # 95% confidence = 1.96 standard deviations
        ellipse = Ellipse(
            xy=(np.mean(x), np.mean(y)),
            width=lambda_[0] * 2 * 1.96,
            height=lambda_[1] * 2 * 1.96,
            angle=np.degrees(np.arctan2(v[1, 0], v[0, 0])),
            facecolor=color,
            alpha=alpha,
            edgecolor=color,
            linewidth=2
        )
        ax.add_patch(ellipse)
    except Exception:
        pass  # Skip if ellipse computation fails


def create_4panel_figure(
    X_all_2d: np.ndarray,
    method_labels: np.ndarray,
    descriptor_labels: np.ndarray,
    tsne_results: Dict,
    output_dir: Path,
    show_ellipses: bool = True
):
    """Create the 4-panel TSNE figure."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # ========== Panel (a): All Combined ==========
    ax = axes[0]
    
    # Plot each descriptor-method combination
    for descriptor in DESCRIPTORS:
        for method in METHODS:
            mask = (descriptor_labels == descriptor) & (method_labels == method)
            if not mask.any():
                continue
            
            ax.scatter(
                X_all_2d[mask, 0], 
                X_all_2d[mask, 1],
                c=METHOD_COLORS[method],
                marker=DESCRIPTOR_MARKERS[descriptor],
                alpha=0.6,
                s=40,
                label=method if descriptor == DESCRIPTORS[0] else None
            )
    
    # Add method centroids
    for method in METHODS:
        mask = method_labels == method
        if mask.any():
            centroid = X_all_2d[mask].mean(axis=0)
            ax.scatter(
                centroid[0], centroid[1],
                c=METHOD_COLORS[method],
                s=200,
                edgecolor='black',
                linewidth=2,
                zorder=10,
                marker='X'
            )
            
            # Add confidence ellipse
            if show_ellipses:
                plot_confidence_ellipse(
                    ax, 
                    X_all_2d[mask, 0], 
                    X_all_2d[mask, 1],
                    METHOD_COLORS[method],
                    alpha=0.15
                )
    
    ax.set_title('(a) All Descriptors Combined', fontsize=12, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add shape legend for panel (a)
    shape_legend = [
        Line2D([0], [0], marker='o', color='gray', linestyle='', markersize=8, label='Cozy'),
        Line2D([0], [0], marker='^', color='gray', linestyle='', markersize=8, label='Calm'),
        Line2D([0], [0], marker='s', color='gray', linestyle='', markersize=8, label='Inviting'),
    ]
    ax.legend(handles=shape_legend, loc='upper right', fontsize=9, title='Descriptor')
    
    # ========== Panels (b), (c), (d): Per Descriptor ==========
    panel_labels = ['(b)', '(c)', '(d)']
    
    for idx, descriptor in enumerate(DESCRIPTORS):
        ax = axes[idx + 1]
        result = tsne_results[descriptor]
        
        for method in METHODS:
            mask = result['methods'] == method
            if not mask.any():
                continue
            
            ax.scatter(
                result['2d'][mask, 0],
                result['2d'][mask, 1],
                c=METHOD_COLORS[method],
                alpha=0.6,
                s=40,
                label=method
            )
        
        # Add method centroids
        for method in METHODS:
            mask = result['methods'] == method
            if mask.any():
                centroid = result['2d'][mask].mean(axis=0)
                ax.scatter(
                    centroid[0], centroid[1],
                    c=METHOD_COLORS[method],
                    s=200,
                    edgecolor='black',
                    linewidth=2,
                    zorder=10,
                    marker='X'
                )
                
                # Add confidence ellipse
                if show_ellipses:
                    plot_confidence_ellipse(
                        ax, 
                        result['2d'][mask, 0], 
                        result['2d'][mask, 1],
                        METHOD_COLORS[method],
                        alpha=0.15
                    )
        
        n_participants = EXPECTED_PARTICIPANTS[descriptor]
        ax.set_title(
            f'{panel_labels[idx]} {descriptor} (n={n_participants})', 
            fontsize=12, 
            fontweight='bold'
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # ========== Global Legend ==========
    # Create method color legend at bottom
    method_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=METHOD_COLORS[m], 
               markersize=10, label=m)
        for m in METHODS
    ]
    fig.legend(
        handles=method_handles,
        loc='lower center',
        ncol=4,
        fontsize=10,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02)
    )
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    
    # Save figures
    pdf_path = output_dir / 'tsne_4panel.pdf'
    png_path = output_dir / 'tsne_4panel.png'
    
    plt.savefig(pdf_path, bbox_inches='tight', dpi=300)
    plt.savefig(png_path, bbox_inches='tight', dpi=300)
    
    print(f"\nSaved: {pdf_path}")
    print(f"Saved: {png_path}")
    
    plt.close()


# ============== Metrics Computation ==============

def compute_metrics(
    embeddings: np.ndarray,
    method_labels: np.ndarray,
    descriptor_labels: np.ndarray
) -> Dict:
    """Compute quantitative metrics for the embeddings."""
    metrics = {}
    
    # Method indices for silhouette score
    method_to_idx = {m: i for i, m in enumerate(METHODS)}
    method_idx = np.array([method_to_idx[m] for m in method_labels])
    
    # Overall silhouette score
    if len(np.unique(method_idx)) > 1:
        metrics['overall_silhouette'] = silhouette_score(embeddings, method_idx)
    else:
        metrics['overall_silhouette'] = np.nan
    
    # Per-descriptor silhouette scores
    for descriptor in DESCRIPTORS:
        mask = descriptor_labels == descriptor
        if mask.sum() > 10 and len(np.unique(method_idx[mask])) > 1:
            metrics[f'{descriptor}_silhouette'] = silhouette_score(
                embeddings[mask], method_idx[mask]
            )
        else:
            metrics[f'{descriptor}_silhouette'] = np.nan
    
    # Method centroids (in embedding space)
    centroids = {}
    for method in METHODS:
        mask = method_labels == method
        if mask.any():
            centroids[method] = embeddings[mask].mean(axis=0)
    
    # Pairwise centroid distances
    if 'Ours' in centroids:
        for other in ['LLM-text', 'LLM-text+tags', 'LLM-text+images']:
            if other in centroids:
                dist = np.linalg.norm(centroids['Ours'] - centroids[other])
                metrics[f'ours_to_{other.replace("-", "_").replace("+", "_")}_distance'] = dist
    
    # Within-method variance
    for method in METHODS:
        mask = method_labels == method
        if mask.sum() > 1:
            distances = pdist(embeddings[mask])
            metrics[f'{method.replace("-", "_").replace("+", "_")}_within_variance'] = distances.mean()
        else:
            metrics[f'{method.replace("-", "_").replace("+", "_")}_within_variance'] = np.nan
    
    return metrics


def save_metrics(metrics: Dict, output_dir: Path):
    """Save metrics to CSV."""
    df = pd.DataFrame([metrics])
    csv_path = output_dir / 'tsne_metrics.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")


def save_coordinates(
    X_all_2d: np.ndarray,
    metadata: List[Dict],
    output_dir: Path
):
    """Save 2D coordinates for reproducibility."""
    data = []
    for i, meta in enumerate(metadata):
        data.append({
            'x': X_all_2d[i, 0],
            'y': X_all_2d[i, 1],
            'descriptor': meta['descriptor'],
            'participant': meta['participant'],
            'location': meta['location'],
            'method': meta['method'],
        })
    
    df = pd.DataFrame(data)
    csv_path = output_dir / 'tsne_coordinates.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")


# ============== Main ==============

def main():
    # Setup paths
    script_dir = Path(__file__).parent
    eval_dir = script_dir.parent
    base_dir = eval_dir / "Eval1_descriptors"
    output_dir = base_dir  # Save outputs in Eval1_descriptors
    
    print(f"Base directory: {base_dir}")
    print(f"Output directory: {output_dir}")
    
    if not base_dir.exists():
        print(f"Error: Base directory does not exist: {base_dir}")
        sys.exit(1)
    
    # Step 1: Discover all images
    print("\n" + "="*60)
    print("STEP 1: Discovering images")
    print("="*60)
    
    images = discover_all_images(base_dir)
    print(f"\nTotal images discovered: {len(images)}")
    
    # Summary by descriptor and method
    for descriptor in DESCRIPTORS:
        for method in METHODS:
            count = sum(1 for img in images if img['descriptor'] == descriptor and img['method'] == method)
            print(f"  {descriptor} / {method}: {count}")
    
    if len(images) == 0:
        print("Error: No images found!")
        sys.exit(1)
    
    # Step 2: Load CLIP and compute embeddings
    print("\n" + "="*60)
    print("STEP 2: Computing CLIP embeddings")
    print("="*60)
    
    model, preprocess, device = load_clip_model()
    embeddings, metadata = compute_all_embeddings(images, model, preprocess, device)
    
    # Convert metadata to arrays for easier indexing
    method_labels = np.array([m['method'] for m in metadata])
    descriptor_labels = np.array([m['descriptor'] for m in metadata])
    participant_labels = np.array([m['participant'] for m in metadata])
    
    # Step 3: Run TSNE
    print("\n" + "="*60)
    print("STEP 3: Running TSNE")
    print("="*60)
    
    # Panel (a): All data combined
    print("\nPanel (a): All descriptors combined")
    X_all_2d = run_tsne(embeddings, perplexity=30)
    
    # Panels (b), (c), (d): Per descriptor
    tsne_results = {}
    for descriptor in DESCRIPTORS:
        print(f"\nPanel for {descriptor}")
        mask = descriptor_labels == descriptor
        X_subset = embeddings[mask]
        
        coords_2d = run_tsne(X_subset, perplexity=30)
        
        tsne_results[descriptor] = {
            '2d': coords_2d,
            'methods': method_labels[mask],
            'participants': participant_labels[mask],
        }
    
    # Step 4: Create visualization
    print("\n" + "="*60)
    print("STEP 4: Creating visualization")
    print("="*60)
    
    create_4panel_figure(
        X_all_2d,
        method_labels,
        descriptor_labels,
        tsne_results,
        output_dir,
        show_ellipses=True
    )
    
    # Step 5: Compute and save metrics
    print("\n" + "="*60)
    print("STEP 5: Computing metrics")
    print("="*60)
    
    metrics = compute_metrics(embeddings, method_labels, descriptor_labels)
    
    print("\nMetrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    save_metrics(metrics, output_dir)
    save_coordinates(X_all_2d, metadata, output_dir)
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)


if __name__ == "__main__":
    main()
