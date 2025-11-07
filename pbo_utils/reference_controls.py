
# pbo_min/generation/reference_controls.py
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image, ImageFilter
import numpy as np

try:
    import cv2  # optional for canny
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

try:
    from transformers import pipeline
    HAS_DEPTH_ESTIMATION = True
except Exception:
    HAS_DEPTH_ESTIMATION = False

def load_reference_image(path: Path):
    """
    Loads the reference image. In a ControlNet/IP-Adapter setup, you may want
    to compute edge/depth maps here. For now, we just return the PIL image.
    """
    img = Image.open(path).convert("RGB")
    return img

def load_reference_for_ip_adapter(path: Path, size: Optional[Tuple[int, int]] = None) -> Image.Image:
    """
    Load the RAW RGB reference strictly for IP-Adapter (never edges/canny).
    Optionally resize to the target render size to avoid aspect/scale artifacts.
    """
    img = load_reference_image(path)
    if size is not None:
        img = img.resize(size, Image.BILINEAR)
    return img

def preprocess_control_image(img: Image.Image, method: str = "canny", size: Optional[Tuple[int, int]] = None) -> Image.Image:
    """
    Prepare a control image from a reference image.
    - method="canny": uses OpenCV Canny if available, else PIL edge filter fallback.
    - method="edges": simple PIL edges.
    - method="depth": uses transformers depth estimation pipeline.
    Returns a single-channel or 3-channel PIL image suitable as control input.
    """
    method = (method or "canny").lower()
    if method == "depth":
        if HAS_DEPTH_ESTIMATION:
            try:
                from transformers import pipeline
                # Initialize depth estimator (caches the model)
                # Alternative models to reduce warnings:
                # - "Intel/dpt-hybrid-midas" (smaller, fewer warnings)
                # - "facebook/dpt-dinov2-small-kitti" (optimized)
                depth_estimator = pipeline('depth-estimation', model="Intel/dpt-large")
                
                # Generate depth map
                depth_result = depth_estimator(img)
                depth_img = depth_result["depth"]
                
                # Convert to RGB for ControlNet
                depth_array = np.array(depth_img)
                # Normalize to 0-255 range
                depth_normalized = ((depth_array - depth_array.min()) / (depth_array.max() - depth_array.min()) * 255).astype(np.uint8)
                depth_rgb = Image.fromarray(np.stack([depth_normalized] * 3, axis=-1))
                
                if size:
                    depth_rgb = depth_rgb.resize(size, Image.BILINEAR)
                return depth_rgb
            except Exception as e:
                print(f"[preprocess_control_image] Depth estimation failed: {e}. Falling back to edges.")
                # Fallback to edges if depth fails
                return preprocess_control_image(img, method="edges", size=size)
        else:
            print("[preprocess_control_image] Depth estimation not available. Falling back to edges.")
            return preprocess_control_image(img, method="edges", size=size)
    elif method == "canny":
        if HAS_CV2:
            arr = np.array(img.convert("RGB"))
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            # Pre-blur to reduce noise and small edges
            gray = cv2.GaussianBlur(gray, (5,5), 1.0)
            # Use thresholds that prefer fewer edges; adjust up if still noisy
            edges = cv2.Canny(gray, 100, 200)
            edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            out = Image.fromarray(edges_rgb)
            # Post-blur lightly to soften outlines
            out = out.filter(ImageFilter.GaussianBlur(radius=0.8))
            if size:
                out = out.resize(size, Image.BILINEAR)
            return out
        else:
            # PIL fallback: FIND_EDGES
            e = img.convert("L").filter(ImageFilter.FIND_EDGES)
            # Light blur to soften lines
            e = e.filter(ImageFilter.GaussianBlur(radius=1))
            out = Image.merge("RGB", (e, e, e))
            if size:
                out = out.resize(size, Image.BILINEAR)
            return out
    else:
        e = img.convert("L").filter(ImageFilter.FIND_EDGES)
        e = e.filter(ImageFilter.GaussianBlur(radius=1))
        out = Image.merge("RGB", (e, e, e))
        if size:
            out = out.resize(size, Image.BILINEAR)
        return out
