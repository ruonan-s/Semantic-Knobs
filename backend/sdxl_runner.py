# backend/sdxl_runner.py
# Wrapper for SDXL generation from concept mixtures.

from typing import List, Dict, Any
import numpy as np
from PIL import Image

# Import utilities from pbo_utils
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pbo_utils.diffusion_runner import DiffusionRunner
from backend.sdxl_embed_fuser import SDXLEmbedFuser
from backend.sdxl_integration import concepts_to_sdxl_phrases
from backend.sdxl_config import get_stage_strength


class SDXLRunner:
    """
    Wrapper for SDXL generation from PBO concept mixtures.

    This class combines:
        - Concept-to-phrase conversion (with gain mapping)
        - Phrase-to-embedding fusion
        - SDXL image generation

    Usage:
        runner = SDXLRunner(model_id="stabilityai/stable-diffusion-xl-base-1.0")
        image = runner.generate_from_mixture(w, concepts, seed=42)
    """

    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
        device: str | None = None,
        height: int = 1024,
        width: int = 1024,
        steps: int = 30,
        guidance_scale: float = 7.5
    ):
        """
        Initialize SDXL runner.

        Args:
            model_id: HuggingFace model ID (default: SDXL base)
            device: Device to use (cuda/mps/cpu). If None, auto-detect.
            height: Default image height (default: 1024)
            width: Default image width (default: 1024)
            steps: Default number of inference steps (default: 30)
            guidance_scale: Default guidance scale (default: 7.5)
        """
        self.model_id = model_id
        self.device = device
        self.default_height = height
        self.default_width = width
        self.default_steps = steps
        self.default_guidance_scale = guidance_scale

        # Initialize diffusion runner
        print(f"[SDXLRunner] Initializing DiffusionRunner with model: {model_id}")
        self.runner = DiffusionRunner(
            model_id=model_id,
            device=device,
            height=height,
            width=width,
            steps=steps,
            guidance_scale=guidance_scale
        )

        # Load pipeline
        print(f"[SDXLRunner] Loading SDXL pipeline...")
        self.runner._ensure_txt2img()

        if self.runner.pipe is None:
            print("⚠️  Warning: SDXL pipeline failed to load. Will generate mock images.")
            self.fuser = None
        else:
            print(f"[SDXLRunner] Pipeline loaded successfully on device: {self.runner.device}")

            # Initialize embed fuser
            self.fuser = SDXLEmbedFuser(self.runner.pipe, device=self.runner.device)
            print(f"[SDXLRunner] SDXLEmbedFuser initialized")

    def generate_from_mixture(
        self,
        w: np.ndarray,
        concepts: List[Dict],
        seed: int = 42,
        height: int | None = None,
        width: int | None = None,
        steps: int | None = None,
        guidance_scale: float | None = None,
        top_k: int = 10,
        num_negatives: int = 5,
        verbose: bool = True,
        init_image: Image.Image | None = None,
        strength: float | None = None,
        stage: str | None = None,
        descriptor: str | None = None,
        tracker: Any | None = None,
        proposal_index: int | None = None,
        generated_image_path: str | None = None
    ) -> Image.Image:
        """
        Generate image from concept mixture.

        Args:
            w: Weight vector (K,)
            concepts: List of concept dicts with 'label' and 'centroid' fields
            seed: Random seed for generation
            height: Image height (default: use initialization value)
            width: Image width (default: use initialization value)
            steps: Number of inference steps (default: use initialization value)
            guidance_scale: Guidance scale (default: use initialization value)
            top_k: Number of positive phrases to include (default: 10)
                   Only top-K concepts by weight appear in positive prompt
            num_negatives: Number of negative phrases (default: 5)
                          Bottom-N concepts (w < uniform/2) appear in negative prompt
            verbose: Print phrase summary (default: True)
            init_image: Reference image for img2img (default: None for txt2img)
            strength: Denoising strength for img2img, 0-1 (default: use stage config)
                     Higher = more deviation from init_image
            stage: Stage name for loading strength from config (e.g., "impression", "spatial")
            descriptor: User description to prepend to concept phrases (e.g., "A comfortable space for reading")
            tracker: GenerationTracker instance for logging (optional)
            proposal_index: Index of this proposal for tracking (optional)
            generated_image_path: Path where image will be saved for tracking (optional)

        Returns:
            Generated PIL Image
        """
        # Use defaults if not specified
        height = height or self.default_height
        width = width or self.default_width
        steps = steps or self.default_steps
        guidance_scale = guidance_scale or self.default_guidance_scale
        
        # Get strength from config if not provided
        if strength is None:
            strength = get_stage_strength(stage) if stage else 0.75

        # Step 1: Convert to phrases
        if verbose:
            print(f"\n[SDXLRunner] Converting mixture to phrases (top_k={top_k}, num_negatives={num_negatives})...")

        pos_phrases, neg_phrases = concepts_to_sdxl_phrases(
            w=w,
            concepts=concepts,
            top_k=top_k,
            num_negatives=num_negatives
        )
        
        # Add global negative constraints (no humans in interior scenes)
        global_negatives = ["people", "person", "human", "man", "woman", "face", "body", "portrait"]
        neg_phrases = global_negatives + neg_phrases
        
        # Step 1.5: Prepend user descriptor if provided
        if descriptor:
            # Add descriptor as the first phrase with highest gain (1.5)
            # This ensures the user's intent is strongly represented
            pos_phrases = [(descriptor, 1.5)] + pos_phrases
            
            if verbose:
                print(f"\n[SDXLRunner] Added user descriptor: '{descriptor}'")

        if verbose:
            print(f"  Positive phrases ({len(pos_phrases)}):")
            for phrase, gain in pos_phrases:
                print(f"    {phrase}: gain={gain:.3f}")

            if neg_phrases:
                print(f"  Negative phrases ({len(neg_phrases)}):")
                for phrase in neg_phrases:
                    print(f"    {phrase}")
            else:
                print(f"  Negative phrases: (none)")

        # Step 2: Fuse embeddings
        if self.fuser is None:
            # Mock generation (pipeline failed to load)
            print("[SDXLRunner] Pipeline not available, generating mock image...")
            return self.runner._mock_image(
                positive_prompt=str(pos_phrases),
                negative_prompt=str(neg_phrases),
                seed=seed
            )

        if verbose:
            print(f"\n[SDXLRunner] Fusing weighted phrase embeddings...")

        prompt_embeds, pooled, neg_embeds, neg_pooled = self.fuser.fuse_weighted_phrases(
            pos_phrases=pos_phrases,
            neg_phrases=neg_phrases
        )

        # Step 3: Generate image
        if init_image is not None:
            # Use img2img with reference image
            if verbose:
                print(f"[SDXLRunner] Generating image (IMG2IMG mode, strength={strength}, seed={seed}, steps={steps}, guidance={guidance_scale})...")
                print(f"[SDXLRunner] Using reference image: {init_image.size}")

            image = self.runner.generate_embeds_img2img(
                init_image=init_image,
                strength=strength,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=neg_embeds,
                pooled_prompt_embeds=pooled,
                negative_pooled_prompt_embeds=neg_pooled,
                seed=seed,
                steps=steps,
                gscale=guidance_scale,
                height=height,
                width=width
            )
        else:
            # Use txt2img
            if verbose:
                print(f"[SDXLRunner] Generating image (TXT2IMG mode, seed={seed}, steps={steps}, guidance={guidance_scale})...")

            image = self.runner.generate_embeds(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=neg_embeds,
                pooled_prompt_embeds=pooled,
                negative_pooled_prompt_embeds=neg_pooled,
                seed=seed,
                steps=steps,
                gscale=guidance_scale,
                height=height,
                width=width
            )

        if verbose:
            print(f"[SDXLRunner] ✅ Image generated successfully")
        
        # Track generation if tracker provided
        if tracker is not None and proposal_index is not None:
            generation_params = {
                "strength": strength,
                "steps": steps,
                "guidance_scale": guidance_scale,
                "height": height,
                "width": width,
                "top_k": top_k,
                "num_negatives": num_negatives,
                "mode": "img2img" if init_image is not None else "txt2img"
            }
            
            tracker.add_proposal(
                proposal_index=proposal_index,
                w_raw=w,
                concepts=concepts,
                descriptor=descriptor,
                pos_phrases=pos_phrases,
                neg_phrases=neg_phrases,
                generated_image_path=generated_image_path or f"proposal_{proposal_index}.png",
                seed=seed,
                generation_params=generation_params
            )

        return image

    def generate_batch_from_proposals(
        self,
        proposals: List[np.ndarray],
        concepts: List[Dict],
        seed_base: int = 42,
        **kwargs
    ) -> List[Image.Image]:
        """
        Generate batch of images from multiple proposals.

        Args:
            proposals: List of weight vectors
            concepts: List of concept dicts
            seed_base: Base seed (each proposal gets seed_base + i)
            **kwargs: Additional arguments for generate_from_mixture()

        Returns:
            List of generated images
        """
        print(f"\n[SDXLRunner] Generating batch of {len(proposals)} images...")

        images = []
        for i, w in enumerate(proposals):
            print(f"\n--- Proposal {i+1}/{len(proposals)} ---")
            img = self.generate_from_mixture(
                w=w,
                concepts=concepts,
                seed=seed_base + i,
                **kwargs
            )
            images.append(img)

        print(f"\n[SDXLRunner] ✅ Batch generation complete ({len(images)} images)")
        return images
