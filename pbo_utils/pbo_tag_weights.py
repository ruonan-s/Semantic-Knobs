
# pbo_min/loops/pbo_tag_weights.py
import numpy as np
from pathlib import Path
from typing import List, Dict

from tags.tag_prompt_builder import load_clusters, build_prompts
from tags.tag_phrase_builder import build_weighted_phrases
from embeddings.sdxl_embed_fuser import SDXLEmbedFuser
from generation.diffusion_runner import DiffusionRunner
from generation.reference_controls import load_reference_image, preprocess_control_image
from ui.chooser import save_round_images, ask_user_choice
from utils.simplex import normalize_simplex, split_concat_beta
from utils.prompt_recorder import create_prompt_data

# Expect an external weight GP implementation in project scope
try:
    from weight_gp import WeightPBO
except Exception as e:
    # Very simple fallback that samples uniformly
    class WeightPBO:
        def __init__(self, dim:int): self.dim=dim
        def add_preference(self, w_win, w_lose): pass
        def fit(self): pass
        def suggest(self, n:int=4, pool:int=128):
            S = np.random.dirichlet([1]*self.dim, size=pool)
            idx = np.random.choice(pool, size=n, replace=False)
            return S[idx]
        def best(self, pool:int=512):
            S = np.random.dirichlet([1]*self.dim, size=pool)
            return S[0]

GROUPS = ["impression","spatial","objects","ambient"]

def run_stage_A_alpha(session_dir: Path, clusters_path: Path, out_dir: Path,
                      model_id: str, seed: int, k: int, rounds: int,
                      height: int, width: int, steps: int, gscale: float):
    clusters = load_clusters(clusters_path)
    gp = WeightPBO(dim=4)
    from config import USE_CONTROLNET, CONTROLNET_MODEL_ID, CONTROL_METHOD, CONTROLNET_SCALE, REF_IMAGE_PATH, CONTROL_START, CONTROL_END, USE_IMG2IMG, IMG2IMG_STRENGTH_DRAFT, INIT_RESIZE_MODE, USE_PROMPT_EMBEDS
    runner = DiffusionRunner(model_id=model_id, height=height, width=width,
                             guidance_scale=gscale, steps=steps,
                             controlnet_model_id=(CONTROLNET_MODEL_ID if USE_CONTROLNET else None))
    ref_img = load_reference_image(Path(REF_IMAGE_PATH))
    control_img = preprocess_control_image(ref_img, method=CONTROL_METHOD, size=(width, height)) if USE_CONTROLNET else None
    
    # Initialize embedding fuser if needed
    fuser = None
    if USE_PROMPT_EMBEDS:
        if USE_IMG2IMG:
            runner._ensure_img2img()  # Ensure img2img pipeline is loaded
            if runner.pipe_i2i is not None:
                fuser = SDXLEmbedFuser(runner.pipe_i2i, device=runner.device)
        else:
            runner._ensure_txt2img()  # Ensure txt2img pipeline is loaded
            if runner.pipe is not None:
                fuser = SDXLEmbedFuser(runner.pipe, device=runner.device)

    # cold start: corners + center
    cold = np.vstack([np.eye(4), np.ones((1,4))/4])
    W = cold[:k]

    for r in range(rounds):
        print(f"\n[Stage A] Round {r+1}/{rounds}")
        print("=" * 50)
        
        # Create round directory
        round_dir = out_dir / f"round_{r:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        
        Ws = []
        prompt_data_list = []
        paths = []
        
        for i, w in enumerate(W):
            w = normalize_simplex(w)
            betas = {g: None for g in GROUPS}
            
            print(f"\n[Stage A] Generating candidate {i+1}/{k}...")
            print(f"Weights: {w}")
            
            # Initialize prompt data variables
            pos_phrases, neg_phrases = None, None
            pos_prompt, neg_prompt = None, None
            
            if USE_PROMPT_EMBEDS and fuser is not None:
                # Use embedding pipeline
                pos_phrases, neg_phrases = build_weighted_phrases(w, betas, clusters)
                if len(pos_phrases) > 0:
                    print(f"Selected phrases:")
                    for phrase, weight in pos_phrases:
                        print(f"  + {phrase} (weight: {weight:.3f})")
                    if neg_phrases:
                        print(f"Negative phrases: {', '.join(neg_phrases)}")
                    print(f"Generating with embeddings...")
                    prompt_embeds, pooled_prompt_embeds, neg_prompt_embeds, neg_pooled_prompt_embeds = fuser.fuse_weighted_phrases(pos_phrases, neg_phrases)
                    if USE_IMG2IMG:
                        img = runner.generate_embeds_img2img(
                            init_image=ref_img,
                            strength=IMG2IMG_STRENGTH_DRAFT,
                            prompt_embeds=prompt_embeds,
                            negative_prompt_embeds=neg_prompt_embeds,
                            pooled_prompt_embeds=pooled_prompt_embeds,
                            negative_pooled_prompt_embeds=neg_pooled_prompt_embeds,
                            seed=seed,
                            steps=steps, gscale=gscale,
                            height=height, width=width,
                            resize_mode=INIT_RESIZE_MODE,
                        )
                    else:
                        img = runner.generate_embeds(
                            prompt_embeds=prompt_embeds,
                            negative_prompt_embeds=neg_prompt_embeds,
                            pooled_prompt_embeds=pooled_prompt_embeds,
                            negative_pooled_prompt_embeds=neg_pooled_prompt_embeds,
                            control_image=control_img,
                            control_scale=CONTROLNET_SCALE,
                            seed=seed,
                            steps=steps, gscale=gscale,
                            height=height, width=width,
                        )
                else:
                    # Fallback to strings if no phrases
                    print(f"No phrases available, using string prompts...")
                    pos_prompt, neg_prompt = build_prompts(w, betas, clusters)
                    print(f"Positive prompt: {pos_prompt}")
                    print(f"Negative prompt: {neg_prompt}")
                    print(f"Generating with string prompts...")
                    if USE_IMG2IMG:
                        img = runner.generate_img2img(
                            init_image=ref_img,
                            strength=IMG2IMG_STRENGTH_DRAFT,
                            positive_prompt=pos_prompt,
                            negative_prompt=neg_prompt,
                            seed=seed,
                            steps=steps, gscale=gscale,
                            height=height, width=width,
                            resize_mode=INIT_RESIZE_MODE,
                        )
                    else:
                        img = runner.generate(
                            pos_prompt, neg_prompt,
                            control_image=control_img if control_img is not None else None,
                            control_scale=CONTROLNET_SCALE,
                            seed=seed,
                            steps=steps, gscale=gscale,
                            height=height, width=width,
                        )
            else:
                # Use string pipeline (original behavior)
                print(f"Using string prompts pipeline...")
                pos_prompt, neg_prompt = build_prompts(w, betas, clusters)
                print(f"Positive prompt: {pos_prompt}")
                print(f"Negative prompt: {neg_prompt}")
                print(f"Generating with string prompts...")
                if USE_IMG2IMG:
                    img = runner.generate_img2img(
                        init_image=ref_img,
                        strength=IMG2IMG_STRENGTH_DRAFT,
                        positive_prompt=pos_prompt,
                        negative_prompt=neg_prompt,
                        seed=seed,
                        steps=steps, gscale=gscale,
                        height=height, width=width,
                        resize_mode=INIT_RESIZE_MODE,
                    )
                else:
                    img = runner.generate(
                        pos_prompt, neg_prompt,
                        control_image=control_img if control_img is not None else None,
                        control_scale=CONTROLNET_SCALE,
                        seed=seed,
                        steps=steps, gscale=gscale,
                        height=height, width=width,
                    )
            
            # Save image immediately
            img_path = round_dir / f"candidate_{i}.png"
            img.save(img_path)
            paths.append(img_path)
            print(f"Saved: {img_path}")
            
            # Create prompt data for this candidate
            prompt_data = create_prompt_data(
                weights=w,
                betas=betas,
                clusters=clusters,
                use_prompt_embeds=USE_PROMPT_EMBEDS and fuser is not None and len(pos_phrases or []) > 0,
                pos_phrases=pos_phrases,
                neg_phrases=neg_phrases,
                pos_prompt=pos_prompt,
                neg_prompt=neg_prompt,
                seed=seed,
                stage="stage_a"
            )
            
            Ws.append(w)
            prompt_data_list.append(prompt_data)
            
            # Clear image from memory
            del img
        
        # Save prompt metadata for the round
        from utils.prompt_recorder import PromptRecorder
        recorder = PromptRecorder(out_dir)
        recorder.record_round_prompts(r, prompt_data_list)
        
        # Ask user for choice
        j = ask_user_choice(paths, round_dir)
        print(f"\n[Stage A] User selected candidate {j}")
        
        # Update Gaussian Process
        for i in range(k):
            if i != j:
                gp.add_preference(Ws[j], Ws[i])
        gp.fit()
        W = gp.suggest(n=k)
        
        print(f"[Stage A] Round {r+1} complete. Generated {k} candidates.")
    
    alpha_star = gp.best()
    return normalize_simplex(alpha_star)

def run_stage_B_beta(session_dir: Path, clusters_path: Path, out_dir: Path,
                     alpha_star: np.ndarray, model_id: str, seed: int, k: int, rounds: int,
                     height: int, width: int, steps: int, gscale: float, top_groups:int=2):
    clusters = load_clusters(clusters_path)
    # pick top groups by alpha
    order = np.argsort(-alpha_star)
    sel = [GROUPS[i] for i in order[:top_groups]]
    cluster_sizes = [len(clusters[g]["clusters"]) for g in sel]
    dim = sum(cluster_sizes)

    gp = WeightPBO(dim=dim)
    from config import USE_CONTROLNET, CONTROLNET_MODEL_ID, CONTROL_METHOD, CONTROLNET_SCALE, REF_IMAGE_PATH, CONTROL_START, CONTROL_END, USE_IMG2IMG, IMG2IMG_STRENGTH_DRAFT, INIT_RESIZE_MODE, USE_PROMPT_EMBEDS
    runner = DiffusionRunner(model_id=model_id, height=height, width=width,
                             guidance_scale=gscale, steps=steps,
                             controlnet_model_id=(CONTROLNET_MODEL_ID if USE_CONTROLNET else None))
    ref_img = load_reference_image(Path(REF_IMAGE_PATH))
    control_img = preprocess_control_image(ref_img, method=CONTROL_METHOD, size=(width, height)) if USE_CONTROLNET else None
    
    # Initialize embedding fuser if needed
    fuser = None
    if USE_PROMPT_EMBEDS:
        if USE_IMG2IMG:
            runner._ensure_img2img()  # Ensure img2img pipeline is loaded
            if runner.pipe_i2i is not None:
                fuser = SDXLEmbedFuser(runner.pipe_i2i, device=runner.device)
        else:
            runner._ensure_txt2img()  # Ensure txt2img pipeline is loaded
            if runner.pipe is not None:
                fuser = SDXLEmbedFuser(runner.pipe, device=runner.device)

    W = gp.suggest(n=k)
    for r in range(rounds):
        print(f"\n[Stage B] Round {r+1}/{rounds}")
        print("=" * 50)
        
        # Create round directory
        round_dir = out_dir / f"round_beta_{r:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        
        images, Ws, prompt_data_list = [], [], []
        paths = []
        
        for i, w in enumerate(W):
            betas_sel = split_concat_beta(w, cluster_sizes, sel)
            # For non-selected groups → None (uniform)
            betas = {g: (betas_sel[g] if g in betas_sel else None) for g in GROUPS}
            
            print(f"\n[Stage B] Generating candidate {i+1}/{k}...")
            print(f"Selected groups: {sel}")
            print(f"Beta weights: {betas_sel}")
            
            # Initialize prompt data variables
            pos_phrases, neg_phrases = None, None
            pos_prompt, neg_prompt = None, None
            
            if USE_PROMPT_EMBEDS and fuser is not None:
                # Use embedding pipeline
                pos_phrases, neg_phrases = build_weighted_phrases(alpha_star, betas, clusters)
                if len(pos_phrases) > 0:
                    print(f"Selected phrases:")
                    for phrase, weight in pos_phrases:
                        print(f"  + {phrase} (weight: {weight:.3f})")
                    if neg_phrases:
                        print(f"Negative phrases: {', '.join(neg_phrases)}")
                    print(f"Generating with embeddings...")
                    prompt_embeds, pooled_prompt_embeds, neg_prompt_embeds, neg_pooled_prompt_embeds = fuser.fuse_weighted_phrases(pos_phrases, neg_phrases)
                    if USE_IMG2IMG:
                        img = runner.generate_embeds_img2img(
                            init_image=ref_img,
                            strength=IMG2IMG_STRENGTH_DRAFT,
                            prompt_embeds=prompt_embeds,
                            negative_prompt_embeds=neg_prompt_embeds,
                            pooled_prompt_embeds=pooled_prompt_embeds,
                            negative_pooled_prompt_embeds=neg_pooled_prompt_embeds,
                            seed=seed,
                            steps=steps, gscale=gscale,
                            height=height, width=width,
                            resize_mode=INIT_RESIZE_MODE,
                        )
                    else:
                        img = runner.generate_embeds(
                            prompt_embeds=prompt_embeds,
                            negative_prompt_embeds=neg_prompt_embeds,
                            pooled_prompt_embeds=pooled_prompt_embeds,
                            negative_pooled_prompt_embeds=neg_pooled_prompt_embeds,
                            control_image=control_img,
                            control_scale=CONTROLNET_SCALE,
                            seed=seed,
                            steps=steps, gscale=gscale,
                            height=height, width=width,
                        )
                else:
                    # Fallback to strings if no phrases
                    print(f"No phrases available, using string prompts...")
                    pos_prompt, neg_prompt = build_prompts(alpha_star, betas, clusters)
                    print(f"Positive prompt: {pos_prompt}")
                    print(f"Negative prompt: {neg_prompt}")
                    print(f"Generating with string prompts...")
                    if USE_IMG2IMG:
                        img = runner.generate_img2img(
                            init_image=ref_img,
                            strength=IMG2IMG_STRENGTH_DRAFT,
                            positive_prompt=pos_prompt,
                            negative_prompt=neg_prompt,
                            seed=seed,
                            steps=steps, gscale=gscale,
                            height=height, width=width,
                            resize_mode=INIT_RESIZE_MODE,
                        )
                    else:
                        img = runner.generate(
                            pos_prompt, neg_prompt,
                            control_image=control_img if control_img is not None else None,
                            control_scale=CONTROLNET_SCALE,
                            seed=seed,
                            steps=steps, gscale=gscale,
                            height=height, width=width,
                        )
            else:
                # Use string pipeline (original behavior)
                print(f"Using string prompts pipeline...")
                pos_prompt, neg_prompt = build_prompts(alpha_star, betas, clusters)
                print(f"Positive prompt: {pos_prompt}")
                print(f"Negative prompt: {neg_prompt}")
                print(f"Generating with string prompts...")
                if USE_IMG2IMG:
                    img = runner.generate_img2img(
                        init_image=ref_img,
                        strength=IMG2IMG_STRENGTH_DRAFT,
                        positive_prompt=pos_prompt,
                        negative_prompt=neg_prompt,
                        seed=seed,
                        steps=steps, gscale=gscale,
                        height=height, width=width,
                        resize_mode=INIT_RESIZE_MODE,
                    )
                else:
                    img = runner.generate(
                        pos_prompt, neg_prompt,
                        control_image=control_img if control_img is not None else None,
                        control_scale=CONTROLNET_SCALE,
                        seed=seed,
                        steps=steps, gscale=gscale,
                        height=height, width=width,
                    )
            
            # Save image immediately
            img_path = round_dir / f"candidate_{i}.png"
            img.save(img_path)
            paths.append(img_path)
            print(f"Saved: {img_path}")
            
            # Create prompt data for this candidate
            prompt_data = create_prompt_data(
                weights=alpha_star,
                betas=betas,
                clusters=clusters,
                use_prompt_embeds=USE_PROMPT_EMBEDS and fuser is not None and len(pos_phrases or []) > 0,
                pos_phrases=pos_phrases,
                neg_phrases=neg_phrases,
                pos_prompt=pos_prompt,
                neg_prompt=neg_prompt,
                seed=seed,
                stage="stage_b"
            )
            
            images.append(img)
            Ws.append(w)
            prompt_data_list.append(prompt_data)
            
            # Clear image from memory
            del img
        
        # Save prompt metadata for the round
        from utils.prompt_recorder import PromptRecorder
        recorder = PromptRecorder(out_dir)
        recorder.record_round_prompts(r, prompt_data_list)
        
        # Ask user for choice
        j = ask_user_choice(paths, round_dir)
        print(f"\n[Stage B] User selected candidate {j}")
        
        # Update Gaussian Process
        for i in range(k):
            if i != j:
                gp.add_preference(Ws[j], Ws[i])
        gp.fit()
        W = gp.suggest(n=k)
        
        print(f"[Stage B] Round {r+1} complete. Generated {k} candidates.")

    # return best betas
    w_best = gp.best()
    betas_star = split_concat_beta(w_best, cluster_sizes, sel)
    for g in GROUPS:
        if g not in betas_star: betas_star[g] = None
    return sel, betas_star

