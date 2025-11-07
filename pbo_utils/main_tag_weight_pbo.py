
# pbo_min/main_tag_weight_pbo.py
from pathlib import Path
from config import (
    SESSION_DIR, CLUSTERS_JSON, OUT_DIR, MODEL_ID,
    GUIDANCE_SCALE, STEPS_DRAFT, STEPS_FINAL, HEIGHT, WIDTH,
    SEED_PER_ROUND, K_PER_ROUND, ROUNDS_A, ROUNDS_B, TOP_GROUPS_B,
    REF_IMAGE_PATH, USE_IMG2IMG, IMG2IMG_STRENGTH_FINAL, INIT_RESIZE_MODE,
    USE_PROMPT_EMBEDS, USE_CONTROLNET, CONTROLNET_MODEL_ID, CONTROL_METHOD, 
    CONTROLNET_SCALE, CONTROL_START, CONTROL_END,
)
from loops.pbo_tag_weights import run_stage_A_alpha, run_stage_B_beta
from tags.tag_prompt_builder import load_clusters, build_prompts
from tags.tag_phrase_builder import build_weighted_phrases
from embeddings.sdxl_embed_fuser import SDXLEmbedFuser
from generation.diffusion_runner import DiffusionRunner
from generation.reference_controls import load_reference_image, preprocess_control_image
from utils.prompt_recorder import create_prompt_data, PromptRecorder

def main():
    session_dir = Path(SESSION_DIR)
    clusters_path = Path(CLUSTERS_JSON)
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PBO Tag Weight Optimization")
    print("=" * 60)
    print(f"Session: {session_dir.name}")
    print(f"Output: {out_dir}")
    print(f"Model: {MODEL_ID}")
    print(f"Prompt Embeds: {USE_PROMPT_EMBEDS}")
    print(f"Img2Img: {USE_IMG2IMG}")
    print(f"ControlNet: {USE_CONTROLNET}")
    print("=" * 60)

    # Stage A: optimize alpha (group weights)
    print(f"\n[MAIN] Starting Stage A: Alpha optimization ({ROUNDS_A} rounds)")
    print("-" * 40)
    alpha_star = run_stage_A_alpha(
        session_dir, clusters_path, out_dir, MODEL_ID,
        seed=SEED_PER_ROUND, k=K_PER_ROUND, rounds=ROUNDS_A,
        height=HEIGHT, width=WIDTH, steps=STEPS_DRAFT, gscale=GUIDANCE_SCALE
    )
    print(f"\n[MAIN] Stage A complete! Optimal alpha* = {alpha_star}")

    # Stage B: optimize beta inside top groups
    print(f"\n[MAIN] Starting Stage B: Beta optimization ({ROUNDS_B} rounds)")
    print("-" * 40)
    sel_groups, betas_star = run_stage_B_beta(
        session_dir, clusters_path, out_dir, alpha_star, MODEL_ID,
        seed=SEED_PER_ROUND, k=K_PER_ROUND, rounds=ROUNDS_B,
        height=HEIGHT, width=WIDTH, steps=STEPS_DRAFT, gscale=GUIDANCE_SCALE,
        top_groups=TOP_GROUPS_B
    )
    print(f"\n[MAIN] Stage B complete!")
    print(f"Selected groups for refinement: {sel_groups}")
    print(f"Optimal betas* = {betas_star}")

    # Final render
    print(f"\n[MAIN] Starting final render...")
    print("-" * 40)
    clusters = load_clusters(clusters_path)
    runner = DiffusionRunner(model_id=MODEL_ID, height=HEIGHT, width=WIDTH,
                             guidance_scale=GUIDANCE_SCALE, steps=STEPS_FINAL,
                             controlnet_model_id=(CONTROLNET_MODEL_ID if USE_CONTROLNET else None))
    ref_img = load_reference_image(Path(REF_IMAGE_PATH))
    control_img = preprocess_control_image(ref_img, method=CONTROL_METHOD, size=(WIDTH, HEIGHT)) if USE_CONTROLNET else None
    
    print(f"[MAIN] Loading diffusion pipeline...")
    # Choose embedding or string pipeline for final render
    if USE_PROMPT_EMBEDS:
        print(f"[MAIN] Using prompt embeddings pipeline")
        # Use embedding pipeline
        if USE_IMG2IMG:
            runner._ensure_img2img()  # Ensure img2img pipeline is loaded
            if runner.pipe_i2i is not None:
                fuser = SDXLEmbedFuser(runner.pipe_i2i, device=runner.device)
                print(f"[MAIN] Initialized img2img embedding fuser")
            else:
                fuser = None
                print(f"[MAIN] Warning: img2img pipeline failed to load")
        else:
            runner._ensure_txt2img()  # Ensure txt2img pipeline is loaded
            if runner.pipe is not None:
                fuser = SDXLEmbedFuser(runner.pipe, device=runner.device)
                print(f"[MAIN] Initialized txt2img embedding fuser")
            else:
                fuser = None
                print(f"[MAIN] Warning: txt2img pipeline failed to load")
        
        if fuser is not None:
            print(f"[MAIN] Building weighted phrases...")
            pos_phrases, neg_phrases = build_weighted_phrases(alpha_star, betas_star, clusters)
            if len(pos_phrases) > 0:
                print(f"[MAIN] Fusing embeddings...")
                prompt_embeds, pooled_prompt_embeds, neg_prompt_embeds, neg_pooled_prompt_embeds = fuser.fuse_weighted_phrases(pos_phrases, neg_phrases)
                if USE_IMG2IMG:
                    print(f"[MAIN] Generating final image with img2img...")
                    final_img = runner.generate_embeds_img2img(
                        init_image=ref_img,
                        strength=IMG2IMG_STRENGTH_FINAL,
                        prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=neg_prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        negative_pooled_prompt_embeds=neg_pooled_prompt_embeds,
                        seed=999,
                        steps=STEPS_FINAL, gscale=GUIDANCE_SCALE,
                        height=HEIGHT, width=WIDTH,
                        resize_mode=INIT_RESIZE_MODE,
                    )
                else:
                    print(f"[MAIN] Generating final image with txt2img...")
                    final_img = runner.generate_embeds(
                        prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=neg_prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        negative_pooled_prompt_embeds=neg_pooled_prompt_embeds,
                        control_image=control_img,
                        control_scale=CONTROLNET_SCALE,
                        seed=999,
                        steps=STEPS_FINAL, gscale=GUIDANCE_SCALE,
                        height=HEIGHT, width=WIDTH,
                    )
                # Save phrases for debugging
                pos_str = ", ".join([f"{phrase} ({weight:.3f})" for phrase, weight in pos_phrases])
                neg_str = ", ".join(neg_phrases)
                print(f"[MAIN] Final positive phrases: {pos_str}")
                print(f"[MAIN] Final negative phrases: {neg_str}")
            else:
                # Fallback to strings if no phrases
                print(f"[MAIN] No phrases available, falling back to string prompts")
                pos, neg = build_prompts(alpha_star, betas_star, clusters)
                pos_str, neg_str = pos, neg
                if USE_IMG2IMG:
                    print(f"[MAIN] Generating final image with img2img (string mode)...")
                    final_img = runner.generate_img2img(
                        init_image=ref_img,
                        strength=IMG2IMG_STRENGTH_FINAL,
                        positive_prompt=pos,
                        negative_prompt=neg,
                        seed=999,
                        steps=STEPS_FINAL, gscale=GUIDANCE_SCALE,
                        height=HEIGHT, width=WIDTH,
                        resize_mode=INIT_RESIZE_MODE,
                    )
                else:
                    print(f"[MAIN] Generating final image with txt2img (string mode)...")
                    final_img = runner.generate(
                        pos, neg,
                        control_image=control_img if control_img is not None else None,
                        control_scale=CONTROLNET_SCALE,
                        seed=999,
                        steps=STEPS_FINAL, gscale=GUIDANCE_SCALE,
                        height=HEIGHT, width=WIDTH,
                    )
        else:
            # Pipeline failed to load, fallback to strings
            print(f"[MAIN] Pipeline failed to load, using string prompts")
            pos, neg = build_prompts(alpha_star, betas_star, clusters)
            pos_str, neg_str = pos, neg
            if USE_IMG2IMG:
                print(f"[MAIN] Generating final image with img2img (string mode)...")
                final_img = runner.generate_img2img(
                    init_image=ref_img,
                    strength=IMG2IMG_STRENGTH_FINAL,
                    positive_prompt=pos,
                    negative_prompt=neg,
                    seed=999,
                    steps=STEPS_FINAL, gscale=GUIDANCE_SCALE,
                    height=HEIGHT, width=WIDTH,
                    resize_mode=INIT_RESIZE_MODE,
                )
            else:
                print(f"[MAIN] Generating final image with txt2img (string mode)...")
                final_img = runner.generate(
                    pos, neg,
                    control_image=control_img if control_img is not None else None,
                    control_scale=CONTROLNET_SCALE,
                    seed=999,
                    steps=STEPS_FINAL, gscale=GUIDANCE_SCALE,
                    height=HEIGHT, width=WIDTH,
                )
    else:
        # Use string pipeline (original behavior)
        print(f"[MAIN] Using string prompts pipeline")
        pos, neg = build_prompts(alpha_star, betas_star, clusters)
        pos_str, neg_str = pos, neg
        if USE_IMG2IMG:
            print(f"[MAIN] Generating final image with img2img...")
            final_img = runner.generate_img2img(
                init_image=ref_img,
                strength=IMG2IMG_STRENGTH_FINAL,
                positive_prompt=pos,
                negative_prompt=neg,
                seed=999,
                steps=STEPS_FINAL, gscale=GUIDANCE_SCALE,
                height=HEIGHT, width=WIDTH,
                resize_mode=INIT_RESIZE_MODE,
            )
        else:
            print(f"[MAIN] Generating final image with txt2img...")
            final_img = runner.generate(
                pos, neg,
                control_image=control_img if control_img is not None else None,
                control_scale=CONTROLNET_SCALE,
                seed=999,
                steps=STEPS_FINAL, gscale=GUIDANCE_SCALE,
                height=HEIGHT, width=WIDTH,
            )
    
    print(f"[MAIN] Saving final image...")
    final_path = out_dir / "final_best.png"
    final_img.save(final_path)
    print(f"[MAIN] Final image saved to: {final_path}")
    
    # Record final prompt with new system
    print(f"[MAIN] Recording final prompt data...")
    recorder = PromptRecorder(out_dir)
    if USE_PROMPT_EMBEDS and 'pos_phrases' in locals():
        final_prompt_data = create_prompt_data(
            weights=alpha_star,
            betas=betas_star,
            clusters=clusters,
            use_prompt_embeds=True,
            pos_phrases=pos_phrases,
            neg_phrases=neg_phrases,
            seed=999,
            stage="final"
        )
    else:
        final_prompt_data = create_prompt_data(
            weights=alpha_star,
            betas=betas_star,
            clusters=clusters,
            use_prompt_embeds=False,
            pos_prompt=pos_str,
            neg_prompt=neg_str,
            seed=999,
            stage="final"
        )
    recorder.record_final_prompt(final_prompt_data)
    
    # Also save the legacy final_prompts.json for backward compatibility
    print(f"[MAIN] Saving legacy prompt format...")
    with open(out_dir / "final_prompts.json", "w", encoding="utf-8") as f:
        import json
        json.dump({"positive_prompt": pos_str, "negative_prompt": neg_str,
                   "alpha_star": alpha_star.tolist(),
                   "betas_star": betas_star,
                   "use_prompt_embeds": USE_PROMPT_EMBEDS}, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE!")
    print("=" * 60)
    print(f"Final image: {final_path}")
    print(f"Final positive prompt: {pos_str}")
    print(f"Final negative prompt: {neg_str}")
    print("=" * 60)

if __name__ == "__main__":
    main()

