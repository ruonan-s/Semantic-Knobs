from PIL import Image, ImageDraw
import textwrap

try:
    import torch
    from diffusers import StableDiffusionXLPipeline
    try:
        from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
        HAS_CONTROLNET = True
    except Exception:
        HAS_CONTROLNET = False
    try:
        from diffusers import StableDiffusionXLImg2ImgPipeline
    except Exception:
        StableDiffusionXLImg2ImgPipeline = None
    HAS_DIFFUSERS = True
except Exception:
    HAS_DIFFUSERS = False
    HAS_CONTROLNET = False
    StableDiffusionXLImg2ImgPipeline = None


class DiffusionRunner:
    def __init__(self, model_id, device=None,
                 height=768, width=768,
                 guidance_scale=7.5, steps=24,
                 controlnet_model_id=None):
        self.model_id = model_id
        self.height = int(height)
        self.width = int(width)
        self.guidance_scale = float(guidance_scale)
        self.steps = int(steps)
        # Prefer MPS on macOS, then CUDA, else CPU
        if device is not None:
            self.device = device
        else:
            if HAS_DIFFUSERS and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            elif HAS_DIFFUSERS and hasattr(torch, "cuda") and torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        self.pipe = None
        self.pipe_i2i = None
        self.controlnet_model_id = controlnet_model_id
        self.using_controlnet = bool(controlnet_model_id and HAS_CONTROLNET)

    def _ensure_txt2img(self):
        if self.pipe is not None:
            return
        if not HAS_DIFFUSERS:
            return
        try:
            dtype = torch.float16 if (self.device == "cuda") else None
            if self.using_controlnet and self.controlnet_model_id:
                controlnet = ControlNetModel.from_pretrained(
                    self.controlnet_model_id,
                    torch_dtype=dtype,
                )
                self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                    self.model_id,
                    controlnet=controlnet,
                    torch_dtype=dtype,
                )
            else:
                self.pipe = StableDiffusionXLPipeline.from_pretrained(
                    self.model_id,
                    torch_dtype=dtype,
                )
            self.pipe = self.pipe.to(self.device)
            if hasattr(self.pipe, "enable_attention_slicing"):
                self.pipe.enable_attention_slicing()
        except Exception as e:
            print(f"[DiffusionRunner] Failed to load txt2img pipeline: {e}. Using mock images for txt2img.")
            self.pipe = None

    def _mock_image(self, positive_prompt, negative_prompt, seed):
        img = Image.new("RGB", (self.width, self.height), color=(245, 245, 245))
        draw = ImageDraw.Draw(img)
        text = f"MOCK GENERATION\nseed={seed}\n\nPOS:\n{textwrap.fill(str(positive_prompt), 60)}\n\nNEG:\n{textwrap.fill(str(negative_prompt), 60)}"
        try:
            draw.multiline_text((20, 20), text, fill=(20, 20, 20))
        except Exception:
            draw.text((20, 20), "Mock image")
        return img

    def generate(self, positive_prompt, negative_prompt,
                 control_image=None, control_scale=1.0,
                 seed=1234, steps=None, gscale=None,
                 height=None, width=None):
        steps = int(steps or self.steps)
        gscale = float(gscale or self.guidance_scale)
        height = int(height or self.height)
        width = int(width or self.width)

        self._ensure_txt2img()
        if self.pipe is None:
            return self._mock_image(positive_prompt, negative_prompt, seed)

        gen = None
        if self.device == "cuda":
            gen = torch.Generator(device=self.device).manual_seed(int(seed))
        try:
            if self.using_controlnet and control_image is not None:
                out = self.pipe(
                    prompt=positive_prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=steps,
                    guidance_scale=gscale,
                    height=height,
                    width=width,
                    generator=gen,
                    image=control_image,
                    controlnet_conditioning_scale=float(control_scale),
                )
            else:
                out = self.pipe(
                    prompt=positive_prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=steps,
                    guidance_scale=gscale,
                    height=height,
                    width=width,
                    generator=gen,
                )
            return out.images[0]
        except Exception as e:
            print(f"[DiffusionRunner] Diffusers call failed ({e}). Returning mock image.")
            return self._mock_image(positive_prompt, negative_prompt, seed)

    def generate_embeds(self,
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        control_image=None,
        control_scale=1.0,
        seed=1234,
        steps=None,
        gscale=None,
        height=None,
        width=None,
    ):
        steps = int(steps or self.steps)
        gscale = float(gscale or self.guidance_scale)
        height = int(height or self.height)
        width = int(width or self.width)

        self._ensure_txt2img()
        if self.pipe is None:
            pos = f"EMBEDS shape={getattr(prompt_embeds,'shape','?')}"
            neg = f"NEG_EMBEDS shape={getattr(negative_prompt_embeds,'shape','?')}"
            return self._mock_image(pos, neg, seed)

        gen = None
        if self.device in ("cuda", "mps"):
            gen = torch.Generator(device=self.device).manual_seed(int(seed))

        try:
            pipe_device = getattr(self.pipe, "_execution_device", None) or getattr(self.pipe, "device", self.device)
            te = getattr(self.pipe, "text_encoder", None)
            unet = getattr(self.pipe, "unet", None)
            target_dtype = getattr(te, "dtype", None) or getattr(unet, "dtype", None)

            if target_dtype is not None:
                prompt_embeds = prompt_embeds.to(pipe_device, dtype=target_dtype)
                negative_prompt_embeds = negative_prompt_embeds.to(pipe_device, dtype=target_dtype)
                if pooled_prompt_embeds is None or negative_pooled_prompt_embeds is None:
                    raise ValueError("SDXL requires pooled_prompt_embeds and negative_pooled_prompt_embeds when providing prompt_embeds.")
                pooled_prompt_embeds = pooled_prompt_embeds.to(pipe_device, dtype=target_dtype)
                negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(pipe_device, dtype=target_dtype)
            else:
                prompt_embeds = prompt_embeds.to(pipe_device)
                negative_prompt_embeds = negative_prompt_embeds.to(pipe_device)
                if pooled_prompt_embeds is None or negative_pooled_prompt_embeds is None:
                    raise ValueError("SDXL requires pooled_prompt_embeds and negative_pooled_prompt_embeds when providing prompt_embeds.")
                pooled_prompt_embeds = pooled_prompt_embeds.to(pipe_device)
                negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(pipe_device)

            # Handle ControlNet parameters if using ControlNet
            if self.using_controlnet and control_image is not None:
                out = self.pipe(
                    prompt=None,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                    num_inference_steps=steps,
                    guidance_scale=gscale,
                    height=height,
                    width=width,
                    generator=gen,
                    image=control_image,
                    controlnet_conditioning_scale=float(control_scale),
                )
            else:
                out = self.pipe(
                    prompt=None,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                    num_inference_steps=steps,
                    guidance_scale=gscale,
                    height=height,
                    width=width,
                    generator=gen,
                )
            return out.images[0]
        except Exception as e:
            print(f"[DiffusionRunner] generate_embeds() failed: {e}. Returning mock image.")
            pos = f"EMBEDS shape={getattr(prompt_embeds,'shape','?')}"
            neg = f"NEG_EMBEDS shape={getattr(negative_prompt_embeds,'shape','?')}"
            return self._mock_image(pos, neg, seed)

    # ---------------------------  SDXL Img2Img  -------------------------------
    def _ensure_img2img(self):
        if self.pipe_i2i is not None:
            return
        if not HAS_DIFFUSERS or StableDiffusionXLImg2ImgPipeline is None:
            self.pipe_i2i = None
            return
        try:
            dtype = torch.float16 if (self.device == "cuda") else None
            self.pipe_i2i = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                self.model_id, torch_dtype=dtype
            ).to(self.device)
            if hasattr(self.pipe_i2i, "enable_attention_slicing"):
                self.pipe_i2i.enable_attention_slicing()
        except Exception as e:
            print(f"[DiffusionRunner] Could not init SDXL Img2Img: {e}. Falling back to mock images.")
            self.pipe_i2i = None

    def preprocess_init_image(self, img, height, width, mode="fit_center_crop"):
        if mode == "stretch":
            return img.resize((int(width), int(height)), Image.BICUBIC)
        w, h = img.size
        target_aspect = float(width) / float(height)
        src_aspect = float(w) / float(h)
        if mode == "letterbox":
            return img.resize((int(width), int(height)), Image.BICUBIC)
        if src_aspect > target_aspect:
            new_h = int(height)
            new_w = int(round(height * src_aspect))
        else:
            new_w = int(width)
            new_h = int(round(width / src_aspect))
        img = img.resize((new_w, new_h), Image.BICUBIC)
        left = (new_w - int(width)) // 2
        top = (new_h - int(height)) // 2
        return img.crop((left, top, left + int(width), top + int(height)))

    def generate_img2img(self, init_image, strength,
                         positive_prompt, negative_prompt="",
                         seed=1234, steps=None, gscale=None,
                         height=None, width=None,
                         resize_mode="fit_center_crop"):
        steps = int(steps or self.steps)
        gscale = float(gscale or self.guidance_scale)
        height = int(height or self.height)
        width = int(width or self.width)
        init_prep = self.preprocess_init_image(init_image, height, width, mode=resize_mode)
        if not HAS_DIFFUSERS:
            pos = f"[I2I] {str(positive_prompt)[:120]}..."
            neg = f"{str(negative_prompt)[:120]}..."
            return self._mock_image(pos, neg, seed)
        self._ensure_img2img()
        if self.pipe_i2i is None:
            return self._mock_image("[I2I Fallback] "+str(positive_prompt), str(negative_prompt), seed)
        gen = torch.Generator(device=self.device).manual_seed(int(seed)) if self.device == "cuda" else None
        try:
            out = self.pipe_i2i(
                image=init_prep,
                strength=float(strength),
                prompt=positive_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=gscale,
                generator=gen,
                height=height,
                width=width,
            )
            return out.images[0]
        except Exception as e:
            print(f"[DiffusionRunner] generate_img2img failed: {e}. Returning mock.")
            return self._mock_image("[I2I ERR] "+str(positive_prompt), str(negative_prompt), seed)

    def generate_embeds_img2img(self, init_image, strength,
                                prompt_embeds, negative_prompt_embeds,
                                pooled_prompt_embeds=None, negative_pooled_prompt_embeds=None,
                                seed=1234, steps=None, gscale=None,
                                height=None, width=None,
                                resize_mode="fit_center_crop"):
        steps = int(steps or self.steps)
        gscale = float(gscale or self.guidance_scale)
        height = int(height or self.height)
        width = int(width or self.width)
        init_prep = self.preprocess_init_image(init_image, height, width, mode=resize_mode)
        if not HAS_DIFFUSERS:
            pos = f"[I2I EMBEDS] shape={getattr(prompt_embeds,'shape','?')}"
            neg = f"NEG shape={getattr(negative_prompt_embeds,'shape','?')}"
            return self._mock_image(pos, neg, seed)
        self._ensure_img2img()
        if self.pipe_i2i is None:
            return self._mock_image("[I2I EMBEDS Fallback]", "neg", seed)
        try:
            dtype = getattr(getattr(self.pipe_i2i, "text_encoder", None), "dtype", None)
        except Exception:
            dtype = None
        dev = self.pipe_i2i.device
        pe = prompt_embeds.to(dev, dtype=dtype) if hasattr(prompt_embeds, "to") else prompt_embeds
        npe = negative_prompt_embeds.to(dev, dtype=dtype) if hasattr(negative_prompt_embeds, "to") else negative_prompt_embeds
        pp = pooled_prompt_embeds.to(dev, dtype=dtype) if (pooled_prompt_embeds is not None and hasattr(pooled_prompt_embeds, "to")) else pooled_prompt_embeds
        npp = negative_pooled_prompt_embeds.to(dev, dtype=dtype) if (negative_pooled_prompt_embeds is not None and hasattr(negative_pooled_prompt_embeds, "to")) else negative_pooled_prompt_embeds
        gen = torch.Generator(device=self.device).manual_seed(int(seed)) if self.device == "cuda" else None
        try:
            out = self.pipe_i2i(
                image=init_prep,
                strength=float(strength),
                prompt=None,
                prompt_embeds=pe,
                negative_prompt_embeds=npe,
                pooled_prompt_embeds=pp,
                negative_pooled_prompt_embeds=npp,
                num_inference_steps=steps,
                guidance_scale=gscale,
                generator=gen,
                height=height,
                width=width,
            )
            return out.images[0]
        except Exception as e:
            print(f"[DiffusionRunner] generate_embeds_img2img failed: {e}. Returning mock.")
            pos = f"[I2I EMBEDS ERR] shape={getattr(prompt_embeds,'shape','?')}"
            neg = f"NEG shape={getattr(negative_prompt_embeds,'shape','?')}"
            return self._mock_image(pos, neg, seed)

