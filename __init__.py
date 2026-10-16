import json
import re
import threading

import comfy.model_management
import comfy.sd
import folder_paths


DEFAULT_CLIP = "qwen3vl_8b_int8_convrot.safetensors"
_cached_clip = None
_cached_path = None
_load_lock = threading.Lock()
import torch
import torch.nn.functional as F

_IMAGE_POLICY = """Evaluate this image for workplace safety (NSFW: nudity, explicit sexual content, exposed genitals/breasts, gore; clothed people/romance: SAFE).
Reply with exactly two lines:
VERDICT: SAFE or VERDICT: NSFW
REASON: brief explanation.
"""

_PROMPT_POLICY = """Evaluate this prompt for workplace safety. Prompt must be in English (otherwise VERDICT: NOT_ENGLISH).
Safety policy: NSFW = nudity, sexual requests, erotic content, exposed body parts; clothed people/everyday scenes = SAFE.
Reply with exactly two lines:
VERDICT: SAFE, VERDICT: NSFW, or VERDICT: NOT_ENGLISH
REASON: brief explanation.
"""

_POLICY = _PROMPT_POLICY


class NSFWDetectedError(RuntimeError):
    pass


class UnsupportedLanguageError(ValueError):
    pass


def _resize_for_guard(img_tensor, max_dim=384):
    """Downsample image for VLM safety moderation to reduce visual tokens from 1024 to 144."""
    if img_tensor is None or not isinstance(img_tensor, torch.Tensor) or img_tensor.ndim != 4:
        return img_tensor
    B, H, W, C = img_tensor.shape
    if max(H, W) <= max_dim:
        return img_tensor
    scale = max_dim / max(H, W)
    new_H = max(32, round(H * scale / 32) * 32)
    new_W = max(32, round(W * scale / 32) * 32)
    x = img_tensor.permute(0, 3, 1, 2)
    resized = F.interpolate(x, size=(new_H, new_W), mode="bilinear", align_corners=False)
    return resized.permute(0, 2, 3, 1)


def _load_clip(clip_name):
    global _cached_clip, _cached_path
    path = folder_paths.get_full_path("text_encoders", clip_name)
    if path is None:
        path = folder_paths.get_full_path("clip", clip_name)
    if path is None:
        raise FileNotFoundError(f"Qwen NSFW Guard: text encoder not found: {clip_name}")
    with _load_lock:
        if _cached_clip is None or _cached_path != path:
            loaded = comfy.sd.load_clip(ckpt_paths=[path])
            _cached_clip, _cached_path = loaded, path
        return _cached_clip


def _check(clip, query, source, image=None):
    tokens = clip.tokenize(query) if image is None else clip.tokenize(query, images=[image])
    generated = clip.generate(tokens, do_sample=False, max_length=32, temperature=0.0)
    response = clip.tokenizer.decode(generated).strip()
    verdicts = re.findall(r"^\s*VERDICT:\s*(NSFW|SAFE|NOT[ _]ENGLISH|UNSUPPORTED[ _]LANGUAGE)\b", response, re.MULTILINE | re.IGNORECASE)
    if any(re.match(r"^(NOT[ _]ENGLISH|UNSUPPORTED[ _]LANGUAGE)$", v, re.IGNORECASE) for v in verdicts):
        comfy.model_management.interrupt_current_processing(True)
        raise UnsupportedLanguageError("language not supported, only support english at this time")
    if any(verdict.upper() == "NSFW" for verdict in verdicts):
        comfy.model_management.interrupt_current_processing(True)
        raise NSFWDetectedError(f"Qwen NSFW Guard blocked {source}: {response}")
    if len(verdicts) != 1 or verdicts[0].upper() != "SAFE":
        raise RuntimeError(f"Qwen NSFW Guard could not classify {source}: {response!r}")
    return {"source": source, "verdict": "SAFE", "response": response}


def _moderate(clip_name, clip=None, prompt=None, image=None):
    if prompt is None and image is None:
        raise ValueError("Qwen NSFW Guard requires a prompt or image.")
    if prompt is not None and not prompt.strip() and image is not None:
        prompt = None
    clip = clip if clip is not None else _load_clip(clip_name)
    checks = []
    if prompt is not None:
        checks.append(_check(clip, _PROMPT_POLICY + "\nClassify this prompt (JSON string):\n" + json.dumps(prompt), "prompt"))
    if image is not None:
        guard_image = _resize_for_guard(image, max_dim=384)
        for i in range(guard_image.shape[0]):
            checks.append(_check(clip, _IMAGE_POLICY + "\nClassify the attached image.", f"image[{i}]", guard_image[i:i + 1]))
    return json.dumps({"verdict": "SAFE", "checks": checks}, ensure_ascii=False)



def _model_inputs():
    return {"clip_name": ("STRING", {"default": DEFAULT_CLIP})}


class QwenNSFWGuard:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": _model_inputs(), "optional": {"prompt": ("STRING", {"multiline": True, "forceInput": True}), "image": ("IMAGE",), "clip": ("CLIP",)}}

    RETURN_TYPES = ("STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("prompt", "image", "report")
    FUNCTION = "check"
    CATEGORY = "moderation/Qwen"
    OUTPUT_NODE = True

    def check(self, clip_name=DEFAULT_CLIP, prompt=None, image=None, clip=None):
        report = _moderate(clip_name, clip=clip, prompt=prompt, image=image)
        return {"ui": {"text": [report]}, "result": (prompt, image, report)}


class QwenNSFWPromptGuard:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"prompt": ("STRING", {"multiline": True}), **_model_inputs()}, "optional": {"clip": ("CLIP",)}}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "report")
    FUNCTION = "check"
    CATEGORY = "moderation/Qwen"
    OUTPUT_NODE = True

    def check(self, prompt, clip_name=DEFAULT_CLIP, clip=None):
        report = _moderate(clip_name, clip=clip, prompt=prompt)
        return {"ui": {"text": [report]}, "result": (prompt, report)}


class QwenNSFWImageGuard:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), **_model_inputs()}, "optional": {"clip": ("CLIP",)}}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "report")
    FUNCTION = "check"
    CATEGORY = "moderation/Qwen"
    OUTPUT_NODE = True

    def check(self, image, clip_name=DEFAULT_CLIP, clip=None):
        report = _moderate(clip_name, clip=clip, image=image)
        return {"ui": {"text": [report]}, "result": (image, report)}


NODE_CLASS_MAPPINGS = {cls.__name__: cls for cls in (QwenNSFWGuard, QwenNSFWPromptGuard, QwenNSFWImageGuard)}
NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenNSFWGuard": "Qwen NSFW Guard (Prompt + Image)",
    "QwenNSFWPromptGuard": "Qwen NSFW Prompt Guard",
    "QwenNSFWImageGuard": "Qwen NSFW Image Guard",
}
