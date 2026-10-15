# ComfyUI Qwen NSFW Guard

A high-performance local NSFW moderation custom node for **ComfyUI** powered by **Qwen3-VL 8B** (`qwen3vl_8b_int8_convrot.safetensors`).

It moderates both text prompts and images (input images and generated output images) entirely locally with zero external network requests or extra model downloads. If NSFW content is detected, the workflow is **terminated immediately**.

---

## Features

- **Local Qwen3-VL 8B Inference**: Reuses the `qwen3vl_8b_int8_convrot.safetensors` model in `models/text_encoders/` or `models/clip/`.
- **Fail-Closed Immediate Interruption**: On NSFW detection, triggers `comfy.model_management.interrupt_current_processing(True)` and raises `NSFWDetectedError`, instantly halting the execution graph so no downstream nodes (like generation or saving) run.
- **Prompt & Image Moderation**: Moderates text prompts, image inputs (e.g. img2img / reference images), and final generated output images.
- **Batch Image Support**: Evaluates all images in a tensor batch (`[B, H, W, C]`).
- **Thread-Safe Model Caching**: Loads the text encoder / VLM once and retains it in memory across workflow runs; supports reusing an upstream `CLIP` connection or auto-loading.
- **Pass-Through Outputs**: When safe, passes through `prompt` and `image` untouched, along with an audit JSON report.
- **Standalone or Inline**: Configured as `OUTPUT_NODE = True`, allowing execution both inline in a pipeline or as a terminal safety inspector.

---

## Nodes Provided

The nodes are registered under the category **`moderation/Qwen`**:

1. **`Qwen NSFW Guard (Prompt + Image)` (`QwenNSFWGuard`)**
   - **Inputs**:
     - `clip_name` (STRING, default: `qwen3vl_8b_int8_convrot.safetensors`)
     - `prompt` (*optional*, STRING, multiline)
     - `image` (*optional*, IMAGE)
     - `clip` (*optional*, CLIP)
   - **Outputs**:
     - `prompt` (STRING)
     - `image` (IMAGE)
     - `report` (STRING, JSON verdict and explanations)

2. **`Qwen NSFW Prompt Guard` (`QwenNSFWPromptGuard`)**
   - Dedicated pass-through node for prompt checks.
   - **Inputs**: `prompt`, `clip_name`, `clip` (*optional*)
   - **Outputs**: `prompt`, `report`

3. **`Qwen NSFW Image Guard` (`QwenNSFWImageGuard`)**
   - Dedicated pass-through node for image checks (inputs or outputs).
   - **Inputs**: `image`, `clip_name`, `clip` (*optional*)
   - **Outputs**: `image`, `report`

---

## Workflow Wiring

### 1. Guarding Text Prompts
```
[Primitive / String] ──> [Qwen NSFW Prompt Guard] ── prompt ──> [CLIPTextEncode]
```

### 2. Guarding Input Images (Img2Img / ControlNet)
```
[Load Image] ── IMAGE ──> [Qwen NSFW Image Guard] ── image ──> [VAE Encode]
```

### 3. Guarding Output Generated Images
```
[VAE Decode] ── IMAGE ──> [Qwen NSFW Image Guard] ── image ──> [Save Image]
```

### 4. Combined Input Guard (All-in-One)
```
[Prompt Text] ───────> prompt ──┐
                                [Qwen NSFW Guard] ── prompt ──> [CLIPTextEncode]
[Load Image]  ───────> image  ──┘                 ── image  ──> [VAE Encode]
```

> [!IMPORTANT]
> To enforce execution order, downstream nodes **must** consume the guard's output (`prompt` / `image`). If independent branches bypass the guard, ComfyUI may execute them in parallel.

---

## Moderation Policy

The moderation prompt instructs Qwen3-VL to classify content strictly for workplace-safe environments:
- **NSFW**: Sexual activity, explicit sexual requests, exposed genitals, erotic nudity, or graphic gore.
- **SAFE**: Ordinary clothed people, landscapes, artwork, and non-explicit romance.

If an NSFW verdict is reached:
```
NSFWDetectedError: Qwen NSFW Guard blocked prompt: VERDICT: NSFW
REASON: Contains explicit sexual content, exposed genitals, and uncensored nudity as defined in NSFW criteria.
```

---

## Testing & Verification

Run the test suite using ComfyUI's virtual environment:

```bash
# Unit & mock tests
python test_guard.py

# End-to-end test against the real local checkpoint
python test_guard.py --real
```

---

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
