# zen-paint

Persistent, offline image generation server — part of the **zen-\*** SLM ecosystem.

Follows the same architecture as [zen-tts](../zen-tts), [zen-stt](../zen-stt), and [zen-separator](../zen-separator): Go HTTP server + ONNX Runtime CGO bindings via `yalue/onnxruntime_go`. No Python. No sidecar.

---

## Features

- **Persistent server** — load once, generate forever (24/7 capable)
- **Multi-model** — hot-swap models at runtime via `POST /load`
- **SLM-first** — optimized for small, task-specific models (<2GB)
- **CPU baseline** — runs on CPU (30s–2min per 512²); GPU via ROCm on Ryzen AI MAX 395+
- **Two backends**:
  - `sdxl` — SDXL-Turbo / LCM (UNet, 4 steps, 512×512)
  - `flux` / `bonsai` — Bonsai Image 4B (MMDiT, 4 steps, up to 1024×1024)

---

## Quick Start

### 1. Link ORT shared library

```bash
make link-ort
# or manually:
mkdir -p piper
ln -s /path/to/libonnxruntime.so.1.24.2 piper/libonnxruntime.so
```

### 2. Download a model

```bash
make fetch-sdxl-turbo
# or place ONNX files manually in models/sdxl-turbo/
```

### 3. Build & run

```bash
make build
make run
# Server on :7890
```

---

## API

### `GET /status`
```json
{ "status": "ok", "model": "sdxl-turbo", "info": "..." }
```

### `GET /models`
```json
{ "models": ["sdxl-turbo", "bonsai-ternary"] }
```

### `POST /load`
```json
{ "model": "bonsai-ternary" }
```

### `POST /generate`
```json
{
  "prompt": "a whiteboard drawing of a tree, black marker, simple",
  "width": 512,
  "height": 512,
  "steps": 4,
  "seed": 42,
  "cfg_scale": 0.0
}
```
Response:
```json
{
  "path": "/tmp/zen-paint/zp_1234567890_42.png",
  "duration_ms": 45200,
  "width": 512,
  "height": 512,
  "seed": 42
}
```

---

## Model Catalog (SLM targets)

| Model | Size | Steps | Use case |
|---|---|---|---|
| `sdxl-turbo` | ~2.1GB | 4 | General, fast preview |
| `lcm-dreamshaper` | ~1.8GB | 4 | 2D illustration / cartoon |
| `bonsai-ternary` | ~1.21GB | 4 | High quality, FLUX-based |
| `bonsai-binary` | ~0.93GB | 4 | Ultra-fast, lower quality |

---

## Adding a Model

Create `models/<name>/model.json`:

```json
{
  "name": "my-model",
  "architecture": "sdxl",
  "files": {
    "text_encoder": "text_encoder.onnx",
    "unet": "unet.onnx",
    "vae_decoder": "vae_decoder.onnx"
  },
  "default_steps": 4,
  "default_cfg": 0.0,
  "max_width": 512,
  "max_height": 512,
  "description": "My custom model"
}
```

Supported architectures: `sdxl`, `sdxl-turbo`, `lcm`, `flux`, `bonsai`

---

## Ecosystem Integration

```
zen-board → POST /generate (zen-paint)  → PNG asset
zen-board → POST /tts     (zen-tts)     → WAV audio  
zen-board → POST /         (zen-stt)    → transcript
```

zen-paint is intentionally stateless across requests — pass `seed` explicitly for reproducible assets.

---

## Configuration (`config.json`)

| Field | Default | Description |
|---|---|---|
| `port` | `7890` | HTTP listen port |
| `default_model` | `sdxl-turbo` | Model loaded at startup |
| `execution_provider` | `cpu` | `cpu` \| `rocm` \| `cuda` |
| `num_threads` | `0` (auto) | ORT intra-op threads |
| `output_dir` | `/tmp/zen-paint` | Where PNGs are saved |
| `ort_lib_path` | `""` (auto) | Path to `libonnxruntime.so` |
