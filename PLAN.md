# zen-paint: Local Image Generation Server

Persistent, offline image generation service following the `zen-*` ecosystem pattern — Go HTTP server, ONNX Runtime CGO bindings via `yalue/onnxruntime_go`, interface-based multi-model engine, designed to run 24/7 alongside zen-tts, zen-stt, and zen-separator.

---

## User Review Required

> [!IMPORTANT]
> **ONNX vs. Sidecar decision**: The DRAFT.md recommends a Python sidecar for Bonsai due to its Gemlite/MLX kernels. However, you explicitly asked for the same CGO pattern as the other zen-* projects. This plan **uses ONNX Runtime** (exactly like zen-stt/zen-sep), which means:
> - Bonsai ONNX format is required (Prism-ML does publish ONNX variants)
> - No Python dependency, pure Go + shared `.so`
> - CPU execution provider is the baseline; ROCm EP added for your Ryzen AI MAX 395+

> [!IMPORTANT]
> **First model target**: Bonsai 4B ONNX is a multi-part pipeline (text encoder + DiT + VAE). For the **first working version**, I recommend starting with **SDXL-Turbo** or **SD 1.5 + LCM** which have well-tested ONNX exports and run feasibly on CPU (30–90s). Bonsai ONNX can be added as a second backend once the pipeline skeleton is proven. Do you agree with this sequencing?

> [!NOTE]
> **Ryzen AI MAX 395+ GPU backend**: AMD ROCm execution provider for ONNX Runtime is supported. The Ryzen AI NPU also has a dedicated EP. We'll add it as a config flag (`execution_provider: "rocm"|"cpu"|"npu"`).

---

## Open Questions

1. Do you want a **TUI** (like zen-tts uses `tview`) or headless-only HTTP server?
2. Should the generated images be saved to disk (with path returned) or streamed as PNG bytes in the HTTP response body? Or both?
3. What's the preferred **model directory layout** — one flat `models/` folder, or named subdirs like `models/sdxl-turbo/`, `models/bonsai-4b/`?

---

## Architecture Overview

```
zen-paint HTTP Server (Go, persistent)
│
├── GET  /status         → server health, loaded model, VRAM
├── POST /generate       → { prompt, model, width, height, steps, seed, cfg }
│                          → { path: "/tmp/zen-paint/abc.png", duration_ms: 1240 }
├── POST /load           → { model: "sdxl-turbo" }  (hot-swap model)
└── GET  /models         → list available models in models/ dir
│
├── internal/
│   ├── engine/          → ImageEngine interface (Initialize/Generate/Close)
│   ├── sdxl/            → SDXL/SDXL-Turbo/LCM ONNX pipeline
│   ├── flux/            → Bonsai/FLUX ONNX pipeline (Phase 2)
│   └── ort/             → shared ORT init (same pattern as zen-stt)
│
├── models/              → model subdirs with config.json per model
├── config.json          → port, default_model, execution_provider, output_dir
└── main.go / server.go  → HTTP server + engine lifecycle
```

---

## Proposed Changes

### Phase 1 — Skeleton & First Working Model

#### [NEW] `go.mod`
- `module zen-paint`, Go 1.24
- `github.com/yalue/onnxruntime_go v1.27.0` (matches zen-sep version)
- `github.com/gdamore/tcell/v2` + `github.com/rivo/tview` (optional TUI, pending Q1)

---

#### [NEW] `internal/ort/init.go`
Identical pattern to zen-stt's ORT init — `sync.Once`, candidate path search, `ORT_SHARED_LIB_PATH` env override. Reuses same `libonnxruntime.so`.

---

#### [NEW] `internal/engine/engine.go`
```go
type ImageEngine interface {
    Initialize(modelDir string, opts Options) error
    Generate(req GenerateRequest) (GenerateResult, error)
    Close() error
}

type Options struct {
    ExecutionProvider string // "cpu", "rocm", "npu"
    NumThreads        int
}

type GenerateRequest struct {
    Prompt         string
    NegativePrompt string
    Width, Height  int
    Steps          int
    Seed           int64
    CFGScale       float32
}

type GenerateResult struct {
    ImagePath   string
    DurationMs  int64
    Width, Height int
}
```

---

#### [NEW] `internal/sdxl/engine.go`
SDXL-Turbo / LCM pipeline:
1. **Text Encoder** — CLIP ViT-L/14, `text_encoder.onnx` → token embeddings
2. **UNet** — `unet.onnx`, conditioned on embeddings + timestep → latents
3. **VAE Decoder** — `vae_decoder.onnx` → RGB pixels
4. **Scheduler** — LCM/DDIM in pure Go (4–8 steps)
5. PNG encode via stdlib `image/png`, write to `output_dir`

Session options: `SetIntraOpNumThreads(min(cpu, 16))`, ROCm EP if configured.

---

#### [NEW] `config.go`
```go
type Config struct {
    Port              int    `json:"port"`
    DefaultModel      string `json:"default_model"`
    ExecutionProvider string `json:"execution_provider"` // "cpu"|"rocm"
    NumThreads        int    `json:"num_threads"`
    OutputDir         string `json:"output_dir"`
    OrtLibPath        string `json:"ort_lib_path"`
}
```

#### [NEW] `config.json`
```json
{
  "port": 7890,
  "default_model": "sdxl-turbo",
  "execution_provider": "cpu",
  "num_threads": 0,
  "output_dir": "/tmp/zen-paint",
  "ort_lib_path": ""
}
```

---

#### [NEW] `server.go`
Same structure as zen-tts `server.go`:
- `StartServer(model, port)` → loads engine, registers HTTP mux, goroutine serve
- `StopServer()` → engine.Close(), shutdown
- `generateHandler` → JSON decode → engine.Generate() → JSON response with path

---

#### [NEW] `main.go`
Minimal: load config, TUI or headless flag, call StartServer.

---

### Phase 2 — Bonsai ONNX Backend

#### [NEW] `internal/flux/engine.go`
MMDiT pipeline (FLUX.2 Klein / Bonsai):
1. **T5 + CLIP encoders** → `text_encoder.onnx`, `text_encoder_2.onnx`
2. **DiT transformer** → `transformer.onnx` (25 blocks, 5 double + 20 single stream)
3. **VAE decoder** → `vae_decoder.onnx`
4. **Scheduler** — FlowMatchEulerDiscrete in Go, 4 steps, shift=3.0, guidance=1.0

> [!NOTE]
> Bonsai's binary (1-bit) and ternary (1.58-bit) weights require the Gemlite custom op if using PyTorch. The **ONNX export** uses standard INT4/INT8 packing that ONNX Runtime understands natively — no Gemlite needed. Verify the Prism-ML ONNX release uses standard quantization before Phase 2.

---

### Phase 3 — SLM Model Catalog

#### [NEW] `models/<name>/model.json`
```json
{
  "name": "sdxl-turbo",
  "architecture": "sdxl",
  "files": {
    "text_encoder": "text_encoder.onnx",
    "unet": "unet.onnx",
    "vae_decoder": "vae_decoder.onnx"
  },
  "default_steps": 4,
  "default_cfg": 0.0,
  "max_resolution": [512, 512],
  "description": "Fast 4-step SDXL, good for icons and diagrams"
}
```

Other catalog targets (all <2GB total):
| Model | Size | Use case |
|---|---|---|
| `sdxl-turbo` | ~2.1GB | General, fast preview |
| `lcm-dreamshaper` | ~1.8GB | Stylized, 2D illustration |
| `bonsai-ternary` | ~1.21GB | High-quality, FLUX-based |
| `bonsai-binary` | ~0.93GB | Ultra-fast, lower quality |
| `tooncrafter-tiny` | ~0.8GB | Cartoon/anime style |

---

## Verification Plan

### Automated
```bash
go build ./...
go test ./internal/...
curl -X POST localhost:7890/generate \
  -d '{"prompt":"a white canvas with a simple tree drawing","width":512,"height":512,"steps":4}'
```

### Manual
- CPU: generate 512×512 image, verify <2min wall time
- Check PNG written to `output_dir`
- Hot-swap model via `/load`, verify memory freed and new model responds
- Confirm server survives 10+ sequential requests without memory leak (`pprof`)
