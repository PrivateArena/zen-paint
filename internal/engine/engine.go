// Package engine defines the ImageEngine interface and shared request/result types
// shared across all zen-paint model backends (SDXL, Bonsai/FLUX, etc.)
package engine

// Options configure backend execution behavior.
type Options struct {
	// ExecutionProvider: "cpu", "rocm", "cuda", "npu"
	ExecutionProvider string
	// NumThreads: 0 means use runtime.NumCPU()
	NumThreads int
	// OutputDir: directory where generated PNGs are written
	OutputDir string
	// OrtLib is an optional path override for the ONNX Runtime shared library.
	OrtLib string
}

// OrtLibPath returns OrtLib, used as the argument to ort.Init().
func (o Options) OrtLibPath() string { return o.OrtLib }

// GenerateRequest carries all parameters for a single image generation call.
type GenerateRequest struct {
	Prompt         string  `json:"prompt"`
	NegativePrompt string  `json:"negative_prompt"`
	Width          int     `json:"width"`
	Height         int     `json:"height"`
	Steps          int     `json:"steps"`
	Seed           int64   `json:"seed"`
	CFGScale       float32 `json:"cfg_scale"`
}

// GenerateResult is returned on successful generation.
type GenerateResult struct {
	ImagePath  string `json:"path"`
	DurationMs int64  `json:"duration_ms"`
	Width      int    `json:"width"`
	Height     int    `json:"height"`
	Seed       int64  `json:"seed"`
}

// ImageEngine is the unified contract for all zen-paint generation backends.
type ImageEngine interface {
	// Initialize loads ONNX sessions and allocates GPU/CPU resources.
	// modelDir must contain the model-specific ONNX files and model.json.
	Initialize(modelDir string, opts Options) error

	// Generate runs the full diffusion pipeline and writes a PNG to OutputDir.
	Generate(req GenerateRequest) (GenerateResult, error)

	// Info returns a human-readable description of the loaded model.
	Info() string

	// Close tears down all ONNX sessions and frees runtime resources.
	Close() error
}
