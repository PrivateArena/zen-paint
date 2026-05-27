// Package flux implements the FLUX.2 Klein / Bonsai Image 4B ONNX pipeline.
//
// Architecture: Multimodal Diffusion Transformer (MMDiT)
//   - 25 blocks: 5 double-stream + 20 single-stream
//   - Sampler: FlowMatchEulerDiscrete, 4 steps, shift=3.0, guidance=1.0
//   - Text: T5-XXL encoder + CLIP-L/14 encoder (dual conditioning)
//   - VAE: FLUX VAE (scaling factor = 0.3611)
//
// Expected model directory layout:
//
//	models/<name>/
//	  model.json
//	  text_encoder.onnx      (CLIP-L/14)
//	  text_encoder_2.onnx    (T5-XXL)
//	  transformer.onnx       (DiT, ternary or binary quantized)
//	  vae_decoder.onnx
package flux

import (
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	ort "github.com/yalue/onnxruntime_go"
	zenort "zen-paint/internal/ort"
	"zen-paint/internal/tokenizer"

	"zen-paint/internal/engine"
)

const (
	fluxVAEScale   = 0.3611  // FLUX-specific VAE scaling factor
	fluxShift      = 3.0     // FlowMatchEuler shift
	fluxGuidance   = 1.0     // distilled — no classifier-free guidance needed
	fluxDefaultRes = 1024    // native resolution
)

// Engine is the Bonsai/FLUX MMDiT backend.
type Engine struct {
	clipEncoder   *ort.DynamicAdvancedSession // CLIP-L/14
	t5Encoder     *ort.DynamicAdvancedSession // T5-XXL
	transformer   *ort.DynamicAdvancedSession // DiT 25-block
	vaeDecoder    *ort.DynamicAdvancedSession
	clipTokenizer *tokenizer.ClipTokenizer
	t5Tokenizer   *tokenizer.T5Tokenizer
	opts          engine.Options
	modelDir      string
	info          string
}

// Initialize loads ONNX sessions from modelDir.
func (e *Engine) Initialize(modelDir string, opts engine.Options) error {
	if err := zenort.Init(opts.OrtLibPath()); err != nil {
		return fmt.Errorf("ort init: %w", err)
	}
	e.modelDir = modelDir
	e.opts = opts

	threads := opts.NumThreads
	if threads <= 0 {
		threads = runtime.NumCPU()
		if threads > 16 {
			threads = 16
		}
	}

	sessOpts, err := ort.NewSessionOptions()
	if err != nil {
		return fmt.Errorf("session options: %w", err)
	}
	_ = sessOpts.SetIntraOpNumThreads(threads)
	_ = sessOpts.SetInterOpNumThreads(1)

	switch strings.ToLower(opts.ExecutionProvider) {
	case "rocm", "hip":
		err = sessOpts.AppendExecutionProvider("ROCM", map[string]string{
			"device_id": "0",
		})
		if err != nil {
			fmt.Printf("[ort] Warning: failed to append ROCm provider: %v. Using CPU fallback.\n", err)
		} else {
			fmt.Println("[ort] Enabled ROCm GPU execution provider")
		}
	case "cuda":
		cudaOpts, err := ort.NewCUDAProviderOptions()
		if err == nil {
			defer cudaOpts.Destroy()
			err = sessOpts.AppendExecutionProviderCUDA(cudaOpts)
		}
		if err != nil {
			fmt.Printf("[ort] Warning: failed to append CUDA provider: %v. Using CPU fallback.\n", err)
		} else {
			fmt.Println("[ort] Enabled CUDA GPU execution provider")
		}
	case "directml":
		err = sessOpts.AppendExecutionProviderDirectML(0)
		if err != nil {
			fmt.Printf("[ort] Warning: failed to append DirectML provider: %v. Using CPU fallback.\n", err)
		} else {
			fmt.Println("[ort] Enabled DirectML GPU execution provider")
		}
	case "openvino":
		err = sessOpts.AppendExecutionProviderOpenVINO(nil)
		if err != nil {
			fmt.Printf("[ort] Warning: failed to append OpenVINO provider: %v. Using CPU fallback.\n", err)
		} else {
			fmt.Println("[ort] Enabled OpenVINO execution provider")
		}
	}

	load := func(name string) (*ort.DynamicAdvancedSession, error) {
		path := filepath.Join(modelDir, name)
		if _, err := os.Stat(path); err != nil {
			return nil, fmt.Errorf("model file not found: %s", path)
		}
		ins, outs, err := ort.GetInputOutputInfo(path)
		if err != nil {
			return nil, fmt.Errorf("get io info %s: %w", name, err)
		}
		inNames := make([]string, len(ins))
		for i, v := range ins {
			inNames[i] = v.Name
		}
		outNames := make([]string, len(outs))
		for i, v := range outs {
			outNames[i] = v.Name
		}
		return ort.NewDynamicAdvancedSession(path, inNames, outNames, sessOpts)
	}

	if e.clipEncoder, err = load("text_encoder.onnx"); err != nil {
		return err
	}
	if e.t5Encoder, err = load("text_encoder_2.onnx"); err != nil {
		return err
	}
	if e.transformer, err = load("transformer.onnx"); err != nil {
		return err
	}
	if e.vaeDecoder, err = load("vae_decoder.onnx"); err != nil {
		return err
	}

	e.clipTokenizer, err = tokenizer.NewClipTokenizer(modelDir)
	if err != nil {
		return fmt.Errorf("clip tokenizer: %w", err)
	}
	e.t5Tokenizer, err = tokenizer.NewT5Tokenizer(modelDir)
	if err != nil {
		return fmt.Errorf("t5 tokenizer: %w", err)
	}

	e.info = fmt.Sprintf("Bonsai/FLUX MMDiT | dir=%s threads=%d provider=%s",
		modelDir, threads, opts.ExecutionProvider)
	return nil
}

// Generate runs the full FLUX pipeline for a single request.
func (e *Engine) Generate(req engine.GenerateRequest) (engine.GenerateResult, error) {
	t0 := time.Now()
	rng := rand.New(rand.NewSource(req.Seed))

	// 1. Dual text encoding: CLIP + T5
	clipEmb, clipPool, err := e.encodeClip(req.Prompt)
	if err != nil {
		return engine.GenerateResult{}, fmt.Errorf("CLIP encode: %w", err)
	}
	t5Emb, err := e.encodeT5(req.Prompt)
	if err != nil {
		return engine.GenerateResult{}, fmt.Errorf("T5 encode: %w", err)
	}

	// 2. Init latent noise — FLUX latent space is H/8 × W/8 × 16 channels
	latentH := req.Height / 8
	latentW := req.Width / 8
	latents := randNormal(rng, 1*16*latentH*latentW)

	// 3. FlowMatchEuler schedule
	sigmas := flowMatchSigmas(req.Steps, fluxShift)

	// 4. Denoising loop (distilled: single guidance, no CFG)
	for i := 0; i < len(sigmas)-1; i++ {
		sigma := sigmas[i]
		sigmaNext := sigmas[i+1]
		latents, err = e.ditStep(latents, clipEmb, clipPool, t5Emb,
			sigma, sigmaNext, latentH, latentW)
		if err != nil {
			return engine.GenerateResult{}, fmt.Errorf("DiT step %d: %w", i, err)
		}
	}

	// 5. VAE decode
	pixels, err := e.decodeLatents(latents, latentH, latentW)
	if err != nil {
		return engine.GenerateResult{}, fmt.Errorf("VAE decode: %w", err)
	}

	// 6. Save PNG
	imgPath, err := savePNG(pixels, req.Width, req.Height, e.opts.OutputDir, req.Seed)
	if err != nil {
		return engine.GenerateResult{}, fmt.Errorf("save PNG: %w", err)
	}

	return engine.GenerateResult{
		ImagePath:  imgPath,
		DurationMs: time.Since(t0).Milliseconds(),
		Width:      req.Width,
		Height:     req.Height,
		Seed:       req.Seed,
	}, nil
}

func (e *Engine) Info() string { return e.info }

func (e *Engine) Close() error {
	for _, s := range []*ort.DynamicAdvancedSession{
		e.clipEncoder, e.t5Encoder, e.transformer, e.vaeDecoder,
	} {
		if s != nil {
			_ = s.Destroy()
		}
	}
	return nil
}

// --- Pipeline internals ---

func (e *Engine) encodeClip(prompt string) (seq []float32, pool []float32, err error) {
	tokens := e.clipTokenizer.Encode(prompt, 77)
	inT, _ := ort.NewTensor(ort.NewShape(1, 77), tokens)
	defer inT.Destroy()

	seq = make([]float32, 1*77*768)
	pool = make([]float32, 1*768)
	seqT, _ := ort.NewTensor(ort.NewShape(1, 77, 768), seq)
	poolT, _ := ort.NewTensor(ort.NewShape(1, 768), pool)
	defer seqT.Destroy()
	defer poolT.Destroy()

	if err = e.clipEncoder.Run([]ort.Value{inT}, []ort.Value{seqT, poolT}); err != nil {
		return nil, nil, err
	}
	out := make([]float32, len(seqT.GetData()))
	copy(out, seqT.GetData())
	p := make([]float32, len(poolT.GetData()))
	copy(p, poolT.GetData())
	return out, p, nil
}

func (e *Engine) encodeT5(prompt string) ([]float32, error) {
	tokens := e.t5Tokenizer.Encode(prompt, 256)
	inT, _ := ort.NewTensor(ort.NewShape(1, 256), tokens)
	defer inT.Destroy()

	out := make([]float32, 1*256*4096)
	outT, _ := ort.NewTensor(ort.NewShape(1, 256, 4096), out)
	defer outT.Destroy()

	if err := e.t5Encoder.Run([]ort.Value{inT}, []ort.Value{outT}); err != nil {
		return nil, err
	}
	result := make([]float32, len(outT.GetData()))
	copy(result, outT.GetData())
	return result, nil
}

func (e *Engine) ditStep(
	latents, clipEmb, clipPool, t5Emb []float32,
	sigma, sigmaNext float64,
	latentH, latentW int,
) ([]float32, error) {
	latentSize := 1 * 16 * latentH * latentW

	latT, _ := ort.NewTensor(ort.NewShape(1, 16, int64(latentH), int64(latentW)), latents)
	clipT, _ := ort.NewTensor(ort.NewShape(1, 77, 768), clipEmb)
	poolT, _ := ort.NewTensor(ort.NewShape(1, 768), clipPool)
	t5T, _ := ort.NewTensor(ort.NewShape(1, 256, 4096), t5Emb)
	sigmaT, _ := ort.NewTensor(ort.NewShape(1), []float32{float32(sigma)})
	defer latT.Destroy()
	defer clipT.Destroy()
	defer poolT.Destroy()
	defer t5T.Destroy()
	defer sigmaT.Destroy()

	outVelocity := make([]float32, latentSize)
	velT, _ := ort.NewTensor(ort.NewShape(1, 16, int64(latentH), int64(latentW)), outVelocity)
	defer velT.Destroy()

	if err := e.transformer.Run(
		[]ort.Value{latT, clipT, poolT, t5T, sigmaT},
		[]ort.Value{velT},
	); err != nil {
		return nil, err
	}

	// Euler step: x_{t+1} = x_t + (sigma_next - sigma) * v
	dt := float32(sigmaNext - sigma)
	vel := velT.GetData()
	updated := make([]float32, latentSize)
	for i := range updated {
		updated[i] = latents[i] + dt*vel[i]
	}
	return updated, nil
}

func (e *Engine) decodeLatents(latents []float32, latentH, latentW int) ([]float32, error) {
	// FLUX VAE scaling
	for i := range latents {
		latents[i] /= float32(fluxVAEScale)
	}

	inT, _ := ort.NewTensor(ort.NewShape(1, 16, int64(latentH), int64(latentW)), latents)
	defer inT.Destroy()

	outH := latentH * 8
	outW := latentW * 8
	out := make([]float32, 1*3*outH*outW)
	outT, _ := ort.NewTensor(ort.NewShape(1, 3, int64(outH), int64(outW)), out)
	defer outT.Destroy()

	if err := e.vaeDecoder.Run([]ort.Value{inT}, []ort.Value{outT}); err != nil {
		return nil, err
	}
	result := make([]float32, len(outT.GetData()))
	copy(result, outT.GetData())
	return result, nil
}

// --- Scheduler ---

// flowMatchSigmas generates sigmas for FlowMatchEulerDiscrete schedule.
func flowMatchSigmas(steps int, shift float64) []float64 {
	sigmas := make([]float64, steps+1)
	for i := 0; i <= steps; i++ {
		t := 1.0 - float64(i)/float64(steps)
		// Shifted cosine schedule
		sigmas[i] = shift * t / (1 + (shift-1)*t)
	}
	return sigmas
}

// --- Tokenizers (placeholder — replace with real BPE) ---

func clipTokenize(prompt string, maxLen int) []int32 {
	tokens := make([]int32, maxLen)
	tokens[0] = 49406
	for i, b := range []byte(prompt) {
		if i+1 >= maxLen-1 {
			break
		}
		tokens[i+1] = int32(b)
	}
	tokens[maxLen-1] = 49407
	return tokens
}

func t5Tokenize(prompt string, maxLen int) []int64 {
	tokens := make([]int64, maxLen)
	for i, b := range []byte(prompt) {
		if i >= maxLen {
			break
		}
		tokens[i] = int64(b)
	}
	return tokens
}

// --- Utilities ---

func randNormal(rng *rand.Rand, n int) []float32 {
	data := make([]float32, n)
	for i := range data {
		data[i] = float32(rng.NormFloat64())
	}
	return data
}

func savePNG(pixels []float32, width, height int, outputDir string, seed int64) (string, error) {
	img := image.NewRGBA(image.Rect(0, 0, width, height))
	pixPerCh := width * height
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			idx := y*width + x
			r := clampByte(pixels[0*pixPerCh+idx])
			g := clampByte(pixels[1*pixPerCh+idx])
			b := clampByte(pixels[2*pixPerCh+idx])
			img.SetRGBA(x, y, color.RGBA{R: r, G: g, B: b, A: 255})
		}
	}
	if err := os.MkdirAll(outputDir, 0o755); err != nil {
		return "", err
	}
	name := fmt.Sprintf("zp_%d_%d.png", time.Now().UnixMilli(), seed)
	path := filepath.Join(outputDir, name)
	f, err := os.Create(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	return path, png.Encode(f, img)
}

func clampByte(v float32) uint8 {
	v = (v + 1.0) / 2.0
	if v < 0 {
		v = 0
	}
	if v > 1 {
		v = 1
	}
	return uint8(math.Round(float64(v) * 255))
}
