package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"time"

	"zen-paint/internal/engine"
	"zen-paint/internal/flux"
	"zen-paint/internal/sdxl"
)

var (
	activeEngine   engine.ImageEngine
	activeEngineMu sync.RWMutex
	activeModel    string
	serverSrv      *http.Server
	serverMu       sync.Mutex
	serverActive   bool
	globalCfg      Config
	generateSem    chan struct{}
)

// --- SERVER LIFECYCLE ---

func StartServer(cfg Config) {
	serverMu.Lock()
	defer serverMu.Unlock()

	if serverActive {
		return
	}
	globalCfg = cfg

	if err := loadEngine(cfg.DefaultModel, cfg); err != nil {
		logf("[red]Engine init failed: %v[-]", err)
		return
	}

	if cfg.MaxConcurrency <= 0 {
		cfg.MaxConcurrency = 1
	}
	generateSem = make(chan struct{}, cfg.MaxConcurrency)

	mux := http.NewServeMux()
	mux.HandleFunc("/status", statusHandler)
	mux.HandleFunc("/models", modelsHandler)
	mux.HandleFunc("/load", loadHandler)
	mux.HandleFunc("/generate", generateHandler)
	mux.Handle("/outputs/", http.StripPrefix("/outputs/", http.FileServer(http.Dir(cfg.OutputDir))))

	serverSrv = &http.Server{
		Addr:         fmt.Sprintf(":%d", cfg.Port),
		Handler:      corsMiddleware(mux),
		ReadTimeout:  10 * time.Minute,
		WriteTimeout: 10 * time.Minute,
	}

	go func() {
		logf("[green]zen-paint listening on :%d (model: %s, provider: %s)[-]",
			cfg.Port, cfg.DefaultModel, cfg.ExecutionProvider)
		if err := serverSrv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			logf("[red]Server error: %v[-]", err)
			serverActive = false
		}
	}()
	serverActive = true
}

func StopServer() {
	serverMu.Lock()
	defer serverMu.Unlock()

	if !serverActive || serverSrv == nil {
		return
	}
	logf("Stopping server...")
	_ = serverSrv.Close()

	activeEngineMu.Lock()
	if activeEngine != nil {
		_ = activeEngine.Close()
		activeEngine = nil
	}
	activeEngineMu.Unlock()

	serverActive = false
	serverSrv = nil
	logf("[yellow]Server stopped[-]")
}

// loadEngine initializes the named model engine and replaces any existing one.
func loadEngine(modelName string, cfg Config) error {
	opts := engine.Options{
		ExecutionProvider: cfg.ExecutionProvider,
		NumThreads:        cfg.NumThreads,
		OutputDir:         cfg.OutputDir,
		OrtLib:            cfg.OrtLibPath,
	}

	var eng engine.ImageEngine
	// Route to backend based on model.json architecture field
	arch, err := readModelArch(cfg.ModelDir(modelName))
	if err != nil {
		// Default to sdxl if model.json missing
		arch = "sdxl"
	}

	switch arch {
	case "sdxl", "sdxl-turbo", "lcm":
		eng = &sdxl.Engine{}
	case "flux", "bonsai":
		eng = &flux.Engine{}
	default:
		return fmt.Errorf("unknown model architecture %q", arch)
	}

	if err := eng.Initialize(cfg.ModelDir(modelName), opts); err != nil {
		return fmt.Errorf("initialize %s: %w", modelName, err)
	}

	activeEngineMu.Lock()
	if activeEngine != nil {
		_ = activeEngine.Close()
	}
	activeEngine = eng
	activeModel = modelName
	activeEngineMu.Unlock()

	logf("[green]Loaded model: %s (%s)[-]", modelName, arch)
	return nil
}

// --- HTTP HANDLERS ---

func corsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
		if r.Method == http.MethodOptions {
			return
		}
		next.ServeHTTP(w, r)
	})
}

func statusHandler(w http.ResponseWriter, r *http.Request) {
	activeEngineMu.RLock()
	info := ""
	if activeEngine != nil {
		info = activeEngine.Info()
	}
	activeEngineMu.RUnlock()

	writeJSON(w, 200, map[string]any{
		"status": "ok",
		"model":  activeModel,
		"info":   info,
	})
}

func modelsHandler(w http.ResponseWriter, r *http.Request) {
	names, err := listModels()
	if err != nil {
		writeJSON(w, 500, map[string]any{"error": err.Error()})
		return
	}
	writeJSON(w, 200, map[string]any{"models": names})
}

func loadHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", 405)
		return
	}
	var req struct {
		Model string `json:"model"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.Model == "" {
		writeJSON(w, 400, map[string]any{"error": "missing model"})
		return
	}
	if err := loadEngine(req.Model, globalCfg); err != nil {
		writeJSON(w, 500, map[string]any{"error": err.Error()})
		return
	}
	writeJSON(w, 200, map[string]any{"status": "loaded", "model": req.Model})
}

func generateHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", 405)
		return
	}

	var req engine.GenerateRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, 400, map[string]any{"error": "bad JSON"})
		return
	}

	select {
	case generateSem <- struct{}{}:
		defer func() { <-generateSem }()
	default:
		writeJSON(w, 429, map[string]any{"error": "server busy: max generation concurrency reached"})
		return
	}

	// Apply defaults
	if req.Width <= 0 {
		req.Width = 512
	}
	if req.Height <= 0 {
		req.Height = 512
	}
	if req.Steps <= 0 {
		req.Steps = 4
	}
	if req.CFGScale <= 0 {
		req.CFGScale = 0.0
	}
	if req.Seed == 0 {
		req.Seed = time.Now().UnixNano()
	}

	activeEngineMu.RLock()
	eng := activeEngine
	activeEngineMu.RUnlock()

	if eng == nil {
		writeJSON(w, 500, map[string]any{"error": "no engine loaded"})
		return
	}

	logf("Generating %dx%d | steps=%d seed=%d | %q", req.Width, req.Height, req.Steps, req.Seed, req.Prompt)
	t0 := time.Now()
	result, err := eng.Generate(req)
	elapsed := time.Since(t0)

	if err != nil {
		logf("[red]Generate error: %v[-]", err)
		writeJSON(w, 500, map[string]any{"error": err.Error()})
		return
	}
	result.DurationMs = elapsed.Milliseconds()
	logf("[green]Done in %v → %s[-]", elapsed, result.ImagePath)
	writeJSON(w, 200, result)
}

// --- HELPERS ---

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func logf(format string, args ...any) {
	fmt.Printf("[zen-paint] "+format+"\n", args...)
}
