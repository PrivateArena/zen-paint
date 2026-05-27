package main

import (
	"encoding/json"
	"fmt"
	"os"
	"runtime"
)

// Config holds the server and engine configuration loaded from config.json.
type Config struct {
	Port              int    `json:"port"`
	DefaultModel      string `json:"default_model"`
	ExecutionProvider string `json:"execution_provider"`
	NumThreads        int    `json:"num_threads"`
	OutputDir         string `json:"output_dir"`
	OrtLibPath        string `json:"ort_lib_path"`
	MaxConcurrency    int    `json:"max_concurrency"`
}

var defaultConfig = Config{
	Port:              7890,
	DefaultModel:      "sdxl-turbo",
	ExecutionProvider: "cpu",
	NumThreads:        0,
	OutputDir:         "/tmp/zen-paint",
	OrtLibPath:        "",
	MaxConcurrency:    1,
}

// LoadConfig reads config.json from the given path.
// Missing values fall back to defaultConfig.
func LoadConfig(path string) (Config, error) {
	cfg := defaultConfig

	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			fmt.Println("[config] config.json not found, using defaults")
			return cfg, nil
		}
		return cfg, fmt.Errorf("read config: %w", err)
	}

	if err := json.Unmarshal(data, &cfg); err != nil {
		return cfg, fmt.Errorf("parse config: %w", err)
	}

	// Auto-detect thread count when set to 0
	if cfg.NumThreads <= 0 {
		cfg.NumThreads = runtime.NumCPU()
		if cfg.NumThreads > 16 {
			cfg.NumThreads = 16
		}
	}

	// Ensure output dir exists
	if err := os.MkdirAll(cfg.OutputDir, 0o755); err != nil {
		return cfg, fmt.Errorf("create output dir: %w", err)
	}

	return cfg, nil
}

// ModelDir returns the resolved path to a model subdirectory.
func (c Config) ModelDir(modelName string) string {
	return "models/" + modelName
}
