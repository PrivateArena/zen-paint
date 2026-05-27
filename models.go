package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// modelMeta mirrors the model.json descriptor in each model subdirectory.
type modelMeta struct {
	Name         string            `json:"name"`
	Architecture string            `json:"architecture"`
	Files        map[string]string `json:"files"`
	DefaultSteps int               `json:"default_steps"`
	DefaultCFG   float32           `json:"default_cfg"`
	MaxWidth     int               `json:"max_width"`
	MaxHeight    int               `json:"max_height"`
	Description  string            `json:"description"`
}

// readModelArch reads only the architecture field from a model directory's model.json.
func readModelArch(modelDir string) (string, error) {
	data, err := os.ReadFile(filepath.Join(modelDir, "model.json"))
	if err != nil {
		return "", err
	}
	var m modelMeta
	if err := json.Unmarshal(data, &m); err != nil {
		return "", err
	}
	if m.Architecture == "" {
		return "", fmt.Errorf("model.json missing architecture field")
	}
	return m.Architecture, nil
}

// listModels returns names of all valid model subdirectories under models/.
func listModels() ([]string, error) {
	entries, err := os.ReadDir("models")
	if err != nil {
		if os.IsNotExist(err) {
			return []string{}, nil
		}
		return nil, err
	}

	var names []string
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		if _, err := os.Stat(filepath.Join("models", e.Name(), "model.json")); err == nil {
			names = append(names, e.Name())
		}
	}
	return names, nil
}
