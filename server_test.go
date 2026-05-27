package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"zen-paint/internal/engine"
)

type mockEngine struct {
	mu   sync.Mutex
	busy bool
}

func (m *mockEngine) Initialize(modelDir string, opts engine.Options) error {
	return nil
}

func (m *mockEngine) Generate(req engine.GenerateRequest) (engine.GenerateResult, error) {
	m.mu.Lock()
	m.busy = true
	m.mu.Unlock()

	// simulate some work
	time.Sleep(100 * time.Millisecond)

	m.mu.Lock()
	m.busy = false
	m.mu.Unlock()

	return engine.GenerateResult{
		ImagePath:  "test.png",
		DurationMs: 100,
		Width:      512,
		Height:     512,
		Seed:       123,
	}, nil
}

func (m *mockEngine) Info() string {
	return "mock engine for unit tests"
}

func (m *mockEngine) Close() error {
	return nil
}

func TestServerRoutesAndConcurrency(t *testing.T) {
	// Create a temp output directory
	tmpDir, err := os.MkdirTemp("", "zen-paint-server-test-*")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	// Create a dummy output file in the output dir to test static serving
	dummyContent := []byte("dummy image content")
	dummyFile := "test_image.png"
	err = os.WriteFile(filepath.Join(tmpDir, dummyFile), dummyContent, 0644)
	if err != nil {
		t.Fatalf("failed to write dummy file: %v", err)
	}

	cfg := Config{
		Port:              7890,
		DefaultModel:      "test-model",
		ExecutionProvider: "cpu",
		NumThreads:        1,
		OutputDir:         tmpDir,
		MaxConcurrency:    1,
	}

	// Set global config
	globalCfg = cfg
	generateSem = make(chan struct{}, cfg.MaxConcurrency)
	activeEngine = &mockEngine{}
	activeModel = "test-model"

	mux := http.NewServeMux()
	mux.HandleFunc("/status", statusHandler)
	mux.HandleFunc("/generate", generateHandler)
	mux.Handle("/outputs/", http.StripPrefix("/outputs/", http.FileServer(http.Dir(cfg.OutputDir))))

	// 1. Test status route
	req, _ := http.NewRequest("GET", "/status", nil)
	rr := httptest.NewRecorder()
	mux.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Errorf("status handler returned wrong status code: got %v want %v", rr.Code, http.StatusOK)
	}

	var statusResp map[string]any
	if err := json.Unmarshal(rr.Body.Bytes(), &statusResp); err != nil {
		t.Fatalf("failed to unmarshal status response: %v", err)
	}
	if statusResp["model"] != "test-model" {
		t.Errorf("expected model test-model, got %v", statusResp["model"])
	}

	// 2. Test static file serving
	reqFile, _ := http.NewRequest("GET", "/outputs/"+dummyFile, nil)
	rrFile := httptest.NewRecorder()
	mux.ServeHTTP(rrFile, reqFile)

	if rrFile.Code != http.StatusOK {
		t.Errorf("static serving returned wrong status code: got %v want %v", rrFile.Code, http.StatusOK)
	}
	if !bytes.Equal(rrFile.Body.Bytes(), dummyContent) {
		t.Errorf("static serving returned wrong content: got %q want %q", rrFile.Body.Bytes(), dummyContent)
	}

	// 3. Test generate concurrency limit (429)
	var wg sync.WaitGroup
	wg.Add(2)

	codes := make([]int, 2)
	reqBody, _ := json.Marshal(engine.GenerateRequest{
		Prompt: "test prompt",
		Width:  512,
		Height: 512,
		Steps:  4,
	})

	// Fire first request
	go func() {
		defer wg.Done()
		reqGen, _ := http.NewRequest("POST", "/generate", bytes.NewBuffer(reqBody))
		rrGen := httptest.NewRecorder()
		mux.ServeHTTP(rrGen, reqGen)
		codes[0] = rrGen.Code
	}()

	// Wait briefly to ensure first request has entered generateSem but not finished
	time.Sleep(20 * time.Millisecond)

	// Fire second request (should get 429)
	go func() {
		defer wg.Done()
		reqGen, _ := http.NewRequest("POST", "/generate", bytes.NewBuffer(reqBody))
		rrGen := httptest.NewRecorder()
		mux.ServeHTTP(rrGen, reqGen)
		codes[1] = rrGen.Code
	}()

	wg.Wait()

	// One should be 200, the other should be 429
	if (codes[0] == http.StatusOK && codes[1] == http.StatusTooManyRequests) ||
		(codes[1] == http.StatusOK && codes[0] == http.StatusTooManyRequests) {
		// Pass!
	} else {
		t.Errorf("expected one 200 and one 429 response, got codes %v", codes)
	}
}
