package ort

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

var (
	initOnce sync.Once
	initErr  error
)

// Init initializes the ONNX Runtime shared library. Safe to call multiple times.
// libPath may be empty to trigger automatic candidate search.
func Init(libPath string) error {
	initOnce.Do(func() {
		// Env override takes priority
		if envLib := os.Getenv("ORT_SHARED_LIB_PATH"); envLib != "" {
			libPath = envLib
		}

		if libPath == "" {
			candidates := []string{
				"./piper/libonnxruntime.so.1.24.2",
				"../piper/libonnxruntime.so.1.24.2",
				// zen-tts ships the same ORT lib — reuse it
				"../../zen-tts/piper/libonnxruntime.so.1.24.2",
				"../../zen-stt/piper/libonnxruntime.so.1.24.2",
				"./piper/libonnxruntime.so",
				"../piper/libonnxruntime.so",
			}
			for _, c := range candidates {
				abs, err := filepath.Abs(c)
				if err != nil {
					continue
				}
				if _, err := os.Stat(abs); err == nil {
					libPath = abs
					break
				}
			}
		}

		if libPath == "" {
			initErr = fmt.Errorf("onnxruntime shared library not found; set ORT_SHARED_LIB_PATH")
			return
		}

		fmt.Println("[ort] using:", libPath)
		ort.SetSharedLibraryPath(libPath)
		initErr = ort.InitializeEnvironment()
	})
	return initErr
}
