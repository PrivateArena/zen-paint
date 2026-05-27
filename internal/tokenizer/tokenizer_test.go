package tokenizer

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestFallbackTokenizers(t *testing.T) {
	// ClipTokenizer fallback test
	clipTok, err := NewClipTokenizer("/nonexistent-dir")
	if err != nil {
		t.Fatalf("expected NewClipTokenizer to succeed on nonexistent dir, got %v", err)
	}

	clipEnc := clipTok.Encode("hello world", 77)
	if len(clipEnc) != 77 {
		t.Errorf("expected length 77, got %d", len(clipEnc))
	}
	if clipEnc[0] != 49406 {
		t.Errorf("expected SOT 49406, got %d", clipEnc[0])
	}

	// T5Tokenizer fallback test
	t5Tok, err := NewT5Tokenizer("/nonexistent-dir")
	if err != nil {
		t.Fatalf("expected NewT5Tokenizer to succeed on nonexistent dir, got %v", err)
	}

	t5Enc := t5Tok.Encode("hello world", 256)
	if len(t5Enc) != 256 {
		t.Errorf("expected length 256, got %d", len(t5Enc))
	}
}

func TestClipTokenizerActualBPE(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "tokenizer-test-*")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	// Create dummy vocab.json
	vocab := map[string]int32{
		"<|startoftext|>": 49406,
		"<|endoftext|>":   49407,
		"he":              100,
		"llo":             101,
		"world":           102,
		"h":               103,
		"e":               104,
		"l":               105,
		"o":               106,
	}
	vocabBytes, _ := json.Marshal(vocab)
	_ = os.WriteFile(filepath.Join(tmpDir, "vocab.json"), vocabBytes, 0644)

	// Create dummy merges.txt
	merges := "#version: 0.2\nh e\nl l\nll o\n"
	_ = os.WriteFile(filepath.Join(tmpDir, "merges.txt"), []byte(merges), 0644)

	clipTok, err := NewClipTokenizer(tmpDir)
	if err != nil {
		t.Fatalf("failed to create ClipTokenizer: %v", err)
	}

	enc := clipTok.Encode("hello world", 10)
	if len(enc) != 10 {
		t.Errorf("expected length 10, got %d", len(enc))
	}

	if enc[0] != 49406 {
		t.Errorf("expected SOT, got %d", enc[0])
	}
}
