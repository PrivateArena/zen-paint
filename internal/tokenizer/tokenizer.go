package tokenizer

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

type Pair struct {
	A, B string
}

type ClipTokenizer struct {
	encoder  map[string]int32
	decoder  map[int32]string
	b2u      map[byte]rune
	u2b      map[rune]byte
	bpeRanks map[Pair]int
	cache    map[string][]string
	re       *regexp.Regexp
}

// NewClipTokenizer loads vocabulary and merges from modelDir, or falls back to simple ASCII.
func NewClipTokenizer(modelDir string) (*ClipTokenizer, error) {
	vocabPath := filepath.Join(modelDir, "vocab.json")
	mergesPath := filepath.Join(modelDir, "merges.txt")

	if _, err := os.Stat(vocabPath); err != nil {
		vocabPath = filepath.Join(modelDir, "tokenizer", "vocab.json")
		mergesPath = filepath.Join(modelDir, "tokenizer", "merges.txt")
	}

	t := &ClipTokenizer{
		encoder:  make(map[string]int32),
		decoder:  make(map[int32]string),
		b2u:      bytesToUnicode(),
		u2b:      make(map[rune]byte),
		bpeRanks: make(map[Pair]int),
		cache:    make(map[string][]string),
		re:       regexp.MustCompile(`(?i)'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+`),
	}

	for k, v := range t.b2u {
		t.u2b[v] = k
	}

	vocabData, err := os.ReadFile(vocabPath)
	if err != nil {
		fmt.Printf("[ClipTokenizer] Warning: %v. Using fallback ascii tokenizer.\n", err)
		return t, nil
	}

	var vocab map[string]int32
	if err := json.Unmarshal(vocabData, &vocab); err != nil {
		return nil, fmt.Errorf("unmarshal vocab.json: %w", err)
	}
	t.encoder = vocab
	for k, v := range vocab {
		t.decoder[v] = k
	}

	mergesFile, err := os.Open(mergesPath)
	if err != nil {
		fmt.Printf("[ClipTokenizer] Warning: %v. Using fallback merges.\n", err)
		return t, nil
	}
	defer mergesFile.Close()

	scanner := bufio.NewScanner(mergesFile)
	rank := 0
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		parts := strings.Fields(line)
		if len(parts) == 2 {
			t.bpeRanks[Pair{A: parts[0], B: parts[1]}] = rank
			rank++
		}
	}

	return t, nil
}

// Encode converts prompt to token IDs. Pad to maxLen (77).
func (t *ClipTokenizer) Encode(text string, maxLen int) []int32 {
	tokens := make([]int32, maxLen)

	if len(t.encoder) == 0 {
		tokens[0] = 49406 // <|startoftext|>
		for i, b := range []byte(text) {
			if i+1 >= maxLen-1 {
				break
			}
			tokens[i+1] = int32(b)
		}
		tokens[maxLen-1] = 49407 // <|endoftext|>
		return tokens
	}

	sotID, ok := t.encoder["<|startoftext|>"]
	if !ok {
		sotID = 49406
	}
	eotID, ok := t.encoder["<|endoftext|>"]
	if !ok {
		eotID = 49407
	}

	tokens[0] = sotID

	text = strings.ToLower(text)
	text = regexp.MustCompile(`\s+`).ReplaceAllString(text, " ")
	text = strings.TrimSpace(text)

	matches := t.re.FindAllString(text, -1)
	tokenIdx := 1

	for _, token := range matches {
		if tokenIdx >= maxLen-1 {
			break
		}

		var bpeToken strings.Builder
		for i := 0; i < len(token); i++ {
			bpeToken.WriteRune(t.b2u[token[i]])
		}

		subwords := t.bpe(bpeToken.String())
		for _, sw := range subwords {
			if tokenIdx >= maxLen-1 {
				break
			}
			id, ok := t.encoder[sw]
			if ok {
				tokens[tokenIdx] = id
				tokenIdx++
			}
		}
	}

	for i := tokenIdx; i < maxLen; i++ {
		tokens[i] = eotID
	}

	return tokens
}

func (t *ClipTokenizer) bpe(token string) []string {
	if val, ok := t.cache[token]; ok {
		return val
	}

	word := make([]string, 0, len(token))
	for _, r := range token {
		word = append(word, string(r))
	}

	if len(word) == 0 {
		return nil
	}

	for {
		pairs := getPairs(word)
		if len(pairs) == 0 {
			break
		}

		var bestPair Pair
		bestRank := -1

		for _, p := range pairs {
			if rank, ok := t.bpeRanks[p]; ok {
				if bestRank == -1 || rank < bestRank {
					bestRank = rank
					bestPair = p
				}
			}
		}

		if bestRank == -1 {
			break
		}

		var newWord []string
		i := 0
		for i < len(word) {
			if i < len(word)-1 && word[i] == bestPair.A && word[i+1] == bestPair.B {
				newWord = append(newWord, bestPair.A+bestPair.B)
				i += 2
			} else {
				newWord = append(newWord, word[i])
				i++
			}
		}
		word = newWord
	}

	t.cache[token] = word
	return word
}

func getPairs(word []string) []Pair {
	pairs := make([]Pair, len(word)-1)
	for i := 0; i < len(word)-1; i++ {
		pairs[i] = Pair{A: word[i], B: word[i+1]}
	}
	return pairs
}

func bytesToUnicode() map[byte]rune {
	b2u := make(map[byte]rune)
	for b := 33; b <= 126; b++ {
		b2u[byte(b)] = rune(b)
	}
	for b := 161; b <= 172; b++ {
		b2u[byte(b)] = rune(b)
	}
	for b := 174; b <= 255; b++ {
		b2u[byte(b)] = rune(b)
	}
	n := 0
	for b := 0; b < 256; b++ {
		inRange1 := (b >= 33 && b <= 126)
		inRange2 := (b >= 161 && b <= 172)
		inRange3 := (b >= 174 && b <= 255)
		if !inRange1 && !inRange2 && !inRange3 {
			b2u[byte(b)] = rune(256 + n)
			n++
		}
	}
	return b2u
}

type T5Tokenizer struct {
	encoder map[string]int64
	decoder map[int64]string
}

// NewT5Tokenizer loads vocabulary from modelDir, or falls back to simple ASCII.
func NewT5Tokenizer(modelDir string) (*T5Tokenizer, error) {
	vocabPath := filepath.Join(modelDir, "vocab_t5.json")
	if _, err := os.Stat(vocabPath); err != nil {
		vocabPath = filepath.Join(modelDir, "tokenizer", "vocab_t5.json")
	}

	t := &T5Tokenizer{
		encoder: make(map[string]int64),
		decoder: make(map[int64]string),
	}

	vocabData, err := os.ReadFile(vocabPath)
	if err != nil {
		fmt.Printf("[T5Tokenizer] Warning: %v. Using fallback ascii tokenizer.\n", err)
		return t, nil
	}

	var vocab map[string]int64
	if err := json.Unmarshal(vocabData, &vocab); err != nil {
		return nil, fmt.Errorf("unmarshal vocab_t5.json: %w", err)
	}
	t.encoder = vocab
	for k, v := range vocab {
		t.decoder[v] = k
	}

	return t, nil
}

// Encode encodes text using greedy longest match, falling back to bytes if empty.
func (t *T5Tokenizer) Encode(text string, maxLen int) []int64 {
	tokens := make([]int64, maxLen)

	if len(t.encoder) == 0 {
		for i, b := range []byte(text) {
			if i >= maxLen {
				break
			}
			tokens[i] = int64(b)
		}
		return tokens
	}

	cleaned := " " + strings.ReplaceAll(text, " ", " ")
	runes := []rune(cleaned)
	tokenIdx := 0

	for i := 0; i < len(runes); {
		if tokenIdx >= maxLen {
			break
		}

		matchLen := 0
		var matchID int64
		found := false

		for j := len(runes); j > i; j-- {
			sub := string(runes[i:j])
			if id, ok := t.encoder[sub]; ok {
				matchLen = j - i
				matchID = id
				found = true
				break
			}
		}

		if found {
			tokens[tokenIdx] = matchID
			tokenIdx++
			i += matchLen
		} else {
			i++
		}
	}

	return tokens
}
