package main

import (
	"flag"
	"fmt"
	"os"
	"os/signal"
	"syscall"
)

func main() {
	cfgPath := flag.String("config", "config.json", "path to config.json")
	model := flag.String("model", "", "override default_model from config")
	port := flag.Int("port", 0, "override port from config")
	flag.Parse()

	cfg, err := LoadConfig(*cfgPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "config error: %v\n", err)
		os.Exit(1)
	}

	if *model != "" {
		cfg.DefaultModel = *model
	}
	if *port > 0 {
		cfg.Port = *port
	}

	fmt.Printf("zen-paint v0.1.0 | model=%s provider=%s threads=%d\n",
		cfg.DefaultModel, cfg.ExecutionProvider, cfg.NumThreads)

	StartServer(cfg)

	// Block until SIGINT / SIGTERM
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	StopServer()
}
