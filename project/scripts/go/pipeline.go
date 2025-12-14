package main

import (
	"context"
	"fmt"

	"dagger.io/dagger"
)

func main() {
	// Create a shared context
	ctx := context.Background()

	// Run the stages of the pipeline
	if err := Build(ctx); err != nil {
		fmt.Println("Error:", err)
		panic(err)
	}
}

func Build(ctx context.Context) error {
	// Initialize Dagger client
	client, err := dagger.Connect(ctx)
	if err != nil {
		return err
	}
	defer client.Close()

	// Define Container with requirements, bash commands and python scripts
	python := client.Container().
		From("python:3.12.2-bookworm").
		WithDirectory("/app", client.Host().Directory("project")).
		WithWorkdir("/app").
		WithExec([]string{"pip", "install", "-r", "requirements.txt"}).
		WithExec([]string{"dvc", "pull"}).
		WithExec([]string{"python", "scripts/python/data_clean.py"}).
		WithExec([]string{"python", "scripts/python/data_preprocess.py"}).
		WithExec([]string{"python", "scripts/python/data_split.py"}).
		WithExec([]string{"python", "scripts/python/model_training.py"}).
		WithExec([]string{"python", "scripts/python/model_selection.py"}).
		WithExec([]string{"python", "scripts/python/deploy.py"})

	_, err = python.Directory("./artifacts").Export(ctx, "./artifacts")
	if err != nil {
		return fmt.Errorf("export failed, %w", err)
	}

	return nil
}
