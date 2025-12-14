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
		WithDirectory("/project", client.Host().Directory("project")). // Set directory
		WithWorkdir("/project").
		WithExec([]string{"git", "init"}).
		WithExec([]string{"pip", "install", "-r", "requirements.txt"}).                         // Install requirements
		WithExec([]string{"bash", "-c", "cd data/raw && dvc update raw_data.csv && dvc pull"}). // Pull data tracked by dvc
		WithExec([]string{"python", "scripts/python/data_clean.py"}).                           // Run data cleaning
		WithExec([]string{"python", "scripts/python/data_preprocess.py"}).                      // Run data preprocessing
		WithExec([]string{"python", "scripts/python/data_split.py"}).                           // Split data for training and validation
		WithExec([]string{"python", "scripts/python/model_training.py"}).                       // Train model

	_, err = python.Directory("models").Export(ctx, "project/models")
	if err != nil {
		return fmt.Errorf("export failed, %w", err)
	}

	return nil
}
