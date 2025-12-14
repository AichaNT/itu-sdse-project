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
		WithExec([]string{""}).

	_, err = python.Directory("output").Export(ctx, "output")
	if err != nil {
		return err
	}

	return nil
}
