# ITU BDS MLOPS'25 - gallop gals
This repository contains our final project for the course Data Science in Production: MLOps and Software Engineering (Autumn 2025) at the IT University of Copenhagen.

For this project, we were given a jupyter notebook containing a complete ML process from data cleaning and preprocessing to model training, selection and deployment.
We were tasked to split this notebook into separate .py scripts and create an automated workflow to train and test the final model.
The picture below provides a detailed overview of the structure that the project is expected to follow:

![project structure](/docs/project-architecture.png)

The final output of the automated pipeline is a single model artifact found in Github Actions called 'model'.


## Project Overview

The goal of this project is to build a machine learning model that classifies website users to identify potential new customers, where:
- Input: Behaviour data collected from users.
- Output: A binary classification that determines whether a user has converted into a customer.

This repository contains an end-to-end **MLOps pipeline** for training and deploying a machine learning model.
The project follows a structured, reproducible workflow combining:

- **Python** for data processing and model training
- **DVC** to fetch data
- **MLflow** for experiment tracking and model registry
- **Dagger (Go)** for pipeline orchestration
- **GitHub Actions** for automated CI training and testing

The pipeline trains a model, exports the final trained model artifact, and validates it using an external model validator.

---

## Repository Structure

```
.
├── .github/
│   └── workflows/
│       └── train-and-test.yml     # GitHub Actions workflow
│
├── project/
|   ├── .dvc/
│   ├── data/
│   │   ├── raw/                   # Raw data
│   │   ├── interim/               # Intermediate datasets
│   │   └── processed/             # Train/test splits
│   │
│   ├── models/
│   │   └── model.pkl              # Final exported model
|   |
│   ├── scripts/
│   │   └── python/
│   │       ├── data_clean.py
│   │       ├── data_preprocess.py
│   │       ├── data_split.py
│   │       ├── model_training.py
|   |       └── production/        # Unused scripts for potential future production use
│   │           ├── model_selection.py
│   │           └── deploy.py
|   |
│   ├── notebooks/                 # Original notebook (for reference)
│   └── requirements.txt
│
├── pipeline.go                    # Dagger pipeline
├── go.mod
├── go.sum
└── README.md
```

---

## Pipeline Description

### Python Scripts

The ML workflow is split into modular Python scripts:

1. **Data cleaning** – removes inconsistencies and prepares raw data
2. **Preprocessing** – feature engineering and transformations
3. **Data splitting** – creates train/test datasets
4. **Model training** – trains models and outputs final model

---

### Go / Dagger Pipeline

The `pipeline.go` file defines a Dagger pipeline that:

- Creates a Python container
- Installs dependencies
- Pulls data using DVC
- Runs all Python scripts in sequence
- Exports the final trained model from the container to the host filesystem

This ensures a reproducible and isolated training environment.

---

### GitHub Actions Workflow

The GitHub Actions workflow:

1. Runs the Dagger pipeline to train the model
2. Uploads the trained model as a workflow artifact
3. Executes a model validation step using a predefined validator action

This makes the entire training and testing process fully automated.

---

## How to Run the Project

### Recommended Method (GitHub Actions)

The project is designed to be run via GitHub Actions.

1. Go to the **Actions** tab in the repository
2. Select **Train and Test Model**
3. Click **Run workflow**
4. Choose the branch and start the run

After completion:

- The trained model is available as a workflow artifact
- The model is automatically validated

---

## Authors
Cæcilie Abildgaard Jeppesen, cjep@itu.dk

Aicha Nadja Thorman, aith@itu.dk

## Acknowledgments
This project was created using the CookieCutter template.