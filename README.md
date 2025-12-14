# ITU BDS MLOPS'25 - Project (unfinished)
This repository contains our final project for the course Data Science in Production: MLOps and Software Engineering (Autumn 2025) at the IT University of Copenhagen.

The objective of the project is to transform a given Python notebook monolith into a modular and production-ready MLOps pipeline. The project includes:
- Modular Codebase: Clear separation of task (data cleaning, preprocessing, splitting as well as model training, selection and deployment).
- Dagger workflow (in Go): Automated pipelines using Dagger (Go) to run tasks locally or in CI/CD environments, ensuring that each step can run reliably and reproducibly.
- GitHub automation workflow: GitHub Actions handle automated training whenever changes are pushed. (??)

Finally the project output a model artifact produced by GitHub workflow and named 'model'.

## How to run the code

## Project Organization / structure

```
├── README.md           <-
│
├── .github             <-
│   └── workflows       <- 
│       ├– dagger.yml   <-
│       └– dagger.yml   <-
│
└── project             <- 
    │         
    ├── .dvc                <- 
    │   ├– .gitignore       <- should be ignored
    │   └– config           <-
    │
    ├── data
    │   ├── interim         <- Intermediate data that has been transformed.
    │   ├── processed       <- The final data sets for modeling.
    │   └── raw             <- The original dataset.
    │
    ├── docs                <- (???)
    │
    ├── models              <- The final trained model
    │
    ├── notebooks           <- Jupyter notebooks.
    │   └── main.ipynb      <- The orignal notebook provided as part of the project.
    │
    ├── artifacts           <- (should the subfolders be inc)
    │   ├── scaler.pkl      <-
    │   ├── metrics         <-
    │   └── temp_models     <-
    │
    ├── requirements.txt    <- The requirements file for reproducing the analysis environment, e.g.
    │                          generated with `pip freeze > requirements.txt`
    │
    └── scripts             <- Source code for use in this project.
        │
        ├── __init__.py             <- Makes scripts a Python module
        │
        ├── go                      <-         
        │   ├── go.mod              <-
        │   ├── go.sum              <-        
        │   └── pipeline.go         <- 
        │
        ├── python                  <-  
        │   ├── data_clean.py       <-
        │   ├── data_preprocess.py  <-
        │   ├── data_split.py       <-
        │   ├── model_training.py   <-      
        │   ├── model_selection.py  <-         
        │   └── deploy.py           <- 
        │
        └── model_inference.py      <- 
```