# ITU BDS MLOPS'25 - Project (unfinished)
This repository contains our final project for the course Data Science in Production: MLOps and Software Engineering (Autumn 2025) at the IT University of Copenhagen.

The objective of the project is to restructure a single Python notebook into a modular MLOps pipeline. Key features of the project include:
- Modular Codebase: Clear separation of tasks, including data cleaning, preprocessing, dataset splitting, model training, selection, and deployment.
- Dagger Workflow (Go): Automated pipelines using Dagger to run tasks locally or in CI/CD environments, ensuring reliable and reproducible execution of each pipeline step.
- GitHub Automation Workflow: GitHub Actions handle automated training whenever changes are pushed. (??)

Finally the project output a model artifact produced by GitHub workflow and named 'model'.

## How to Run the Code (unfinished)


## Project Structure (unfinished)
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
    ├── requirements.txt    <- The requirements file for reproducing the analysia environment, e.g.
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

## Authors
Aicha Nadja Thorman, aith@itu.dk

Cæcilie Abildgaard Jeppesen, cjep@itu.dk

## Acknowledgments
This project was created using the CookieCutter template.