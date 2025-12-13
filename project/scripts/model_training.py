# Imports
import os
import pandas as pd
import datetime
import mlflow
import mlflow.pyfunc
import json
import joblib
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report


# Saving constant variables
current_date = datetime.datetime.now().strftime("%Y_%B_%d")
data_version = "00000"
experiment_name = current_date

# Create paths
os.makedirs("./models/mlruns", exist_ok=True)
os.makedirs("/models/mlruns/.trash", exist_ok=True)

# Supress warnings
warnings.filterwarnings('ignore')

mlflow.set_experiment(experiment_name)

# Load training split
train = pd.read_csv("./data/processed/train.csv")
test = pd.read_csv("./data/processed/test.csv")

X_train = train.drop(columns=["lead_indicator"])
y_train = train["lead_indicator"]

X_test = test.drop(columns=["lead_indicator"])
y_test = test["lead_indicator"]

# SKLearn logistic regression
class lr_wrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model 
    
    def predict(self, context, model_input):
        return self.model.predict_proba(model_input)[:, 1]

mlflow.sklearn.autolog(log_input_examples=True, log_models=False) 
experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

with mlflow.start_run(experiment_id=experiment_id) as run: 
    model = LogisticRegression() 
    lr_model_path = "./models/lead_model_lr.pkl"

    params = {
              'solver': ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
              'penalty':  ["none", "l1", "l2", "elasticnet"],
              'C' : [100, 10, 1.0, 0.1, 0.01]
    }

    model_grid = RandomizedSearchCV(model, param_distributions= params, verbose=3, n_iter=10, cv=3)
    model_grid.fit(X_train, y_train)

    best_model = model_grid.best_estimator_ 

    y_pred_train = model_grid.predict(X_train)
    y_pred_test = model_grid.predict(X_test)

    # log artifacts
    mlflow.log_metric('f1_score', f1_score(y_test, y_pred_test))
    mlflow.log_artifacts("artifacts", artifact_path="model")
    mlflow.log_param("data_version", "00000")
    
    # store model for model interpretability
    joblib.dump(value=best_model, filename=lr_model_path)
        
    # Custom python model for predicting probability
    mlflow.pyfunc.log_model('model', python_model=lr_wrapper(best_model)) # changed from model to best_model

# Classification report for test data
model_classification_report = classification_report(y_test, y_pred_test, output_dict=True)

model_results = {
    lr_model_path: model_classification_report
}

# Save columns and model results
column_list_path = './artifacts/columns_list.json'
with open(column_list_path, 'w+') as columns_file:
    columns = {'column_names': list(X_train.columns)}
    json.dump(columns, columns_file)

model_results_path = "./models/model_results.json"
with open(model_results_path, 'w+') as results_file:
    json.dump(model_results, results_file)