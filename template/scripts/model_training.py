# Imports
import os
import datetime
import mlflow
import mlflow.pyfunc
import json
import joblib 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report


# Saving constant variables
current_date = datetime.datetime.now().strftime("%Y_%B_%d")
data_gold_path = "./artifacts/train_data_gold.csv" 
data_version = "00000"
experiment_name = current_date

# Create paths
os.makedirs("mlruns", exist_ok=True)
os.makedirs("mlruns/.trash", exist_ok=True)

mlflow.set_experiment(experiment_name)

# Load training split
#

# SKLearn logistic regression
class lr_wrapper(mlflow.pyfunc.PythonModel): #C: custom wrapper for MLflow to log predict_proba outputs
    def __init__(self, model):
        self.model = model #C: wrapper stores model internally for use in prediction.
    
    def predict(self, context, model_input):
        return self.model.predict_proba(model_input)[:, 1] #C: probability for each class, extracting only probability of "class 1"

mlflow.sklearn.autolog(log_input_examples=True, log_models=False) #C: automatically log parameters, metrics, and input examples (log_input_examples=True)
experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id #C: get experiment ID

with mlflow.start_run(experiment_id=experiment_id) as run: #C: start MLflow run
    model = LogisticRegression() #C: create logistic regression (LR) model
    lr_model_path = "./artifacts/lead_model_lr.pkl" #C: path to save LR model in artifacts folder

    #C: define hyperparameter search space
    params = {
              'solver': ["newton-cg", "lbfgs", "liblinear", "sag", "saga"], #C: optimization algorithm
              'penalty':  ["none", "l1", "l2", "elasticnet"], #C: regularization type
              'C' : [100, 10, 1.0, 0.1, 0.01] #C: regularization strength (inverse)
    }

    #C: random search for best hyperparameters
    model_grid = RandomizedSearchCV(model, param_distributions= params, verbose=3, n_iter=10, cv=3)
    model_grid.fit(X_train, y_train) #C: fit model grid on training data

    best_model = model_grid.best_estimator_ #C: get best LR model

    y_pred_train = model_grid.predict(X_train) #C: predict training labels
    y_pred_test = model_grid.predict(X_test) #C: predict test labels

    # log artifacts - C: to MLflow
    mlflow.log_metric('f1_score', f1_score(y_test, y_pred_test))
    mlflow.log_artifacts("artifacts", artifact_path="model")
    mlflow.log_param("data_version", "00000")
    
    # store model for model interpretability
    joblib.dump(value=model, filename=lr_model_path)
        
    # Custom python model for predicting probability - C: logging to MLflow
    mlflow.pyfunc.log_model('model', python_model=lr_wrapper(best_model)) # changed from model to best_model

#C: classification report for test data
model_classification_report = classification_report(y_test, y_pred_test, output_dict=True)

model_results = {
    lr_model_path: model_classification_report
}

# Save columns and model results
column_list_path = './artifacts/columns_list.json' #C: path to save list of feature columns in articats folder
with open(column_list_path, 'w+') as columns_file:
    columns = {'column_names': list(X_train.columns)} #C: create dictionary of column names from training data
    json.dump(columns, columns_file) #C: save column names to JSON file

model_results_path = "./artifacts/model_results.json" #C: path to save model results in articats folder
with open(model_results_path, 'w+') as results_file:
    json.dump(model_results, results_file) #C: save model evaluation metrics to JSON file