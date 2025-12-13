# Imports
import time
import mlflow
import datetime
from mlflow.tracking.client import MlflowClient #C: MLflow client to interact with tracking server
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus #C: enumeration for model version status

# Helper function
def wait_until_ready(model_name, model_version):
    client = MlflowClient() #C: create MLflow client instance
    for _ in range(10):
        model_version_details = client.get_model_version( #C: fetch details of specific model version
          name=model_name, 
          version=model_version,
        )
        status = ModelVersionStatus.from_string(model_version_details.status) #C: convert status string to enumeration
        if status == ModelVersionStatus.READY: #C: stop waiting if model is ready
            break
        time.sleep(1) #C: wait 1 second before checking again

# Constants used:
current_date = datetime.datetime.now().strftime("%Y_%B_%d") #C: gets current date and time and formats it as yyyy_monthname_dd
artifact_path = "model" #C: folder path to save MLflow artifacts
model_name = "lead_model" #C: name of the model
experiment_name = current_date #C: uses the current date as the experiment name

experiment_ids = [mlflow.get_experiment_by_name(experiment_name).experiment_id] #C: get the ID of the experiment

experiment_best = mlflow.search_runs(
    experiment_ids=experiment_ids, #C: search runs within this experiment
    order_by=["metrics.f1_score DESC"], #C: sort runs by F1-score descending
    max_results=1 #C: return only the top run
).iloc[0] #C: select the first row (best run)

# Get production model
client = MlflowClient() #C: create MLflow client instance
prod_model = [model for model in client.search_model_versions(f"name='{model_name}'") if dict(model)['current_stage']=='Production'] #C: search for all registered versions of the model that are in 'Production' stage
prod_model_exists = len(prod_model)>0  #C: check if any production model exists

if prod_model_exists:
    prod_model_version = dict(prod_model[0])['version'] #C: get version number of first production model
    prod_model_run_id = dict(prod_model[0])['run_id'] #C: get run ID of first production model

# Compare porductionand best trained model
train_model_score = experiment_best["metrics.f1_score"] #C: F1-score of the best run from current experiment - this is only LR runs (XGBOOSt was not wrapped in MLflow run)
model_details = {}
model_status = {} #C: placeholder to track current vs production scores
run_id = None

if prod_model_exists:
    data, details = mlflow.get_run(prod_model_run_id) #C: fetch metrics and details of current production model
    prod_model_score = data[1]["metrics.f1_score"] #C: extract F1-score of production model

    model_status["current"] = train_model_score #C: store F1-score of candidate model
    model_status["prod"] = prod_model_score #C: store F1-score of production model

    if train_model_score>prod_model_score: #C: compare scores - candidate model is better; mark for registration
        run_id = experiment_best["run_id"] #C: store run ID of best candidate model
else:
    run_id = experiment_best["run_id"] #C: register the best candidate model by default

# Register best model
if run_id is not None:
    model_uri = "runs:/{run_id}/{artifact_path}".format( #C: create MLflow URI to locate model artifacts
        run_id=run_id,
        artifact_path=artifact_path
    )
    model_details = mlflow.register_model(model_uri=model_uri, name=model_name) #C: register the model in MLflow Model Registry
    wait_until_ready(model_details.name, model_details.version) #C: wait until the model version is fully available in the registry
    model_details = dict(model_details) #C: convert model details to dictionary for easier access/logging
