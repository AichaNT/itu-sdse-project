# Imports
import time
import mlflow
import datetime
import json
from mlflow.tracking.client import MlflowClient 
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus

# Helper function
def wait_until_ready(model_name, model_version):
    client = MlflowClient()
    for _ in range(10):
        model_version_details = client.get_model_version( 
          name=model_name, 
          version=model_version,
        )
        status = ModelVersionStatus.from_string(model_version_details.status) 
        if status == ModelVersionStatus.READY: 
            break
        time.sleep(1) 

# Constants used:
current_date = datetime.datetime.now().strftime("%Y_%B_%d")
artifact_path = "model" 
model_name = "lead_model" 
experiment_name = current_date 

mlflow.set_tracking_uri("file:./artifacts/mlruns")
experiment_ids = [mlflow.get_experiment_by_name(experiment_name).experiment_id] 

experiment_best = mlflow.search_runs(
    experiment_ids=experiment_ids,
    order_by=["metrics.f1_score DESC"], 
    max_results=1 
).iloc[0] 

# Get production model
client = MlflowClient() 
prod_model = [model for model in client.search_model_versions(f"name='{model_name}'") if dict(model)['current_stage']=='Production']
prod_model_exists = len(prod_model)>0  

if prod_model_exists:
    prod_model_version = dict(prod_model[0])['version']
    prod_model_run_id = dict(prod_model[0])['run_id'] 

# Compare porductionand best trained model
train_model_score = experiment_best["metrics.f1_score"] 
model_status = {} 
run_id = None

if prod_model_exists:
    data, details = mlflow.get_run(prod_model_run_id) 
    prod_model_score = data[1]["metrics.f1_score"] 

    model_status["current"] = train_model_score 
    model_status["prod"] = prod_model_score 

    if train_model_score>prod_model_score: 
        run_id = experiment_best["run_id"]
else:
    run_id = experiment_best["run_id"]

# Register best model
if run_id is not None:
    model_uri = "runs:/{run_id}/{artifact_path}".format( 
        run_id=run_id,
        artifact_path=artifact_path
    )
    model_details = mlflow.register_model(model_uri=model_uri, name=model_name) 
    wait_until_ready(model_details.name, model_details.version) 
    model_details = dict(model_details) 

# Saving model information
with open('./artifacts/best_model.json', 'w') as f:
    json.dump({
        "model_name": model_name,
        "model_version": model_details['version']
    }, f)