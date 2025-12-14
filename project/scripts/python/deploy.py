# Imports
import mlflow
import json
import time
from mlflow.tracking import MlflowClient

# Helper function
def wait_for_deployment(model_name, model_version, stage='Staging'): 
    status = False 
    while not status:
        model_version_details = dict( 
            client.get_model_version(name=model_name,version=model_version) 
            )
        if model_version_details['current_stage'] == stage: 
            status = True
            break
        else:
            time.sleep(2) 
    return status 

# Load model info from selection step
with open('./artifacts/metrics/best_model.json') as f:
    best_model = json.load(f)

model_name = best_model['model_name']
model_version = best_model['model_version']
 
mlflow.set_tracking_uri("file:./artifacts/mlruns")
client = MlflowClient()

model_version_details = dict(client.get_model_version(name=model_name,version=model_version)) 
model_status = True 
if model_version_details['current_stage'] != 'Staging': 
    client.transition_model_version_stage(
        name=model_name,
        version=model_version,stage="Staging", 
        archive_existing_versions=True 
    )
    model_status = wait_for_deployment(model_name, model_version, 'Staging') 

# Save the model path
local_path = f'./models/{model_name}'

# Load the model from the registry
model_uri = f"models:/{model_name}/{model_version}"
local_model_path = mlflow.artifacts.download_artifacts(model_uri, dst_path=local_path)


