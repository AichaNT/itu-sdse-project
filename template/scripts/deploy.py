# Imports
import mlflow
from mlflow.tracking import MlflowClient

# Helper function
def wait_for_deployment(model_name, model_version, stage='Staging'): #C: waits until the specified model version reaches the desired stage
    status = False #C: flag to track deployment status
    while not status:
        model_version_details = dict( 
            client.get_model_version(name=model_name,version=model_version) #C: get details of the model version
            )
        if model_version_details['current_stage'] == stage: #C: check if model reached target stage
            print(f'Transition completed to {stage}') #C: remove or log(?)
            status = True
            break
        else:
            time.sleep(2) #C: wait 2 seconds before checking again
    return status #C: return True when deployment is complete


model_version = 1 #C: manually specify the version number of the model

client = MlflowClient()

model_version_details = dict(client.get_model_version(name=model_name,version=model_version)) #C: fetch current model version details
model_status = True #C: flag to track whether model transition happened successfully
if model_version_details['current_stage'] != 'Staging': #C: check if model is already in Staging
    client.transition_model_version_stage( #C: move a registered model version to a different stage in the MLflow Model Registry
        name=model_name,
        version=model_version,stage="Staging", 
        archive_existing_versions=True #C: archive any other versions currently in Staging
    )
    model_status = wait_for_deployment(model_name, model_version, 'Staging') #C: wait until model reaches Staging
else:
    print('Model already in staging') #C: remove or log(?)