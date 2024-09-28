import mlflow.pyfunc
from mlflow import MlflowClient
run_id = "e389609f9f1b44678ea7fea020453f94"
model_artifact_path = "pytorch model"
model = mlflow.pyfunc.load_model(model_uri=f"runs:/{run_id}/{model_artifact_path}")
print(model.metadata)
# OR
model_name = "pytorch_model"
model_version = 1
model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
print(model.metadata)
# OR
client = MlflowClient()
client.set_registered_model_alias(name = model_name, alias
= "staging", version = "1")
model_name = "pytorch_model"
model_alias = "staging"
model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}@{model_alias}")
print(model.metadata)