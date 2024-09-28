import mlflow 
import mlflow.sklearn 
import mlflow.exceptions
from mlflow.models import infer_signature
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mlflow.autolog()

data = pd.read_csv('../datasets/CO2_emission.csv')

def preprocess_input(input_data: pd.DataFrame) -> pd.DataFrame:
    input_data.rename(columns={'Fuel_Consumption_in_City(L/100 km)': 'Fuel_Consumption_in_City'}, inplace=True)

    input_data.rename(columns={'Fuel_Consumption_in_City_Hwy(L/100 km)': 'Fuel_Consumption_in_City_Hwy'}, inplace=True)

    input_data.rename(columns={'Fuel_Consumption_in_City_comb(L/100 km)': 'Fuel_Consumption_in_City_comb'}, inplace=True)

    input_data = input_data[['Make', 'Model', 'Vehicle_Class', 'Engine_Size',
        'Transmission', 'Fuel_Consumption_in_City', 'Smog_Level', 'CO2_Emissions']]


    # Initialize LabelEncoder
    label_encoder = LabelEncoder()

    # Encode categorical columns
    input_data['Make'] = label_encoder.fit_transform(input_data['Make'])
    input_data['Model'] = label_encoder.fit_transform(input_data['Model'])
    input_data['Vehicle_Class'] = label_encoder.fit_transform(input_data['Vehicle_Class'])
    input_data['Transmission'] = label_encoder.fit_transform(input_data['Transmission'])

y = data['CO2_Emissions']
X = data.drop(columns=['CO2_Emissions'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Parameters for grid search
param_grid = {
    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
    'max_depth': [None, 3, 5, 7, 10, 13],
    'max_features': [None, 'auto', 'sqrt', 'log2'],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4, 6]
}

# Training model with grid search
cv = GridSearchCV(DecisionTreeRegressor(), param_grid)
cv.fit(X_train, y_train)
best_model = cv.best_estimator_
y_pred = best_model.predict(X_test)
r2 = r2_score(y_test, y_pred)

print(r2)
print(best_model)

experiment_name = "CO2 Emission Prediction"
run_name = "run 01"

try:
    experiment_id = mlflow.create_experiment(name=experiment_name)
except mlflow.exceptions.MlflowException as e:
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
print(experiment_id)

with mlflow.start_run(run_name=run_name, experiment_id=experiment_id) as run:
    # Predictions
    y_pred = best_model.predict(X_test)

    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    # Log parameters
    mlflow.log_params(cv.best_params_)
    
    # Log metrics
    mlflow.log_metric("R2_Score", r2)
    mlflow.log_metric("Mean_Absolute_Error", mae)
    mlflow.log_metric("Mean_Squared_Error", mse)

    # Infer model signature
    signature = infer_signature(X_test, y_test)

    # Log the model
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="co2_emission_model",
        signature=signature,
        input_example=X_test.iloc[:5]  # Example input for model logging
    )

    # Save the model locally
    local_model_path = "./models/co2_emission_model"
    mlflow.sklearn.save_model(best_model, local_model_path)

    print(f"Best model: {best_model}")
    print(f"R² Score: {r2}")
    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Squared Error: {mse}")
