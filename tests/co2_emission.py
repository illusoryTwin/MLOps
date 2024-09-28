import mlflow 
import mlflow.sklearn 
import mlflow.exceptions
from mlflow.models import infer_signature
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

mlflow.autolog()

data = pd.read_csv('data/CO2_emission.csv')

data = data[['Make', 'Model', 'Vehicle_Class', 'Engine_Size',
       'Transmission','Fuel_Consumption_in_City_Hwy(L/100 km)', 'Smog_Level', 'CO2_Emissions']]

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Encode categorical columns
data['Make'] = label_encoder.fit_transform(data['Make'])
data['Model'] = label_encoder.fit_transform(data['Model'])
data['Vehicle_Class'] = label_encoder.fit_transform(data['Vehicle_Class'])
data['Transmission'] = label_encoder.fit_transform(data['Transmission'])

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

#Training model with grid search
cv=GridSearchCV(DecisionTreeRegressor(),param_grid)
cv.fit(X_train,y_train)
best_model = cv.best_estimator_
y_pred = best_model.predict(X_test)
print(y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
r2 = r2_score(y_test,y_pred)

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

    
    mlflow.log_params(params=param_grid)

    # Log the performance metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1", f1)
    mlflow.log_metric("r2", r2)
    mlflow.log_metrics({
        "accuracy": accuracy,
        "f1": f1,
        "r2": r2
    })


