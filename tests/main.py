from sklearn import datasets
from sklearn.datasets import load_wine
import pandas as pd
from sklearn.decomposition import PCA 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler

data = datasets.load_wine()
# print(data.target)

df = pd.DataFrame(data.data, columns=data.feature_names)
df['label'] = data.target
y = df['label'].values

print(y)
# X = df.drop('label', axis=1).values

# X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# lr = LogisticRegression(max_iter=200)
# lr.fit(X_train, y_train)
# preds = lr.predict(X_test)
# print(preds)


# print(len(data.feature_names))
# print(len(data.target))



# import mlflow 
# from sklearn import datasets
# from sklearn.model_selection import train_test_split 
# from sklearn.datasets import load_diabetes 
# from sklearn.ensemble import RandomForestRegressor 
# from sklearn.linear_model import LogisticRegression 
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# mlflow.autolog()

# X, y = datasets.load_iris(return_X_y=True)
# X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# params = {
#     "solver": "lbfgs",
#     "max_iter": 1000, 
#     "random_state": 8888,
# }
# print(y_train)
# lr = LogisticRegression(**params)
# lr.fit(X_train, y_train)

# # y_pred = lr.predict(X_test)

# # # Calculate metrics
# # accuracy = accuracy_score(y_test, y_pred)
# # precision = precision_score(y_test, y_pred, average="macro")
# # recall = recall_score(y_test, y_pred, average="macro")
# # f1 = f1_score(y_test, y_pred, average="macro")
# # print(accuracy, precision, recall, f1)

# # experiment_name = "MLFlow experiment 01"
# # run_name = "run 01"
# # try:
# #     experiment_id = mlflow.create_experiment(name=experiment_name) 
# # except mlflow.exceptions.MLflowException as e:
# #     experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

# # print(experiment_id)

# # # db = load_diabetes()
# # # X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

# # # rf = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
# # # rf.fit(X_train, y_train)
# # # print(db)