from sklearn.datasets import load_wine
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression

# Step 1: Load the Wine dataset
data = load_wine()
X = data.data  # Features
y = data.target  # Target (wine type)

# Step 2: Shuffle the dataset
X, y = shuffle(X, y, random_state=42)

# Step 3: Split the dataset into training (60%), test (20%), and validation (20%) sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Step 4: Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)

# Step 5: Train the model (Random Forest Classifier)
# model = RandomForestClassifier(random_state=42)
model = LogisticRegression
model.fit(X_train, y_train)

# Step 6: Predict on the validation set
y_val_pred = model.predict(X_val)

# Step 7: Evaluate the model on validation set
val_accuracy = accuracy_score(y_val, y_val_pred)
val_report = classification_report(y_val, y_val_pred, target_names=data.target_names)

print(f"Validation Accuracy: {val_accuracy:.2f}")
print("Validation Classification Report:")
print(val_report)

# Step 8: Predict on the test set
y_test_pred = model.predict(X_test)

# Step 9: Evaluate the model on test set
test_accuracy = accuracy_score(y_test, y_test_pred)
test_report = classification_report(y_test, y_test_pred, target_names=data.target_names)

print(f"Test Accuracy: {test_accuracy:.2f}")
print("Test Classification Report:")
print(test_report)
