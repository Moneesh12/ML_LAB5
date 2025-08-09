import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("DCT_mal.csv")

X = df.drop(columns=["LABEL"])  
y = df["LABEL"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# kNN classifier (example k=5)
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Predict for train and test sets
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Convert classification outputs to numerical for regression metrics
y_train_pred_num = y_train_pred.astype(float)
y_test_pred_num = y_test_pred.astype(float)
y_train_num = y_train.astype(float)
y_test_num = y_test.astype(float)

def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mape, r2

# Calculate metrics
train_metrics = calculate_metrics(y_train_num, y_train_pred_num)
test_metrics = calculate_metrics(y_test_num, y_test_pred_num)

print("Train Set Metrics:")
print(f"MSE: {train_metrics[0]:.4f}")
print(f"RMSE: {train_metrics[1]:.4f}")
print(f"MAPE: {train_metrics[2]:.4f}%")
print(f"R² Score: {train_metrics[3]:.4f}")

print("\nTest Set Metrics:")
print(f"MSE: {test_metrics[0]:.4f}")
print(f"RMSE: {test_metrics[1]:.4f}")
print(f"MAPE: {test_metrics[2]:.4f}%")
print(f"R² Score: {test_metrics[3]:.4f}")
