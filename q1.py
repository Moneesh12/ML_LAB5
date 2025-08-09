import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

df = pd.read_csv("DCT_mal.csv")

X = df.drop(columns=["LABEL"]) 
y = df["LABEL"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Train kNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_train_pred = knn.predict(X_train)
y_test_pred = knn.predict(X_test)

# accuracy,precision,f1,recall score for testing data
train_acc = accuracy_score(y_train, y_train_pred)
train_prec = precision_score(y_train, y_train_pred, average="weighted")
train_rec = recall_score(y_train, y_train_pred, average="weighted")
train_f1 = f1_score(y_train, y_train_pred, average="weighted")

# accuracy,precision,f1,recall score for training data
test_acc = accuracy_score(y_test, y_test_pred)
test_prec = precision_score(y_test, y_test_pred, average="weighted")
test_rec = recall_score(y_test, y_test_pred, average="weighted")
test_f1 = f1_score(y_test, y_test_pred, average="weighted")

print("TRAIN METRICS")
print(f"Accuracy: {train_acc:.4f}")
print(f"Precision: {train_prec:.4f}")
print(f"Recall: {train_rec:.4f}")
print(f"F1 Score: {train_f1:.4f}")

print("\nTEST METRICS")
print(f"Accuracy: {test_acc:.4f}")
print(f"Precision: {test_prec:.4f}")
print(f"Recall: {test_rec:.4f}")
print(f"F1 Score: {test_f1:.4f}")

# 10. Confusion Matrix
print("\nConfusion Matrix (Test):")
print(confusion_matrix(y_test, y_test_pred))
