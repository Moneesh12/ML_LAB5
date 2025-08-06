import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,root_mean_squared_error,r2_score,mean_absolute_error

df = pd.read_csv('DCT_mal.csv')

X = df.iloc[:, [0]]  
y = df.iloc[:, -1]  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

reg = LinearRegression().fit(X_train, y_train)

y_train_pred = reg.predict(X_train)
y_test_pred = reg.predict(X_test)

me = mean_squared_error(y_test,y_test_pred)
rme = root_mean_squared_error(y_test,y_test_pred)

mape = mean_absolute_error