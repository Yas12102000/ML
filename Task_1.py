import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
df = pd.read_csv('Salary_Data.csv')
df.head()
df.plot.scatter(x='YearsExperience', y='Salary', title='Salary_Data')
print(df.corr())
print(df.describe())

y = df['Salary'].values.reshape(-1, 1)
X = df['YearsExperience'].values.reshape(-1, 1)

print(df['YearsExperience'].values) # [ 1.1  1.3  1.5  2.   2.2  2.9  3.   3.2  3.2  3.7  3.9  4.   4.   4.1
 # 4.5  4.9  5.1  5.3  5.9  6.   6.8  7.1  7.9  8.2  8.7  9.   9.5  9.6
 # 10.3 10.5]
print(df['YearsExperience'].values.shape) # (30,)

print(X.shape) #(30, 1)
print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
print(X_train)
print(y_train)

reg = LinearRegression()
reg.fit(X_train, y_train)
print(reg.intercept_)

print(reg.coef_)

Score = reg.predict([[2.2]])
print(Score)

y_pred = reg.predict(X_test)

df_preds = pd.DataFrame({'Actual': y_test.squeeze(), 'Predicted':y_pred.squeeze()})
print(df_preds)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f'Mean absolute error: {mae:.2f}')
print(f'Mean squared error: {mse:.2f}')
print(f'Root mean squared error: {rmse:.2f}')