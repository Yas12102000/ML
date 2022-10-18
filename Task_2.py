import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
df = pd.read_csv('50_Startups.csv')
# print(df.head())
# print(df.shape)
# print(df.describe().round(2).T)

variables = ['R&D Spend', 'Administration', 'Marketing Spend']
for var in variables:
    plt.figure()
    sns.regplot(x=var, y='Profit', data=df).set(title=f'Regression plot of {var} and Profit');

correlations = df.corr()
print(correlations)

y = df['R&D Spend']
X = df[['Administration', 'Marketing Spend', 'Profit']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# print(X.shape)
reg = LinearRegression()
reg.fit(X_train, y_train)
print(reg.intercept_)
print(reg.coef_)

feature_names = X.columns
# print(feature_names)

model_coefficients = reg.coef_
coefficients_df = pd.DataFrame(data=model_coefficients, index=feature_names, columns=['Coefficient value'])
print(coefficients_df)

y_pred = reg.predict(X_test)
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(results)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f'Mean absolute error: {mae:.2f}')
print(f'Mean squared error: {mse:.2f}')
print(f'Root mean squared error: {rmse:.2f}')

actual_minus_predicted = sum((y_test - y_pred)**2)
actual_minus_actual_mean = sum((y_test - y_test.mean())**2)
r2 = 1 - actual_minus_predicted/actual_minus_actual_mean
print('R^2:', r2)

print(reg.score(X_test, y_test))
print(reg.score(X_train, y_train))