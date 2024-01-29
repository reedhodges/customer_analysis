import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from data_processing import train_df, test_df

independent_vars = ['Year_Birth', 'Income', 'Kidhome', 'Teenhome']
dependent_var = 'MntWines'

X_train = train_df[independent_vars]
y_train = train_df[dependent_var]

X_test = test_df[independent_vars]
y_test = test_df[dependent_var]

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")