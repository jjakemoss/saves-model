import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('combined_simplified.csv')

# Drop rows with missing values
data.dropna(inplace=True)

# Separate features and target
X = data.drop(['gameID', 'gameDate', 'opponent', 'team', 'teamSaves', 'splitGame', 'corsiFor', 'corsiAgainst', 'fenwickFor', 'fenwickAgainst'], axis=1)
y = data['teamSaves']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid
param_grid = {
    'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100],  # More alpha values
    'max_iter': [500, 1000, 2000, 5000, 10000, 20000],  # More max_iter values
    'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],  # More tolerance values
    'positive': [True, False],
    'selection': ['cyclic', 'random'],
    'warm_start': [True, False],
    'random_state': [None, 42] # Adding random state for reproducibility
}

# Initialize the Lasso model
lasso_model = Lasso()

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=lasso_model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)

# Perform grid search
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print("Best parameters:", best_params)

# Train the model with the best parameters
best_lasso_model = Lasso(**best_params)
best_lasso_model.fit(X_train, y_train)

# Make predictions
y_pred = best_lasso_model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print the results
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R-squared:", r2)