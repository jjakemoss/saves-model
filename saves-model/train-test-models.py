import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
data = pd.read_csv('combined_simplified.csv')

# Drop rows with missing values
data.dropna(inplace=True)

# Define the features and target
X_home_columns = ['isHome',
                  'teamSaves_last',
                  'teamSaves_rolling_5',
                  'teamSaves_rolling_3',
                  'teamSaves_rolling_10',
                  'teamSaves_rolling_15',
                  'opponentOpponentSaves_last',
                  'opponentOpponentSaves_rolling_5',
                  'opponentOpponentSaves_rolling_3',
                  'opponentOpponentSaves_rolling_10',
                  'opponentOpponentSaves_rolling_15',
                  'opponentCorsi_last',
                  'opponentCorsi_rolling_5',
                  'opponentCorsi_rolling_3',
                  'opponentCorsi_rolling_10',
                  'opponentCorsi_rolling_15',
                  'opponentTeamCorsi_last',
                  'opponentTeamCorsi_rolling_5',
                  'opponentTeamCorsi_rolling_3',
                  'opponentTeamCorsi_rolling_10',
                  'opponentTeamCorsi_rolling_15',
                  'opponentFenwick_last',
                  'opponentFenwick_rolling_5',
                  'opponentFenwick_rolling_3',
                  'opponentFenwick_rolling_10',
                  'opponentFenwick_rolling_15',
                  'opponentTeamFenwick_last',
                  'opponentTeamFenwick_rolling_5',
                  'opponentTeamFenwick_rolling_3',
                  'opponentTeamFenwick_rolling_10',
                  'opponentTeamFenwick_rolling_15',
                  'backToBack']

y_home_column = 'teamSaves'

# Prepare data
X = data[X_home_columns]
y = data[y_home_column]

# Best parameters for Lasso: {'alpha': 0.5, 'max_iter': 1000, 'selection': 'random', 'tol': 0.1}

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define Lasso hyperparameter grid
lasso_param_grid = {
    'alpha': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100],  # Most important parameter
    'max_iter': [500, 1000, 5000, 10000],  # Helps with convergence
    'tol': [1e-4, 1e-3, 1e-2, 1e-1],  # Convergence tolerance
    'selection': ['cyclic', 'random']  # Optimization method
}

# Perform Grid Search for Lasso model
lasso_model = Lasso()
grid_search = GridSearchCV(lasso_model, lasso_param_grid, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get best model
best_lasso = grid_search.best_estimator_
print('Best parameters for Lasso:', grid_search.best_params_)

# Evaluate the best model
y_pred = best_lasso.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print('\nLasso Model Performance:')
print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
print(f'R-squared: {r2}')