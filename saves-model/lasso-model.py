import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import ElasticNet, Lars, Lasso, LassoLars
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

df = pd.read_csv('S:/Documents/GitHub/saves-model/combined_simplified.csv')

# Drop rows with NA values
df_cleaned = df.dropna(subset=X_home_columns + [y_home_column])

# Separate features and target
X = df_cleaned[X_home_columns]
y = df_cleaned[y_home_column]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Lasso model
lasso_model = LassoLars()

lasso_model.fit(X_train, y_train)

# Make predictions
y_pred = lasso_model.predict(X_test)

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