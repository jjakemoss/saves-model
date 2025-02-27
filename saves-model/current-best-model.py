from datetime import datetime, timedelta
import json
import pandas as pd
import logging
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
import os
from itertools import product

model_home_path = "S:/Documents/GitHub/saves-model/model_home.pkl"
error_stats_path = "S:/Documents/GitHub/saves-model/error_stats.pkll"

model_home = None

target_columns = ["teamSaves"]

hidden_layer_sizes_options = [(50, 25, 10), (50, 10), (25, 10), (100, 50, 25)]
learning_rate_init_options = [0.00025, 0.0005, 0.001]
alpha_options = [0.00001, 0.00005, 0.0001, 0.001]

# Generate all combinations
param_grid = list(product(hidden_layer_sizes_options, learning_rate_init_options, alpha_options))

best_mae = float('inf')
best_params = None

# Function to calculate additional error metrics
def calculate_error_metrics(true_values, predicted_values):
    # Mean Absolute Error
    mae = mean_absolute_error(true_values, predicted_values)

    # Mean Squared Error
    mse = mean_squared_error(true_values, predicted_values)

    # Root Mean Squared Error
    rmse = np.sqrt(mse)

    # R-squared
    r2 = r2_score(true_values, predicted_values)

    error_std = np.std(np.array(true_values) - np.array(predicted_values))

    return mae, mse, rmse, r2, error_std

if os.path.exists(model_home_path) and os.path.exists(error_stats_path):
    use_saved_model = input("Saved models found. Do you want to use the existing models? (yes/no): ").strip().lower()

    if use_saved_model == "yes" or use_saved_model == "y":
        print("Loading saved models...")
        print("Loading saved models and error statistics...")
        model_home = joblib.load(model_home_path)
        error_stats = joblib.load(error_stats_path)

        # Retrieve saved error statistics
        home_error_mean = error_stats["home_mae"]
        home_error_std = error_stats["home_std"]

# Load the combined dataset with rolling averages
combined_df = pd.read_csv("S:/Documents/GitHub/saves-model/combined_simplified.csv")

# Select the columns to normalize (numeric features)
numeric_columns = ['teamSaves_last', 'opponentSaves_last', 'teamSaves_rolling', 'opponentSaves_rolling', 'teamSaves_rolling_3',
                  'opponentSaves_rolling_3', 'teamSaves_rolling_10', 'opponentSaves_rolling_10',
                  'teamSaves_rolling_15', 'opponentSaves_rolling_15', 'opponentTeamSaves_last', 'opponentOpponentSaves_last',
                  'opponentTeamSaves_rolling', 'opponentOpponentSaves_rolling', 'opponentTeamSaves_rolling_3',
                  'opponentOpponentSaves_rolling_3', 'opponentTeamSaves_rolling_10', 'opponentOpponentSaves_rolling_10',
                  'opponentTeamSaves_rolling_15', 'opponentOpponentSaves_rolling_15']

# Initialize the scaler
scaler = MinMaxScaler()

# Fit and transform the data
combined_df[numeric_columns] = scaler.fit_transform(combined_df[numeric_columns])

# Update the X_home_columns and X_away_columns lists to include the one-hot encoded columns
X_home_columns = ['isHome', 'teamSaves_last', 'opponentSaves_last', 'teamSaves_rolling', 'opponentSaves_rolling', 'teamSaves_rolling_3',
                  'opponentSaves_rolling_3', 'teamSaves_rolling_10', 'opponentSaves_rolling_10',
                  'teamSaves_rolling_15', 'opponentSaves_rolling_15', 'opponentTeamSaves_last', 'opponentOpponentSaves_last',
                  'opponentTeamSaves_rolling', 'opponentOpponentSaves_rolling', 'opponentTeamSaves_rolling_3',
                  'opponentOpponentSaves_rolling_3', 'opponentTeamSaves_rolling_10', 'opponentOpponentSaves_rolling_10',
                  'opponentTeamSaves_rolling_15', 'opponentOpponentSaves_rolling_15',
                  'backToBack']

combined_df_sorted = combined_df.sort_values(by='gameDate')

home_saves_std = combined_df_sorted["teamSaves"].std()
away_saves_std = combined_df_sorted["opponentSaves"].std()

if model_home == None:
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger()


    # Initialize MLPRegressor models for both teams with tuned parameters
    model_home = MLPRegressor(
        hidden_layer_sizes=(50, 25, 10),  # Fewer layers and neurons to reduce model complexity
        max_iter=20000,                  # A moderate number of iterations
        warm_start=True,                # Keep training from previous model
        activation='relu',             # Use relu activation function
        solver='adam',                 # Using the default optimizer 'adam'
        learning_rate='adaptive',      # Keep learning rate constant for stability
        learning_rate_init=0.0005,      # Moderate learning rate
        alpha=0.0001,                    # Slightly increased regularization to prevent overfitting
    )

    # Initialize lists to track errors
    home_predicted = []

    home_actual = []

    total_games = len(combined_df_sorted)

    for hidden_layers, lr_init, alpha in param_grid:
        print(f"Testing: hidden_layers={hidden_layers}, learning_rate_init={lr_init}, alpha={alpha}")
        
        model_home = MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            max_iter=20000,
            warm_start=True,
            activation='relu',
            solver='adam',
            learning_rate='adaptive',
            learning_rate_init=lr_init,
            alpha=alpha,
        )
        
        home_predicted = []
        home_actual = []

        for i, row in enumerate(combined_df_sorted.itertuples(index=False), start=1):
            current_gameID = row.gameDate
            past_games = combined_df_sorted[combined_df_sorted["gameDate"] < current_gameID]
            
            if past_games.empty:
                continue
            
            X_train_home = past_games[X_home_columns]
            y_train_home = past_games[target_columns[0]]
            model_home.fit(X_train_home, y_train_home)
            
            X_home_game = pd.DataFrame([getattr(row, col) for col in X_home_columns], index=X_home_columns).T
            home_pred = model_home.predict(X_home_game)[0]
            home_predicted.append(home_pred)
            home_actual.append(getattr(row, target_columns[0]))
        
        home_error_mean, _, _, _, _ = calculate_error_metrics(home_actual, home_predicted)
        print(f"MAE for this config: {home_error_mean:.2f}")
        
        if home_error_mean < best_mae:
            best_mae = home_error_mean
            best_params = (hidden_layers, lr_init, alpha)

    print(f"Best params: hidden_layers={best_params[0]}, learning_rate_init={best_params[1]}, alpha={best_params[2]} with MAE={best_mae:.2f}")
