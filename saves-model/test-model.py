from datetime import datetime, timedelta
from itertools import product
import json
import pandas as pd
import logging
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
import os

model_home_path = "S:/Documents/GitHub/saves-model/model_home.pkl"
error_stats_path = "S:/Documents/GitHub/saves-model/error_stats.pkll"

model_home = None

target_columns = ["teamSaves"]

hidden_layer_sizes_options = [
    (100, 50, 25),  # Larger network to capture complex relationships
    (75, 40, 15),   # Slightly smaller, but still robust
    (50, 25),        # A more moderate size
    (100, 50)         # A wider, shallower network
]

learning_rate_init_options = [
    0.0001,  # Lower end for stability, especially with more features
    0.0005,  # Moderate learning rate
    0.001,   # Standard starting point
    0.002,   # Higher end to see if faster convergence is possible
]

alpha_options = [
    0.0001,  # Starting point, mild regularization
    0.0005,  # Increased regularization to combat overfitting
    0.001,   # Moderate regularization
    0.005,   # Higher regularization, good for complex models
]

# Generate all combinations
param_grid = list(product(hidden_layer_sizes_options, learning_rate_init_options, alpha_options))

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

# Load the combined dataset with rolling averages
combined_df = pd.read_csv("S:/Documents/GitHub/saves-model/combined_simplified.csv")

# Select the columns to normalize (numeric features)
numeric_columns = ['teamSaves_last', 'opponentSaves_last', 
                   'teamSaves_rolling', 'opponentSaves_rolling', 
                   'teamSaves_rolling_3', 'opponentSaves_rolling_3', 
                   'teamSaves_rolling_10', 'opponentSaves_rolling_10', 
                   'teamSaves_rolling_15', 'opponentSaves_rolling_15', 

                   'opponentTeamSaves_last', 'opponentOpponentSaves_last', 
                   'opponentTeamSaves_rolling', 'opponentOpponentSaves_rolling', 
                   'opponentTeamSaves_rolling_3', 'opponentOpponentSaves_rolling_3', 
                   'opponentTeamSaves_rolling_10', 'opponentOpponentSaves_rolling_10', 
                   'opponentTeamSaves_rolling_15', 'opponentOpponentSaves_rolling_15',

                   'teamCorsi_last', 'opponentCorsi_last', 
                   'teamCorsi_rolling', 'opponentCorsi_rolling', 
                   'teamCorsi_rolling_3', 'opponentCorsi_rolling_3', 
                   'teamCorsi_rolling_10', 'opponentCorsi_rolling_10', 
                   'teamCorsi_rolling_15', 'opponentCorsi_rolling_15',

                   'opponentTeamCorsi_last', 'opponentOpponentCorsi_last', 
                   'opponentTeamCorsi_rolling', 'opponentOpponentCorsi_rolling', 
                   'opponentTeamCorsi_rolling_3', 'opponentOpponentCorsi_rolling_3', 
                   'opponentTeamCorsi_rolling_10', 'opponentOpponentCorsi_rolling_10', 
                   'opponentTeamCorsi_rolling_15', 'opponentOpponentCorsi_rolling_15',

                   'teamFen_last', 'opponentFenwick_last', 
                   'teamFenwick_rolling', 'opponentFenwick_rolling', 
                   'teamFenwick_rolling_3', 'opponentFenwick_rolling_3', 
                   'teamFenwick_rolling_10', 'opponentFenwick_rolling_10', 
                   'teamFenwick_rolling_15', 'opponentFenwick_rolling_15',

                   'opponentTeamFenwick_last', 'opponentOpponentFenwick_last', 
                   'opponentTeamFenwick_rolling', 'opponentOpponentFenwick_rolling', 
                   'opponentTeamFenwick_rolling_3', 'opponentOpponentFenwick_rolling_3', 
                   'opponentTeamFenwick_rolling_10', 'opponentOpponentFenwick_rolling_10', 
                   'opponentTeamFenwick_rolling_15', 'opponentOpponentFenwick_rolling_15']


# Initialize the scaler
scaler = MinMaxScaler()

# Fit and transform the data
combined_df[numeric_columns] = scaler.fit_transform(combined_df[numeric_columns])

# Update the X_home_columns and X_away_columns lists to include the one-hot encoded columns
X_home_columns = ['isHome', 
                  'teamSaves_last', 'opponentSaves_last', 
                  'teamSaves_rolling', 'opponentSaves_rolling', 
                  'teamSaves_rolling_3', 'opponentSaves_rolling_3', 
                  'teamSaves_rolling_10', 'opponentSaves_rolling_10', 
                  'teamSaves_rolling_15', 'opponentSaves_rolling_15', 

                  'opponentTeamSaves_last', 'opponentOpponentSaves_last', 
                  'opponentTeamSaves_rolling', 'opponentOpponentSaves_rolling', 
                  'opponentTeamSaves_rolling_3', 'opponentOpponentSaves_rolling_3', 
                  'opponentTeamSaves_rolling_10', 'opponentOpponentSaves_rolling_10', 
                  'opponentTeamSaves_rolling_15', 'opponentOpponentSaves_rolling_15',

                  'teamCorsi_last', 'opponentCorsi_last', 
                  'teamCorsi_rolling', 'opponentCorsi_rolling', 
                  'teamCorsi_rolling_3', 'opponentCorsi_rolling_3', 
                  'teamCorsi_rolling_10', 'opponentCorsi_rolling_10', 
                  'teamCorsi_rolling_15', 'opponentCorsi_rolling_15',

                  'opponentTeamCorsi_last', 'opponentOpponentCorsi_last', 
                  'opponentTeamCorsi_rolling', 'opponentOpponentCorsi_rolling', 
                  'opponentTeamCorsi_rolling_3', 'opponentOpponentCorsi_rolling_3', 
                  'opponentTeamCorsi_rolling_10', 'opponentOpponentCorsi_rolling_10', 
                  'opponentTeamCorsi_rolling_15', 'opponentOpponentCorsi_rolling_15',

                  'teamFen_last', 'opponentFenwick_last', 
                  'teamFenwick_rolling', 'opponentFenwick_rolling', 
                  'teamFenwick_rolling_3', 'opponentFenwick_rolling_3', 
                  'teamFenwick_rolling_10', 'opponentFenwick_rolling_10', 
                  'teamFenwick_rolling_15', 'opponentFenwick_rolling_15',

                  'opponentTeamFenwick_last', 'opponentOpponentFenwick_last', 
                  'opponentTeamFenwick_rolling', 'opponentOpponentFenwick_rolling', 
                  'opponentTeamFenwick_rolling_3', 'opponentOpponentFenwick_rolling_3', 
                  'opponentTeamFenwick_rolling_10', 'opponentOpponentFenwick_rolling_10', 
                  'opponentTeamFenwick_rolling_15', 'opponentOpponentFenwick_rolling_15',

                  'backToBack']


combined_df_sorted = combined_df.sort_values(by='gameDate')

home_saves_std = combined_df_sorted["teamSaves"].std()
away_saves_std = combined_df_sorted["opponentSaves"].std()

best_mae = 0

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

