import pandas as pd
import logging
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.stats import norm
import numpy as np
import joblib
import os

model_home_path = "model_home.pkl"
model_away_path = "model_away.pkl"
error_stats_path = "error_stats.pkl"

model_home = None
model_away = None

X_home_columns = ['home_backToBack', 'isHome_x', 'home_teamSaves_rolling', 'home_teamSaves_rolling_3', 'home_teamSaves_rolling_10', 'home_teamSaves_rolling_15', 'away_opponentSaves_rolling', 'away_opponentSaves_rolling_3', 'away_opponentSaves_rolling_10', 'away_opponentSaves_rolling_15',
                    'away_backToBack', 'isHome_y']

X_away_columns = ['away_backToBack', 'isHome_y', 'home_opponentSaves_rolling', 'home_opponentSaves_rolling_3', 'home_opponentSaves_rolling_10', 'home_opponentSaves_rolling_15', 'away_teamSaves_rolling', 'away_teamSaves_rolling_3', 'away_teamSaves_rolling_10', 'away_teamSaves_rolling_15',
                    'home_backToBack', 'isHome_x']

target_columns = ["home_teamSaves", "away_teamSaves"]

if os.path.exists(model_home_path) and os.path.exists(model_away_path) and os.path.exists(error_stats_path):
    use_saved_model = input("Saved models found. Do you want to use the existing models? (yes/no): ").strip().lower()

    if use_saved_model == "yes" or use_saved_model == "y":
        print("Loading saved models...")
        print("Loading saved models and error statistics...")
        model_home = joblib.load(model_home_path)
        model_away = joblib.load(model_away_path)
        error_stats = joblib.load(error_stats_path)

        # Retrieve saved error statistics
        home_error_mean = error_stats["home_error_mean"]
        home_error_std = error_stats["home_error_std"]
        away_error_mean = error_stats["away_error_mean"]
        away_error_std = error_stats["away_error_std"]

# Load the combined dataset with rolling averages
combined_df = pd.read_csv("combined_simplified.csv")

if model_home == None and model_away == None:
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger()

    model_home = HistGradientBoostingRegressor(warm_start=True, max_iter=1000, learning_rate=0.01,)
    model_away = HistGradientBoostingRegressor(warm_start=True, max_iter=1000, learning_rate=0.05,)

    # Sort the data by 'gameID' to ensure chronological order
    combined_df_sorted = combined_df.sort_values(by='gameID')

    # Iterate over each game sequentially for training and testing
    mae_home_list = []
    mae_away_list = []

    X_home = combined_df_sorted[X_home_columns]
    X_away = combined_df_sorted[X_away_columns]
    y_home = combined_df_sorted['home_teamSaves']
    y_away = combined_df_sorted['away_teamSaves']

    for i in range(len(combined_df_sorted)):
        if i == 0:
            continue  # Skip first game since no prior data is available
        
        # Use all prior games for training
        X_home_train = X_home.iloc[:i]
        X_away_train = X_away.iloc[:i]
        y_home_train = y_home.iloc[:i]
        y_away_train = y_away.iloc[:i]
        
        # Use current game for testing
        X_home_test = X_home.iloc[i:i+1]
        X_away_test = X_away.iloc[i:i+1]
        y_home_test = y_home.iloc[i:i+1]
        y_away_test = y_away.iloc[i:i+1]
        
        # Train models
        model_home.fit(X_home_train, y_home_train)
        model_away.fit(X_away_train, y_away_train)
        
        # Make predictions
        y_home_pred = model_home.predict(X_home_test)
        y_away_pred = model_away.predict(X_away_test)
        
        # Evaluate performance
        mae_home = mean_absolute_error(y_home_test, y_home_pred)
        mae_away = mean_absolute_error(y_away_test, y_away_pred)
        mae_home_list.append(mae_home)
        mae_away_list.append(mae_away)

    # Save trained models
    joblib.dump(model_home, 'histgb_home_model.pkl')
    joblib.dump(model_away, 'histgb_away_model.pkl')

    # Print average MAE
    print(f'Average MAE Home: {np.mean(mae_home_list):.2f}, Average MAE Away: {np.mean(mae_away_list):.2f}')

    print("Training complete and model updated with new data.")

# Function to get matchup data based on team names and game_id
def get_matchup_data(team1, team2):
    # Get the relevant data for team1 (home team)
    team1_data = combined_df[(combined_df['home_team'] == team1)]
    team1_backToBack = team1_data['home_backToBack'].tail(1).values[0]
    team1_isHome = team1_data['isHome_x'].tail(1).values[0]
    team1_rolling_saves = team1_data['home_teamSaves_rolling'].tail(1).values[0]
    team1_rolling_opponent_saves = team1_data['home_opponentSaves_rolling'].tail(1).values[0]
    team1_rolling_saves_3 = team1_data['home_teamSaves_rolling_3'].tail(1).values[0]
    team1_rolling_opponent_saves_3 = team1_data['home_opponentSaves_rolling_3'].tail(1).values[0]
    team1_rolling_saves_10 = team1_data['home_teamSaves_rolling_10'].tail(1).values[0]
    team1_rolling_opponent_saves_10 = team1_data['home_opponentSaves_rolling_10'].tail(1).values[0]
    team1_rolling_saves_15 = team1_data['home_teamSaves_rolling_15'].tail(1).values[0]
    team1_rolling_opponent_saves_15 = team1_data['home_opponentSaves_rolling_15'].tail(1).values[0]

    # Get the relevant data for team2 (away team)
    team2_data = combined_df[(combined_df['away_team'] == team2)]
    team2_backToBack = team2_data['away_backToBack'].tail(1).values[0]
    team2_isHome = team2_data['isHome_y'].tail(1).values[0]
    team2_rolling_saves = team2_data['away_teamSaves_rolling'].tail(1).values[0]
    team2_rolling_saves_3 = team2_data['away_teamSaves_rolling_3'].tail(1).values[0]
    team2_rolling_opponent_saves_3 = team2_data['away_opponentSaves_rolling_3'].tail(1).values[0]
    team2_rolling_opponent_saves = team2_data['away_opponentSaves_rolling'].tail(1).values[0]
    team2_rolling_saves_10 = team2_data['away_teamSaves_rolling_10'].tail(1).values[0]
    team2_rolling_opponent_saves_10 = team2_data['away_opponentSaves_rolling_10'].tail(1).values[0]
    team2_rolling_saves_15 = team2_data['away_teamSaves_rolling_15'].tail(1).values[0]
    team2_rolling_opponent_saves_15 = team2_data['away_opponentSaves_rolling_15'].tail(1).values[0]

    # Return a dictionary with the features for the matchup
    return {
        'team1_backToBack': team1_backToBack,
        'team1_isHome': team1_isHome,
        'team1_rolling_saves': team1_rolling_saves,
        'team1_rolling_saves_3': team1_rolling_saves_3,
        'team1_rolling_saves_10': team1_rolling_saves_10,
        'team1_rolling_saves_15': team1_rolling_saves_15,
        'team2_backToBack': team2_backToBack,
        'team2_isHome': team2_isHome,
        'team2_rolling_opponent_saves': team2_rolling_opponent_saves,
        'team2_rolling_opponent_saves_3': team2_rolling_opponent_saves_3,
        'team2_rolling_opponent_saves_10': team2_rolling_opponent_saves_10,
        'team2_rolling_opponent_saves_15': team2_rolling_opponent_saves_15,
    }

# Function to get matchup data based on team names and game_id
def get_matchup_data_away(team1, team2):
    # Get the relevant data for team1 (home team)
    team1_data = combined_df[(combined_df['home_team'] == team1)]
    team1_backToBack = team1_data['home_backToBack'].tail(1).values[0]
    team1_isHome = team1_data['isHome_x'].tail(1).values[0]
    team1_rolling_saves = team1_data['home_teamSaves_rolling'].tail(1).values[0]
    team1_rolling_opponent_saves = team1_data['home_opponentSaves_rolling'].tail(1).values[0]
    team1_rolling_saves_3 = team1_data['home_teamSaves_rolling_3'].tail(1).values[0]
    team1_rolling_opponent_saves_3 = team1_data['home_opponentSaves_rolling_3'].tail(1).values[0]
    team1_rolling_saves_10 = team1_data['home_teamSaves_rolling_10'].tail(1).values[0]
    team1_rolling_opponent_saves_10 = team1_data['home_opponentSaves_rolling_10'].tail(1).values[0]
    team1_rolling_saves_15 = team1_data['home_teamSaves_rolling_15'].tail(1).values[0]
    team1_rolling_opponent_saves_15 = team1_data['home_opponentSaves_rolling_15'].tail(1).values[0]

    # Get the relevant data for team2 (away team)
    team2_data = combined_df[(combined_df['away_team'] == team2)]
    team2_backToBack = team2_data['away_backToBack'].tail(1).values[0]
    team2_isHome = team2_data['isHome_y'].tail(1).values[0]
    team2_rolling_saves = team2_data['away_teamSaves_rolling'].tail(1).values[0]
    team2_rolling_opponent_saves = team2_data['away_opponentSaves_rolling'].tail(1).values[0]
    team2_rolling_saves_3 = team2_data['away_teamSaves_rolling_3'].tail(1).values[0]
    team2_rolling_opponent_saves_3 = team2_data['away_opponentSaves_rolling_3'].tail(1).values[0]
    team2_rolling_saves_10 = team2_data['away_teamSaves_rolling_10'].tail(1).values[0]
    team2_rolling_opponent_saves_10 = team2_data['away_opponentSaves_rolling_10'].tail(1).values[0]
    team2_rolling_saves_15 = team2_data['away_teamSaves_rolling_15'].tail(1).values[0]
    team2_rolling_opponent_saves_15 = team2_data['away_opponentSaves_rolling_15'].tail(1).values[0]

    # Return a dictionary with the features for the matchup
    return {
        'team1_backToBack': team1_backToBack,
        'team1_isHome': team1_isHome,
        'team1_rolling_opponent_saves': team1_rolling_opponent_saves,
        'team1_rolling_opponent_saves_3': team1_rolling_opponent_saves_3,
        'team1_rolling_opponent_saves_10': team1_rolling_opponent_saves_10,
        'team1_rolling_opponent_saves_15': team1_rolling_opponent_saves_15,
        'team2_backToBack': team2_backToBack,
        'team2_isHome': team2_isHome,
        'team2_rolling_saves': team2_rolling_saves,
        'team2_rolling_saves_3': team2_rolling_saves_3,
        'team2_rolling_saves_10': team2_rolling_saves_10,
        'team2_rolling_saves_15': team2_rolling_saves_15
    }

# Function to predict saves for a new matchup
def predict_saves(team1, team2, home_threshold, away_threshold):
    # Get the features for the matchup using the get_matchup_data function
    matchup_input_home = get_matchup_data(team1, team2)
    matchup_input_away = get_matchup_data_away(team1, team2)
    
    # Convert the input dictionaries to DataFrames with appropriate column names
    home_input_df = pd.DataFrame([list(matchup_input_home.values())], columns=X_home_columns)
    away_input_df = pd.DataFrame([list(matchup_input_away.values())], columns=X_away_columns)
    
    # Make prediction for the home team and away team
    home_prediction = model_home.predict(home_input_df)
    away_prediction = model_away.predict(away_input_df)
    
    print(f"Predicted saves for {team1} vs {team2}:")

    home_prob = 1 - norm.cdf(home_threshold, loc=home_prediction + home_error_mean, scale=home_error_std)
    away_prob = 1 - norm.cdf(away_threshold, loc=away_prediction + away_error_mean, scale=away_error_std)
    
    print(f"Predicted saves for {team1}: {home_prediction[0]:.2f}, Probability of reaching {home_threshold}: {home_prob[0]:.2%}")
    print(f"Predicted saves for {team2}: {away_prediction[0]:.2f}, Probability of reaching {away_threshold}: {away_prob[0]:.2%}")

# Continuous User Input Loop
while True:
    user_input = input("\nEnter matchup and thresholds (Format: 'BOS 24 ANA 25.5') or type 'exit' to quit: ").strip()
    
    if user_input.lower() == "exit":
        print("Exiting program.")
        break

    team1, home_threshold, team2, away_threshold = user_input.split()
    home_threshold = float(home_threshold)
    away_threshold = float(away_threshold)
    predict_saves(team1, team2, home_threshold, away_threshold)
