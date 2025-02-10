import pandas as pd
import logging
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from scipy.stats import norm
import numpy as np
import joblib
import os

model_home_path = "model_home.pkl"
model_away_path = "model_away.pkl"
error_stats_path = "error_stats.pkl"

model_home = None
model_away = None

X_home_columns = ['home_team', 'away_team', 'home_backToBack', 'isHome_x', 'home_teamSaves_rolling', 'home_teamSaves_rolling_3', 'home_teamSaves_rolling_10', 'home_teamSaves_rolling_15', 'away_opponentSaves_rolling', 'away_opponentSaves_rolling_3', 'away_opponentSaves_rolling_10', 'away_opponentSaves_rolling_15',
                    'away_backToBack', 'isHome_y']

X_away_columns = ['away_team', 'home_team', 'away_backToBack', 'isHome_y', 'home_opponentSaves_rolling', 'home_opponentSaves_rolling_3', 'home_opponentSaves_rolling_10', 'home_opponentSaves_rolling_15', 'away_teamSaves_rolling', 'away_teamSaves_rolling_3', 'away_teamSaves_rolling_10', 'away_teamSaves_rolling_15',
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

team_encoder = LabelEncoder()

# Fit and transform the 'home_team' and 'away_team' columns
combined_df['home_team'] = team_encoder.fit_transform(combined_df['home_team'])
combined_df['away_team'] = team_encoder.transform(combined_df['away_team'])

if model_home == None and model_away == None:
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger()

    # Sort the data by 'gameID' to ensure chronological order
    combined_df_sorted = combined_df.sort_values(by='gameID')

    # combined_df_sorted['home_teamSaves_rolling_3'] *= 2
    # combined_df_sorted['away_teamSaves_rolling_3'] *= 2
    # combined_df_sorted['away_opponentSaves_rolling_3'] *= 2
    # combined_df_sorted['home_opponentSaves_rolling_3'] *= 2

    # combined_df_sorted['home_teamSaves_rolling_15'] *= 0.5
    # combined_df_sorted['away_teamSaves_rolling_15'] *= 0.5
    # combined_df_sorted['away_opponentSaves_rolling_15'] *= 0.5
    # combined_df_sorted['home_opponentSaves_rolling_15'] *= 0.5

    # Initialize MLPRegressor models for both teams with tuned parameters
    model_home = MLPRegressor(
        hidden_layer_sizes=(50, 25, 10),  # Fewer layers and neurons to reduce model complexity
        max_iter=20000,                  # A moderate number of iterations
        warm_start=True,                # Keep training from previous model
        activation='relu',             # Use relu activation function
        solver='adam',                 # Using the default optimizer 'adam'
        learning_rate='adaptive',      # Keep learning rate constant for stability
        learning_rate_init=0.0005 ,      # Moderate learning rate
        alpha=0.001,                    # Slightly increased regularization to prevent overfitting
    )

    model_away = MLPRegressor(
        hidden_layer_sizes=(50, 25, 10),  # Fewer layers and neurons to reduce model complexity
        max_iter=20000,                  # A moderate number of iterations
        warm_start=True,                # Keep training from previous model
        activation='relu',             # Use relu activation function
        solver='adam',                 # Using the default optimizer 'adam'
        learning_rate='adaptive',      # Keep learning rate constant for stability
        learning_rate_init=0.0005 ,
        alpha=0.001,                    # Slightly increased regularization to prevent overfitting
    )

    # Initialize lists to track errors
    home_errors = []
    away_errors = []

    home_errors_end = []
    away_errors_end = []

    total_games = len(combined_df_sorted)

   # Iterate through the rows of the sorted dataframe for training
    for i, row in enumerate(combined_df_sorted.itertuples(index=False), start=1):
        current_gameID = row.gameID  # Get the current gameID

        # Filter past games for training (exclude the current game)
        past_games = combined_df_sorted[
            (combined_df_sorted["gameID"] < current_gameID) & 
            (
                (combined_df_sorted["home_team"] == row.home_team) | 
                (combined_df_sorted["home_team"] == row.away_team) | 
                (combined_df_sorted["away_team"] == row.home_team) | 
                (combined_df_sorted["away_team"] == row.away_team)
            )
        ]

        if past_games.empty:
            continue  # Skip training for the first game since no past data exists

        # Prepare training data (only past games)
        X_train_home = past_games[X_home_columns]
        y_train_home = past_games[target_columns[0]]

        X_train_away = past_games[X_away_columns]
        y_train_away = past_games[target_columns[1]]

        # Train models on all past games before making predictions
        model_home.fit(X_train_home, y_train_home)
        model_away.fit(X_train_away, y_train_away)

        # Prepare current game features for prediction
        X_home_game = pd.DataFrame([getattr(row, col) for col in X_home_columns], index=X_home_columns).T
        X_away_game = pd.DataFrame([getattr(row, col) for col in X_away_columns], index=X_away_columns).T


        # Make predictions
        home_pred = model_home.predict(X_home_game)[0]
        away_pred = model_away.predict(X_away_game)[0]

        # Get actual values
        home_y = getattr(row, target_columns[0])
        away_y = getattr(row, target_columns[1])

        # Track errors
        home_errors.append(abs(home_pred - home_y))
        away_errors.append(abs(away_pred - away_y))

        # Track only later-game errors
        if i > 600:
            home_errors_end.append(abs(home_pred - home_y))
            away_errors_end.append(abs(away_pred - away_y))

        # Log progress every 100 games
        if i % 100 == 0 or i == total_games:
            logging.info(f"Processed {i}/{total_games} games.")

    # Calculate and print overall MAE
    home_mae = np.mean(home_errors)
    away_mae = np.mean(away_errors)
    print(f"Overall Home MAE: {home_mae:.2f}")
    print(f"Overall Away MAE: {away_mae:.2f}")

    home_mae2 = np.mean(home_errors_end)
    away_mae2 = np.mean(away_errors_end)
    print(f"Overall Home MAE2: {home_mae2:.2f}")
    print(f"Overall Away MAE2: {away_mae2:.2f}")

    # Save the models
    joblib.dump(model_home, model_home_path)
    joblib.dump(model_away, model_away_path)
    print("Models saved for future use.")

    home_error_mean, home_error_std = norm.fit(home_errors)
    away_error_mean, away_error_std = norm.fit(away_errors)

    # Save error statistics
    error_stats = {
        "home_error_mean": home_mae,
        "home_error_std": home_error_std,
        "away_error_mean": away_mae,
        "away_error_std": away_error_std
    }
    joblib.dump(error_stats, "error_stats.pkl")

    print("Models and error statistics saved for future use.")

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
    team1_encoded = team_encoder.transform([team1])[0]
    team2_encoded = team_encoder.transform([team2])[0]

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
        'home_team': team1_encoded,
        'away_team': team2_encoded,
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
    team1_encoded = team_encoder.transform([team1])[0]
    team2_encoded = team_encoder.transform([team2])[0]

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
        'home_team': team1_encoded,
        'away_team': team2_encoded,
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
