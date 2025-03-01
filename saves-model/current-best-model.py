from datetime import datetime, timedelta
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

if model_home == None:
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger()


    # Initialize MLPRegressor models for both teams with tuned parameters
    model_home = MLPRegressor(
        hidden_layer_sizes=(25, 10),  # Fewer layers and neurons to reduce model complexity
        max_iter=20000,                  # A moderate number of iterations
        warm_start=True,                # Keep training from previous model
        activation='relu',             # Use relu activation function
        solver='adam',                 # Using the default optimizer 'adam'
        learning_rate='adaptive',      # Keep learning rate constant for stability
        learning_rate_init=0.001,      # Moderate learning rate
        alpha=0.0001,                    # Slightly increased regularization to prevent overfitting
    )

    # Initialize lists to track errors
    home_predicted = []

    home_actual = []

    total_games = len(combined_df_sorted)

   # Iterate through the rows of the sorted dataframe for training
    for i, row in enumerate(combined_df_sorted.itertuples(index=False), start=1):
        current_gameID = row.gameDate  # Get the current gameID

        # Filter past games
        past_games = combined_df_sorted[
            (combined_df_sorted["gameDate"] < current_gameID)
        ]

        if past_games.empty:
            continue  # Skip training for the first game since no past data exists

        # Prepare training data (only past games)
        X_train_home = past_games[X_home_columns]
        y_train_home = past_games[target_columns[0]]

        # Train models on all past games before making predictions
        model_home.fit(X_train_home, y_train_home)

        # Prepare current game features for prediction
        X_home_game = pd.DataFrame([getattr(row, col) for col in X_home_columns], index=X_home_columns).T

        # Make predictions
        home_pred = model_home.predict(X_home_game)[0]

        home_predicted.append(home_pred)

        # Get actual values
        home_y = getattr(row, target_columns[0])

        # Track errors
        home_actual.append(home_y)

        # Log progress every 100 games
        if i % 100 == 0 or i == total_games:
            logging.info(f"Processed {i}/{total_games} games.")

    home_error_mean, home_mse, home_rmse, home_r2, home_error_std = calculate_error_metrics(home_actual, home_predicted)

    # Print the error metrics for both home and away saves
    print(f"Home Saves - MAE: {home_error_mean:.2f}, MSE: {home_mse:.2f}, RMSE: {home_rmse:.2f}, R²: {home_r2:.2f}")

    # Save the models
    joblib.dump(model_home, model_home_path)
    print("Models saved for future use.")

    # Save error statistics
    error_stats = {
        "home_mae": home_error_mean,
        "home_mse": home_mse,
        "home_rmse": home_rmse,
        "home_r2": home_r2,
        "home_std": home_error_std
    }
    joblib.dump(error_stats, error_stats_path)

    print("Models and error statistics saved for future use.")

def get_rolling_stat(df, column, group_col, span, min_periods=3):
    """Helper function to calculate rolling EWMA and return the last value."""
    rolling_values = (
        df.sort_values(by=['gameDate'])
        .groupby(group_col)[column]
        .ewm(span=span, adjust=False, min_periods=min_periods)
        .mean()
    )
    return rolling_values.iloc[-1] if not rolling_values.empty else None

# Function to get matchup data based on team names and game_id
def get_matchup_data(team1, team2, isHome: bool):
    # Get the relevant data for team1 (home team) and team2 (away team)
    team1_data = combined_df_sorted[combined_df_sorted['team'] == team1]
    team2_data = combined_df_sorted[combined_df_sorted['team'] == team2]

    # Convert the last game's date from Pandas Series to a datetime object
    last_game_date = pd.to_datetime(team1_data['gameDate']).iloc[-1] if not team1_data.empty else None

    yesterday = datetime.now().date() - timedelta(days=1)
    isBacktoBack = last_game_date.date() == yesterday if last_game_date else False

    # Get the last game's saves for team1 and team2
    teamSaves_last = team1_data.sort_values(by=['gameDate'])['teamSaves'].iloc[-1] if not team1_data.empty else None
    opponentSaves_last = team1_data.sort_values(by=['gameDate'])['opponentSaves'].iloc[-1] if not team1_data.empty else None

    opponentTeamSaves_last = team2_data.sort_values(by=['gameDate'])['teamSaves'].iloc[-1] if not team2_data.empty else None
    opponentOpponentSaves_last = team2_data.sort_values(by=['gameDate'])['opponentSaves'].iloc[-1] if not team2_data.empty else None

    # Get the latest rolling values for saves
    def get_saves_rolling(span):
        return {
            f'teamSaves_rolling_{span}': get_rolling_stat(team1_data, 'teamSaves', 'team', span=span),
            f'opponentSaves_rolling_{span}': get_rolling_stat(team1_data, 'opponentSaves', 'team', span=span),
            f'opponentTeamSaves_rolling_{span}': get_rolling_stat(team2_data, 'teamSaves', 'team', span=span),
            f'opponentOpponentSaves_rolling_{span}': get_rolling_stat(team2_data, 'opponentSaves', 'team', span=span)
        }

    # Get the latest rolling values for Corsi
    def get_corsi_rolling(span):
        return {
            f'teamCorsi_rolling_{span}': get_rolling_stat(team1_data, 'corsiFor', 'team', span=span),
            f'opponentCorsi_rolling_{span}': get_rolling_stat(team1_data, 'corsiAgainst', 'team', span=span),
            f'opponentTeamCorsi_rolling_{span}': get_rolling_stat(team2_data, 'corsiFor', 'team', span=span),
            f'opponentOpponentCorsi_rolling_{span}': get_rolling_stat(team2_data, 'corsiAgainst', 'team', span=span)
        }

    # Get the latest rolling values for Fenwick
    def get_fenwick_rolling(span):
        return {
            f'teamFenwick_rolling_{span}': get_rolling_stat(team1_data, 'fenwickFor', 'team', span=span),
            f'opponentFenwick_rolling_{span}': get_rolling_stat(team1_data, 'fenwickAgainst', 'team', span=span),
            f'opponentTeamFenwick_rolling_{span}': get_rolling_stat(team2_data, 'fenwickFor', 'team', span=span),
            f'opponentOpponentFenwick_rolling_{span}': get_rolling_stat(team2_data, 'fenwickAgainst', 'team', span=span)
        }

    # Generate rolling statistics for spans 3, 5, 10, 15
    rolling_spans = [3, 5, 10, 15]
    rolling_data = {}
    
    for span in rolling_spans:
        rolling_data.update(get_saves_rolling(span))
        rolling_data.update(get_corsi_rolling(span))
        rolling_data.update(get_fenwick_rolling(span))

    return {
        'isHome': isHome,
        'teamSaves_last': teamSaves_last,
        'opponentSaves_last': opponentSaves_last,
        'opponentTeamSaves_last': opponentTeamSaves_last,
        'opponentOpponentSaves_last': opponentOpponentSaves_last,
        'teamCorsi_last': team1_data.sort_values(by=['gameDate'])['corsiFor'].iloc[-1] if not team1_data.empty else None,
        'opponentCorsi_last': team1_data.sort_values(by=['gameDate'])['corsiAgainst'].iloc[-1] if not team1_data.empty else None,
        'opponentTeamCorsi_last': team2_data.sort_values(by=['gameDate'])['corsiFor'].iloc[-1] if not team2_data.empty else None,
        'opponentOpponentCorsi_last': team2_data.sort_values(by=['gameDate'])['corsiAgainst'].iloc[-1] if not team2_data.empty else None,
        'teamFen_last': team1_data.sort_values(by=['gameDate'])['fenwickFor'].iloc[-1] if not team1_data.empty else None,
        'opponentFenwick_last': team1_data.sort_values(by=['gameDate'])['fenwickAgainst'].iloc[-1] if not team1_data.empty else None,
        'opponentTeamFenwick_last': team2_data.sort_values(by=['gameDate'])['fenwickFor'].iloc[-1] if not team2_data.empty else None,
        'opponentOpponentFenwick_last': team2_data.sort_values(by=['gameDate'])['fenwickAgainst'].iloc[-1] if not team2_data.empty else None,
        'backToBack': isBacktoBack,
        **rolling_data
    }

# Convert NumPy types to native Python types
def convert_numpy(obj):
    if isinstance(obj, np.integer):
        return int(obj)  # Convert int64 -> int
    elif isinstance(obj, np.floating):
        return float(obj)  # Convert float64 -> float
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert NumPy arrays -> lists
    else:
        return obj  # Return the object as-is if it's already serializable

# Function to predict saves for a new matchup
def predict_saves(team1, team2, home_threshold, away_threshold):
    # Get the features for the matchup using the get_matchup_data function
    matchup_input_home = get_matchup_data(team1, team2, True)
    matchup_input_away = get_matchup_data(team2, team1, False)

    # Combine both matchup inputs into a dictionary
    matchup_data = {
        team1: {k: convert_numpy(v) for k, v in matchup_input_home.items()},
        team2: {k: convert_numpy(v) for k, v in matchup_input_away.items()}
    }

    # Define a file path
    file_path = f"prediction-games/{team1}-{team2}.json"

    # Write the data to a file in a readable format
    with open(file_path, "w") as f:
        json.dump(matchup_data, f, indent=4)  # indent=4 makes it more readable

    team1_data = combined_df_sorted[combined_df_sorted['team'] == team1]
    # Calculate the overall percentage of split games for team1
    splitGame_rate = team1_data["splitGame"].mean()

    team2_data = combined_df_sorted[combined_df_sorted['team'] == team2]
    # Calculate the overall percentage of split games for team1
    splitGame_rate_2 = team2_data["splitGame"].mean()

    # Convert the input dictionaries to DataFrames with appropriate column names
    home_input_df = pd.DataFrame([list(matchup_input_home.values())], columns=X_home_columns)
    away_input_df = pd.DataFrame([list(matchup_input_away.values())], columns=X_home_columns)

    home_input_df[numeric_columns] = scaler.transform(home_input_df[numeric_columns])
    away_input_df[numeric_columns] = scaler.transform(away_input_df[numeric_columns])

    # Make prediction for the home team and away team[
    home_prediction = model_home.predict(home_input_df)
    away_prediction = model_home.predict(away_input_df)

    print(f"Predicted saves for {team1} vs {team2}:")

    home_prob = 1 - norm.cdf(home_threshold, loc=home_prediction, scale=home_saves_std)
    home_prob *= (1 - splitGame_rate)  # Adjust for split games

    away_prob = 1 - norm.cdf(away_threshold, loc=away_prediction, scale=away_saves_std)
    away_prob *= (1 - splitGame_rate_2)

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
