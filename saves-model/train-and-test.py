import pandas as pd
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from scipy.stats import norm
import joblib
import os

model_home_path = "model_home.pkl"
model_away_path = "model_away.pkl"
error_stats_path = "error_stats.pkl"

model_home = None
model_away = None

X_home_columns = ['team1_backToBack', 'team1_isHome', 'team1_rolling_saves', 'team1_rolling_saves_10', 'team1_rolling_saves_15', 'team2_rolling_opponent_saves', 'team2_rolling_opponent_saves_10', 'team2_rolling_opponent_saves_15',
                    'team2_backToBack', 'team2_isHome']

X_away_columns = ['team2_backToBack', 'team2_isHome', 'team1_rolling_opponent_saves', 'team1_rolling_opponent_saves_10', 'team1_rolling_opponent_saves_15', 'team2_rolling_saves', 'team2_rolling_saves_10', 'team2_rolling_saves_15',
                    'team1_backToBack', 'team1_isHome']

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

    # Prepare features and target variables for both teams
    X_home = []
    X_away = []
    y_home = []
    y_away = []

    # Sort the data by 'gameID' to ensure chronological order
    combined_df_sorted = combined_df.sort_values(by='gameID')

    # Initialize empty lists for features and targets
    X_home, X_away, y_home, y_away = [], [], [], []

    # Iterate through the rows of the sorted dataframe for training
    for i, row in combined_df_sorted.iterrows():
        # Home team features
        team1_backToBack = row['home_backToBack']
        team1_isHome = row['isHome_x']
        team1_rolling_saves = row['home_teamSaves_rolling']
        team1_rolling_opponent_saves = row['home_opponentSaves_rolling']
        team1_rolling_saves_10 = row['home_teamSaves_rolling_10']
        team1_rolling_opponent_saves_10 = row['home_opponentSaves_rolling_10']
        team1_rolling_saves_15 = row['home_teamSaves_rolling_15']
        team1_rolling_opponent_saves_15 = row['home_opponentSaves_rolling_15']
        
        # Away team features
        team2_backToBack = row['away_backToBack']
        team2_isHome = row['isHome_y']
        team2_rolling_saves = row['away_teamSaves_rolling']
        team2_rolling_opponent_saves = row['away_opponentSaves_rolling']
        team2_rolling_saves_10 = row['away_teamSaves_rolling_10']
        team2_rolling_opponent_saves_10 = row['away_opponentSaves_rolling_10']
        team2_rolling_saves_15 = row['away_teamSaves_rolling_15']
        team2_rolling_opponent_saves_15 = row['away_opponentSaves_rolling_15']
        
        # Add features for the home team (team1)
        X_home.append([team1_backToBack, team1_isHome, team1_rolling_saves, team1_rolling_saves_10, team1_rolling_saves_15, team2_rolling_opponent_saves, team2_rolling_opponent_saves_10, team2_rolling_opponent_saves_15,
                    team2_backToBack, team2_isHome])
        # Update targets to use the non-rolling saves columns if you removed the rolling fields
        y_home.append(row['home_teamSaves'])  # Use home_teamSaves (non-rolling)
        
        # Add features for the away team (team2)
        X_away.append([team2_backToBack, team2_isHome, team1_rolling_opponent_saves, team1_rolling_opponent_saves_10, team1_rolling_opponent_saves_15, team2_rolling_saves, team2_rolling_saves_10, team2_rolling_saves_15,
                    team1_backToBack, team1_isHome])
        y_away.append(row['away_teamSaves'])  # Target is away team's saves

    # Convert lists to DataFrames
    X_home = pd.DataFrame(X_home, columns=X_home_columns)
    X_away = pd.DataFrame(X_away, columns=X_away_columns)
    y_home = pd.Series(y_home)
    y_away = pd.Series(y_away)

    # Initialize the RandomForest models
    model_home = RandomForestRegressor(n_estimators=1000, max_depth=20, min_samples_split=10, min_samples_leaf=5, random_state=42, n_jobs=-1)
    model_away = RandomForestRegressor(n_estimators=1000, max_depth=20, min_samples_split=10, min_samples_leaf=5, random_state=42, n_jobs=-1)

    # Iterate through each game for testing
    home_mae_list = []
    away_mae_list = []

    # Lists to store prediction errors
    home_errors, away_errors = [], []

    for i in range(1, len(X_home)):  # Starting from the second game to have previous games to train on
        # Log current game for testing
        logger.info(f"Training on {i}/{len(X_home)} games, predicting game {i+1}")
        
        # Use all games before the current game for training
        X_home_train = X_home.iloc[:i]
        y_home_train = y_home.iloc[:i]
        X_away_train = X_away.iloc[:i]
        y_away_train = y_away.iloc[:i]

        # Train the models
        model_home.fit(X_home_train, y_home_train)
        model_away.fit(X_away_train, y_away_train)

        # Test on the current game (i)
        X_home_test = X_home.iloc[i:i+1]
        y_home_test = y_home.iloc[i:i+1]
        X_away_test = X_away.iloc[i:i+1]
        y_away_test = y_away.iloc[i:i+1]

        # Make predictions
        y_home_pred = model_home.predict(X_home_test)
        y_away_pred = model_away.predict(X_away_test)

        # Calculate and store MAE
        home_mae_list.append(mean_absolute_error(y_home_test, y_home_pred))
        away_mae_list.append(mean_absolute_error(y_away_test, y_away_pred))

        home_errors.append(y_home.iloc[i] - y_home_pred[0])
        away_errors.append(y_away.iloc[i] - y_away_pred[0])

    # Calculate the average MAE across all games
    avg_home_mae = sum(home_mae_list) / len(home_mae_list)
    avg_away_mae = sum(away_mae_list) / len(away_mae_list)

    # Fit normal distributions to the errors
    home_error_mean, home_error_std = norm.fit(home_errors)
    away_error_mean, away_error_std = norm.fit(away_errors)

    print("Average Home team saves prediction MAE:", avg_home_mae)
    print("Average Away team saves prediction MAE:", avg_away_mae)

    print("Model training complete.")

    # Save the models
    joblib.dump(model_home, model_home_path)
    joblib.dump(model_away, model_away_path)
    print("Models saved for future use.")

    # Save error statistics
    error_stats = {
        "home_error_mean": home_error_mean,
        "home_error_std": home_error_std,
        "away_error_mean": away_error_mean,
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
    team2_rolling_saves_10 = team2_data['away_teamSaves_rolling_10'].tail(1).values[0]
    team2_rolling_opponent_saves_10 = team2_data['away_opponentSaves_rolling_10'].tail(1).values[0]
    team2_rolling_saves_15 = team2_data['away_teamSaves_rolling_15'].tail(1).values[0]
    team2_rolling_opponent_saves_15 = team2_data['away_opponentSaves_rolling_15'].tail(1).values[0]

    # Return a dictionary with the features for the matchup
    return {
        'team1_backToBack': team1_backToBack,
        'team1_isHome': team1_isHome,
        'team1_rolling_saves': team1_rolling_saves,
        'team1_rolling_saves_10': team1_rolling_saves_10,
        'team1_rolling_saves_15': team1_rolling_saves_15,
        'team2_backToBack': team2_backToBack,
        'team2_isHome': team2_isHome,
        'team2_rolling_opponent_saves': team2_rolling_opponent_saves,
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
    team2_rolling_saves_10 = team2_data['away_teamSaves_rolling_10'].tail(1).values[0]
    team2_rolling_opponent_saves_10 = team2_data['away_opponentSaves_rolling_10'].tail(1).values[0]
    team2_rolling_saves_15 = team2_data['away_teamSaves_rolling_15'].tail(1).values[0]
    team2_rolling_opponent_saves_15 = team2_data['away_opponentSaves_rolling_15'].tail(1).values[0]

    # Return a dictionary with the features for the matchup
    return {
        'team1_backToBack': team1_backToBack,
        'team1_isHome': team1_isHome,
        'team1_rolling_opponent_saves': team1_rolling_opponent_saves,
        'team1_rolling_opponent_saves_10': team1_rolling_opponent_saves_10,
        'team1_rolling_opponent_saves_15': team1_rolling_opponent_saves_15,
        'team2_backToBack': team2_backToBack,
        'team2_isHome': team2_isHome,
        'team2_rolling_saves': team2_rolling_saves,
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
