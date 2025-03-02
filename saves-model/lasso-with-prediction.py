import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from datetime import datetime, timedelta
import json
import pandas as pd
import logging
from sklearn.linear_model import Lasso
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

# Load the dataset
data = pd.read_csv('S:/Documents/GitHub/saves-model/combined_simplified.csv')

# Drop rows with missing values
data.dropna(inplace=True)

# Separate features and target
X = data.drop(['gameID', 'gameDate', 'opponent', 'team', 'teamSaves', 'opponentSaves', 'splitGame', 'corsiFor', 'corsiAgainst', 'fenwickFor', 'fenwickAgainst'], axis=1)
y = data['teamSaves']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Lasso model
lasso_model = Lasso()

# Train the model
lasso_model.fit(X_train, y_train)

# Make predictions
y_pred = lasso_model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

home_saves_std = data["teamSaves"].std()
away_saves_std = data["opponentSaves"].std()

# Print the evaluation metrics
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R2): {r2}")

# Retrain the model on the entire dataset
lasso_model.fit(X, y)

X_home_columns = ['isHome', 
                  'teamSaves_last', 'opponentSaves_last', 
                   'teamSaves_rolling_5', 'opponentSaves_rolling_5', 
                   'teamSaves_rolling_3', 'opponentSaves_rolling_3', 
                   'teamSaves_rolling_10', 'opponentSaves_rolling_10', 
                   'teamSaves_rolling_15', 'opponentSaves_rolling_15', 

                   'opponentTeamSaves_last', 'opponentOpponentSaves_last', 
                   'opponentTeamSaves_rolling_5', 'opponentOpponentSaves_rolling_5', 
                   'opponentTeamSaves_rolling_3', 'opponentOpponentSaves_rolling_3', 
                   'opponentTeamSaves_rolling_10', 'opponentOpponentSaves_rolling_10', 
                   'opponentTeamSaves_rolling_15', 'opponentOpponentSaves_rolling_15',

                   'teamCorsi_last', 'opponentCorsi_last', 
                   'teamCorsi_rolling_5', 'opponentCorsi_rolling_5', 
                   'teamCorsi_rolling_3', 'opponentCorsi_rolling_3', 
                   'teamCorsi_rolling_10', 'opponentCorsi_rolling_10', 
                   'teamCorsi_rolling_15', 'opponentCorsi_rolling_15',

                   'opponentTeamCorsi_last', 'opponentOpponentCorsi_last', 
                   'opponentTeamCorsi_rolling_5', 'opponentOpponentCorsi_rolling_5', 
                   'opponentTeamCorsi_rolling_3', 'opponentOpponentCorsi_rolling_3', 
                   'opponentTeamCorsi_rolling_10', 'opponentOpponentCorsi_rolling_10', 
                   'opponentTeamCorsi_rolling_15', 'opponentOpponentCorsi_rolling_15',

                   'teamFenwick_last', 'opponentFenwick_last', 
                   'teamFenwick_rolling_5', 'opponentFenwick_rolling_5', 
                   'teamFenwick_rolling_3', 'opponentFenwick_rolling_3', 
                   'teamFenwick_rolling_10', 'opponentFenwick_rolling_10', 
                   'teamFenwick_rolling_15', 'opponentFenwick_rolling_15',

                   'opponentTeamFenwick_last', 'opponentOpponentFenwick_last', 
                   'opponentTeamFenwick_rolling_5', 'opponentOpponentFenwick_rolling_5', 
                   'opponentTeamFenwick_rolling_3', 'opponentOpponentFenwick_rolling_3', 
                   'opponentTeamFenwick_rolling_10', 'opponentOpponentFenwick_rolling_10', 
                   'opponentTeamFenwick_rolling_15', 'opponentOpponentFenwick_rolling_15', 

                  'backToBack']

print("Model retrained on the entire dataset.")

def get_rolling_stat(df, column, group_col, span, min_periods=3):
    """Helper function to calculate rolling EWMA and return the last value."""
    rolling_values = (
        df.sort_values(by=['gameDate'])
        .groupby(group_col)[column]
        .rolling(span, min_periods=min_periods)
        .mean()
        .reset_index(0, drop=True)
    )
    return rolling_values.iloc[-1] if not rolling_values.empty else None

# Function to get matchup data based on team names and game_id
def get_matchup_data(team1, team2, isHome: bool):
    # Get the relevant data for team1 (home team) and team2 (away team)
    team1_data = data[data['team'] == team1]
    team2_data = data[data['team'] == team2]

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
        'teamFenwick_last': team1_data.sort_values(by=['gameDate'])['fenwickFor'].iloc[-1] if not team1_data.empty else None,
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

    team1_data = data[data['team'] == team1]
    # Calculate the overall percentage of split games for team1
    splitGame_rate = team1_data["splitGame"].mean()

    team2_data = data[data['team'] == team2]
    # Calculate the overall percentage of split games for team1
    splitGame_rate_2 = team2_data["splitGame"].mean()

    # Convert the input dictionaries to DataFrames with appropriate column names
    home_input_df = pd.DataFrame([list(matchup_input_home.values())], columns=X_home_columns)
    away_input_df = pd.DataFrame([list(matchup_input_away.values())], columns=X_home_columns)

    # Make prediction for the home team and away team[
    home_prediction = lasso_model.predict(home_input_df)
    away_prediction = lasso_model.predict(away_input_df)

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