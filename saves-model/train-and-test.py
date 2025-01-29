import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Load the combined dataset with rolling averages
combined_df = pd.read_csv("combined_simplified.csv")

# Prepare features and target variables for both teams
X_home = []
X_away = []
y_home = []
y_away = []

# Function to get matchup data based on team names and game_id
def get_matchup_data(team1, team2, game_id):
    # Get the relevant data for team1 (home team)
    team1_data = combined_df[(combined_df['home_team'] == team1) & (combined_df['gameID'] <= game_id)]
    team1_rolling_saves = team1_data['home_teamSaves_rolling'].tail(1).values[0]
    team1_rolling_opponent_saves = team1_data['home_opponentSaves_rolling'].tail(1).values[0]
    team1_backToBack = team1_data['home_backToBack'].tail(1).values[0]
    team1_isHome = team1_data['isHome_x'].tail(1).values[0]
    team1_goalsFor = team1_data['home_goalsFor'].tail(1).values[0]
    team1_goalsAgainst = team1_data['home_goalsAgainst'].tail(1).values[0]
    team1_shotsFor = team1_data['home_shotsFor'].tail(1).values[0]
    team1_shotsAgainst = team1_data['home_shotsAgainst'].tail(1).values[0]

    # Get the relevant data for team2 (away team)
    team2_data = combined_df[(combined_df['away_team'] == team2) & (combined_df['gameID'] <= game_id)]
    team2_rolling_saves = team2_data['away_teamSaves_rolling'].tail(1).values[0]
    team2_rolling_opponent_saves = team2_data['away_opponentSaves_rolling'].tail(1).values[0]
    team2_backToBack = team2_data['away_backToBack'].tail(1).values[0]
    team2_isHome = team2_data['isHome_y'].tail(1).values[0]
    team2_goalsFor = team2_data['away_goalsFor'].tail(1).values[0]
    team2_goalsAgainst = team2_data['away_goalsAgainst'].tail(1).values[0]
    team2_shotsFor = team2_data['away_shotsFor'].tail(1).values[0]
    team2_shotsAgainst = team2_data['away_shotsAgainst'].tail(1).values[0]

    # Return a dictionary with the features for the matchup
    return {
        'team1_rolling_saves': team1_rolling_saves,
        'team1_rolling_opponent_saves': team1_rolling_opponent_saves,
        'team1_backToBack': team1_backToBack,
        'team1_isHome': team1_isHome,
        'team1_goalsFor': team1_goalsFor,
        'team1_goalsAgainst': team1_goalsAgainst,
        'team1_shotsFor': team1_shotsFor,
        'team1_shotsAgainst': team1_shotsAgainst,
        'team2_rolling_saves': team2_rolling_saves,
        'team2_rolling_opponent_saves': team2_rolling_opponent_saves,
        'team2_backToBack': team2_backToBack,
        'team2_isHome': team2_isHome,
        'team2_goalsFor': team2_goalsFor,
        'team2_goalsAgainst': team2_goalsAgainst,
        'team2_shotsFor': team2_shotsFor,
        'team2_shotsAgainst': team2_shotsAgainst,
    }

# Iterate through the combined dataframe to build the training data
for i, row in combined_df.iterrows():
    # Home team features
    team1_rolling_saves = row['home_teamSaves_rolling']
    team1_rolling_opponent_saves = row['home_opponentSaves_rolling']
    team1_backToBack = row['home_backToBack']
    team1_isHome = row['isHome_x']
    team1_goalsFor = row['home_goalsFor']
    team1_goalsAgainst = row['home_goalsAgainst']
    team1_shotsFor = row['home_shotsFor']
    team1_shotsAgainst = row['home_shotsAgainst']
    
    # Away team features
    team2_rolling_saves = row['away_teamSaves_rolling']
    team2_rolling_opponent_saves = row['away_opponentSaves_rolling']
    team2_backToBack = row['away_backToBack']
    team2_isHome = row['isHome_y']
    team2_goalsFor = row['away_goalsFor']
    team2_goalsAgainst = row['away_goalsAgainst']
    team2_shotsFor = row['away_shotsFor']
    team2_shotsAgainst = row['away_shotsAgainst']
    
    # Add features for the home team (team1)
    X_home.append([team1_rolling_saves, team1_rolling_opponent_saves, team1_backToBack, team1_isHome, team1_goalsFor, team1_goalsAgainst, team1_shotsFor, team1_shotsAgainst,
                  team2_rolling_saves, team2_rolling_opponent_saves, team2_backToBack, team2_isHome, team2_goalsFor, team2_goalsAgainst, team2_shotsFor, team2_shotsAgainst])
    y_home.append(row['home_teamSaves_rolling'])  # Target is home team's saves
    
    # Add features for the away team (team2)
    X_away.append([team2_rolling_saves, team2_rolling_opponent_saves, team2_backToBack, team2_isHome, team2_goalsFor, team2_goalsAgainst, team2_shotsFor, team2_shotsAgainst,
                  team1_rolling_saves, team1_rolling_opponent_saves, team1_backToBack, team1_isHome, team1_goalsFor, team1_goalsAgainst, team1_shotsFor, team1_shotsAgainst])
    y_away.append(row['away_teamSaves_rolling'])  # Target is away team's saves

# Convert lists to DataFrames
X_home = pd.DataFrame(X_home, columns=['team1_rolling_saves', 'team1_rolling_opponent_saves', 'team1_backToBack', 'team1_isHome', 'team1_goalsFor', 'team1_goalsAgainst', 'team1_shotsFor', 'team1_shotsAgainst',
                                       'team2_rolling_saves', 'team2_rolling_opponent_saves', 'team2_backToBack', 'team2_isHome', 'team2_goalsFor', 'team2_goalsAgainst', 'team2_shotsFor', 'team2_shotsAgainst'])
X_away = pd.DataFrame(X_away, columns=['team2_rolling_saves', 'team2_rolling_opponent_saves', 'team2_backToBack', 'team2_isHome', 'team2_goalsFor', 'team2_goalsAgainst', 'team2_shotsFor', 'team2_shotsAgainst',
                                       'team1_rolling_saves', 'team1_rolling_opponent_saves', 'team1_backToBack', 'team1_isHome', 'team1_goalsFor', 'team1_goalsAgainst', 'team1_shotsFor', 'team1_shotsAgainst'])
y_home = pd.Series(y_home)
y_away = pd.Series(y_away)

# Train-test split for both home and away teams
X_home_train, X_home_test, y_home_train, y_home_test = train_test_split(X_home, y_home, test_size=0.2, random_state=42)
X_away_train, X_away_test, y_away_train, y_away_test = train_test_split(X_away, y_away, test_size=0.2, random_state=42)

# Initialize and train the RandomForest model for home team saves prediction
model_home = RandomForestRegressor(n_estimators=100, random_state=42)
model_home.fit(X_home_train, y_home_train)

# Initialize and train the RandomForest model for away team saves prediction
model_away = RandomForestRegressor(n_estimators=100, random_state=42)
model_away.fit(X_away_train, y_away_train)

# Make predictions for both home and away teams
y_home_pred = model_home.predict(X_home_test)
y_away_pred = model_away.predict(X_away_test)

# Evaluate the models
print("Home team saves prediction MAE:", mean_absolute_error(y_home_test, y_home_pred))
print("Away team saves prediction MAE:", mean_absolute_error(y_away_test, y_away_pred))

# Function to predict saves for a new matchup
def predict_saves(team1, team2, game_id):
    # Get the features for the matchup using the get_matchup_data function
    matchup_input_home = get_matchup_data(team1, team2, game_id)
    matchup_input_away = get_matchup_data(team2, team1, game_id)
    
    # Convert the input dictionaries to DataFrames with appropriate column names
    home_input_df = pd.DataFrame([list(matchup_input_home.values())], columns=X_home.columns)
    away_input_df = pd.DataFrame([list(matchup_input_away.values())], columns=X_away.columns)
    
    # Make prediction for the home team and away team
    home_prediction = model_home.predict(home_input_df)
    away_prediction = model_away.predict(away_input_df)
    
    print(f"Predicted saves for {team1} vs {team2}:")
    print(f"Home team ({team1}) saves: {home_prediction[0]:.2f}")
    print(f"Away team ({team2}) saves: {away_prediction[0]:.2f}")

# Example: Predict saves for a new matchup
predict_saves('ANA', 'CGY', 2023020450)  # Example gameID (update with a real one)
