import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Load your combined dataset with rolling averages
combined_df = pd.read_csv("combined_with_rolling_averages.csv")

# Prepare features and target variables for both teams
X_home = []
X_away = []
y_home = []
y_away = []

# Function to get matchup data based on team names and game_id
def get_matchup_data(team1, team2, game_id):
    # Get the relevant data for team1 (home team)
    team1_data = combined_df[(combined_df['team'] == team1) & (combined_df['gameID'] <= game_id)]
    team1_rolling_saves = team1_data['teamSaves_rolling'].tail(1).values[0]
    team1_rolling_opponent_saves = team1_data['opponentSaves_rolling'].tail(1).values[0]
    team1_backToBack = team1_data['backToBack'].tail(1).values[0]
    team1_isHome = team1_data['isHome'].tail(1).values[0]
    team1_goalsFor = team1_data['goalsFor'].tail(1).values[0]
    team1_goalsAgainst = team1_data['goalsAgainst'].tail(1).values[0]
    team1_shotsFor = team1_data['shotsFor'].tail(1).values[0]
    team1_shotsAgainst = team1_data['shotsAgainst'].tail(1).values[0]

    # Get the relevant data for team2 (away team)
    team2_data = combined_df[(combined_df['team'] == team2) & (combined_df['gameID'] <= game_id)]
    team2_rolling_saves = team2_data['teamSaves_rolling'].tail(1).values[0]
    team2_rolling_opponent_saves = team2_data['opponentSaves_rolling'].tail(1).values[0]
    team2_backToBack = team2_data['backToBack'].tail(1).values[0]
    team2_isHome = team2_data['isHome'].tail(1).values[0]
    team2_goalsFor = team2_data['goalsFor'].tail(1).values[0]
    team2_goalsAgainst = team2_data['goalsAgainst'].tail(1).values[0]
    team2_shotsFor = team2_data['shotsFor'].tail(1).values[0]
    team2_shotsAgainst = team2_data['shotsAgainst'].tail(1).values[0]

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
    team1 = row['team']
    team1_rolling_saves = row['teamSaves_rolling']
    team1_rolling_opponent_saves = row['opponentSaves_rolling']
    team1_backToBack = row['backToBack']
    team1_isHome = row['isHome']
    team1_goalsFor = row['goalsFor']
    team1_goalsAgainst = row['goalsAgainst']
    team1_shotsFor = row['shotsFor']
    team1_shotsAgainst = row['shotsAgainst']
    
    # Get the latest opponent data (away team)
    opponent_data = combined_df[(combined_df['team'] == row['opponent']) & (combined_df['gameID'] <= row['gameID'])].tail(1)
    if not opponent_data.empty:
        team2_rolling_saves = opponent_data['teamSaves_rolling'].values[0]
        team2_rolling_opponent_saves = opponent_data['opponentSaves_rolling'].values[0]
        team2_backToBack = opponent_data['backToBack'].values[0]
        team2_isHome = opponent_data['isHome'].values[0]
        team2_goalsFor = opponent_data['goalsFor'].values[0]
        team2_goalsAgainst = opponent_data['goalsAgainst'].values[0]
        team2_shotsFor = opponent_data['shotsFor'].values[0]
        team2_shotsAgainst = opponent_data['shotsAgainst'].values[0]
    else:
        team2_rolling_saves = 0
        team2_rolling_opponent_saves = 0
        team2_backToBack = False
        team2_isHome = False
        team2_goalsFor = 0
        team2_goalsAgainst = 0
        team2_shotsFor = 0
        team2_shotsAgainst = 0
    
    # Add features for the home team (team1)
    X_home.append([team1_rolling_saves, team1_rolling_opponent_saves, team1_backToBack, team1_isHome, team1_goalsFor, team1_goalsAgainst, team1_shotsFor, team1_shotsAgainst,
                  team2_rolling_saves, team2_rolling_opponent_saves, team2_backToBack, team2_isHome, team2_goalsFor, team2_goalsAgainst, team2_shotsFor, team2_shotsAgainst])
    y_home.append(row['teamSaves'])  # Target is home team's saves
    
    # Add features for the away team (team2)
    X_away.append([team2_rolling_saves, team2_rolling_opponent_saves, team2_backToBack, team2_isHome, team2_goalsFor, team2_goalsAgainst, team2_shotsFor, team2_shotsAgainst,
                  team1_rolling_saves, team1_rolling_opponent_saves, team1_backToBack, team1_isHome, team1_goalsFor, team1_goalsAgainst, team1_shotsFor, team1_shotsAgainst])
    y_away.append(row['opponentSaves'])  # Target is away team's saves

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
predict_saves('ANA', 'CGY', 2023020031)  # Example gameID (update with a real one)
