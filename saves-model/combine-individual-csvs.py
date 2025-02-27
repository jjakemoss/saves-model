import pandas as pd
import os
from abbreviations import nhl_team_dict

# Folder containing all team CSVs (update with your actual path)
csv_folder = "S:/Documents/GitHub/saves-model/team_schedules"

# List all CSV files in the folder
csv_files = [f for f in os.listdir(csv_folder) if f.endswith(".csv")]

# Initialize an empty list to store dataframes
df_list = []

# Read each CSV file, add team column, and append to list
for file in csv_files:
    # Extract the 3-letter team abbreviation from the file name
    team_abbr = file[:3]  # Assumes the first 3 characters are the team abbreviation
    
    # Read the CSV into a DataFrame
    team_df = pd.read_csv(os.path.join(csv_folder, file))
    
    # Add a new column 'team' with the team abbreviation
    team_df['team'] = team_abbr
    
    # Append the DataFrame to the list
    df_list.append(team_df)

# Combine all dataframes into a single dataset
combined_df = pd.concat(df_list, ignore_index=True)

# Get the last game's saves for each team
combined_df["teamSaves_last"] = (
    combined_df.sort_values(by=['gameDate', 'gameID'])  # Ensure chronological order
    .groupby('team')['teamSaves']
    .shift(1)  # Shift by 1 to get the last game's value
)

combined_df["opponentSaves_last"] = (
    combined_df.sort_values(by=['gameDate', 'gameID'])  # Ensure chronological order
    .groupby('team')['opponentSaves']
    .shift(1)
)

# Get the last game's saves for each team
combined_df["opponentTeamSaves_last"] = (
    combined_df.sort_values(by=['gameDate', 'gameID'])  # Ensure chronological order
    .groupby('opponent')['opponentSaves']
    .shift(1)  # Shift by 1 to get the last game's value
)

combined_df["opponentOpponentSaves_last"] = (
    combined_df.sort_values(by=['gameDate', 'gameID'])  # Ensure chronological order
    .groupby('opponent')['teamSaves']
    .shift(1)
)

# Rolling averages for team saves (for each team)
combined_df['teamSaves_rolling'] = (
    combined_df.sort_values(by=['gameDate', 'gameID'])  # Ensure order
    .groupby('team')['teamSaves']
    .ewm(span=5, adjust=False, min_periods=3)  # Exponential weighting
    .mean()
    .reset_index(0, drop=True)
)

combined_df.at[combined_df.index[-1], 'teamSaves_rolling'] = None
combined_df['teamSaves_rolling'] = combined_df['teamSaves_rolling'].shift(1)

# Rolling averages for opponent saves (for each team)
combined_df['opponentSaves_rolling'] = (
    combined_df.sort_values(by=['gameDate', 'gameID'])  # Ensure order
    .groupby('team')['opponentSaves']
    .ewm(span=5, adjust=False, min_periods=3)  # Exponential weighting
    .mean()
    .reset_index(0, drop=True)
)

combined_df.at[combined_df.index[-1], 'opponentSaves_rolling'] = None
combined_df['opponentSaves_rolling'] = combined_df['opponentSaves_rolling'].shift(1)

combined_df['teamSaves_rolling_3'] = (
    combined_df.sort_values(by=['gameDate', 'gameID'])  # Ensure order
    .groupby('team')['teamSaves']
    .ewm(span=3, adjust=False, min_periods=3)  # Exponential weighting
    .mean()
    .reset_index(0, drop=True)
)

combined_df.at[combined_df.index[-1], 'teamSaves_rolling_3'] = None
combined_df['teamSaves_rolling_3'] = combined_df['teamSaves_rolling_3'].shift(1)

# Rolling averages for opponent saves (for each team)
combined_df['opponentSaves_rolling_3'] = (
    combined_df.sort_values(by=['gameDate', 'gameID'])  # Ensure order
    .groupby('team')['opponentSaves']
    .ewm(span=3, adjust=False, min_periods=3)  # Exponential weighting
    .mean()
    .reset_index(0, drop=True)
)

combined_df.at[combined_df.index[-1], 'opponentSaves_rolling_3'] = None
combined_df['opponentSaves_rolling_3'] = combined_df['opponentSaves_rolling_3'].shift(1)

# Rolling averages for team saves (for each team)
combined_df['teamSaves_rolling_10'] = (
    combined_df.sort_values(by=['gameDate', 'gameID'])  # Ensure order
    .groupby('team')['teamSaves']
    .ewm(span=10, adjust=False, min_periods=3)  # Exponential weighting
    .mean()
    .reset_index(0, drop=True)
)

combined_df.at[combined_df.index[-1], 'teamSaves_rolling_10'] = None
combined_df['teamSaves_rolling_10'] = combined_df['teamSaves_rolling_10'].shift(1)

# Rolling averages for opponent saves (for each team)
combined_df['opponentSaves_rolling_10'] = (
    combined_df.sort_values(by=['gameDate', 'gameID'])  # Ensure order
    .groupby('team')['opponentSaves']
    .ewm(span=10, adjust=False, min_periods=3)  # Exponential weighting
    .mean()
    .reset_index(0, drop=True)
)

combined_df.at[combined_df.index[-1], 'opponentSaves_rolling_10'] = None
combined_df['opponentSaves_rolling_10'] = combined_df['opponentSaves_rolling_10'].shift(1)

# Rolling averages for team saves (for each team)
combined_df['teamSaves_rolling_15'] = (
    combined_df.sort_values(by=['gameDate', 'gameID'])  # Ensure order
    .groupby('team')['teamSaves']
    .ewm(span=15, adjust=False, min_periods=3)  # Exponential weighting
    .mean()
    .reset_index(0, drop=True)
)

combined_df.at[combined_df.index[-1], 'teamSaves_rolling_15'] = None
combined_df['teamSaves_rolling_15'] = combined_df['teamSaves_rolling_15'].shift(1)

# Rolling averages for opponent saves (for each team)
combined_df['opponentSaves_rolling_15'] = (
    combined_df.sort_values(by=['gameDate', 'gameID'])  # Ensure order
    .groupby('team')['opponentSaves']
    .ewm(span=15, adjust=False, min_periods=3)  # Exponential weighting
    .mean()
    .reset_index(0, drop=True)
)

combined_df.at[combined_df.index[-1], 'opponentSaves_rolling_15'] = None
combined_df['opponentSaves_rolling_15'] = combined_df['opponentSaves_rolling_15'].shift(1)

# Rolling averages for team saves (for each team)
combined_df['opponentTeamSaves_rolling'] = (
    combined_df.sort_values(by=['gameDate', 'gameID'])  # Ensure order
    .groupby('opponent')['opponentSaves']
    .ewm(span=5, adjust=False, min_periods=3)  # Exponential weighting
    .mean()
    .reset_index(0, drop=True)
)

combined_df.at[combined_df.index[-1], 'opponentTeamSaves_rolling'] = None
combined_df['opponentTeamSaves_rolling'] = combined_df['opponentTeamSaves_rolling'].shift(1)

# Rolling averages for opponent saves (for each team)
combined_df['opponentOpponentSaves_rolling'] = (
    combined_df.sort_values(by=['gameDate', 'gameID'])  # Ensure order
    .groupby('opponent')['teamSaves']
    .ewm(span=5, adjust=False, min_periods=3)  # Exponential weighting
    .mean()
    .reset_index(0, drop=True)
)

combined_df.at[combined_df.index[-1], 'opponentOpponentSaves_rolling'] = None
combined_df['opponentOpponentSaves_rolling'] = combined_df['opponentOpponentSaves_rolling'].shift(1)

combined_df['opponentTeamSaves_rolling_3'] = (
    combined_df.sort_values(['gameDate'])  # Ensure order
    .groupby('opponent')['opponentSaves']
    .ewm(span=3, adjust=False, min_periods=3)  # Exponential weighting
    .mean()
    .reset_index(0, drop=True)
)

combined_df.at[combined_df.index[-1], 'opponentTeamSaves_rolling_3'] = None
combined_df['opponentTeamSaves_rolling_3'] = combined_df['opponentTeamSaves_rolling_3'].shift(1)

# Rolling averages for opponent saves (for each team)
combined_df['opponentOpponentSaves_rolling_3'] = (
    combined_df.sort_values(by=['gameDate', 'gameID'])  # Ensure order
    .groupby('opponent')['teamSaves']
    .ewm(span=3, adjust=False, min_periods=3)  # Exponential weighting
    .mean()
    .reset_index(0, drop=True)
)

combined_df.at[combined_df.index[-1], 'opponentOpponentSaves_rolling_3'] = None
combined_df['opponentOpponentSaves_rolling_3'] = combined_df['opponentOpponentSaves_rolling_3'].shift(1)

# Rolling averages for team saves (for each team)
combined_df['opponentTeamSaves_rolling_10'] = (
    combined_df.sort_values(by=['gameDate', 'gameID'])  # Ensure order
    .groupby('opponent')['opponentSaves']
    .ewm(span=10, adjust=False, min_periods=3)  # Exponential weighting
    .mean()
    .reset_index(0, drop=True)
)

combined_df.at[combined_df.index[-1], 'opponentTeamSaves_rolling_10'] = None
combined_df['opponentTeamSaves_rolling_10'] = combined_df['opponentTeamSaves_rolling_10'].shift(1)

# Rolling averages for opponent saves (for each team)
combined_df['opponentOpponentSaves_rolling_10'] = (
    combined_df.sort_values(by=['gameDate', 'gameID'])  # Ensure order
    .groupby('opponent')['teamSaves']
    .ewm(span=10, adjust=False, min_periods=3)  # Exponential weighting
    .mean()
    .reset_index(0, drop=True)
)

combined_df.at[combined_df.index[-1], 'opponentOpponentSaves_rolling_10'] = None
combined_df['opponentOpponentSaves_rolling_10'] = combined_df['opponentOpponentSaves_rolling_10'].shift(1)

# Rolling averages for team saves (for each team)
combined_df['opponentTeamSaves_rolling_15'] = (
    combined_df.sort_values(by=['gameDate', 'gameID'])  # Ensure order
    .groupby('opponent')['opponentSaves']
    .ewm(span=15, adjust=False, min_periods=3)  # Exponential weighting
    .mean()
    .reset_index(0, drop=True)
)

combined_df.at[combined_df.index[-1], 'opponentTeamSaves_rolling_15'] = None
combined_df['opponentTeamSaves_rolling_15'] = combined_df['opponentTeamSaves_rolling_15'].shift(1)

# Rolling averages for opponent saves (for each team)
combined_df['opponentOpponentSaves_rolling_15'] = (
    combined_df.sort_values(by=['gameDate', 'gameID'])  # Ensure order
    .groupby('opponent')['teamSaves']
    .ewm(span=15, adjust=False, min_periods=3)  # Exponential weighting
    .mean()
    .reset_index(0, drop=True)
)

combined_df.at[combined_df.index[-1], 'opponentOpponentSaves_rolling_15'] = None
combined_df['opponentOpponentSaves_rolling_15'] = combined_df['opponentOpponentSaves_rolling_15'].shift(1)

combined_df = combined_df.sort_values(by="gameDate").reset_index(drop=True)

# Filter out the necessary columns to preserve the team-specific features
home_team_columns = ['gameID', "gameDate", 'isHome', 'opponent', 'team', 'teamSaves_last', 'opponentSaves_last', 'teamSaves_rolling', 'opponentSaves_rolling', 'teamSaves_rolling_3', 'opponentSaves_rolling_3', 'teamSaves_rolling_10', 'opponentSaves_rolling_10', 'teamSaves_rolling_15', 'opponentSaves_rolling_15',
                     'opponentTeamSaves_last', 'opponentOpponentSaves_last', 'opponentTeamSaves_rolling', 'opponentOpponentSaves_rolling', 'opponentTeamSaves_rolling_3', 'opponentOpponentSaves_rolling_3', 'opponentTeamSaves_rolling_10', 'opponentOpponentSaves_rolling_10', 'opponentTeamSaves_rolling_15', 'opponentOpponentSaves_rolling_15',
                      'backToBack', 'teamSaves', 'opponentSaves', 'splitGame']

# Split the data into home and away teams
home_games = combined_df[combined_df['isHome'] != None][home_team_columns]

merged_df = home_games.dropna().reset_index(drop=True)

# The final dataset now contains one row per game, with both home and away team data
merged_df.to_csv('S:/Documents/GitHub/saves-model/combined_simplified.csv', index=False)