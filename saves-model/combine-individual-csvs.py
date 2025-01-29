import pandas as pd
import os

# Folder containing all team CSVs (update with your actual path)
csv_folder = "team_schedules"

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

# Calculate rolling averages for teamSaves and opponentSaves
window_size = 5  # Define the window size for rolling average

# Rolling averages for team saves (for each team)
combined_df['teamSaves_rolling'] = combined_df.groupby('team')['teamSaves'].rolling(window=window_size, min_periods=1).mean().reset_index(0, drop=True)

# Rolling averages for opponent saves (for each team)
combined_df['opponentSaves_rolling'] = combined_df.groupby('team')['opponentSaves'].rolling(window=window_size, min_periods=1).mean().reset_index(0, drop=True)

combined_df = combined_df.sort_values(by="gameID").reset_index(drop=True)

# Save the updated dataframe to a new CSV file
combined_df.to_csv("combined_with_rolling_averages.csv", index=False)

print("Updated dataset with rolling averages has been saved to 'combined_with_rolling_averages.csv'")

# Load the combined dataset
combined_df = pd.read_csv("combined_with_rolling_averages.csv")

# Filter out the necessary columns to preserve the team-specific features
home_team_columns = ['gameID', 'isHome', 'opponent', 'team', 'teamSaves_rolling', 'opponentSaves_rolling', 'backToBack', 'goalsFor', 'goalsAgainst', 'shotsFor', 'shotsAgainst']
away_team_columns = ['gameID', 'isHome', 'opponent', 'team', 'teamSaves_rolling', 'opponentSaves_rolling', 'backToBack', 'goalsFor', 'goalsAgainst', 'shotsFor', 'shotsAgainst']

# Split the data into home and away teams
home_games = combined_df[combined_df['isHome'] == True][home_team_columns]
away_games = combined_df[combined_df['isHome'] == False][away_team_columns]

# Rename columns for home and away teams to avoid conflict
home_games = home_games.rename(columns={
    'teamSaves_rolling': 'home_teamSaves_rolling',
    'opponentSaves_rolling': 'home_opponentSaves_rolling',
    'backToBack': 'home_backToBack',
    'goalsFor': 'home_goalsFor',
    'goalsAgainst': 'home_goalsAgainst',
    'shotsFor': 'home_shotsFor',
    'shotsAgainst': 'home_shotsAgainst',
    'team': 'home_team',
    'opponent': 'home_opponent'
})

away_games = away_games.rename(columns={
    'teamSaves_rolling': 'away_teamSaves_rolling',
    'opponentSaves_rolling': 'away_opponentSaves_rolling',
    'backToBack': 'away_backToBack',
    'goalsFor': 'away_goalsFor',
    'goalsAgainst': 'away_goalsAgainst',
    'shotsFor': 'away_shotsFor',
    'shotsAgainst': 'away_shotsAgainst',
    'team': 'away_team',
    'opponent': 'away_opponent'
})

# Merge the home and away games on 'gameID' to get one row per game
merged_df = pd.merge(home_games, away_games, on='gameID', how='inner')

# The final dataset now contains one row per game, with both home and away team data
merged_df.to_csv('combined_simplified.csv', index=False)
