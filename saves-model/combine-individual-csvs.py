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

# Sort by team and gameID to ensure chronological order
combined_df = combined_df.sort_values(by=['gameID'])

# Optionally, reset the index after dropping rows
combined_df.reset_index(drop=True, inplace=True)

# Save the updated dataframe to a new CSV file
combined_df.to_csv("combined_with_rolling_averages.csv", index=False)

print("Updated dataset with rolling averages has been saved to 'combined_with_rolling_averages.csv'")

# Load the combined dataset
combined_df = pd.read_csv("combined_with_rolling_averages.csv")

# Filter out the necessary columns to preserve the team-specific features
home_team_columns = ['gameID', 'isHome', 'opponent', 'team', 'backToBack', 'goalsFor', 'goalsAgainst', 'shotsFor', 'shotsAgainst', 'teamSaves', 'opponentSaves']
away_team_columns = ['gameID', 'isHome', 'opponent', 'team', 'backToBack', 'goalsFor', 'goalsAgainst', 'shotsFor', 'shotsAgainst', 'teamSaves', 'opponentSaves']

# Split the data into home and away teams
home_games = combined_df[combined_df['isHome'] == True][home_team_columns]
away_games = combined_df[combined_df['isHome'] == False][away_team_columns]

# Rename columns for home and away teams to avoid conflict
home_games = home_games.rename(columns={
    'backToBack': 'home_backToBack',
    'goalsFor': 'home_goalsFor',
    'goalsAgainst': 'home_goalsAgainst',
    'shotsFor': 'home_shotsFor',
    'shotsAgainst': 'home_shotsAgainst',
    'team': 'home_team',
    'opponent': 'home_opponent',
    'teamSaves': 'home_teamSaves',
    'opponentSaves': 'home_opponentSaves'
})

away_games = away_games.rename(columns={
    'backToBack': 'away_backToBack',
    'goalsFor': 'away_goalsFor',
    'goalsAgainst': 'away_goalsAgainst',
    'shotsFor': 'away_shotsFor',
    'shotsAgainst': 'away_shotsAgainst',
    'team': 'away_team',
    'opponent': 'away_opponent',
    'teamSaves': 'away_teamSaves',
    'opponentSaves': 'away_opponentSaves'
})

# Merge the home and away games on 'gameID' to get one row per game
merged_df = pd.merge(home_games, away_games, on='gameID', how='inner')

# The final dataset now contains one row per game, with both home and away team data
merged_df.to_csv('combined_simplified.csv', index=False)
