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

combined_df["teamCorsi_last"] = (
    combined_df.sort_values(by=['gameDate', 'gameID'])  # Ensure chronological order
    .groupby('team')['corsiFor']
    .shift(1)  # Shift by 1 to get the last game's value
)

combined_df["opponentCorsi_last"] = (
    combined_df.sort_values(by=['gameDate', 'gameID'])  # Ensure chronological order
    .groupby('team')['corsiAgainst']
    .shift(1)
)

combined_df["teamFenwick_last"] = (
    combined_df.sort_values(by=['gameDate', 'gameID'])  # Ensure chronological order
    .groupby('team')['fenwickFor']
    .shift(1)  # Shift by 1 to get the last game's value
)

combined_df["opponentFenwick_last"] = (
    combined_df.sort_values(by=['gameDate', 'gameID'])  # Ensure chronological order
    .groupby('team')['fenwickAgainst']
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

# Get the last game's saves for each team
combined_df["opponentTeamCorsi_last"] = (
    combined_df.sort_values(by=['gameDate', 'gameID'])  # Ensure chronological order
    .groupby('opponent')['corsiAgainst']
    .shift(1)  # Shift by 1 to get the last game's value
)

combined_df["opponentOpponentCorsi_last"] = (
    combined_df.sort_values(by=['gameDate', 'gameID'])  # Ensure chronological order
    .groupby('opponent')['corsiFor']
    .shift(1)
)

# Get the last game's saves for each team
combined_df["opponentTeamFenwick_last"] = (
    combined_df.sort_values(by=['gameDate', 'gameID'])  # Ensure chronological order
    .groupby('opponent')['fenwickAgainst']
    .shift(1)  # Shift by 1 to get the last game's value
)

combined_df["opponentOpponentFenwick_last"] = (
    combined_df.sort_values(by=['gameDate', 'gameID'])  # Ensure chronological order
    .groupby('opponent')['fenwickFor']
    .shift(1)
)

team_last_indices = combined_df.groupby('team').tail(1).index
opponent_last_indices = combined_df.groupby('opponent').tail(1).index

# Rolling averages for team saves (for each team)
combined_df['teamSaves_rolling_5'] = (
    combined_df.sort_values(by=['gameDate', 'gameID'])  # Ensure order
    .groupby('team')['teamSaves']
    .rolling(5, min_periods=3)  # Exponential weighting
    .mean()
    .reset_index(0, drop=True)
)

combined_df.loc[team_last_indices, 'teamSaves_rolling_5'] = None
combined_df['teamSaves_rolling_5'] = combined_df['teamSaves_rolling_5'].shift(1)

# Rolling averages for opponent saves (for each team)
combined_df['opponentSaves_rolling_5'] = (
    combined_df.sort_values(by=['gameDate', 'gameID'])  # Ensure order
    .groupby('team')['opponentSaves']
    .rolling(5, min_periods=3)  # Exponential weighting
    .mean()
    .reset_index(0, drop=True)
)

combined_df.loc[team_last_indices, 'opponentSaves_rolling_5'] = None
combined_df['opponentSaves_rolling_5'] = combined_df['opponentSaves_rolling_5'].shift(1)

combined_df['teamSaves_rolling_3'] = (
    combined_df.sort_values(by=['gameDate', 'gameID'])  # Ensure order
    .groupby('team')['teamSaves']
    .rolling(3, min_periods=3)  # Exponential weighting
    .mean()
    .reset_index(0, drop=True)
)

combined_df.loc[team_last_indices, 'teamSaves_rolling_3'] = None
combined_df['teamSaves_rolling_3'] = combined_df['teamSaves_rolling_3'].shift(1)

# Rolling averages for opponent saves (for each team)
combined_df['opponentSaves_rolling_3'] = (
    combined_df.sort_values(by=['gameDate', 'gameID'])  # Ensure order
    .groupby('team')['opponentSaves']
    .rolling(3, min_periods=3)  # Exponential weighting
    .mean()
    .reset_index(0, drop=True)
)

combined_df.loc[team_last_indices, 'opponentSaves_rolling_3'] = None
combined_df['opponentSaves_rolling_3'] = combined_df['opponentSaves_rolling_3'].shift(1)

# Rolling averages for team saves (for each team)
combined_df['teamSaves_rolling_10'] = (
    combined_df.sort_values(by=['gameDate', 'gameID'])  # Ensure order
    .groupby('team')['teamSaves']
    .rolling(10, min_periods=3)  # Exponential weighting
    .mean()
    .reset_index(0, drop=True)
)

combined_df.loc[team_last_indices, 'teamSaves_rolling_10'] = None
combined_df['teamSaves_rolling_10'] = combined_df['teamSaves_rolling_10'].shift(1)

# Rolling averages for opponent saves (for each team)
combined_df['opponentSaves_rolling_10'] = (
    combined_df.sort_values(by=['gameDate', 'gameID'])  # Ensure order
    .groupby('team')['opponentSaves']
    .rolling(10, min_periods=3)  # Exponential weighting
    .mean()
    .reset_index(0, drop=True)
)

combined_df.loc[team_last_indices, 'opponentSaves_rolling_10'] = None
combined_df['opponentSaves_rolling_10'] = combined_df['opponentSaves_rolling_10'].shift(1)

# Rolling averages for team saves (for each team)
combined_df['teamSaves_rolling_15'] = (
    combined_df.sort_values(by=['gameDate', 'gameID'])  # Ensure order
    .groupby('team')['teamSaves']
    .rolling(15, min_periods=3)  # Exponential weighting
    .mean()
    .reset_index(0, drop=True)
)

combined_df.loc[team_last_indices, 'teamSaves_rolling_15'] = None
combined_df['teamSaves_rolling_15'] = combined_df['teamSaves_rolling_15'].shift(1)

# Rolling averages for opponent saves (for each team)
combined_df['opponentSaves_rolling_15'] = (
    combined_df.sort_values(by=['gameDate', 'gameID'])  # Ensure order
    .groupby('team')['opponentSaves']
    .rolling(15, min_periods=3)  # Exponential weighting
    .mean()
    .reset_index(0, drop=True)
)

combined_df.loc[team_last_indices, 'opponentSaves_rolling_15'] = None
combined_df['opponentSaves_rolling_15'] = combined_df['opponentSaves_rolling_15'].shift(1)

# Rolling averages for team saves (for each team)
combined_df['teamCorsi_rolling_5'] = (
    combined_df.sort_values(by=['gameDate', 'gameID'])  # Ensure order
    .groupby('team')['corsiFor']
    .rolling(5, min_periods=3)  # Exponential weighting
    .mean()
    .reset_index(0, drop=True)
)

combined_df.loc[team_last_indices, 'teamCorsi_rolling_5'] = None
combined_df['teamCorsi_rolling_5'] = combined_df['teamCorsi_rolling_5'].shift(1)

# Rolling averages for opponent saves (for each team)
combined_df['opponentCorsi_rolling_5'] = (
    combined_df.sort_values(by=['gameDate', 'gameID'])  # Ensure order
    .groupby('team')['corsiAgainst']
    .rolling(5, min_periods=3)  # Exponential weighting
    .mean()
    .reset_index(0, drop=True)
)

combined_df.loc[team_last_indices, 'opponentCorsi_rolling_5'] = None
combined_df['opponentCorsi_rolling_5'] = combined_df['opponentCorsi_rolling_5'].shift(1)

combined_df['teamCorsi_rolling_3'] = (
    combined_df.sort_values(by=['gameDate', 'gameID'])  # Ensure order
    .groupby('team')['corsiFor']
    .rolling(3, min_periods=3)  # Exponential weighting
    .mean()
    .reset_index(0, drop=True)
)

combined_df.loc[team_last_indices, 'teamCorsi_rolling_3'] = None
combined_df['teamCorsi_rolling_3'] = combined_df['teamCorsi_rolling_3'].shift(1)

# Rolling averages for opponent saves (for each team)
combined_df['opponentCorsi_rolling_3'] = (
    combined_df.sort_values(by=['gameDate', 'gameID'])  # Ensure order
    .groupby('team')['corsiAgainst']
    .rolling(3, min_periods=3)  # Exponential weighting
    .mean()
    .reset_index(0, drop=True)
)

combined_df.loc[team_last_indices, 'opponentCorsi_rolling_3'] = None
combined_df['opponentCorsi_rolling_3'] = combined_df['opponentCorsi_rolling_3'].shift(1)

# Rolling averages for team saves (for each team)
combined_df['teamCorsi_rolling_10'] = (
    combined_df.sort_values(by=['gameDate', 'gameID'])  # Ensure order
    .groupby('team')['corsiFor']
    .rolling(10, min_periods=3)  # Exponential weighting
    .mean()
    .reset_index(0, drop=True)
)

combined_df.loc[team_last_indices, 'teamCorsi_rolling_10'] = None
combined_df['teamCorsi_rolling_10'] = combined_df['teamCorsi_rolling_10'].shift(1)

# Rolling averages for opponent saves (for each team)
combined_df['opponentCorsi_rolling_10'] = (
    combined_df.sort_values(by=['gameDate', 'gameID'])  # Ensure order
    .groupby('team')['corsiAgainst']
    .rolling(10, min_periods=3)  # Exponential weighting
    .mean()
    .reset_index(0, drop=True)
)

combined_df.loc[team_last_indices, 'opponentCorsi_rolling_10'] = None
combined_df['opponentCorsi_rolling_10'] = combined_df['opponentCorsi_rolling_10'].shift(1)

# Rolling averages for team saves (for each team)
combined_df['teamCorsi_rolling_15'] = (
    combined_df.sort_values(by=['gameDate', 'gameID'])  # Ensure order
    .groupby('team')['corsiFor']
    .rolling(15, min_periods=3)  # Exponential weighting
    .mean()
    .reset_index(0, drop=True)
)

combined_df.loc[team_last_indices, 'teamCorsi_rolling_15'] = None
combined_df['teamCorsi_rolling_15'] = combined_df['teamCorsi_rolling_15'].shift(1)

# Rolling averages for opponent saves (for each team)
combined_df['opponentCorsi_rolling_15'] = (
    combined_df.sort_values(by=['gameDate', 'gameID'])  # Ensure order
    .groupby('team')['corsiAgainst']
    .rolling(15, min_periods=3)  # Exponential weighting
    .mean()
    .reset_index(0, drop=True)
)

combined_df.loc[team_last_indices, 'opponentCorsi_rolling_15'] = None
combined_df['opponentCorsi_rolling_15'] = combined_df['opponentCorsi_rolling_15'].shift(1)

# Rolling averages for team saves (for each team)
combined_df['teamFenwick_rolling_5'] = (
    combined_df.sort_values(by=['gameDate', 'gameID'])  # Ensure order
    .groupby('team')['fenwickFor']
    .rolling(5, min_periods=3)  # Exponential weighting
    .mean()
    .reset_index(0, drop=True)
)

combined_df.loc[team_last_indices, 'teamFenwick_rolling_5'] = None
combined_df['teamFenwick_rolling_5'] = combined_df['teamFenwick_rolling_5'].shift(1)

# Rolling averages for opponent saves (for each team)
combined_df['opponentFenwick_rolling_5'] = (
    combined_df.sort_values(by=['gameDate', 'gameID'])  # Ensure order
    .groupby('team')['fenwickAgainst']
    .rolling(5, min_periods=3)  # Exponential weighting
    .mean()
    .reset_index(0, drop=True)
)

combined_df.loc[team_last_indices, 'opponentFenwick_rolling_5'] = None
combined_df['opponentFenwick_rolling_5'] = combined_df['opponentFenwick_rolling_5'].shift(1)

combined_df['teamFenwick_rolling_3'] = (
    combined_df.sort_values(by=['gameDate', 'gameID'])  # Ensure order
    .groupby('team')['fenwickFor']
    .rolling(3, min_periods=3)  # Exponential weighting
    .mean()
    .reset_index(0, drop=True)
)

combined_df.loc[team_last_indices, 'teamFenwick_rolling_3'] = None
combined_df['teamFenwick_rolling_3'] = combined_df['teamFenwick_rolling_3'].shift(1)

# Rolling averages for opponent saves (for each team)
combined_df['opponentFenwick_rolling_3'] = (
    combined_df.sort_values(by=['gameDate', 'gameID'])  # Ensure order
    .groupby('team')['fenwickAgainst']
    .rolling(3, min_periods=3)  # Exponential weighting
    .mean()
    .reset_index(0, drop=True)
)

combined_df.loc[team_last_indices, 'opponentFenwick_rolling_3'] = None
combined_df['opponentFenwick_rolling_3'] = combined_df['opponentFenwick_rolling_3'].shift(1)

# Rolling averages for team saves (for each team)
combined_df['teamFenwick_rolling_10'] = (
    combined_df.sort_values(by=['gameDate', 'gameID'])  # Ensure order
    .groupby('team')['fenwickFor']
    .rolling(10, min_periods=3)  # Exponential weighting
    .mean()
    .reset_index(0, drop=True)
)

combined_df.loc[team_last_indices, 'teamFenwick_rolling_10'] = None
combined_df['teamFenwick_rolling_10'] = combined_df['teamFenwick_rolling_10'].shift(1)

# Rolling averages for opponent saves (for each team)
combined_df['opponentFenwick_rolling_10'] = (
    combined_df.sort_values(by=['gameDate', 'gameID'])  # Ensure order
    .groupby('team')['fenwickAgainst']
    .rolling(10, min_periods=3)  # Exponential weighting
    .mean()
    .reset_index(0, drop=True)
)

combined_df.loc[team_last_indices, 'opponentFenwick_rolling_10'] = None
combined_df['opponentFenwick_rolling_10'] = combined_df['opponentFenwick_rolling_10'].shift(1)

# Rolling averages for team saves (for each team)
combined_df['teamFenwick_rolling_15'] = (
    combined_df.sort_values(by=['gameDate', 'gameID'])  # Ensure order
    .groupby('team')['fenwickFor']
    .rolling(15, min_periods=3)  # Exponential weighting
    .mean()
    .reset_index(0, drop=True)
)

combined_df.loc[team_last_indices, 'teamFenwick_rolling_15'] = None
combined_df['teamFenwick_rolling_15'] = combined_df['teamFenwick_rolling_15'].shift(1)

# Rolling averages for opponent saves (for each team)
combined_df['opponentFenwick_rolling_15'] = (
    combined_df.sort_values(by=['gameDate', 'gameID'])  # Ensure order
    .groupby('team')['fenwickAgainst']
    .rolling(15, min_periods=3)  # Exponential weighting
    .mean()
    .reset_index(0, drop=True)
)

combined_df.loc[team_last_indices, 'opponentFenwick_rolling_15'] = None
combined_df['opponentFenwick_rolling_15'] = combined_df['opponentFenwick_rolling_15'].shift(1)

# Remove previous opponent rolling calculations
combined_df = combined_df.sort_values(by=['gameDate', 'gameID'])

# Create a mapping of gameID to its corresponding rows
id_map = combined_df.groupby('gameID').apply(lambda x: x.index.tolist()).to_dict()

# Iterate through the dataframe and match values based on gameID
for game_id, indices in id_map.items():
    if len(indices) == 2:
        idx1, idx2 = indices
        
        # Loop through different spans to set rolling values dynamically
        for span in [3, 5, 10, 15]:
            team_col = f'teamSaves_rolling_{span}'
            opponent_col = f'opponentSaves_rolling_{span}'
            opp_team_col = f'opponentTeamSaves_rolling_{span}'
            opp_opp_col = f'opponentOpponentSaves_rolling_{span}'

            team_c_col = f'teamCorsi_rolling_{span}'
            opponent_c_col = f'opponentCorsi_rolling_{span}'
            opp_team_c_col = f'opponentTeamCorsi_rolling_{span}'
            opp_opp_c_col = f'opponentOpponentCorsi_rolling_{span}'

            team_f_col = f'teamFenwick_rolling_{span}'
            opponent_f_col = f'opponentFenwick_rolling_{span}'
            opp_team_f_col = f'opponentTeamFenwick_rolling_{span}'
            opp_opp_f_col = f'opponentOpponentFenwick_rolling_{span}'
            
            combined_df.at[idx1, opp_team_c_col] = combined_df.at[idx2, team_c_col]
            combined_df.at[idx1, opp_opp_c_col] = combined_df.at[idx2, opponent_c_col]
            combined_df.at[idx2, opp_team_c_col] = combined_df.at[idx1, team_c_col]
            combined_df.at[idx2, opp_opp_c_col] = combined_df.at[idx1, opponent_c_col]

            combined_df.at[idx1, opp_team_f_col] = combined_df.at[idx2, team_f_col]
            combined_df.at[idx1, opp_opp_f_col] = combined_df.at[idx2, opponent_f_col]
            combined_df.at[idx2, opp_team_f_col] = combined_df.at[idx1, team_f_col]
            combined_df.at[idx2, opp_opp_f_col] = combined_df.at[idx1, opponent_f_col]

            combined_df.at[idx1, opp_team_col] = combined_df.at[idx2, team_col]
            combined_df.at[idx1, opp_opp_col] = combined_df.at[idx2, opponent_col]
            combined_df.at[idx2, opp_team_col] = combined_df.at[idx1, team_col]
            combined_df.at[idx2, opp_opp_col] = combined_df.at[idx1, opponent_col]

# Filter out the necessary columns to preserve the team-specific features
home_team_columns = ['gameID', "gameDate", 'isHome', 'opponent', 'team', 
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

                     'backToBack', 'teamSaves', 'opponentSaves', 'corsiFor', 'corsiAgainst', 'fenwickFor', 'fenwickAgainst', 'splitGame']


# Split the data into home and away teams
home_games = combined_df[combined_df['isHome'] != None][home_team_columns]

# merged_df = home_games.dropna().reset_index(drop=True)

# The final dataset now contains one row per game, with both home and away team data
home_games.to_csv('S:/Documents/GitHub/saves-model/combined_simplified.csv', index=False)