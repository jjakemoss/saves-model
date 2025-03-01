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

combined_df["teamFen_last"] = (
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
combined_df['teamCorsi_rolling'] = (
    combined_df.sort_values(by=['gameDate', 'gameID'])  # Ensure order
    .groupby('team')['corsiFor']
    .ewm(span=5, adjust=False, min_periods=3)  # Exponential weighting
    .mean()
    .reset_index(0, drop=True)
)

combined_df.at[combined_df.index[-1], 'teamCorsi_rolling'] = None
combined_df['teamCorsi_rolling'] = combined_df['teamCorsi_rolling'].shift(1)

# Rolling averages for opponent saves (for each team)
combined_df['opponentCorsi_rolling'] = (
    combined_df.sort_values(by=['gameDate', 'gameID'])  # Ensure order
    .groupby('team')['corsiAgainst']
    .ewm(span=5, adjust=False, min_periods=3)  # Exponential weighting
    .mean()
    .reset_index(0, drop=True)
)

combined_df.at[combined_df.index[-1], 'opponentCorsi_rolling'] = None
combined_df['opponentCorsi_rolling'] = combined_df['opponentCorsi_rolling'].shift(1)

combined_df['teamCorsi_rolling_3'] = (
    combined_df.sort_values(by=['gameDate', 'gameID'])  # Ensure order
    .groupby('team')['corsiFor']
    .ewm(span=3, adjust=False, min_periods=3)  # Exponential weighting
    .mean()
    .reset_index(0, drop=True)
)

combined_df.at[combined_df.index[-1], 'teamCorsi_rolling_3'] = None
combined_df['teamCorsi_rolling_3'] = combined_df['teamCorsi_rolling_3'].shift(1)

# Rolling averages for opponent saves (for each team)
combined_df['opponentCorsi_rolling_3'] = (
    combined_df.sort_values(by=['gameDate', 'gameID'])  # Ensure order
    .groupby('team')['corsiAgainst']
    .ewm(span=3, adjust=False, min_periods=3)  # Exponential weighting
    .mean()
    .reset_index(0, drop=True)
)

combined_df.at[combined_df.index[-1], 'opponentCorsi_rolling_3'] = None
combined_df['opponentCorsi_rolling_3'] = combined_df['opponentCorsi_rolling_3'].shift(1)

# Rolling averages for team saves (for each team)
combined_df['teamCorsi_rolling_10'] = (
    combined_df.sort_values(by=['gameDate', 'gameID'])  # Ensure order
    .groupby('team')['corsiFor']
    .ewm(span=10, adjust=False, min_periods=3)  # Exponential weighting
    .mean()
    .reset_index(0, drop=True)
)

combined_df.at[combined_df.index[-1], 'teamCorsi_rolling_10'] = None
combined_df['teamCorsi_rolling_10'] = combined_df['teamCorsi_rolling_10'].shift(1)

# Rolling averages for opponent saves (for each team)
combined_df['opponentCorsi_rolling_10'] = (
    combined_df.sort_values(by=['gameDate', 'gameID'])  # Ensure order
    .groupby('team')['corsiAgainst']
    .ewm(span=10, adjust=False, min_periods=3)  # Exponential weighting
    .mean()
    .reset_index(0, drop=True)
)

combined_df.at[combined_df.index[-1], 'opponentCorsi_rolling_10'] = None
combined_df['opponentCorsi_rolling_10'] = combined_df['opponentCorsi_rolling_10'].shift(1)

# Rolling averages for team saves (for each team)
combined_df['teamCorsi_rolling_15'] = (
    combined_df.sort_values(by=['gameDate', 'gameID'])  # Ensure order
    .groupby('team')['corsiFor']
    .ewm(span=15, adjust=False, min_periods=3)  # Exponential weighting
    .mean()
    .reset_index(0, drop=True)
)

combined_df.at[combined_df.index[-1], 'teamCorsi_rolling_15'] = None
combined_df['teamCorsi_rolling_15'] = combined_df['teamCorsi_rolling_15'].shift(1)

# Rolling averages for opponent saves (for each team)
combined_df['opponentCorsi_rolling_15'] = (
    combined_df.sort_values(by=['gameDate', 'gameID'])  # Ensure order
    .groupby('team')['corsiAgainst']
    .ewm(span=15, adjust=False, min_periods=3)  # Exponential weighting
    .mean()
    .reset_index(0, drop=True)
)

combined_df.at[combined_df.index[-1], 'opponentCorsi_rolling_15'] = None
combined_df['opponentCorsi_rolling_15'] = combined_df['opponentCorsi_rolling_15'].shift(1)

# Rolling averages for team saves (for each team)
combined_df['teamFenwick_rolling'] = (
    combined_df.sort_values(by=['gameDate', 'gameID'])  # Ensure order
    .groupby('team')['fenwickFor']
    .ewm(span=5, adjust=False, min_periods=3)  # Exponential weighting
    .mean()
    .reset_index(0, drop=True)
)

combined_df.at[combined_df.index[-1], 'teamFenwick_rolling'] = None
combined_df['teamFenwick_rolling'] = combined_df['teamFenwick_rolling'].shift(1)

# Rolling averages for opponent saves (for each team)
combined_df['opponentFenwick_rolling'] = (
    combined_df.sort_values(by=['gameDate', 'gameID'])  # Ensure order
    .groupby('team')['fenwickAgainst']
    .ewm(span=5, adjust=False, min_periods=3)  # Exponential weighting
    .mean()
    .reset_index(0, drop=True)
)

combined_df.at[combined_df.index[-1], 'opponentFenwick_rolling'] = None
combined_df['opponentFenwick_rolling'] = combined_df['opponentFenwick_rolling'].shift(1)

combined_df['teamFenwick_rolling_3'] = (
    combined_df.sort_values(by=['gameDate', 'gameID'])  # Ensure order
    .groupby('team')['fenwickFor']
    .ewm(span=3, adjust=False, min_periods=3)  # Exponential weighting
    .mean()
    .reset_index(0, drop=True)
)

combined_df.at[combined_df.index[-1], 'teamFenwick_rolling_3'] = None
combined_df['teamFenwick_rolling_3'] = combined_df['teamFenwick_rolling_3'].shift(1)

# Rolling averages for opponent saves (for each team)
combined_df['opponentFenwick_rolling_3'] = (
    combined_df.sort_values(by=['gameDate', 'gameID'])  # Ensure order
    .groupby('team')['fenwickAgainst']
    .ewm(span=3, adjust=False, min_periods=3)  # Exponential weighting
    .mean()
    .reset_index(0, drop=True)
)

combined_df.at[combined_df.index[-1], 'opponentFenwick_rolling_3'] = None
combined_df['opponentFenwick_rolling_3'] = combined_df['opponentFenwick_rolling_3'].shift(1)

# Rolling averages for team saves (for each team)
combined_df['teamFenwick_rolling_10'] = (
    combined_df.sort_values(by=['gameDate', 'gameID'])  # Ensure order
    .groupby('team')['fenwickFor']
    .ewm(span=10, adjust=False, min_periods=3)  # Exponential weighting
    .mean()
    .reset_index(0, drop=True)
)

combined_df.at[combined_df.index[-1], 'teamFenwick_rolling_10'] = None
combined_df['teamFenwick_rolling_10'] = combined_df['teamFenwick_rolling_10'].shift(1)

# Rolling averages for opponent saves (for each team)
combined_df['opponentFenwick_rolling_10'] = (
    combined_df.sort_values(by=['gameDate', 'gameID'])  # Ensure order
    .groupby('team')['fenwickAgainst']
    .ewm(span=10, adjust=False, min_periods=3)  # Exponential weighting
    .mean()
    .reset_index(0, drop=True)
)

combined_df.at[combined_df.index[-1], 'opponentFenwick_rolling_10'] = None
combined_df['opponentFenwick_rolling_10'] = combined_df['opponentFenwick_rolling_10'].shift(1)

# Rolling averages for team saves (for each team)
combined_df['teamFenwick_rolling_15'] = (
    combined_df.sort_values(by=['gameDate', 'gameID'])  # Ensure order
    .groupby('team')['fenwickFor']
    .ewm(span=15, adjust=False, min_periods=3)  # Exponential weighting
    .mean()
    .reset_index(0, drop=True)
)

combined_df.at[combined_df.index[-1], 'teamFenwick_rolling_15'] = None
combined_df['teamFenwick_rolling_15'] = combined_df['teamFenwick_rolling_15'].shift(1)

# Rolling averages for opponent saves (for each team)
combined_df['opponentFenwick_rolling_15'] = (
    combined_df.sort_values(by=['gameDate', 'gameID'])  # Ensure order
    .groupby('team')['fenwickAgainst']
    .ewm(span=15, adjust=False, min_periods=3)  # Exponential weighting
    .mean()
    .reset_index(0, drop=True)
)

combined_df.at[combined_df.index[-1], 'opponentFenwick_rolling_15'] = None
combined_df['opponentFenwick_rolling_15'] = combined_df['opponentFenwick_rolling_15'].shift(1)

# Remove previous opponent rolling calculations
combined_df = combined_df.sort_values(by=['gameDate', 'gameID'])

# Create a mapping of gameID to its corresponding rows
id_map = combined_df.groupby('gameID').apply(lambda x: x.index.tolist()).to_dict()

# Iterate through the dataframe and match values based on gameID
for game_id, indices in id_map.items():
    if len(indices) == 2:
        idx1, idx2 = indices
        
        # Assign opponent values from the other row with the same gameID
        combined_df.at[idx1, 'opponentTeamSaves_rolling'] = combined_df.at[idx2, 'teamSaves_rolling']
        combined_df.at[idx1, 'opponentOpponentSaves_rolling'] = combined_df.at[idx2, 'opponentSaves_rolling']
        combined_df.at[idx2, 'opponentTeamSaves_rolling'] = combined_df.at[idx1, 'teamSaves_rolling']
        combined_df.at[idx2, 'opponentOpponentSaves_rolling'] = combined_df.at[idx1, 'opponentSaves_rolling']

        combined_df.at[idx1, 'opponentTeamCorsi_rolling'] = combined_df.at[idx2, 'teamCorsi_rolling']
        combined_df.at[idx1, 'opponentOpponentCorsi_rolling'] = combined_df.at[idx2, 'opponentCorsi_rolling']
        combined_df.at[idx2, 'opponentTeamCorsi_rolling'] = combined_df.at[idx1, 'teamCorsi_rolling']
        combined_df.at[idx2, 'opponentOpponentCorsi_rolling'] = combined_df.at[idx1, 'opponentCorsi_rolling']

        combined_df.at[idx1, 'opponentTeamFenwick_rolling'] = combined_df.at[idx2, 'teamFenwick_rolling']
        combined_df.at[idx1, 'opponentOpponentFenwick_rolling'] = combined_df.at[idx2, 'opponentFenwick_rolling']
        combined_df.at[idx2, 'opponentTeamFenwick_rolling'] = combined_df.at[idx1, 'teamFenwick_rolling']
        combined_df.at[idx2, 'opponentOpponentFenwick_rolling'] = combined_df.at[idx1, 'opponentFenwick_rolling']
        
        # Loop through different spans to set rolling values dynamically
        for span in [3, 10, 15]:
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

                     'backToBack', 'teamSaves', 'opponentSaves', 'splitGame']


# Split the data into home and away teams
home_games = combined_df[combined_df['isHome'] != None][home_team_columns]

merged_df = home_games.dropna().reset_index(drop=True)

# The final dataset now contains one row per game, with both home and away team data
merged_df.to_csv('S:/Documents/GitHub/saves-model/combined_simplified.csv', index=False)