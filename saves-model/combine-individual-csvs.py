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

# Save the updated dataframe to a new CSV file
combined_df.to_csv("combined_with_rolling_averages.csv", index=False)

print("Updated dataset with rolling averages has been saved to 'combined_with_rolling_averages.csv'")
