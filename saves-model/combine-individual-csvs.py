import pandas as pd
import os

# Folder containing all team CSVs
csv_folder = "team_schedules"  # Update with your actual path

# List all CSV files
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

# Sort by gameID in ascending order
combined_df = combined_df.sort_values(by="gameID").reset_index(drop=True)

# Save to a new CSV file
combined_df.to_csv("NHL_combined_data_with_team.csv", index=False)

print("Combined dataset with team column created successfully!")
