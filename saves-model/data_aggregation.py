import os
import pandas as pd
import numpy as np
from abbreviations import *

def compute_weighted_moving_average(series, weights=None):
    if weights is None:
        weights = np.arange(1, len(series) + 1)
    weights = weights / weights.sum()  # Normalize weights

    # Compute the convolution
    result = np.convolve(series, weights[::-1], mode="valid")

    # Pad the result to match the original series length
    padding = [np.nan] * (len(series) - len(result))
    return pd.Series(padding + list(result), index=series.index)


def process_team_schedule(team_file, team_id):
    """
    Process an individual team's schedule file to compute features.
    """
    df = pd.read_csv(team_file)

    # Compute weighted moving averages
    df["shots_against_wma"] = compute_weighted_moving_average(df["shotsAgainst"], weights=np.arange(1, 6))

    # Compute home/away splits
    df["shots_against_home"] = df.loc[df["isHome"] == 1, "shotsAgainst"].expanding().mean()
    df["shots_against_away"] = df.loc[df["isHome"] == 0, "shotsAgainst"].expanding().mean()

    # Compute back-to-back splits
    df["shots_against_b2b"] = df.loc[df["backToBack"] == 1, "shotsAgainst"].expanding().mean()
    df["shots_against_non_b2b"] = df.loc[df["backToBack"] == 0, "shotsAgainst"].expanding().mean()

    # Forward-fill missing values
    df.fillna(method="ffill", inplace=True)

    # Add team identifier
    df["team_id"] = team_id

    return df

def build_dataset(team_files, nhl_team_abbreviations):
    """
    Build the complete dataset by combining all team schedules.
    """
    dataframes = []

    for team_id, team_file in zip(nhl_team_abbreviations, team_files):
        print(f"Processing {team_file} for team {team_id}")
        team_df = process_team_schedule(team_file, team_id)
        dataframes.append(team_df)

    # Combine all team dataframes
    full_df = pd.concat(dataframes, ignore_index=True)

    # Add opponent features
    full_df = add_opponent_features(full_df)

    return full_df

def add_opponent_features(df):
    """
    Add opponent-related features to the dataset.
    """
    opponent_stats = (
        df.groupby("team_id")[["shotsFor"]]
        .expanding()
        .mean()
        .reset_index(level=0)
        .rename(columns={"shotsFor": "opponent_shots_for_wma"})
    )
    df = df.merge(opponent_stats, left_on="opponent", right_on="team_id", suffixes=("", "_opponent"))
    return df

# Main script
if __name__ == "__main__":
    # List your team schedule files
    team_files = [f"team_schedules/{team}_schedule.csv" for team in nhl_team_abbreviations_2024]

    # Build the dataset
    full_dataset = build_dataset(team_files, nhl_team_abbreviations_2024)

    # Save the dataset to a CSV file
    full_dataset.to_csv("goalie_saves_dataset.csv", index=False)
    print("Dataset saved as 'goalie_saves_dataset.csv'")
