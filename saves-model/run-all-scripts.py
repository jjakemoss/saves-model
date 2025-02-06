import subprocess
import get_team_stats_simplified  # Assuming this is the module where create_schedule_csvs() is defined

def main():
    # Step 1: Generate individual team CSVs
    print("Generating individual team CSVs...")
    get_team_stats_simplified.create_schedule_csvs()

    # Step 2: Combine individual CSVs into a single dataset
    print("Combining individual CSVs into 'combined_simplified.csv'...")
    subprocess.run(["python", "combine-individual-csvs.py"], check=True)

    # Step 3: Predict saves for a user-input game
    home_team = input("Enter home team abbreviation (e.g., ANA): ").strip().upper()
    away_team = input("Enter away team abbreviation (e.g., BOS): ").strip().upper()

    print(f"Predicting saves for {home_team} vs {away_team}...")
    subprocess.run(["python", "train-and-test.py", home_team, away_team], check=True)

if __name__ == "__main__":
    main()
