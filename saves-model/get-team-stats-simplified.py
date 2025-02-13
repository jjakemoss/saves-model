import os
import csv
import logging
from nhlpy import NHLClient
from abbreviations import *
from simplifiedTeamFileClass import HockeyGameSimplified
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def create_shedule_csvs():
    # Ensure the directory for CSV files exists
    os.makedirs("S:/Documents/GitHub/saves-model/team_schedules", exist_ok=True)

    client = NHLClient()

    # Create a ThreadPoolExecutor with a number of worker threads
    with ThreadPoolExecutor(max_workers=8) as executor:
        # Submit a task for each team to be processed in parallel
        for team in nhl_team_abbreviations_2025:
            executor.submit(process_team_schedule, team, client)


def process_team_schedule(team: str, client: NHLClient):
    csv_file_path = f"S:/Documents/GitHub/saves-model/team_schedules/{team}_schedule.csv"
    existing_game_ids = set()

    # Check if the CSV file exists and load existing game IDs
    if os.path.exists(csv_file_path):
        with open(csv_file_path, mode="r", newline="") as file:
            csv_reader = csv.reader(file)
            header = next(csv_reader, None)  # Skip header if it exists
            for row in csv_reader:
                if row:  # Ensure the row is not empty
                    existing_game_ids.add(row[0])  # Assuming game_id is the first column

    # Open the file in append mode, so we don't overwrite existing data
    with open(csv_file_path, mode="a", newline="") as file:
        csv_writer = csv.writer(file)
        
        # Write the header row only if the file was just created
        if not existing_game_ids:
            csv_writer.writerow([
                "gameID", "gameDate", "isHome", "opponent", "shotsFor", "shotsAgainst", 
                "goalsFor", "goalsAgainst", "teamSaves", "opponentSaves", 
                "backToBack",  "splitGame"
            ])
        
        logging.info(f"Processing schedule for team: {team}")
        try:
            # Fetch the schedule and process each game
            schedule = client.schedule.get_season_schedule(team_abbr=team, season="20242025")
            prev_game = None
            
            for game in schedule['games']:
                if game['gameType'] != 2:
                    logging.info(f"Skipping non-regular-season game")
                    continue
                
                curr_game_date = datetime.strptime(game['gameDate'], '%Y-%m-%d').date()
                if (datetime.now().date() - curr_game_date).days <= 0:
                    logging.info(f"Skipping future game {game['id']}.")
                    continue
                
                game_id = game['id']
                if str(game_id) in existing_game_ids:
                    logging.info(f"Skipping already processed game {game_id} for team {team}.")
                    continue
                
                try:
                    boxscore = client.game_center.boxscore(game_id=game_id)
                    if not boxscore or 'homeTeam' not in boxscore or 'awayTeam' not in boxscore:
                        logging.warning(f"Skipping game {game_id} due to incomplete data.")
                        continue

                    game_stats = HockeyGameSimplified()
                    game_stats.game_id = game_id
                    home_team = boxscore['homeTeam']
                    away_team = boxscore['awayTeam']
                    
                    if home_team['abbrev'] == team:
                        stats = parse_team_stats(True, prev_game, boxscore, game_stats, home_team, away_team)
                    else:
                        stats = parse_team_stats(False, prev_game, boxscore, game_stats, home_team, away_team)
                    
                    # Write the game stats to the CSV file
                    csv_writer.writerow([
                        stats.game_id, stats.game_date, stats.is_home, stats.opponent, stats.shots_for, stats.shots_against,
                        stats.goals_for, stats.goals_against, stats.team_saves, stats.opponent_saves,
                        stats.back_to_back, stats.split_game
                    ])
                    
                    logging.info(f"Added game {game_id} for team {team}.")
                    prev_game = boxscore

                except Exception as game_error:
                    logging.error(f"Error processing game {game_id}: {game_error}")

        except Exception as team_error:
            logging.error(f"Error processing team {team}: {team_error}")


def parse_team_stats(
    is_home: bool, 
    prev_game: dict, 
    boxscore: dict, 
    game_stats: HockeyGameSimplified, 
    home_team: dict, 
    away_team: dict
) -> HockeyGameSimplified:
    """
    Parse team stats and populate the HockeyGame object.
    """
    try:
        # Determine if this game is a back-to-back game
        if prev_game:
            prev_game_date = datetime.strptime(prev_game['gameDate'], '%Y-%m-%d').date()
            curr_game_date = datetime.strptime(boxscore['gameDate'], '%Y-%m-%d').date()
            if (curr_game_date - prev_game_date).days <= 1:
                game_stats.back_to_back = True

        # Populate stats for the current game
        game_stats.is_home = is_home
        game_stats.opponent = away_team['abbrev'] if is_home else home_team['abbrev']
        game_stats.shots_for = home_team['sog'] if is_home else away_team['sog']
        game_stats.shots_against = away_team['sog'] if is_home else home_team['sog']
        game_stats.goals_for = home_team['score'] if is_home else away_team['score']
        game_stats.goals_against = away_team['score'] if is_home else home_team['score']

        game_stats.game_date = datetime.strptime(boxscore['gameDate'], '%Y-%m-%d').date()

        # Find the goalie with the most saves
        team_goalies = boxscore['playerByGameStats']['homeTeam']['goalies'] if is_home else boxscore['playerByGameStats']['awayTeam']['goalies']
        for goalie in team_goalies:
            if game_stats.team_saves >= 3 and goalie['saves'] >= 3:
                game_stats.split_game = True
            game_stats.team_saves += goalie['saves']

        opponent_goalies = boxscore['playerByGameStats']['homeTeam']['goalies'] if not is_home else boxscore['playerByGameStats']['awayTeam']['goalies']
        for goalie in opponent_goalies:
            game_stats.opponent_saves += goalie['saves']

    except Exception as parse_error:
        logging.error(f"Error parsing stats: {parse_error}")
    
    return game_stats


if __name__ == "__main__":
    create_shedule_csvs()
