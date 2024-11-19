import csv
from nhlpy import NHLClient
from abbreviations import *
from teamFileClass import HockeyGame
from datetime import datetime, timedelta


def create_shedule_csvs():
    client = NHLClient()
    for team in nhl_team_abbreviations:
        # Create or overwrite the CSV file for the current team
        with open(f"{team}_schedule.csv", mode="w", newline="") as file:
            csv_writer = csv.writer(file)
            
            # Write the header row
            csv_writer.writerow([
                "isHome", "opponent", "shotsFor", "shotsAgainst", 
                "goalsFor", "goalsAgainst", "goalie", "goalieSaves", 
                "goalieShots", "backToBack"
            ])
            
            # Fetch the schedule and process each game
            schedule = client.schedule.get_season_schedule(team_abbr=team, season="20242025")
            prev_game = None
            
            for game in schedule['games']:
                boxscore = client.game_center.boxscore(game_id=game['id'])
                curr_game_date = datetime.strptime(boxscore['gameDate'], '%Y-%m-%d').date()
                if boxscore and (datetime.now().date() - curr_game_date).days > 0:
                    game_stats = HockeyGame()
                    game_stats.game_id = game['id']
                    home_team = boxscore['homeTeam']
                    away_team = boxscore['awayTeam']
                    
                    if home_team['abbrev'] == team:
                        stats = parse_team_stats(True, prev_game, boxscore, game_stats, home_team, away_team)
                    else:
                        stats = parse_team_stats(False, prev_game, boxscore, game_stats, home_team, away_team)
                    
                    # Write the game stats to the CSV file
                    csv_writer.writerow([
                        stats.is_home, stats.opponent, stats.shots_for, stats.shots_against,
                        stats.goals_for, stats.goals_against, stats.goalie, stats.goalie_saves,
                        stats.goalie_shots, stats.back_to_back
                    ])
                    
                    prev_game = boxscore


def parse_team_stats(is_home, prev_game, boxscore, game_stats, home_team, away_team):
    """
    Parse team stats and populate the HockeyGame object.
    """
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
    
    # Find the goalie with the most saves
    team_goalies = boxscore['playerByGameStats']['homeTeam']['goalies'] if is_home else boxscore['playerByGameStats']['awayTeam']['goalies']
    for goalie in team_goalies:
        saves = goalie['saves']
        if saves > 0 and saves > game_stats.goalie_saves:
            game_stats.goalie = goalie['playerId']
            game_stats.goalie_saves = saves
            game_stats.goalie_shots = goalie['shotsAgainst']
    
    return game_stats


if __name__ == "__main__":
    create_shedule_csvs()
