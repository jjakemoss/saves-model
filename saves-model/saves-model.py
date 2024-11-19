from nhlpy import NHLClient
from abbreviations import *
from teamFileClass import HockeyGame
from datetime import datetime, timedelta

def create_shedule_csvs():
    client = NHLClient()
    for team in nhl_team_abbreviations:
        schedule = client.schedule.get_season_schedule(team_abbr=team, season="20242025")
        prev_game = None
        for game in schedule['games']:
            boxscore = client.game_center.boxscore(game_id=game['id'])
            if boxscore and boxscore['gameState'] == 'FINAL':
                gameStats = HockeyGame()
                homeTeam = boxscore['homeTeam']
                awayTeam = boxscore['awayTeam']
                if (homeTeam['abbrev'] == team):
                    stats = parse_team_stats(True, prev_game, boxscore, gameStats, homeTeam, awayTeam)
                    # Write stats to CSV file
                else:
                    stats = parse_team_stats(False, prev_game, boxscore, gameStats, homeTeam, awayTeam)
                    # Write stats to CSV file
                prev_game = boxscore
                            
            var = "Client"
        var = "Client"

def parse_team_stats(isHome, prev_game, boxscore, gameStats, homeTeam, awayTeam):
    if (prev_game):
        prev_game_date = datetime.strptime(prev_game['gameDate'], '%Y-%m-%d').date()
        curr_game_date = datetime.strptime(boxscore['gameDate'], '%Y-%m-%d').date()
        if (abs(curr_game_date - prev_game_date) <= timedelta(days=1, hours=12)):
            gameStats.back_to_back = True
    gameStats.is_home = isHome
    gameStats.shots_for = homeTeam['sog'] if isHome else awayTeam['sog']
    gameStats.shots_against = awayTeam['sog'] if isHome else homeTeam['sog']
    gameStats.goals_for = homeTeam['score'] if isHome else awayTeam['score']
    gameStats.goals_against = awayTeam['score'] if isHome else homeTeam['score']
    homeTeamGoalies = boxscore['playerByGameStats']['homeTeam']['goalies'] if isHome else boxscore['playerByGameStats']['awayTeam']['goalies']
    for goalie in homeTeamGoalies:
        saves = goalie['saves']
        if saves > 0 and saves > gameStats.goalie_saves:
            gameStats.goalie = goalie['playerId']
            gameStats.goalie_saves = saves
            gameStats.goalie_shots = goalie['shotsAgainst']
    
    return gameStats


if __name__=="__main__":
    create_shedule_csvs()