from nhlpy import NHLClient

client = NHLClient()
schedule = client.schedule.get_season_schedule(team_abbr="BUF", season="20242025")
boxscore = client.game_center.boxscore(game_id="2024010001")
var = "kdjsalf"