from datetime import datetime


class HockeyGameSimplified:
    def __init__(
        self,
        game_id: str = "",
        game_date: datetime = None,
        is_home: bool = False, 
        opponent: str = "", 
        shots_for: int = 0, 
        shots_against: int = 0,
        goals_for: int = 0,
        goals_against: int = 0,
        team_saves: int = 0,
        opponent_saves: int = 0,
        back_to_back: bool = False,
        split_game: bool = False,
        corsi_for: int = 0,
        corsi_against: int = 0,
        fenwick_for: int = 0,
        fenwick_against: int = 0
    ):
        """
        Initialize a HockeyGame instance.

        :param is_home: (bool) Whether the team is playing at home
        :param opponent: (str) Abbreviation of the opponent team
        :param shots_for: (int) Number of shots taken by the team
        :param shots_against: (int) Number of shots taken by the opponent
        :param goalie: (str) Name of the goalie
        :param goalie_saves: (int) Number of saves made by the goalie
        :param back_to_back: (bool) Whether the team is playing on back-to-back days
        """
        self.game_id: str = game_id
        self.game_date = game_date
        self.is_home: bool = is_home
        self.opponent: str = opponent
        self.shots_for: int = shots_for
        self.shots_against: int = shots_against
        self.goals_for: int = goals_for
        self.goals_against: int = goals_against
        self.team_saves: int = team_saves
        self.opponent_saves: int = opponent_saves
        self.back_to_back: bool = back_to_back
        self.split_game: bool = split_game
        self.corsi_for: int = corsi_for
        self.corsi_against: int = corsi_against
        self.fenwick_for: int = fenwick_for
        self.fenwick_against: int = fenwick_against

    def __str__(self) -> str:
        """
        String representation of the HockeyGame instance.
        """
        return (f"HockeyGame(home={self.is_home}, opponent='{self.opponent}', shots_for={self.shots_for}, "
                f"shots_against={self.shots_against}, team_saves='{self.team_saves}', opponent_saves={self.opponent_saves}, "
                f"back_to_back={self.back_to_back})")