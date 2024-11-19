class HockeyGame:
    def __init__(
        self, 
        is_home: bool = False, 
        opponent: str = "", 
        shots_for: int = 0, 
        shots_against: int = 0,
        goals_for: int = 0,
        goals_against: int = 0, 
        goalie: int = 0, 
        goalie_saves: int = 0,
        goalie_shots: int = 0,
        back_to_back: bool = False
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
        self.is_home: bool = is_home
        self.opponent: str = opponent
        self.shots_for: int = shots_for
        self.shots_against: int = shots_against
        self.goals_for: int = goals_for
        self.goals_against: int = goals_against
        self.goalie: int = goalie
        self.goalie_saves: int = goalie_saves
        self.goalie_shots: int = goalie_shots
        self.back_to_back: bool = back_to_back

    def __str__(self) -> str:
        """
        String representation of the HockeyGame instance.
        """
        return (f"HockeyGame(home={self.is_home}, opponent='{self.opponent}', shots_for={self.shots_for}, "
                f"shots_against={self.shots_against}, goalie='{self.goalie}', goalie_saves={self.goalie_saves}, "
                f"back_to_back={self.back_to_back})")