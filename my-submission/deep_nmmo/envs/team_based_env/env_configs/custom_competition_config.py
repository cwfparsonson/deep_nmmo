import nmmo

import neurips2022nmmo

class CustomCompetitionConfig(neurips2022nmmo.CompetitionConfig):
    NUM_TEAMS = 2
    PLAYER_TEAM_SIZE = 8 # number of players per team

    PLAYERS = [nmmo.Agent] * NUM_TEAMS # A.K.A. in TeamBasedEnv, PLAYERS == TEAMS, will overwrite with players with instantiated teams
    PLAYER_N = int(len(PLAYERS) * PLAYER_TEAM_SIZE) # total number of players in the environment

    @property
    def PLAYER_SPAWN_FUNCTION(self):
        return nmmo.spawn.spawn_concurrent

