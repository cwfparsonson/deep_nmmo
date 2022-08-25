from deep_nmmo.utils import get_class_from_path

import copy


class EnvLoop:
    def __init__(self,
                 path_to_env_cls: str,
                 path_to_env_config_cls: str,
                 teams_config: dict,
                 teams_copies: list = None,
                 env_config_kwargs: dict = None,
                 **kwargs):
        '''
        Args:
            teams_config (dict): Dict mapping path_to_team_cls to team_kwargs dict.
                E.g. To use two teams (one Mixture and one Combat):
                    teams_config = {
                        'neurips2022nmmo.scripted.CombatTeam': {'team_id': 'Combat'},
                        'neurips2022nmmo.scripted.MixtureTeam': {'team_id': 'Mixture'},
                    }
                N.B. The 'env_config' argument required by nmmo teams and agents will be added automatically.
            teams_copies (list): List of ints indicating how many copies to make of the teams in
                teams_config. If None, will not make any copies. teams_copies must be same length
                as teams_config. If specified, will add an ID to each team_id when instantiating.
                E.g. In above teams_config, if supply the following teams_copies:
                    teams_copies = [
                        1,
                        2 
                    ]
                    Then this will result in the following teams being instantiated:
                    teams = [
                        'neurips2022nmmo.scripted.CombatTeam': {'team_id': 'Combat-1'},
                        'neurips2022nmmo.scripted.MixtureTeam': {'team_id': 'Mixture-1'},
                        'neurips2022nmmo.scripted.MixtureTeam': {'team_id': 'Mixture-2'},
                    ]
        '''
        # init env config
        if env_config_kwargs is None:
            self.env_config_kwargs = {}
        else:
            self.env_config_kwargs = env_config_kwargs
        self.env_config = get_class_from_path(path_to_env_config_cls)(**self.env_config_kwargs)

        # init env
        self.env = get_class_from_path(path_to_env_cls)(self.env_config)

        # init number of copies to make of each team
        if teams_copies is None:
            self.teams_copies = [1 for _ in range(len(teams_config))]
        else:
            if len(teams_config) != len(teams_copies):
                raise Exception(f'Length of teams_copies ({len(teams_copies)}) must equal length of teams_config ({len(teams_config)}).')
            self.teams_copies = teams_copies

        # init teams
        self.teams_config = teams_config
        self.teams = []
        team_idx = 0
        for path_to_team_cls, team_kwargs in self.teams_config.items():
            # check team kwargs provided are valid
            if 'team_id' not in team_kwargs:
                raise Exception(f'Kwarg \'team_id\' missing from team {path_to_team_cls} team kwargs in teams_config.')

            # update team kwargs as required and initialise team(s)
            team_kwargs['env_config'] = self.env_config
            for i in range(self.teams_copies[team_idx]):
                _team_kwargs = copy.deepcopy(team_kwargs)
                _team_kwargs['team_id'] = f'{team_kwargs["team_id"]}-{i}'
                self.teams.append(get_class_from_path(path_to_team_cls)(**_team_kwargs))
            team_idx += 1

    
    def run(self,
            verbose: bool = False,
            **kwargs):
        '''Runs one episode.'''
        if verbose:
            print(f'Starting environment episode...')
        
        observations = self.env.reset()
        step_counter = 1
        while observations:
            team_to_player_to_actions = self._get_team_to_player_to_actions(observations)
            
            observations, rewards, dones, infos = self.env.step(team_to_player_to_actions)

            if verbose:
                print(f'\nStep {step_counter} | Rewards: {rewards} | Dones: {dones}')

            step_counter += 1
    
    def _get_team_to_player_to_actions(self, observations):
        team_to_player_to_actions = {}
        for team_idx, team_observations in observations.items():
            team_to_player_to_actions[team_idx] = self.teams[team_idx].act(team_observations)
        return team_to_player_to_actions

    






