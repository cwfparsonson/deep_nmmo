from deep_nmmo.utils import get_class_from_path

from neurips2022nmmo.evaluation import analyzer
import openskill
from neurips2022nmmo.evaluation.rating import RatingSystem

import nmmo

import copy

from collections import defaultdict
import pandas as pd
import numpy as np

import time

import ray
import psutil
NUM_CPUS = psutil.cpu_count(logical=False)
try:
    ray.init(num_cpus=NUM_CPUS)
except RuntimeError:
    # already initialised ray in script calling dcn sim, no need to init again
    pass



@ray.remote
def get_team_action_asynchronous(team, team_observations):
    # return team.act(team_observations)
    return get_team_action_synchronous(team, team_observations)

def get_team_action_synchronous(team, team_observations):
    return team.act(team_observations)

@ray.remote
def run_asynchronous(*args, **kwargs):
    return run_synchronous(*args, **kwargs)

def run_synchronous(episode_idx,
                    path_to_env_config_cls,
                    env_config_kwargs,
                    path_to_env_cls,
                    teams_copies,
                    teams_config,
                    verbose: bool = False,
                    seed: int = None,
                    **kwargs):

    if verbose:
        print(f'Starting environment episode index {episode_idx}...')

    # init env params
    env_config, env, teams_copies, teams = init_env(path_to_env_config_cls=path_to_env_config_cls,
                                                    env_config_kwargs=env_config_kwargs,
                                                    path_to_env_cls=path_to_env_cls,
                                                    teams_copies=teams_copies,
                                                    teams_config=teams_config)

    # reset env
    if seed is not None:
        env.seed(seed)
    observations = env.reset()

    # run episode
    episode_log = defaultdict(lambda: 0)
    episode_log['episode_start_time'] = time.time()
    while observations:
        observations, rewards, dones, infos, episode_log = _step_episode(env=env,
                                                                         observations=observations,
                                                                         teams=teams,
                                                                         episode_log=episode_log,
                                                                         verbose=verbose)

    if verbose:
        print(f'Completed episode in {episode_log["episode_run_time"]:.3f} s')

    # extract env stats (since cannot pickle environment when running in parallel, need to extract stats before returning)
    metrics_by_team, stats_by_team, policy_id_by_team, n_timeout_by_team, result_by_team = get_env_stats(env, teams)

    return {'episode_idx': episode_idx, 
            'episode_log': episode_log, 
            'metrics_by_team': metrics_by_team,
            'stats_by_team': stats_by_team,
            'policy_id_by_team': policy_id_by_team,
            'n_timeout_by_team': n_timeout_by_team,
            'result_by_team': result_by_team,
            'teams': teams,
            }

def get_env_stats(env, teams):
    metrics_by_team = env.metrices_by_team()
    stats_by_team = env.stat_by_team() # maps team_id -> min/max/sum/avg player_metric 
    policy_id_by_team = {
        i: teams[i].policy_id
        for i in metrics_by_team.keys()
    }
    n_timeout_by_team = {
            i: teams[i].n_timeout
        for i in metrics_by_team.keys()
    }
    result_by_team = analyzer.gen_result(
                                            policy_id_by_team, 
                                            metrics_by_team,
                                            n_timeout_by_team
                                          )

    return metrics_by_team, stats_by_team, policy_id_by_team, n_timeout_by_team, result_by_team

def _step_episode(env,
                  observations,
                  teams,
                  episode_log,
                  verbose: bool = False):

    # get actions of each team
    get_action_start_time = time.time()
    team_to_player_to_actions = _get_team_to_player_to_actions(observations, teams)
    get_action_time = time.time() - get_action_start_time 
    
    # step env
    step_env_start_time = time.time()
    observations, rewards, dones, infos = env.step(team_to_player_to_actions)
    step_env_time = time.time() - step_env_start_time

    # record timing stats
    episode_log['episode_run_time'] = time.time() - episode_log['episode_start_time']
    episode_log['timer/episode_total_run_time'] = episode_log['episode_run_time']
    episode_log['timer/get_action_total_time'] += get_action_time
    episode_log['timer/step_env_total_time'] += step_env_time

    if verbose:
        print(f'Step: {episode_log["env_step_counter"]} | Get action time: {get_action_time:.3f} s | Step env time: {step_env_time:.3f} s | Episode run time: {episode_log["episode_run_time"]:.3f} s')

    episode_log['env_step_counter'] += 1

    return observations, rewards, dones, infos, episode_log

def _get_team_to_player_to_actions(observations, teams, team_action_parallel: bool = False):
    if team_action_parallel:
        # run team action selecions in team_action_parallel
        max_num_processes = min(len(list(observations.keys())), NUM_CPUS)
        i, team_idxs, result_ids = 0, list(observations.keys()), []
        while i < len(team_idxs):
            num_processes = min(len(team_idxs) - i, max_num_processes)
            for _ in range(num_processes):
                team_idx = team_idxs[i]
                result_ids.append(
                            get_team_action_asynchronous.remote(
                                    team=teams[team_idx],
                                    team_observations=observations[team_idx]
                                )
                        )
                i += 1
        # collect results
        team_to_player_to_actions = {team_idxs[i]: player_to_actions for i, player_to_actions in enumerate(ray.get(result_ids))}

    else:
        # run team action selections sequentially
        team_to_player_to_actions = {}
        for team_idx, team_observations in observations.items():
            team_to_player_to_actions[team_idx] = get_team_action_synchronous(
                        team=teams[team_idx],
                        team_observations=team_observations
                    )

    return team_to_player_to_actions


def init_env(path_to_env_config_cls,
             env_config_kwargs,
             path_to_env_cls,
             teams_copies,
             teams_config):
    # init env config
    if env_config_kwargs is None:
        env_config_kwargs = {}
    else:
        env_config_kwargs = env_config_kwargs
    env_config = get_class_from_path(path_to_env_config_cls)(**env_config_kwargs)

    # init env
    env = get_class_from_path(path_to_env_cls)(env_config)

    # init number of copies to make of each team
    teams_config = teams_config
    if teams_copies is None:
        teams_copies = [1 for _ in range(len(teams_config))]
    else:
        if len(teams_config) != len(teams_copies):
            raise Exception(f'Length of teams_copies ({len(teams_copies)}) must equal length of teams_config ({len(teams_config)}).')
        teams_copies = teams_copies

    # init teams
    teams = []
    team_idx = 0
    for team_id, team_kwargs in teams_config.items():
        path_to_team_cls = team_kwargs['path_to_team_cls']

        # update team kwargs as required and initialise team(s)
        team_kwargs['env_config'] = env_config
        team_kwargs['team_id'] = team_id
        for i in range(teams_copies[team_idx]):
            _team_kwargs = copy.deepcopy(team_kwargs)
            _team_kwargs['team_id'] = f'{team_kwargs["team_id"]}-{i}'
            teams.append(get_class_from_path(path_to_team_cls)(**_team_kwargs))
        team_idx += 1

    # check for conflicts with env config
    assert len(teams) == len(
        env_config.PLAYERS
    ), f'Number of teams ({len(teams)}) is different with config ({len(env_config.PLAYERS)}). Change teams_config ({teams_config}) or teams_config ({teams_copies}) or change env config.'
    # for team in teams:
        # assert len(team) == len(
            # env_config.PLAYER_TEAM_SIZE
        # ), f'Player team size ({len(team)}) of team {team} is different with config ({len(env_config.PLAYER_TEAM_SIZE)}). Change teams_config ({teams_config}) or change env config.'


    # overwrite env players with teams
    # use team.id as agent.name, so players can be identified in replay
    for i, team in enumerate(teams):
        class Agent(nmmo.Agent):
            name = f"{team.id}_"
            policy = f"{team.id}_"
        env_config.PLAYERS[i] = Agent

    return env_config, env, teams_copies, teams


class EnvLoop:
    def __init__(self,
                 path_to_env_cls: str,
                 path_to_env_config_cls: str,
                 teams_config: dict,
                 teams_copies: list = None,
                 env_config_kwargs: dict = None,
                 team_action_parallel: bool = False,
                 run_parallel: bool = False,
                 wandb=None,
                 **kwargs):
        '''
        Args:
            # TODO: Update below teams_config doctstring since is wrong; is now teams_config[team_id] = {path_to_team_cls} etc

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
        self.path_to_env_config_cls = path_to_env_config_cls
        self.env_config_kwargs = env_config_kwargs
        self.path_to_env_cls = path_to_env_cls
        self.teams_copies = teams_copies
        self.teams_config = teams_config

        self.team_action_parallel = team_action_parallel
        self.run_parallel = run_parallel
        self.wandb = wandb

        self.reset()

    def reset(self):
        if self.wandb is not None:
            self._init_log()

    def _init_log(self):
        self.log = defaultdict(lambda: 0)

        team_result_columns = ['team_id', 'episode']
        team_result_columns.extend(analyzer.TeamResult.names())
        self.log['analyzer/result_by_team_table'] = self.wandb.Table(columns=team_result_columns)

        self.team_results = [] # record analyzer.TeamResult objects at end of each episode
    
    def run(self,
            num_episodes: int = 1,
            verbose: bool = False,
            seed: int = None,
            **kwargs):
        '''Runs one episode.'''
        
        if not self.run_parallel:
            run_func = run_synchronous
        else:
            run_func = run_asynchronous.remote

        # run experiments
        # for episode_idx in range(num_episodes):
        i, episode_idxs, result_ids = 0, [idx for idx in range(num_episodes)], []
        while i < len(episode_idxs):
            num_processes = min(len(episode_idxs) - i, NUM_CPUS)
            for _ in range(num_processes):
                episode_idx = episode_idxs[i]
                result_ids.append(run_func(episode_idx=episode_idx,
                                           path_to_env_config_cls=self.path_to_env_config_cls,
                                           env_config_kwargs=self.env_config_kwargs,
                                           path_to_env_cls=self.path_to_env_cls,
                                           teams_copies=self.teams_copies,
                                           teams_config=self.teams_config,
                                           verbose=verbose,
                                           seed=seed,
                                         )
                                        )
                i += 1

        # collect results
        if self.run_parallel:
            results = ray.get(result_ids)
        else:
            results = result_ids

        return results


    def update_log(self, 
                   metrics_by_team,
                   stats_by_team,
                   policy_id_by_team,
                   result_by_team,
                   teams,
                   episode_log: dict,
                   external_log: dict = None,
                   **kwargs):
        '''Call this after each call to EnvLoop.run() to update the W&B log after each episode.

        If have any custom metrics to log which are external to EnvLoop (e.g. RLlib training stats),
        can call EnvLoop.update_log(external_log) to log any metrics in addition to those
        logged by default by EnvLoop. Note that the log should be compatible with
        wandb.log(external_log), and that the external_log stats will be logged at the
        end of each episode (i.e. update_log should only be called after each episode).
        '''
        if external_log is not None:
            # update EnvLoop log with externally-provided additional log data
            self.log.update(external_log)

        # update tracker stats
        self.log['tracker/num_env_episodes'] += 1
        self.log['tracker/num_env_steps'] += episode_log['env_step_counter']

        # update timer stats
        for metric in ['timer/get_action_total_time', 'timer/step_env_total_time', 'timer/episode_total_run_time']:
            self.log[metric] = episode_log[metric]
        self.log['timer/get_action_mean_time'] = self.log['timer/get_action_total_time'] / episode_log['env_step_counter']
        self.log['timer/step_env_mean_time'] = self.log['timer/step_env_total_time'] / episode_log['env_step_counter']

        # update env stats
        # metrics_by_team = env.metrices_by_team()
        # stats_by_team = env.stat_by_team() # maps team_id -> min/max/sum/avg player_metric 
        for team_id in metrics_by_team.keys():
            self.log[f'env/player_stats_by_team_{team_id}/'] = metrics_by_team[team_id]
            self.log[f'env/team_stats_by_team_{team_id}/'] = stats_by_team[team_id]

        # update analyzer per-team per-attr TeamResult
        # policy_id_by_team = {
            # i: teams[i].policy_id
            # for i in metrics_by_team.keys()
        # }
        # n_timeout_by_team = {
                # i: teams[i].n_timeout
            # for i in metrics_by_team.keys()
        # }
        # result_by_team = analyzer.gen_result(
                                                # policy_id_by_team, 
                                                # metrics_by_team,
                                                # n_timeout_by_team
                                              # )
        # record TeamResult object for this episode
        self.team_results.append(result_by_team)
        # process TeamResult objects into JSON-serialisable format for wandb
        for team_id, team_result in result_by_team.items():
            self.log[f'analyzer/result_by_team_{team_id}/'] = {attr: getattr(team_result, attr) for attr in analyzer.TeamResult.names()}

        # update analyzer TeamResult stats table
        teams_rows = []
        for team_id, team_result in result_by_team.items():
            team_row = [team_id, self.log['tracker/num_env_episodes']]
            # process TeamResult objects into JSON-serialisable format for wandb
            for attr in analyzer.TeamResult.names():
                team_row.append(getattr(team_result, attr))
            self.log['analyzer/result_by_team_table'].add_data(*team_row)

        # update aggregated teams per-attr analyzer plots with TeamResult
        table_df = {}
        for col in self.log['analyzer/result_by_team_table'].columns:
            table_df[col] = self.log['analyzer/result_by_team_table'].get_column(col)
        table_df = pd.DataFrame(table_df)
        xs = list(table_df.groupby("team_id")['episode'].apply(list).to_dict().values())
        for attr in analyzer.TeamResult.names():
            team_id_to_attrs = table_df.groupby("team_id")[attr].apply(list).to_dict()
            self.log[f'analyzer/result_by_attr_{attr}'] = self.wandb.plot.line_series(
                                                                                xs=xs,
                                                                                ys=list(team_id_to_attrs.values()),
                                                                                keys=list(team_id_to_attrs.keys()),
                                                                                title=attr,
                                                                                xname='Episode',
                                                                            )

        # update mean team result summary table
        mean_result_by_team = analyzer.avg_results(self.team_results)
        # for n in range(1, self.env_config.NUM_TEAMS):
            # self.log[f'analyzer/top_{n}_ratio_by_team'] = analyzer.topn_probs([result_by_team], n=n)
        topn_probs = analyzer.topn_probs(self.team_results, n=1)
        team_summary_columns = ['team_id', 'top_1_ratio']
        team_summary_columns.extend([f'mean_{attr}' for attr in analyzer.TeamResult.names()])
        self.log['analyzer/mean_result_by_team_table/'] = self.wandb.Table(columns=team_summary_columns)
        for team_id, topn_prob in topn_probs.items():
            row, attr_to_val = [team_id, topn_prob], {}
            team_result = mean_result_by_team[team_id]
            for attr in analyzer.TeamResult.names():
                row.append(getattr(team_result, attr))
                attr_to_val[attr] = getattr(team_result, attr)
            self.log[f'analyzer/mean_result_by_team_table/'].add_data(*row)
            self.log[f'analyzer/mean_result_by_team_{team_id}/'] = attr_to_val

        # update wandb log
        self.wandb.log(self.log)
