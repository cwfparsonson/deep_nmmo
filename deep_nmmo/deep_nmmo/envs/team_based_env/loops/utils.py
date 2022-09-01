from deep_nmmo.utils import get_class_from_path

import nmmo

from neurips2022nmmo.evaluation import analyzer
import openskill
from neurips2022nmmo.evaluation.rating import RatingSystem

import ray
import psutil
NUM_CPUS = psutil.cpu_count(logical=False)
try:
    ray.init(num_cpus=NUM_CPUS)
except RuntimeError:
    # already initialised ray in another script, no need to init again
    pass

from collections import defaultdict
import copy
import time

import numpy as np



@ray.remote
def get_team_action_asynchronous(team, team_observations):
    return get_team_action_synchronous(team, team_observations)

def get_team_action_synchronous(team, team_observations):
    return team.act(team_observations)

@ray.remote
def run_asynchronous(*args, **kwargs):
    return run_synchronous(*args, **kwargs)

def reset_env(env, 
              seed: int = None):
    if seed is not None:
        env.seed(seed)
    observations = env.reset()
    return observations

    return observations

def reset_teams(teams):
    for team in teams:
        team.reset()
        team.n_timeout = 0
    return teams

def run_synchronous(episode_idx,
                    path_to_env_config_cls,
                    env_config_kwargs,
                    path_to_env_cls,
                    teams_copies,
                    teams_config,
                    team_action_parallel: bool = False,
                    verbose: bool = False,
                    seed: int = None,
                    **kwargs):

    if verbose:
        print(f'Starting environment episode index {episode_idx}...')

    # init env params
    env_config, teams_copies, teams = init_env_params(path_to_env_config_cls=path_to_env_config_cls,
                                                      env_config_kwargs=env_config_kwargs,
                                                      teams_copies=teams_copies,
                                                      teams_config=teams_config)

    # init env
    env = get_class_from_path(path_to_env_cls)(env_config)

    # reset env and teams
    teams = reset_teams(teams)
    observations = reset_env(env, seed=seed)

    # run episode
    episode_log = defaultdict(lambda: 0)
    episode_log['episode_start_time'] = time.time()
    while observations:
        observations, rewards, dones, infos, episode_log = step_episode(env=env,
                                                                        observations=observations,
                                                                        teams=teams,
                                                                        episode_log=episode_log,
                                                                        team_action_parallel=team_action_parallel,
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

def step_episode(env,
                 observations,
                 teams,
                 episode_log,
                 team_action_parallel: bool = False,
                 verbose: bool = False):

    # get actions of each team
    get_action_start_time = time.time()
    team_to_player_to_actions = get_team_to_player_to_actions(observations, teams, team_action_parallel=team_action_parallel)
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

def get_team_to_player_to_actions(observations, teams, team_action_parallel: bool = False):
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


def init_env_params(path_to_env_config_cls,
                    env_config_kwargs,
                    teams_copies,
                    teams_config):
    # init env config
    if env_config_kwargs is None:
        env_config_kwargs = {}
    else:
        env_config_kwargs = env_config_kwargs
    env_config = get_class_from_path(path_to_env_config_cls)(**env_config_kwargs)

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

    # overwrite env players with teams
    # use team.id as agent.name, so players can be identified in replay
    for i, team in enumerate(teams):
        class Agent(nmmo.Agent):
            name = f"{team.id}_"
            policy = f"{team.id}_"
        env_config.PLAYERS[i] = Agent

    return env_config, teams_copies, teams
