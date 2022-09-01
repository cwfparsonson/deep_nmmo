from deep_nmmo.envs.team_based_env.loops.utils import EnvLoop, get_env_stats, reset_teams

import ray
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker

from collections import defaultdict
import numpy as np


class RLlibEnvLoopCallback(DefaultCallbacks):
    '''
    NOTE: RLLib callbacks are hard-coded to store only the min/mean/max values
    for each attribute you track (i.e. if you try to track a list of metrics at each step,
    RLLib will save the min/mean/max of this list).

    Think RLLib does not store per-step/episode/epoch stats, but rather stores the rolling mean/min/max 
    per-step/episode/epoch stats. Seen somewhere online that this is done with a history window 
    of size 100 https://discuss.ray.io/t/custom-metrics-only-mean-value/636/3. Don't think there's 
    a way to change this, and also means plots may look different from what you'd expect...
    '''
    def __init__(self,
                 teams: list,
                 *args,
                 **kwargs):
        DefaultCallbacks.__init__(self, *args, **kwargs)
        self.teams = teams

    def on_episode_start(self,
                         *,
                         worker: 'RolloutWorker',
                         base_env: BaseEnv,
                         policies: dict,
                         episode: Episode,
                         **kwargs):
        self.teams = reset_teams(self.teams)

    def on_episode_step(self,
                         *,
                         worker: 'RolloutWorker',
                         base_env: BaseEnv,
                         policies: dict,
                         episode: Episode,
                         **kwargs):
        pass
        
    def on_episode_end(self,
                       *,
                       worker: 'RolloutWorker',
                       base_env: BaseEnv,
                       policies: dict,
                       episode: Episode,
                       **kwargs):
        # store data of each env in a temporary dict
        episode.user_data['env_loop_log'], episode.user_data['team_results'] = EnvLoop.init_log() # store data in temporary dict
        for env in base_env.get_sub_environments():
            metrics_by_team, stats_by_team, policy_id_by_team, n_timeout_by_team, result_by_team = get_env_stats(env=env, teams=self.teams)
            episode.user_data['env_loop_log'], episode.user_data['team_results'] = EnvLoop.update_log(
                                                                                                    loop_log=episode.user_data['env_loop_log'],
                                                                                                    team_results=episode.user_data['team_results'],
                                                                                                    metrics_by_team=metrics_by_team,
                                                                                                    stats_by_team=stats_by_team,
                                                                                                    policy_id_by_team=policy_id_by_team,
                                                                                                    result_by_team=result_by_team,
                                                                                                    teams=self.teams,
                                                                                                    log_timer=False, # let RLlib log timer stats
                                                                                                    log_tracker=False, # let RLlib log tracker stats
                                                                                                ) # store data in temporary dict
        # log data across all envs
        for key, val in episode.user_data['env_loop_log'].items():
            episode.custom_metrics[key] = val
