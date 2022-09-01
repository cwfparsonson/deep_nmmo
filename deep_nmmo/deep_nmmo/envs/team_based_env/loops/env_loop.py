from deep_nmmo.envs.team_based_env.loops.utils import get_team_action_asynchronous, get_team_action_synchronous, run_asynchronous, run_synchronous, get_env_stats, step_episode, get_team_to_player_to_actions, init_env_params

from neurips2022nmmo.evaluation import analyzer

import copy

from collections import defaultdict
import pandas as pd
import numpy as np

import time

import wandb

import ray
import psutil
NUM_CPUS = psutil.cpu_count(logical=False)
try:
    ray.init(num_cpus=NUM_CPUS)
except RuntimeError:
    # already initialised ray in another script, no need to init again
    pass





class EnvLoop:
    def __init__(self,
                 path_to_env_cls: str,
                 path_to_env_config_cls: str,
                 teams_config: dict,
                 teams_copies: list = None,
                 env_config_kwargs: dict = None,
                 team_action_parallel: bool = False,
                 run_parallel: bool = False,
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

        self.reset()

    def reset(self):
        return EnvLoop.init_log()

    @staticmethod
    def init_log(*args, **kwargs):
        loop_log = defaultdict(lambda: 0)

        team_result_columns = ['team_id', 'episode']
        team_result_columns.extend(analyzer.TeamResult.names())
        loop_log['analyzer/result_by_team_table'] = wandb.Table(columns=team_result_columns)

        team_results = [] # record analyzer.TeamResult objects at end of each episode

        return loop_log, team_results
    
    def run(self,
            num_episodes: int = 1,
            verbose: bool = False,
            seed: int = None,
            **kwargs):
        '''Runs episode(s).'''
        
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
                                           team_action_parallel=self.team_action_parallel,
                                           verbose=verbose,
                                           seed=seed,
                                         )
                                        )
                i += 1

        # collect results
        if self.run_parallel:
            self.results = ray.get(result_ids)
        else:
            self.results = result_ids

        return self.results


    @staticmethod
    def update_log(loop_log, 
                   team_results,
                   metrics_by_team,
                   stats_by_team,
                   policy_id_by_team,
                   result_by_team,
                   teams,
                   episode_log: dict,
                   external_log: dict = None,
                   log_tracker: bool = True,
                   log_timer: bool = True,
                   log_env: bool = True,
                   log_analyzer: bool = True,
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
            loop_log.update(external_log)

        if log_tracker:
            # update tracker stats
            loop_log['tracker/num_env_episodes'] += 1
            loop_log['tracker/num_env_steps'] += episode_log['env_step_counter']

        if log_timer:
            # update timer stats
            for metric in ['timer/get_action_total_time', 'timer/step_env_total_time', 'timer/episode_total_run_time']:
                loop_log[metric] = episode_log[metric]
            loop_log['timer/get_action_mean_time'] = loop_log['timer/get_action_total_time'] / episode_log['env_step_counter']
            loop_log['timer/step_env_mean_time'] = loop_log['timer/step_env_total_time'] / episode_log['env_step_counter']

        if log_env:
            # update env stats
            for team_id in metrics_by_team.keys():
                loop_log[f'env/player_stats_by_team_{team_id}/'] = metrics_by_team[team_id]
                loop_log[f'env/team_stats_by_team_{team_id}/'] = stats_by_team[team_id]

        if log_analyzer:
            # record TeamResult object for this episode
            team_results.append(result_by_team)

            # process TeamResult objects into JSON-serialisable format for wandb
            for team_id, team_result in result_by_team.items():
                loop_log[f'analyzer/result_by_team_{team_id}/'] = {attr: getattr(team_result, attr) for attr in analyzer.TeamResult.names()}

            # update analyzer TeamResult stats table
            teams_rows = []
            for team_id, team_result in result_by_team.items():
                team_row = [team_id, loop_log['tracker/num_env_episodes']]
                # process TeamResult objects into JSON-serialisable format for wandb
                for attr in analyzer.TeamResult.names():
                    team_row.append(getattr(team_result, attr))
                loop_log['analyzer/result_by_team_table'].add_data(*team_row)

            # update aggregated teams per-attr analyzer plots with TeamResult
            table_df = {}
            for col in loop_log['analyzer/result_by_team_table'].columns:
                table_df[col] = loop_log['analyzer/result_by_team_table'].get_column(col)
            table_df = pd.DataFrame(table_df)
            xs = list(table_df.groupby("team_id")['episode'].apply(list).to_dict().values())
            for attr in analyzer.TeamResult.names():
                team_id_to_attrs = table_df.groupby("team_id")[attr].apply(list).to_dict()
                loop_log[f'analyzer/result_by_attr_{attr}'] = wandb.plot.line_series(
                                                                                xs=xs,
                                                                                ys=list(team_id_to_attrs.values()),
                                                                                keys=list(team_id_to_attrs.keys()),
                                                                                title=attr,
                                                                                xname='Episode',
                                                                                )

            # update mean team result summary table
            mean_result_by_team = analyzer.avg_results(team_results)
            # for n in range(1, self.env_config.NUM_TEAMS):
                # self.log[f'analyzer/top_{n}_ratio_by_team'] = analyzer.topn_probs([result_by_team], n=n)
            topn_probs = analyzer.topn_probs(team_results, n=1)
            team_summary_columns = ['team_id', 'top_1_ratio']
            team_summary_columns.extend([f'mean_{attr}' for attr in analyzer.TeamResult.names()])
            loop_log['analyzer/mean_result_by_team_table/'] = wandb.Table(columns=team_summary_columns)
            for team_id, topn_prob in topn_probs.items():
                row, attr_to_val = [team_id, topn_prob], {}
                team_result = mean_result_by_team[team_id]
                for attr in analyzer.TeamResult.names():
                    row.append(getattr(team_result, attr))
                    attr_to_val[attr] = getattr(team_result, attr)
                loop_log[f'analyzer/mean_result_by_team_table/'].add_data(*row)
                loop_log[f'analyzer/mean_result_by_team_{team_id}/'] = attr_to_val

            return loop_log, team_results

