from deep_nmmo.utils import get_module_from_path, get_class_from_path
from deep_nmmo.envs.team_based_env.loops.utils import get_team_action_asynchronous, get_team_action_synchronous, run_asynchronous, run_synchronous, get_env_stats, step_episode, get_team_to_player_to_actions, init_env_params

import ray
ray.shutdown()
ray.init()
from ray.rllib.agents import ppo
from ray.tune.registry import register_env

from ray.rllib.models import ModelCatalog

import gym

from collections import defaultdict
from omegaconf import OmegaConf
import time
import hydra

import pickle
import gzip

import threading

import numpy as np


class RLlibEpochLoop:
    def __init__(self,
                 path_to_env_cls: str,
                 path_to_env_config_cls: str,
                 teams_config: dict,
                 path_to_rllib_trainer_cls: str,
                 rllib_config: dict,
                 path_to_model_cls: str = None,
                 teams_copies: list = None,
                 env_config_kwargs: dict = None,
                 team_action_parallel: bool = False):
        rllib_config = OmegaConf.to_container(rllib_config, resolve=False)

        if path_to_model_cls is not None:
            # register model with rllib
            ModelCatalog.register_custom_model(rllib_config['model']['custom_model'], get_class_from_path(path_to_model_cls))

        # init env params
        self.env_config, self.teams_copies, self.teams = init_env_params(path_to_env_config_cls=path_to_env_config_cls,
                                                                          env_config_kwargs=env_config_kwargs,
                                                                          teams_copies=teams_copies,
                                                                          teams_config=teams_config)

        # if 'env' in rllib_config:
        # register env with ray
        register_env(path_to_env_cls.split('.')[-1], lambda env_config: get_class_from_path(path_to_env_cls)(self.env_config))

        # init rllib callbacks
        if 'callbacks' in rllib_config:
            if isinstance(rllib_config['callbacks'], str):
                # get callbacks class from string path
                rllib_config['callbacks'] = get_class_from_path(rllib_config['callbacks'])(teams=self.teams)

        # merge rllib trainer's default config with specified config
        path_to_agent = '.'.join(path_to_rllib_trainer_cls.split('.')[:-1])
        self.rllib_config = get_module_from_path(path_to_agent).DEFAULT_CONFIG.copy()
        self.rllib_config.update(rllib_config)

        # init rllib trainer
        self.trainer = get_class_from_path(path_to_rllib_trainer_cls)(config=self.rllib_config)
        self.last_agent_checkpoint = None

        # init other kwargs
        self.team_action_parallel = team_action_parallel

    def run(self,
            num_epochs: int = 1,
            verbose: bool = False,
            seed: int = None,
            **kwargs):
        '''Runs epoch(s).'''
        for epoch_counter in range(num_epochs):
            self.results = self.trainer.train()
            # TODO: Implement custom SingleAgent model policy which can handle observations etc. Use RLlib multi agent env with reward, obs, policy etc function mapping? Can try in notebook?
            # TODO: Test RLlib script for one epoch
            # TODO: Logging

        print(f'results: {self.results}')


















