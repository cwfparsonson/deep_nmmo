# TODO: Clean up any imports which are not actually used
from deep_nmmo.utils import get_class_from_path, get_module_from_path
from deep_nmmo.envs.team_based_env.utils import convert_nmmo_action_objects_to_action_indices
from deep_nmmo.envs.team_based_env.loops.utils import init_env_params, reset_teams, reset_env

import nmmo
from nmmo import config
from nmmo.io import action
from nmmo import scripting, material, Serialized
from nmmo.systems import skill, item
from nmmo.lib import colors
from nmmo import action as Action

import neurips2022nmmo
from neurips2022nmmo.scripted import baselines
from neurips2022nmmo import Team
from neurips2022nmmo import CompetitionConfig, scripted, RollOut, TeamBasedEnv
from neurips2022nmmo.scripted import attack, move

import ray

from ray.rllib.agents import ppo
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog

# from ray.rllib.env.multi_agent_env import make_multi_agent
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.policy.policy import PolicySpec

from ray.rllib.models.modelv2 import restore_original_dimensions
from collections.abc import Mapping

from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import (
    MultiAgentDict,
)

import gym

import numpy as np

import copy
from collections import defaultdict


class RLlibMultiAgentTeamBasedEnv(MultiAgentEnv):
    def __init__(self, 
                 env_config,
                 # path_to_env_cls,
                 # path_to_env_config_cls,
                 teams_config,
                 verbose=False,
                ):
        '''
        Notes on interacting with _env TeamBasedEnv internally:
            - TeamBasedEnv.players: Dict mapping player ID to player object
            - TeamBasedEnv.player_team_map: Dict mapping player ID to team ID
            - TeamBasedEnv.team_player_map: Dict mapping team ID to list of player IDs
        '''
        from ray.rllib.utils.spaces.space_utils import get_base_struct_from_space
        from neurips2022nmmo.scripted import RandomTeam

        self.verbose = verbose
        # self.verbose = True # DEBUG
        
        # inherit from RLlib multi-agent env
        MultiAgentEnv.__init__(self)
        
        # # init env config
        # env_config = get_class_from_path(path_to_env_config_cls)()
        
        # HACK: init dummy teams and env so can init RLlib policies with obs and action space
        dummy_teams = [RandomTeam(team_id=i, env_config=env_config) for i in range(len(list(teams_config.keys())))]
        for i, team in enumerate(dummy_teams):
            class Agent(nmmo.Agent):
                name = f'{team.id}'
                policy = f'{team.id}'
            env_config.PLAYERS[i] = Agent
        # dummy_env = get_class_from_path(path_to_env_cls)(env_config)
        dummy_env = get_class_from_path('neurips2022nmmo.TeamBasedEnv')(env_config)
        _ = dummy_env.reset() # need to call this for RLlib to make obs and action space accessible
        
        # use dummy env to init single agent observation and action space
        dummy_team_id = list(dummy_env.team_players_map.keys())[0]
        self.observation_space = dummy_env.observation_space(dummy_team_id)
        self.observation_space_struct = get_base_struct_from_space(self.observation_space)
        self.action_space = dummy_env.action_space(dummy_team_id)
        self.action_space_struct = get_base_struct_from_space(self.action_space)
        if self.verbose:
            print(f'Dummy env observation_space: {self.observation_space}')
            print(f'Dummy env action_space: {self.action_space}')

        # init teams
        self.teams = []
        for team_id, params in teams_config.items():
            team_cls, team_kwargs = params['cls'], params['kwargs']
            team_kwargs['env_config'] = env_config
            team_kwargs['team_id'] = team_id
            for player_idx in range(len(team_kwargs['agents_cls'])):
                team_kwargs['agents_kwargs'][player_idx] = {'observation_space': self.observation_space, 'action_space': self.action_space}
            # team_kwargs['observation_space'] = self.observation_space
            # team_kwargs['action_space'] = self.action_space
            if 'config' not in team_kwargs:
                team_kwargs['config'] = {}
            self.teams.append(team_cls(**team_kwargs))
        if self.verbose:
            print(f'Teams: {self.teams}')

        # update env config with instantiated teams
        for i, team in enumerate(self.teams):
            class Agent(nmmo.Agent):
                name = f'{team.id}'
                policy = f'{team.id}'
            env_config.PLAYERS[i] = Agent
            if self.verbose:
                print(f'env_config.PLAYERS[{i}]: {Agent}')
        if self.verbose:
            print(f'env_config: {env_config}')
                
        # init TeamBasedEnv with env config
        # self._env = get_class_from_path(path_to_env_cls)(env_config)
        self._env = get_class_from_path('neurips2022nmmo.TeamBasedEnv')(env_config)
        _ = self._env.reset() # need to call this for RLlib to make obs and action space accessible
        if self.verbose:
            print(f'_env: {self._env}')
            print(f'players: {self._env.players}')
            print(f'player_team_map: {self._env.player_team_map}')
            print(f'team_players_map: {self._env.team_players_map}')
        
        # init agent info
        self._agent_ids = list(self._env.players.keys())
        self.agents = list(self._env.players.values())
        if self.verbose:
            print(f'agents: {self.agents}')
            print(f'_agent_ids: {self._agent_ids}')

        
    @override(MultiAgentEnv)
    def observation_space_sample(self, agent_ids: list = None) -> MultiAgentDict:
        if agent_ids is None:
            # agent_ids = list(range(len(self.agents)))
            agent_ids = self._agent_ids
        obs = {agent_id: self.observation_space.sample() for agent_id in agent_ids}
        return obs

    @override(MultiAgentEnv)
    def action_space_sample(self, agent_ids: list = None) -> MultiAgentDict:
        if agent_ids is None:
            # agent_ids = list(range(len(self.agents)))
            agent_ids = self._agent_ids
        actions = {agent_id: self.action_space.sample() for agent_id in agent_ids}
        return actions

    @override(MultiAgentEnv)
    def action_space_contains(self, x: MultiAgentDict) -> bool:
        if not isinstance(x, dict):
            return False
        return all(self.action_space.contains(val) for val in x.values())

    @override(MultiAgentEnv)
    def observation_space_contains(self, x: MultiAgentDict) -> bool:
        if not isinstance(x, dict):
            return False
        return all(self.observation_space.contains(val) for val in x.values())

    @override(MultiAgentEnv)
    def reset(self):
        self.dones = set()
        # return {i: a.reset() for i, a in enumerate(self.agents)}
        
        self.teams = reset_teams(self.teams)
        
        self.agent_id_to_agent = {}
        for team_idx, team in enumerate(self.teams):
            for player_idx, player_id in enumerate(self._env.team_players_map[team_idx]):
                self.agent_id_to_agent[player_id] = team.agents[player_idx]
                
        nmmo_obs = reset_env(self._env)
        self.init_obs = self._flatten_teams_dict(nmmo_obs)
        # self.init_obs = self.post_process_obs(self.init_obs)

        
        # self.agent_id_to_agent = self._env.players

        self.step_counter = 0

        # print(self.observation_space.sample())
        # print(self.action_space.sample())

        self._check_obs_shape(self.init_obs) # DEBUG

        # DEBUG
        print(f'\n~~~ RESET ~~~')
        print(f'RLlibMultiAgentTeamBasedEnv obs keys: {len(self.init_obs)} {self.init_obs.keys()}')

        return self.init_obs

    # def post_process_obs(self, _obs):
        # from ray.rllib.utils.spaces.space_utils import get_base_struct_from_space
        # self.observation_space_struct = get_base_struct_from_space(self.observation_space)
        # padded_obs = defaultdict(lambda: defaultdict(dict))
        # for i in _obs:
            # # print(f'i: {i}') # DEBUG
            # for obs_type, _dict in self.observation_space_struct.items():
                # # print(f'obs_type: {obs_type}')
                # # print(f'_dict: {_dict}')
                # for obs_dim in _dict.keys():
                    # # print(f'obs_type {obs_type} obs_dim {obs_dim}: {_obs[i][obs_type][obs_dim].shape}') # DEBUG
                    # if obs_type in _obs[i]:
                        # if obs_dim in _obs[i][obs_type]:
                            # padded_obs[i][obs_type][obs_dim] = _obs[i][obs_type][obs_dim]
                        # else:
                            # print(f'NOT FOUND') # DEBUG
                            # # padded_obs[i][obs_type][obs_dim] = None
                            # padded_obs[i][obs_type][obs_dim] = self.observation_space[obs_type][obs_dim].sample()
                    # else:
                        # print(f'NOT FOUND') # DEBUG
                        # # padded_obs[i][obs_type][obs_dim] = None
                        # padded_obs[i][obs_type][obs_dim] = self.observation_space[obs_type][obs_dim].sample()
        # _obs = padded_obs
        # # raise Exception()
        # return _obs

    @override(MultiAgentEnv)
    def step(self, action_dict):
        import copy
        from deep_nmmo.envs.rllib_multi_agent_team_based_env.utils import gen_team_based_env_dummy_obs, check_if_dummy_obs

        if self.verbose:
            print(f'player to actions:\n{action_dict.keys()}\n{action_dict}')

        # print(self.observation_space.sample())
        # print(self.action_space.sample())
        # raise Exception()

        # # DEBUG
        # for i in action_dict:
            # for action_type, _dict in action_dict[i].items():
                # for action_dim in _dict.keys():
                    # # print(f'action_type {action_type} action_dim {action_dim}: {action_dict[i][action_type][action_dim].shape}') # DEBUG
                    # print(f'action_type {action_type} action_dim {action_dim}: {action_dict[i][action_type][action_dim]}') # DEBUG
        
        # convert player dict into team dict
        team_actions = self._unflatten_players_dict(action_dict)
        if self.verbose:
            print(f'team to actions:\n{team_actions.keys()}\n{team_actions}')
        
        # do post processing of actions
        for team_idx, actions in team_actions.items():
            team_actions[team_idx] = self.post_process_team_actions(self.teams[team_idx], actions)

        # step TeamBasedEnv
        nmmo_obs, nmmo_rew, nmmo_done, nmmo_info = self._env.step(team_actions)
        self.step_counter += 1
        print(f'\n~~~ Step {self.step_counter} ~~~') # DEBUG
        if self.verbose:
            print(f'TeamBasedEnv obs keys: {obs.keys()}')
            for key in obs.keys():
                print(f'TeamBasedEnv obs[{key}] keys: {obs[key].keys()}')
        
        # convert team dicts into player dicts
        _obs, _rew, _done, _info = [self._flatten_teams_dict(_dict) for _dict in [nmmo_obs, nmmo_rew, nmmo_done, nmmo_info]]

        # # DEBUG
        # copy_flat_nmmo_obs = copy.deepcopy(_obs) 
        # copy_flat_nmmo_rew = copy.deepcopy(_rew) 
        # copy_flat_nmmo_done = copy.deepcopy(_done)
        # copy_flat_nmmo_info = copy.deepcopy(_info)

        # DEBUG
        print(f'action_dict keys: {len(action_dict)} {action_dict.keys()}')
        # print(f'TeamBasedEnv obs keys: {len(_obs)} {_obs.keys()}')
        # print(f'TeamBasedEnv rew: {len(_rew)} {_rew}')
        # print(f'TeamBasedEnv done: {len(_done)} {_done}')
        # print(f'TeamBasedEnv info keys: {len(_info)} {_info.keys()}')

        # rllib requires that obs, rew, done, info all have keys for each agent
        # in this step's action_dict (i.e. number of agents cannot change within 
        # a step, whereas they do with NMMO). Generate dummy values which meet
        # the shape requirements of the environment for RLlib (these will be filtered
        # by the agent and then by the next step).
        obs, rew, done = _obs, _rew, _done
        for i in _rew.keys():
            if i not in _obs:
                # agent i has been filtered from obs by NMMO this step, need to add dummy obs
                _obs[i] = gen_team_based_env_dummy_obs(self.observation_space)
            if done[i]:
                self.dones.add(i)

        # rllib requires __all__ key in done to indicate if whole episode has finished
        done["__all__"] = len(self.dones) == len(self.agents)

        info = {}
        for i in obs.keys():
            # print(f'{i} in obs, adding to info')
            info[i] = {}



        # TODO TEMP DEBUG: Commented out below to overwrit obs etc for debugging below
        # # rllib requires info keys to be subset of obs keys
        # info = {}
        # obs, rew, done = _obs, _rew, _done
        # for i in _info:
            # if i in _obs:
                # # info[i] = _info[i]
                # info[i] = {}
            # else:
                # pass

        # for i in rew.keys():
            # if done[i]:
                # self.dones.add(i)
                # # HACK: RLlib expects observations and infos for agents on the step at which they are done, so make a dummy obs
                # obs[i] = self.init_obs[i]
                # info[i] = {}

        # # rllib requires done to have __all__ attribute to see if episode has completed for all agents
        # done["__all__"] = len(self.dones) == len(self.agents)

        # if self.verbose:
            # print(f'RLlibMultiAgentTeamBasedEnv obs keys: {len(obs)} {obs.keys()}')
            # print(f'RLlibMultiAgentTeamBasedEnv rew: {len(req)} {rew}')
            # print(f'RLlibMultiAgentTeamBasedEnv done: {len(done)} {done}')
            # print(f'RLlibMultiAgentTeamBasedEnv info keys: {len(info)} {info.keys()}')




        # TODO TEMP DEBUG: Overwriting obs etc for debugging
        # obs, rew, done, info = {}, {}, {}, {}
        # for i in action_dict:
            # done[i] = i not in copy_flat_nmmo_obs
            # if done[i]:
                # dummy_obs = self.observation_space.sample()
                # for obs_type in dummy_obs:
                    # for obs_dim in dummy_obs[obs_type]:
                        # dummy_obs[obs_type][obs_dim] = np.zeros(self.observation_space[obs_type][obs_dim].shape)
                # obs[i] = dummy_obs
            # else:
                # obs[i] = self.observation_space.sample()
            # rew[i] = 0
            # info[i] = {}
        # obs = {}
        # for i in copy_flat_nmmo_obs:
            # # obs[i] = self.observation_space.sample()
            # obs[i] = copy_flat_nmmo_obs[i]
            # # done[i] = False
            # # rew[i] = 0
        # done[i] = copy_flat_nmmo_done[i]
        # rew[i] = copy_flat_nmmo_rew[i]
        # info = {}
        # for i in obs.keys():
            # print(f'{i} in obs, adding to info')
            # info[i] = {}
        # # info[i] = {i: {} for i in obs.keys()}

        # done["__all__"] = len(self.dones) == len(self.agents)

        # DEBUG
        print(f'RLlibMultiAgentTeamBasedEnv obs keys: {len(obs)} {obs.keys()}')
        print(f'RLlibMultiAgentTeamBasedEnv rew: {len(rew)} {rew}')
        print(f'RLlibMultiAgentTeamBasedEnv done: {len(done)} {done}')
        print(f'RLlibMultiAgentTeamBasedEnv info keys: {len(info)} {info.keys()}')

        self._check_obs_shape(obs) # DEBUG
        
        return obs, rew, done, info

    def _check_obs_shape(self, obs):
        for i in obs:
            for obs_type, _dict in self.observation_space_struct.items():
                for obs_dim in _dict.keys():
                    if obs[i][obs_type][obs_dim].shape != self.observation_space[obs_type][obs_dim].shape:
                        raise Exception(f'obs obs_type {obs_type} obs_dim {obs_dim} has shape ({obs[obs_type][obs_dim][i]}) != observation_space shape ({self.observation_space[obs_type][obs_dim][i]})')
    
    def _flatten_teams_dict(self, _dict):
        flat_dict = {}
        # print(f'dict to flatten: {_dict}')
        for team_idx, player_ids in self._env.team_players_map.items():
            if team_idx in _dict:
                if 'stat' in _dict[team_idx]:
                    _ = _dict[team_idx].pop('stat')
                for player_idx in _dict[team_idx].keys():
                    player_id = player_ids[player_idx]
                    flat_dict[player_id] = _dict[team_idx][player_idx]
        return flat_dict
    
    def _unflatten_players_dict(self, _dict):
        unflat_dict = defaultdict(dict)
        for player_id in _dict.keys():
            team_idx = self._env.player_team_map[player_id]
            player_idx = self._env.team_players_map[team_idx].index(player_id)
            unflat_dict[team_idx][player_idx] = _dict[player_id]
        return unflat_dict
    
    def post_process_team_actions(self, team, _actions, verbose=False):
        import numpy as np
        # verbose = True # DEBUG

        if verbose:
            print(f'\nteam: {team}')
        
        # filter out any None actions (need as None for RLlib, need to remove for NMMO)
        if verbose:
            print(f'Filtering None actions...')
        actions = defaultdict(lambda: defaultdict(dict))
        for i in _actions:
            if verbose:
                print(f'\ti: {i}')
            for action_type, _dict in _actions[i].items():
                # print(f'action_type: {action_type} | _dict: {_dict}') # DEBUG
                for action_dim in _dict.keys():
                    # print(f'action_dim: {action_dim}') # DEBUG
                    if _actions[i][action_type][action_dim] is not None:
                    # if np.any(_actions[i][action_type][action_dim]):
                        if verbose:
                            print(f'agent {i} action_type {action_type} action_dim {action_dim} is {_actions[i][action_type][action_dim]}, adding')
                        actions[i][action_type][action_dim] = _actions[i][action_type][action_dim]
                    else:
                        if verbose:
                            print(f'agent {i} action_type {action_type} action_dim {action_dim} is dummy action (_actions[i][action_type][action_dim]), filtering')
                        pass

        # # convert nmmo action classes to action indices
        # if verbose:
            # print(f'Setting action indices...')
        # for i in actions:
            # if verbose:
                # print(f'\ti: {i}')
            # for atn, args in actions[i].items():
                # if verbose:
                    # print(f'atn: {atn}')
                    # print(f'args: {args}')
                # for arg, val in args.items():
                    # if verbose:
                        # print(f'arg: {arg}')
                        # print(f'val: {val}')
                        # print(f'edges: {arg.edges}')
                    # try:
                        # if arg.argType == nmmo.action.Fixed:
                            # actions[i][atn][arg] = arg.edges.index(val)
                        # elif arg == nmmo.action.Target:
                            # actions[i][atn][arg] = self.get_target_index(
                                # val, team.agents[i].ob.agents)
                        # elif atn in (nmmo.action.Sell,
                                     # nmmo.action.Use) and arg == nmmo.action.Item:
                            # actions[i][atn][arg] = self.get_item_index(
                                # val, team.agents[i].ob.items)
                        # elif atn == nmmo.action.Buy and arg == nmmo.action.Item:
                            # actions[i][atn][arg] = self.get_item_index(
                                # val, team.agents[i].ob.market)
                    # except ValueError:
                        # # TODO TEMP HACK: RLlib Epoch loop seems to return actions as ints rather than objects, somewhere under hood is this conversion done already?!
                        # # need to figure this out
                        # actions[i][atn][arg] = val

        # convert nmmo action objects to action indices
        if verbose:
            print(f'Setting action indices...')
        from deep_nmmo.envs.team_based_env.utils import convert_nmmo_action_objects_to_action_indices
        for i in actions:
            agent = team.agents[i]
            agent_actions = actions[i]
            actions[i] = convert_nmmo_action_objects_to_action_indices(agent=agent, agent_actions=agent_actions, verbose=verbose)

        return actions

    # @staticmethod
    # def get_item_index(instance: int, items: np.ndarray) -> int:
        # for i, itm in enumerate(items):
            # id_ = nmmo.scripting.Observation.attribute(itm,
                                                       # nmmo.Serialized.Item.ID)
            # if id_ == instance:
                # return i
        # raise ValueError(f"Instance {instance} not found")

    # @staticmethod
    # def get_target_index(target: int, agents: np.ndarray) -> int:
        # targets = [
            # x for x in [
                # nmmo.scripting.Observation.attribute(
                    # agent, nmmo.Serialized.Entity.ID) for agent in agents
            # ] if x
        # ]
        # return targets.index(target)
