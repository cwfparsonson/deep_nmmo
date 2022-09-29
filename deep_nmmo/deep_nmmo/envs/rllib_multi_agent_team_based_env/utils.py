from ray.rllib.utils.spaces.space_utils import get_base_struct_from_space

from collections import defaultdict, Mapping

import numpy as np


def gen_team_based_env_dummy_obs(observation_space):
    dummy_obs = observation_space.sample()
    for obs_type in dummy_obs:
        for obs_dim in dummy_obs[obs_type]:
            dummy_obs[obs_type][obs_dim] = np.zeros(observation_space[obs_type][obs_dim].shape)
    return dummy_obs

def check_if_dummy_obs(obs):
    if isinstance(obs, Mapping):
        is_dummy = True
        print(f'obs: {type(obs)} {obs}')
        for obs_type in obs:
            for obs_dim in obs[obs_type]:
                if np.any(obs[obs_type][obs_dim]):
                    # non-zero array found in obs, not a done dummy obs
                    is_dummy = False
                    break
                else:
                    continue
    else:
        # obs is a numpy array
        if np.any(obs):
            is_dummy = False
        else:
            is_dummy = True
    return is_dummy

def gen_team_based_env_dummy_actions(action_space):
    action_space_struct = get_base_struct_from_space(action_space)
    dummy_actions = defaultdict(dict)
    for action_type, _dict in action_space_struct.items():
        for action_dim in _dict.keys():
            dummy_actions[action_type][action_dim] = gen_team_based_env_dummy_action()
            # dummy_actions[action_type][action_dim] = np.zeros(action_space_struct[action_type][action_dim].shape)
    return dummy_actions

def gen_team_based_env_dummy_action():
    return None

def check_if_dummy_action(action):
    return action is None

