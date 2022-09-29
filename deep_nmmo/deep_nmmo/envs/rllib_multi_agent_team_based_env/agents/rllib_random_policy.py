from deep_nmmo.envs.rllib_multi_agent_team_based_env.agents.rllib_nmmo_policy import RLlibNMMOPolicy
from deep_nmmo.envs.team_based_env.utils import convert_nmmo_action_objects_to_action_indices
from deep_nmmo.envs.rllib_multi_agent_team_based_env.utils import check_if_dummy_obs, gen_team_based_env_dummy_actions, gen_team_based_env_dummy_action

from neurips2022nmmo.scripted import attack, move

import numpy as np

from collections import defaultdict

from ray.rllib.utils.spaces.space_utils import get_base_struct_from_space

class RLlibRandomPolicy(RLlibNMMOPolicy):
    def __init__(self,
                 *args,
                 **kwargs):
        RLlibNMMOPolicy.__init__(self, *args, **kwargs)

        
    def compute_actions(self, obs, *args, **kwargs):
        if check_if_dummy_obs(obs):
            # TODO: Not sure if this would ever be called since agent should have done == True and therefore RLlib shouldn't ask it to make additional actions? check
            # returned action for RLlib, need to return dummy action
            print(f'DUMMY OBS PASSED TO COMPUTE ACTIONS!!!')
            rllib_actions = gen_team_based_env_dummy_actions(self.action_space)
        else:
            # compute nmmo agent actions
            # print(f'Computing nmmo actions from obs {obs}')
            super().compute_actions(obs)

            # compute specialised agent actions
            # print(f'Computing random actions from ob {self.ob}')
            move.random(self.config, self.ob, self.actions)
            # print(f'Computed agent {self} actions:')
            # print(self.actions)
            # for key, val in self.actions.items():
                # print(f'key: {type(key)} {key} | val: {type(val)} {val}')
                # for k, v in val.items():
                    # print(f'k: {type(k)} {k} | v: {type(v)} {v}')

            # # TODO TEMP DEBUG: Convert action objects to indices
            # from deep_nmmo.envs.team_based_env.utils import convert_nmmo_action_objects_to_action_indices
            # self.actions = convert_nmmo_action_objects_to_action_indices(agent=self, agent_actions=self.actions)


            # rllib requires the agent actions to have the same structure as the original action space with no missing entries, so need to fill empty actions with dummy actions
            self.action_space_struct = get_base_struct_from_space(self.action_space)
            rllib_actions = defaultdict(dict)
            for action_type, _dict in self.action_space_struct.items():
                for action_dim in _dict.keys():
                    if action_type in self.actions:
                        if action_dim in self.actions[action_type]:
                            rllib_actions[action_type][action_dim] = self.actions[action_type][action_dim]
                        else:
                            rllib_actions[action_type][action_dim] = None
                            # rllib_actions[action_type][action_dim] = np.zeros(self.action_space_struct[action_type][action_dim].shape)
                    else:
                        rllib_actions[action_type][action_dim] = None
                        # rllib_actions[action_type][action_dim] = np.zeros(self.action_space_struct[action_type][action_dim].shape)

            # TODO TEMP DEBUG: Overwrite None action filler with sample action
            sample_action = self.action_space.sample()
            for action_type, _dict in self.actions.items():
                for action_dim in _dict.keys():
                    sample_action[action_type][action_dim] = self.actions[action_type][action_dim]
            # print(f'rllib_actions: {rllib_actions}')
            # print(f'action_space.sample(): {self.action_space.sample()}')
            rllib_actions = sample_action

            # rllib expects actions to be returned as a batch of actions
            batched_rllib_actions = [rllib_actions]

            # rllib expects a tensor or numpy array
            rllib_actions = np.array([rllib_actions])

        print(f'rllib_actions obs of agent: {obs.shape} {obs}') # DEBUG
        print(f'rllib_actions returned by agent: {rllib_actions}') # DEBUG
        print(f'ob of agent: {self.ob}')

        return rllib_actions, [], {}
