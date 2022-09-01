'''
All code must be in the my-submission folder (i.e. copy -r deep_nmmo/deep_nmmo/
into this folder).

Instructions on how to submit: https://gitlab.aicrowd.com/neural-mmo/neurips2022-nmmo-starter-kit

python tool.py test --startby=docker
python tool.py submit <unique-submission-name> --startby=docker
'''

# from deep_nmmo.envs.team_based_env.teams.custom_team import CustomTeam

# from deep_nmmo.utils import get_class_from_path

from neurips2022nmmo import Team

import nmmo

from typing import Any, Dict, Type, List
import numpy as np

def get_class_from_path(path):
    '''
    Path must be the path to the class **without** the .py extension.

    E.g. deep_nmmo.module_name.ModuleClass
    '''
    ClassName = path.split('.')[-1]
    path_to_class = '.'.join(path.split('.')[:-1])
    module = __import__(path_to_class, fromlist=[ClassName])
    return getattr(module, ClassName)


class CustomTeam(Team):
    def __init__(self, 
                 team_id: str,
                 env_config, 
                 paths_to_agents_cls: list,
                 **kwargs):
        '''
        Args:
            paths_to_agents_cls (dict): Dict mapping agents indices to paths to agent classes.
                N.B. Indices must be able to be converted into ints e.g. can be '10' or 10 but 
                not 'agent'.
        '''
        super().__init__(team_id, env_config)
        self.id = team_id
        # self.agents = [get_class_from_path(path_to_agent_cls)(config=env_config, idx=idx) for idx, path_to_agent_cls in enumerate(paths_to_agents_cls)]
        self.agents = [get_class_from_path(path_to_agent_cls)(config=env_config, idx=int(idx)) for idx, path_to_agent_cls in paths_to_agents_cls.items()]
            
    def reset(self):
        pass
    
    def act(self, observations: Dict[Any, dict]) -> Dict[int, dict]:
        if "stat" in observations:
            stat = observations.pop("stat")
        actions = {i: self.agents[i](obs) for i, obs in observations.items()}
        for i in actions:
            for atn, args in actions[i].items():
                for arg, val in args.items():
                    if arg.argType == nmmo.action.Fixed:
                        actions[i][atn][arg] = arg.edges.index(val)
                    elif arg == nmmo.action.Target:
                        actions[i][atn][arg] = self.get_target_index(
                            val, self.agents[i].ob.agents)
                    elif atn in (nmmo.action.Sell,
                                 nmmo.action.Use) and arg == nmmo.action.Item:
                        actions[i][atn][arg] = self.get_item_index(
                            val, self.agents[i].ob.items)
                    elif atn == nmmo.action.Buy and arg == nmmo.action.Item:
                        actions[i][atn][arg] = self.get_item_index(
                            val, self.agents[i].ob.market)
        return actions

    @staticmethod
    def get_item_index(instance: int, items: np.ndarray) -> int:
        for i, itm in enumerate(items):
            id_ = nmmo.scripting.Observation.attribute(itm,
                                                       nmmo.Serialized.Item.ID)
            if id_ == instance:
                return i
        raise ValueError(f"Instance {instance} not found")

    @staticmethod
    def get_target_index(target: int, agents: np.ndarray) -> int:
        targets = [
            x for x in [
                nmmo.scripting.Observation.attribute(
                    agent, nmmo.Serialized.Entity.ID) for agent in agents
            ] if x
        ]
        return targets.index(target)

class Submission:
    team_klass = CustomTeam
    init_params = {'paths_to_agents_cls': 
                    {
                        '0': 'neurips2022nmmo.scripted.baselines.Herbalist',
                        '1': 'neurips2022nmmo.scripted.baselines.Melee',
                        '2': 'neurips2022nmmo.scripted.baselines.Range',
                        '3': 'neurips2022nmmo.scripted.baselines.Fisher',
                        '4': 'neurips2022nmmo.scripted.baselines.Carver',
                        '5': 'neurips2022nmmo.scripted.baselines.Range',
                        '6': 'neurips2022nmmo.scripted.baselines.Alchemist',
                        '7': 'neurips2022nmmo.scripted.baselines.Fisher',
                    }
                  }
