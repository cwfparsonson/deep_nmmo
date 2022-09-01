'''
All code must be in the my-submission folder (i.e. copy -r deep_nmmo/deep_nmmo/
into this folder).

Instructions on how to submit: https://gitlab.aicrowd.com/neural-mmo/neurips2022-nmmo-starter-kit

python tool.py test --startby=docker
python tool.py submit <unique-submission-name> --startby=docker
'''
from neurips2022nmmo import Team

import nmmo

from typing import Any, Dict, Type, List
import numpy as np

def get_class_from_path(path):
    ClassName = path.split('.')[-1]
    path_to_class = '.'.join(path.split('.')[:-1])
    module = __import__(path_to_class, fromlist=[ClassName])
    return getattr(module, ClassName)

def setup():
    from nmmo.systems import item
    from nmmo.io.action import Price
    for itm in [
            item.Gold,
            item.Hat,
            item.Top,
            item.Bottom,
            item.Sword,
            item.Bow,
            item.Wand,
            item.Rod,
            item.Gloves,
            item.Pickaxe,
            item.Chisel,
            item.Arcane,
            item.Scrap,
            item.Shard,
            item.Shaving,
            item.Ration,
            item.Poultice,
    ]:

        item.ItemID.register(itm, itm.ITEM_ID)
    Price.init(None)


setup()


class CustomTeam(Team):
    def __init__(self, 
                 team_id: str,
                 env_config, 
                 paths_to_agents_cls: dict,
                 **kwargs):
        if "policy_id" not in kwargs:
            kwargs["policy_id"] = self.__class__.__name__
        super().__init__(team_id, env_config, **kwargs)
        self.agent_klass = [get_class_from_path(path_to_agent_cls) for path_to_agent_cls in paths_to_agents_cls.values()]
        self.reset()

    def reset(self):
        assert self.agent_klass
        self.agents = []
        for i in range(self.n_player):
            idx = i % len(self.agent_klass)
            agent = self.agent_klass[idx](self.env_config, i)
            self.agents.append(agent)
    
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
                        '0': 'neurips2022nmmo.scripted.baselines.Mage',
                        '1': 'neurips2022nmmo.scripted.baselines.Mage',
                        '2': 'neurips2022nmmo.scripted.baselines.Mage',
                        '3': 'neurips2022nmmo.scripted.baselines.Mage',
                        '4': 'neurips2022nmmo.scripted.baselines.Mage',
                        '5': 'neurips2022nmmo.scripted.baselines.Mage',
                        '6': 'neurips2022nmmo.scripted.baselines.Mage',
                        '7': 'neurips2022nmmo.scripted.baselines.Mage',
                    }
                  }
