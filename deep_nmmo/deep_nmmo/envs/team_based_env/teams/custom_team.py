from deep_nmmo.utils import get_class_from_path

from neurips2022nmmo import Team

import nmmo

from typing import Any, Dict, Type, List
import numpy as np

class CustomTeam(Team):
    def __init__(self, 
                 team_id: str,
                 env_config, 
                 agents_cls: list, # either string path or raw cls
                 agents_kwargs: list, # corresponding kwargs for agents when inititalising
                 **kwargs):
        if "policy_id" not in kwargs:
            kwargs["policy_id"] = self.__class__.__name__
        super().__init__(team_id, env_config, **kwargs)
        # self.agent_klass = [get_class_from_path(path_to_agent_cls) for path_to_agent_cls in paths_to_agents_cls if isinstance(path_to_agent_cls, str) else path_to_agent_cls]
        self.agent_klass = []
        for agent_cls in agents_cls:
            if isinstance(agents_cls, str):
                self.agent_klass.append(get_class_from_path(agent_cls))
            else:
                self.agent_klass.append(agent_cls)
        self.agents_kwargs = agents_kwargs
        self.reset()

    def reset(self):
        assert self.agent_klass
        self.agents = []
        for i in range(self.n_player):
            idx = i % len(self.agent_klass)
            # agent = self.agent_klass[idx](config=self.env_config, idx=i, **self.agents_kwargs[idx])
            agent = self.agent_klass[idx](config={'env_config': self.env_config, 'idx': i}, **self.agents_kwargs[idx])
            self.agents.append(agent)
            # # TODO TEMP HACK: NMMO requires various calls such as self.ob etc which only exist inside nmmo_agent, so storing that here. This may lead to problems when batching, since self.obs set on each call but not saved across batches?
            # if hasattr(agent, 'nmmo_agent'):
                # self.agents.append(agent.nmmo_agent)
            # else:
                # self.agents.append(agent)
    
    def act(self, observations: Dict[Any, dict]) -> Dict[int, dict]:
        # filter obs as desired
        if "stat" in observations:
            _ = observations.pop("stat")
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

# class CustomTeam(Team):
    # def __init__(self, 
                 # team_id: str,
                 # env_config, 
                 # paths_to_agents_cls: dict,
                 # **kwargs):
        # '''
        # Args:
            # paths_to_agents_cls (dict): Dict mapping agents indices to paths to agent classes.
                # N.B. Indices must be able to be converted into ints e.g. can be '10' or 10 but 
                # not 'agent'.
        # '''
        # super().__init__(team_id, env_config)
        # self.id = team_id
        # # self.agents = [get_class_from_path(path_to_agent_cls)(config=env_config, idx=idx) for idx, path_to_agent_cls in enumerate(paths_to_agents_cls)]
        # self.agents = [get_class_from_path(path_to_agent_cls)(config=env_config, idx=int(idx)) for idx, path_to_agent_cls in paths_to_agents_cls.items()]
            
    # def reset(self):
        # pass
    
    # def act(self, observations: Dict[Any, dict]) -> Dict[int, dict]:
        # if "stat" in observations:
            # stat = observations.pop("stat")
        # actions = {i: self.agents[i](obs) for i, obs in observations.items()}
        # for i in actions:
            # for atn, args in actions[i].items():
                # for arg, val in args.items():
                    # if arg.argType == nmmo.action.Fixed:
                        # actions[i][atn][arg] = arg.edges.index(val)
                    # elif arg == nmmo.action.Target:
                        # actions[i][atn][arg] = self.get_target_index(
                            # val, self.agents[i].ob.agents)
                    # elif atn in (nmmo.action.Sell,
                                 # nmmo.action.Use) and arg == nmmo.action.Item:
                        # actions[i][atn][arg] = self.get_item_index(
                            # val, self.agents[i].ob.items)
                    # elif atn == nmmo.action.Buy and arg == nmmo.action.Item:
                        # actions[i][atn][arg] = self.get_item_index(
                            # val, self.agents[i].ob.market)
        # return actions

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
