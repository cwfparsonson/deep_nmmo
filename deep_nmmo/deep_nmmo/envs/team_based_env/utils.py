import nmmo
import numpy as np

# def convert_nmmo_action_objects_to_action_indices(team, actions, verbose=False):
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
                    # if arg.argType == nmmo.action.Fixedj
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
def convert_nmmo_action_objects_to_action_indices(agent, agent_actions, verbose=False):
    '''
    For a team actions, do:

    for i in actions:
        agent = team.agents[i]
        agent_actions = actions[i]
        actions[i] = convert_nmmo_action_objects_to_action_indices(team=team, actions=actions, verbose=verbose)
    '''
    # convert nmmo action classes to action indices
    if verbose:
        print(f'Setting action indices for agent {agent}...')
    for atn, args in agent_actions.items():
        if verbose:
            print(f'atn: {atn}')
            print(f'args: {args}')
        for arg, val in args.items():
            if verbose:
                print(f'arg: {arg}')
                print(f'val: {val}')
                print(f'edges: {arg.edges}')
            try:
                if arg.argType == nmmo.action.Fixed:
                    agent_actions[atn][arg] = arg.edges.index(val)
                elif arg == nmmo.action.Target:
                    agent_actions[atn][arg] = get_target_index(
                        val, agent.ob.agents)
                elif atn in (nmmo.action.Sell,
                             nmmo.action.Use) and arg == nmmo.action.Item:
                    agent_actions[atn][arg] = get_item_index(
                        val, agent.ob.items)
                elif atn == nmmo.action.Buy and arg == nmmo.action.Item:
                    agent_actions[atn][arg] = get_item_index(
                        val, agent.ob.market)
            except ValueError:
                # TODO TEMP HACK: RLlib Epoch loop seems to return actions as ints rather than objects, somewhere under hood is this conversion done already?!
                # need to figure this out
                agent_actions[atn][arg] = val
    return agent_actions

def get_item_index(instance: int, items: np.ndarray) -> int:
    for i, itm in enumerate(items):
        id_ = nmmo.scripting.Observation.attribute(itm,
                                                   nmmo.Serialized.Item.ID)
        if id_ == instance:
            return i
    raise ValueError(f"Instance {instance} not found")

def get_target_index(target: int, agents: np.ndarray) -> int:
    targets = [
        x for x in [
            nmmo.scripting.Observation.attribute(
                agent, nmmo.Serialized.Entity.ID) for agent in agents
        ] if x
    ]
    return targets.index(target)
