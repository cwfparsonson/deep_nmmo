# TODO: Clean up any imports which are not actually used
from deep_nmmo.utils import get_class_from_path, get_module_from_path
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
from neurips2022nmmo.scripted.baselines import Scripted

import ray

from ray.rllib.agents import ppo
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import Policy

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



# class RLlibNMMOPolicy(Policy, Scripted):
    # """Pass NMMO agent to make compatible with RLlib envs."""
    # def __init__(self, 
                 # observation_space, 
                 # action_space, 
                 # config, # must include 'env_config' and 'idx'
                 # **kwargs):
        # # init RLlib policy
        # Policy.__init__(self, observation_space=observation_space, action_space=action_space, config={})
        # self.observation_space = observation_space
        # self.action_space = action_space

        # # init NMMO scripted policy
        # for key in ['env_config', 'idx']:
            # if key not in config:
                # raise Exception(f'Config must include key {key}')
        # self.idx = config['idx']
        # self.config = config['env_config'] # need this for Scripted
        # Scripted.__init__(self, config=self.config, idx=self.idx)

    # def compute_single_action(self, *args, **kwargs):
        # raise Exception(f'COMPUTE SINGLE ACTION CALLED, NOT IMPLEMENTED')

    # def compute_actions(self,
                        # obs_batch,
                        # state_batches=None,
                        # prev_action_batch=None,
                        # prev_reward_batch=None,
                        # info_batch=None,
                        # episodes=None,
                        # verbose=False,
                        # **kwargs):
        # """Compute actions on a batch of observations."""
        # # verbose = True # DEBUG
        
        # # HACK: Assume not batching (since not supported by nmmo agents as far as I can see)
        # # if isinstance(obs_batch, list):
        # #     raise Exception(f'Have not implemented observation batching, currently assumes obs_batch is just obs')
        # if not isinstance(obs_batch, Mapping):
            # # rllib trainer.train() automatically flattens obs, need to restore original obs dict to work with NMMO
            # if verbose:
                # print(f'OBS BATCH IS NOT MAPPING FOUND')
                # print(f'obs before restoring original:\n{obs_batch}')
            # import torch
            # obs = np.expand_dims(obs_batch[0], axis=0)
            # obs = torch.from_numpy(obs)
            # obs = restore_original_dimensions(obs, self.observation_space, 'torch')
            # # unpack
            # unpacked_obs = defaultdict(dict)
            # for key, val in obs.items():
                # for k, v in val.items():
                    # unpacked_obs[key][k] = np.array(v[0])
            # obs = unpacked_obs
            # if verbose:
                # print(f'obs after restoring original:\n{obs}')
        # else:
            # # obs is already a dict
            # obs = obs_batch
        

        # self(obs) # Scripted.__call__(obs)


class RLlibNMMOPolicy(Policy):
    """Pass NMMO agent to make compatible with RLlib envs."""
    def __init__(self, 
                 observation_space, 
                 action_space, 
                 config, # must include 'env_config' and 'idx'
                 **kwargs):
        '''
        Have to copy-past neurips2022nmmo.scripted.baselines.Scripted code here
        so that can change Scripted.config to Scripted.env_config so that does not
        conflict with RLlib.Policy.config
        '''
        Policy.__init__(self, observation_space=observation_space, action_space=action_space, config={})
        self.observation_space = observation_space
        self.action_space = action_space

        for key in ['env_config', 'idx']:
            if key not in config:
                raise Exception(f'Config must include key {key}')
        self.idx = config['idx']
        self.env_config = config['env_config']
        
        self.health_max = self.env_config.PLAYER_BASE_HEALTH

        if self.env_config.RESOURCE_SYSTEM_ENABLED:
            self.food_max = self.env_config.RESOURCE_BASE
            self.water_max = self.env_config.RESOURCE_BASE

        self.spawnR = None
        self.spawnC = None

    def compute_single_action(self, *args, **kwargs):
        print(f'COMPUTE SINGLE ACTION CALLED')

    def compute_actions(self,
                        obs_batch,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        verbose=False,
                        **kwargs):
        """Compute actions on a batch of observations."""
        # verbose = True # DEBUG
        
        # HACK: Assume not batching (since not supported by nmmo agents as far as I can see)
        # if isinstance(obs_batch, list):
        #     raise Exception(f'Have not implemented observation batching, currently assumes obs_batch is just obs')
        if not isinstance(obs_batch, Mapping):
            # rllib trainer.train() automatically flattens obs, need to restore original obs dict to work with NMMO
            if verbose:
                print(f'OBS BATCH IS NOT MAPPING FOUND')
                print(f'obs before restoring original:\n{obs_batch}')
            import torch
            obs = np.expand_dims(obs_batch[0], axis=0)
            obs = torch.from_numpy(obs)
            obs = restore_original_dimensions(obs, self.observation_space, 'torch')
            # unpack
            unpacked_obs = defaultdict(dict)
            for key, val in obs.items():
                for k, v in val.items():
                    unpacked_obs[key][k] = np.array(v[0])
            obs = unpacked_obs
            if verbose:
                print(f'obs after restoring original:\n{obs}')
        else:
            # obs is already a dict
            obs = obs_batch
        
        self.actions = {}

        if verbose:
            print(f'Computing obs from obs:\n{obs}')
            print(f'obs len: {len(obs)}')
            print(f'obs shape: {np.array(obs).shape}')
            print(f'obs keys: {obs.keys()}')
            for key in obs.keys():
                print(f'obs[{key}] shape: {np.array(obs[key])}')
        self.ob = scripting.Observation(self.env_config, obs)
        if verbose:
            print(f'Computed obs!!!')
        agent = self.ob.agent

        # Time Alive
        self.timeAlive = scripting.Observation.attribute(
            agent, Serialized.Entity.TimeAlive)

        # Pos
        self.r = scripting.Observation.attribute(agent, Serialized.Entity.R)
        self.c = scripting.Observation.attribute(agent, Serialized.Entity.C)

        #Resources
        self.health = scripting.Observation.attribute(agent,
                                                      Serialized.Entity.Health)
        self.food = scripting.Observation.attribute(agent,
                                                    Serialized.Entity.Food)
        self.water = scripting.Observation.attribute(agent,
                                                     Serialized.Entity.Water)

        #Skills
        self.melee = scripting.Observation.attribute(agent,
                                                     Serialized.Entity.Melee)
        self.range = scripting.Observation.attribute(agent,
                                                     Serialized.Entity.Range)
        self.mage = scripting.Observation.attribute(agent,
                                                    Serialized.Entity.Mage)
        self.fishing = scripting.Observation.attribute(
            agent, Serialized.Entity.Fishing)
        self.herbalism = scripting.Observation.attribute(
            agent, Serialized.Entity.Herbalism)
        self.prospecting = scripting.Observation.attribute(
            agent, Serialized.Entity.Prospecting)
        self.carving = scripting.Observation.attribute(
            agent, Serialized.Entity.Carving)
        self.alchemy = scripting.Observation.attribute(
            agent, Serialized.Entity.Alchemy)

        #Combat level
        # TODO: Get this from agent properties
        self.level = max(self.melee, self.range, self.mage, self.fishing,
                         self.herbalism, self.prospecting, self.carving,
                         self.alchemy)

        self.skills = {
            skill.Melee: self.melee,
            skill.Range: self.range,
            skill.Mage: self.mage,
            skill.Fishing: self.fishing,
            skill.Herbalism: self.herbalism,
            skill.Prospecting: self.prospecting,
            skill.Carving: self.carving,
            skill.Alchemy: self.alchemy
        }

        if self.spawnR is None:
            self.spawnR = scripting.Observation.attribute(
                agent, Serialized.Entity.R)
        if self.spawnC is None:
            self.spawnC = scripting.Observation.attribute(
                agent, Serialized.Entity.C)

        # When to run from death fog in BR configs
        self.fog_criterion = None
        if self.env_config.PLAYER_DEATH_FOG is not None:
            step_per_tile = max(1, int(1 / self.env_config.PLAYER_DEATH_FOG_SPEED))
            start_running = self.timeAlive > self.env_config.PLAYER_DEATH_FOG - step_per_tile
            run_now = (self.timeAlive % step_per_tile == 0)
            self.fog_criterion = start_running and run_now
    
    
    

    # def compute_single_action(self, obs, *args, **kwargs):
    #     return self.nmmo_agent(obs)
    
    def learn_on_batch(self, samples):
        """No learning."""
        #return {}
        pass

    def get_weights(self):
        pass

    def set_weights(self, weights):
        pass

    @property
    def policy(self):
        return self.__class__.__name__

    @property
    def forage_criterion(self) -> bool:
        '''Return true if low on food or water'''
        min_level = 7 * self.env_config.RESOURCE_DEPLETION_RATE
        return self.food <= min_level or self.water <= min_level

    def forage(self):
        '''Min/max food and water using Dijkstra's algorithm'''
        move.forageDijkstra(self.env_config, self.ob, self.actions, self.food_max,
                            self.water_max)

    def gather(self, resource):
        '''BFS search for a particular resource'''
        return move.gatherBFS(self.env_config, self.ob, self.actions, resource)

    def explore(self):
        '''Route away from spawn'''
        sz = self.env_config.MAP_SIZE
        centR, centC = sz // 2, sz // 2
        if self.timeAlive < sz // 4:
            move.explore(self.env_config, self.ob, self.actions, self.spawnR,
                         self.spawnC)
        elif self.fog_criterion:
            move.explore(self.env_config, self.ob, self.actions, self.r, self.c)
        else:
            move.explore(self.env_config, self.ob, self.actions, centR, centC)
        #move.explore(self.env_config, self.ob, self.actions, self.spawnR, self.spawnC)

    @property
    def downtime(self):
        '''Return true if agent is not occupied with a high-priority action'''
        return not self.forage_criterion and self.attacker is None

    def evade(self):
        '''Target and path away from an attacker'''
        move.evade(self.env_config, self.ob, self.actions, self.attacker)
        self.target = self.attacker
        self.targetID = self.attackerID
        self.targetDist = self.attackerDist

    def attack(self):
        '''Attack the current target'''
        if self.target is not None:
            assert self.targetID is not None
            style = random.choice(self.style)
            attack.target(self.env_config, self.actions, style, self.targetID)

    def target_weak(self):
        '''Target the nearest agent if it is weak'''
        if self.closest is None:
            return False

        selfLevel = scripting.Observation.attribute(self.ob.agent,
                                                    Serialized.Entity.Level)
        targLevel = scripting.Observation.attribute(self.closest,
                                                    Serialized.Entity.Level)
        population = scripting.Observation.attribute(
            self.closest, Serialized.Entity.Population)

        if population == -1 or targLevel <= selfLevel <= 5 or selfLevel >= targLevel + 3:
            self.target = self.closest
            self.targetID = self.closestID
            self.targetDist = self.closestDist

    def scan_agents(self):
        '''Scan the nearby area for agents'''
        self.closest, self.closestDist = attack.closestTarget(
            self.env_config, self.ob)
        self.attacker, self.attackerDist = attack.attacker(
            self.env_config, self.ob)

        self.closestID = None
        if self.closest is not None:
            self.closestID = scripting.Observation.attribute(
                self.closest, Serialized.Entity.ID)

        self.attackerID = None
        if self.attacker is not None:
            self.attackerID = scripting.Observation.attribute(
                self.attacker, Serialized.Entity.ID)

        self.target = None
        self.targetID = None
        self.targetDist = None

    def adaptive_control_and_targeting(self, explore=True):
        '''Balanced foraging, evasion, and exploration'''
        self.scan_agents()

        if self.attacker is not None:
            self.evade()
            return

        if self.fog_criterion:
            self.explore()
        elif self.forage_criterion or not explore:
            self.forage()
        else:
            self.explore()

        self.target_weak()

    def process_inventory(self):
        if not self.env_config.ITEM_SYSTEM_ENABLED:
            return

        self.inventory = set()
        self.best_items = {}
        self.item_counts = defaultdict(int)

        self.item_levels = {
            item.Hat: self.level,
            item.Top: self.level,
            item.Bottom: self.level,
            item.Sword: self.melee,
            item.Bow: self.range,
            item.Wand: self.mage,
            item.Rod: self.fishing,
            item.Gloves: self.herbalism,
            item.Pickaxe: self.prospecting,
            item.Chisel: self.carving,
            item.Arcane: self.alchemy,
            item.Scrap: self.melee,
            item.Shaving: self.range,
            item.Shard: self.mage
        }

        self.gold = scripting.Observation.attribute(self.ob.agent,
                                                    Serialized.Entity.Gold)

        for item_ary in self.ob.items:
            itm = Item(item_ary)
            cls = itm.cls

            assert itm.cls.__name__ == 'Gold' or itm.quantity != 0
            #if itm.quantity == 0:
            #   continue

            self.item_counts[cls] += itm.quantity
            self.inventory.add(itm)

            #Too high level to equip
            if cls in self.item_levels and itm.level > self.item_levels[cls]:
                continue

            #Best by default
            if cls not in self.best_items:
                self.best_items[cls] = itm

            best_itm = self.best_items[cls]

            if itm.level > best_itm.level:
                self.best_items[cls] = itm

            if __debug__:
                err = 'Key {} must be an Item object'.format(cls)
                assert isinstance(self.best_items[cls], Item), err

    def upgrade_heuristic(self, current_level, upgrade_level, price):
        return (upgrade_level - current_level) / max(price, 1)

    def process_market(self):
        if not self.env_config.EXCHANGE_SYSTEM_ENABLED:
            return

        self.market = set()
        self.best_heuristic = {}

        for item_ary in self.ob.market:
            itm = Item(item_ary)
            cls = itm.cls

            self.market.add(itm)

            #Prune Unaffordable
            if itm.price > self.gold:
                continue

            #Too high level to equip
            if cls in self.item_levels and itm.level > self.item_levels[cls]:
                continue

            #Current best item level
            current_level = 0
            if cls in self.best_items:
                current_level = self.best_items[cls].level

            itm.heuristic = self.upgrade_heuristic(current_level, itm.level,
                                                   itm.price)

            #Always count first item
            if cls not in self.best_heuristic:
                self.best_heuristic[cls] = itm
                continue

            #Better heuristic value
            if itm.heuristic > self.best_heuristic[cls].heuristic:
                self.best_heuristic[cls] = itm

    def equip(self, items: set):
        for cls, itm in self.best_items.items():
            if cls not in items:
                continue

            if itm.equipped:
                continue

            self.actions[Action.Use] = {Action.Item: itm.instance}

            return True

    def consume(self):
        if self.health <= self.health_max // 2 and item.Poultice in self.best_items:
            itm = self.best_items[item.Poultice]
        elif (self.food == 0
              or self.water == 0) and item.Ration in self.best_items:
            itm = self.best_items[item.Ration]
        else:
            return

        self.actions[Action.Use] = {Action.Item: itm.instance}

    def sell(self, keep_k: dict, keep_best: set):
        for itm in self.inventory:
            price = itm.level
            cls = itm.cls

            if cls == item.Gold:
                continue

            assert itm.quantity > 0

            if cls in keep_k:
                owned = self.item_counts[cls]
                k = keep_k[cls]
                if owned <= k:
                    continue

            #Exists an equippable of the current class, best needs to be kept, and this is the best item
            if cls in self.best_items and cls in keep_best and itm.instance == self.best_items[
                    cls].instance:
                continue

            self.actions[Action.Sell] = {
                Action.Item: itm.instance,
                Action.Price: Action.Price.edges[int(price)],
            }

            return itm

    def buy(self, buy_k: dict, buy_upgrade: set):
        if len(self.inventory) >= self.env_config.ITEM_INVENTORY_CAPACITY:
            return

        purchase = None
        best = list(self.best_heuristic.items())
        random.shuffle(best)
        for cls, itm in best:
            #Buy top k
            if cls in buy_k:
                owned = self.item_counts[cls]
                k = buy_k[cls]
                if owned < k:
                    purchase = itm

            #Check if item desired
            if cls not in buy_upgrade:
                continue

            #Check is is an upgrade
            if itm.heuristic <= 0:
                continue

            #Buy best heuristic upgrade
            self.actions[Action.Buy] = {Action.Item: itm.instance}

            return itm

    def exchange(self):
        if not self.env_config.EXCHANGE_SYSTEM_ENABLED:
            return

        self.process_market()
        self.sell(keep_k=self.supplies, keep_best=self.wishlist)
        self.buy(buy_k=self.supplies, buy_upgrade=self.wishlist)

    def use(self):
        self.process_inventory()
        if self.env_config.EQUIPMENT_SYSTEM_ENABLED and not self.consume():
            self.equip(items=self.wishlist)
