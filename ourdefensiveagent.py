import inspect

from neural_nets import *
import pickle as pkl
import numpy as np
from configs import *
from pprint import pprint
from bline_CybORGAgent import CybORGAgent as bline_CybORGAgent
from CybORGAgent import CybORGAgent

from CybORG import CybORG
from CybORG.Agents.SimpleAgents.BaseAgent import BaseAgent
from CybORG.Agents.Wrappers.EnumActionWrapper import EnumActionWrapper
from CybORG.Agents.Wrappers.FixedFlatWrapper import FixedFlatWrapper
from CybORG.Agents.Wrappers.OpenAIGymWrapper import OpenAIGymWrapper
from CybORG.Agents.Wrappers.ReduceActionSpaceWrapper import ReduceActionSpaceWrapper
from CybORG.Agents.Wrappers import ChallengeWrapper

import ray.rllib.agents.ppo as ppo
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog

class ourdefensiveagent(BaseAgent):

    def end_episode(self):
        self.set = False

    #constructor for the agent object, made with reference to the mindrake implementation.
    def __init__(self, model_file: str = None):

        #configuration
        ModelCatalog.register_custom_model("CybORG_hier_Model", TorchModel)
        RM_config = meander_config
        RM_config["in_evaluation"] = True
        RM_config["explore"] = False

        #variable intialization
        self.state = [np.zeros(256, np.float32),
                      np.zeros(256, np.float32)]
        self.step = 0
        self.set = False
        self.observations = []
        self.adversary = 0

        #Loads the weights from mindrakes meander implementation
        self.RM_def = PPOTrainer(config=RM_config, env=CybORGAgent)
        self.RM_def.restore("./checkpoint-1829")


    #Feeds in state to neural network, which outputs an action which it performs.
    def get_action(self, obs, action_space):
        agent_action, state, _ = self.RM_def.compute_single_action(obs[2:], self.state)
        # print('meander defence')
        self.state = state
        return agent_action, self.adversary


