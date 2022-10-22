import inspect

from neural_nets import *
import pickle as pkl
import numpy as np
from configs import *
from pprint import pprint
from bline_CybORGAgent import CyborgAgent

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
    # agent that loads a StableBaselines3 PPO model file
    def train(self, results):
        pass

    def end_episode(self):
        pass

    def set_initial_values(self, action_space, observation):
        pass

    def __init__(self, model_file: str = None):

        #keep here until other code in get_action is working
        self.model = None


        ModelCatalog.register_custom_model("CybORG_hier_Model", TorchModel)
        with open("checkpoint-1829", "rb") as controller_chkpt:  # Must open file in binary mode for pickle
            self.controller = pkl.load(controller_chkpt)
        self.bandit_observation = np.array([], dtype=int)

        RM_config = meander_config
        RM_config["in_evaluation"] = True
        RM_config["explore"] = False

        self.RM_def = PPOTrainer(config=RM_config, env=CybORGAgent)
        self.RM_def.restore("checkpoint-1829")

    def get_action(self, observation, action_space):
        """gets an action from the agent that should be performed based on the agent's internal state and provided observation and action space"""
        if self.model is None:
            path = str(inspect.getfile(CybORG))
            path = path[:-10] + '/Shared/Scenarios/Scenario1b.yaml'
            cyborg = ChallengeWrapper(env=CybORG(path, 'sim'), agent_name='Blue')
            #self.model = PPO('MlpPolicy', cyborg)
        #action, _states = self.model.predict
        action = 0
        return action
