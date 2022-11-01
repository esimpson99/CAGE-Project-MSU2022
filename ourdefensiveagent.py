
import numpy as np
from configs import *
from CybORGAgent import CybORGAgent
from CybORG.Agents.SimpleAgents.BaseAgent import BaseAgent
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

        #variable intialization. State is used to store previous observations made.
        self.state = [np.zeros(256, np.float32),np.zeros(256, np.float32)]
        self.set = False
        self.adversary = 0

        #Loads the weights from mindrakes meander implementation
        self.PPONN = PPOTrainer(config=RM_config, env=CybORGAgent)
        self.PPONN.restore("./general_weights")


    #Feeds in state to neural network, which outputs an action which it returns.
    def get_action(self, obs, action_space):
        agent_action, state, _ = self.PPONN.compute_single_action(obs[2:], self.state)
        self.state = state
        return agent_action, self.adversary