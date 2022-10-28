from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG as PPO_CONFIG
from ray.rllib.models import ModelCatalog
from ray.rllib.agents import Trainer
from CybORGAgent import CybORGAgent
from bline_CybORGAgent import CybORGAgent as bline_CybORGAgent
import os
from neural_nets import *
from ray import tune
ModelCatalog.register_custom_model("CybORG_Torch", TorchModel)



meander_config = {
    "env": CybORGAgent,
    "env_config": {
        "null": 0,
    },
    "gamma": 0.99,
    # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
    "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", 0)),
    "num_envs_per_worker": 4,
    "entropy_coeff": 0.001,
    "num_sgd_iter": 10,
    "horizon": 100,
    "rollout_fragment_length": 100,
    #"vf_loss_coeff": 1e-5,
    #"vf_share_layers": False,
    "model": {
        # Attention net wrapping (for tf) can already use the native keras
        # model versions. For torch, this will have no effect.
        "_use_default_native_models": True,
        "custom_model": "CybORG_Torch",
        'fcnet_hiddens': [256, 256, 52],
        "use_attention": not True,
        "use_lstm":  not True,
        "max_seq_len": 10,
        "lstm_use_prev_action": True,
        "lstm_use_prev_reward": True,

    },
    "framework": 'torch',
}



