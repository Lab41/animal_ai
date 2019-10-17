import argparse

import torch
#import sys
#sys.path.insert(1,'/aaio/data/a3c_src')
#from model import ActorCritic

from a3c_src.model import ActorCritic
import torch.nn.functional as F
import numpy as np
from animalai.envs import UnityEnvironment
from animalai.envs.arena_config import ArenaConfig

class Agent(object):

    def __init__(self):
        """
         Load your agent here and initialize anything needed
         WARNING: any path to files you wish to access on the docker should be ABSOLUTE PATHS
        """


        self.device = torch.device("cpu")
        self.num_states = 3
        self.num_actions = 5
        self.actions_array = np.array([[0,0],[0,1],[0,2],[1,0],[2,0]])
        self.brain_name = 'Learner'

        self.saved_filepath = '/aaio/data/trained_models/a3c_base_99500'
        #self.saved_filepath = 'data/trained_models/a3c_base_99500'


        self.model = ActorCritic(self.num_states, self.num_actions)
        self.model.load_state_dict(torch.load(self.saved_filepath))
        self.model = self.model.to(self.device)
        self.model.eval()
        self.h_0 = torch.zeros((1, 512), dtype=torch.float)
        self.c_0 = torch.zeros((1, 512), dtype=torch.float)


    def reset(self, t=250):
        """
        Reset is called before each episode begins
        Leave blank if nothing needs to happen there
        :param t the number of timesteps in the episode
        """
        self.h_0 = torch.zeros((1, 512), dtype=torch.float)
        self.c_0 = torch.zeros((1, 512), dtype=torch.float)

    def step(self, obs, reward, done, info):
        """
        A single step the agent should take based on the current state of the environment
        We will run the Gym environment (AnimalAIEnv) and pass the arguments returned by env.step() to
        the agent.

        Note that should if you prefer using the BrainInfo object that is usually returned by the Unity
        environment, it can be accessed from info['brain_info'].

        :param obs: agent's observation of the current environment
        :param reward: amount of reward returned after previous action
        :param done: whether the episode has ended.
        :param info: contains auxiliary diagnostic information, including BrainInfo.
        :return: the action to take, a list or size 2
        """
        self.h_0 = self.h_0.detach()
        self.c_0 = self.c_0.detach()

        self.h_0 = self.h_0.to(self.device)
        self.c_0 = self.c_0.to(self.device)

        #action_info = info[self.brain_name]
        action_info = info['brain_info']
        state = action_info.visual_observations[0]
        state = torch.from_numpy(np.moveaxis(state, -1, 1)).float().to(self.device)
        state = state.to(self.device)

        logits, value, self.h_0, self.c_0 = self.model(state, self.h_0, self.c_0)
        policy = F.softmax(logits, dim=1)
        #print(policy)
        action_idx = torch.argmax(policy).item()
        action_idx = int(action_idx)
        action = self.actions_array[action_idx]


        return action
