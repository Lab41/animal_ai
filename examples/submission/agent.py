import argparse
import torch
#from aaio.data.a3c_src.model import ActorCritic
from data.a3c_src.model import ActorCritic
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

        #self.saved_filepath = '/aaio/data/trained_models/a3c_base'
        self.saved_filepath = '/data/trained_models/a3c_base'
        

        self.model = ActorCritic(num_states, num_actions)



        pass

    def reset(self, t=250):
        """
        Reset is called before each episode begins
        Leave blank if nothing needs to happen there
        :param t the number of timesteps in the episode
        """

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
        action = [0, 0]

        return action
