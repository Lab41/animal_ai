"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""

import os

os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import torch
from a3c_src.model import ActorCriticAutoencoder
import torch.nn.functional as F
import numpy as np
from animalai.envs import UnityEnvironment
from animalai.envs.arena_config import ArenaConfig
from conv_autoencoder import autoencoder


from env_utils import *

import matplotlib.pyplot as plt


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of model described in the paper: Asynchronous Methods for Deep Reinforcement Learning for Super Mario Bros""")
    parser.add_argument("--saved_filepath", type=str, default="trained_models/a3c_animalai")
    args = parser.parse_args()
    return args


def test(opt):


    # AnimalAI
    device = torch.device("cpu")
    num_states = 3
    num_actions = 5
    actions_array = np.array([[0,0],[0,1],[0,2],[1,0],[2,0]])
    brain_name = 'Learner'
    # AnimalAI

    autoenc_model = autoencoder()
    autoenc_model.load_state_dict(torch.load('conv_autoencoder.pth'))

    for param in autoenc_model.parameters():
        param.requires_grad = False


    torch.manual_seed(123)


    #env=UnityEnvironment(file_name='../env/AnimalAI', n_arenas=1, worker_id=np.random.randint(1,100), play=False,inference=True)
    b_env = better_env(n_arenas = 1, walls=2, t=200, inference=True)
    env = b_env.env
    #arena_config_in = b_env.env_config
    #start_positions, start_rotations = b_env.get_start_positions()
    #ps = position_tracker(start_positions, start_rotations)




    model = ActorCriticAutoencoder(num_states, num_actions)

    basepath = opt.saved_filepath.split('/')[0]
    basename = opt.saved_filepath.split('/')[1]

    found_models =  [int(filenames.split('_')[-1]) for filenames in os.listdir(basepath) if basename in filenames]
    if len(found_models) > 0:
        latest = max(found_models)
        model.load_state_dict(torch.load("{}_{}".format(opt.saved_filepath, latest)))
        model = model.to(device)
        print("Loaded saved model from {}_{}".format(opt.saved_filepath, latest))
    else:
        print("Could not find model to load.")
        raise


    '''
    if torch.cuda.is_available():
        model.load_state_dict(torch.load("{}/a3c_super_mario_bros_{}_{}".format(opt.saved_path, opt.world, opt.stage)))
        model.cuda()
    else:
        model.load_state_dict(torch.load("{}/a3c_super_mario_bros_{}_{}".format(opt.saved_path, opt.world, opt.stage),
                                         map_location=lambda storage, loc: storage))
    '''

    model.eval()

    action_info = env.reset(arenas_configurations=b_env.env_config, train_mode=False)
    state = action_info[brain_name].visual_observations[0]
    #ax1 = plt.subplot(111)
    #im1 = ax1.imshow(state[0])
    #plt.ion()


    ax1 = plt.subplot(111)
    im1 = ax1.imshow(np.zeros((40,40)), cmap='gray', vmin=-1, vmax=1)
    plt.ion()

    state = torch.from_numpy(np.moveaxis(state, -1, 1)).float().to(device)
    done = True



    while True:
        if done:
            h_0 = torch.zeros((1, 512), dtype=torch.float)
            c_0 = torch.zeros((1, 512), dtype=torch.float)
            #b_env = better_env(n_arenas = 1)
            #arena_config_in = b_env.env_config
            b_env.generate_new_config()
            action_info = env.reset(arenas_configurations=b_env.env_config, train_mode=False)
            state = action_info[brain_name].visual_observations[0]
            state = torch.from_numpy(np.moveaxis(state, -1, 1)).float().to(device)
        else:
            h_0 = h_0.detach()
            c_0 = c_0.detach()

        h_0 = h_0.to(device)
        c_0 = c_0.to(device)
        state = state.to(device)

        logits, value, h_0, c_0, mapper  = model(autoenc_model.encoder(state), h_0, c_0)
        policy = F.softmax(logits, dim=1)

        action_idx = torch.argmax(policy).item()
        action_idx = int(action_idx)
        action = actions_array[action_idx]
        action_info = env.step(vector_action=action)

        state = action_info[brain_name].visual_observations[0]
        #im1.set_data(state[0])
        #plt.pause(0.01)

        print(mapper.detach().cpu().numpy())
        print(b_env.position_tracker.good_goal_start[0][[0,2]])
        im1.set_data(mapper.detach().cpu().numpy())
        plt.pause(0.01)
        state = torch.from_numpy(np.moveaxis(state, -1, 1)).float().to(device)

        velocity_obs = action_info[brain_name].vector_observations
        b_env.position_tracker.position_step(velocity_obs, action)


        #print("{}__{}".format(b_env.position_tracker.current_rotation,b_env.position_tracker.angle_to_goal()))
        #print("Current position = {}".format(b_env.position_tracker.current_position))


        arenas_done       = action_info[brain_name].local_done
        done = any(arenas_done)



if __name__ == "__main__":
    opt = get_args()
    test(opt)
