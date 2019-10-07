"""
modified by: Lucas Tindall
@author: Viet Nguyen <nhviet1009@gmail.com>
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import torch
#from src.env import create_train_env
from a3c_src.model import ActorCritic
from a3c_src.optimizer import GlobalAdam
from a3c_src.process import local_train, local_test
import torch.multiprocessing as _mp
import shutil

from animalai.envs import UnityEnvironment
from animalai.envs.arena_config import ArenaConfig


from env_utils import *

def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of model described in the paper: Asynchronous Methods for Deep Reinforcement Learning for Super Mario Bros""")
    #parser.add_argument('--config', type=str, default='configs/1-Food.yaml', help='Environment config file. Default: "configs/1-Food.yaml"')
    #parser.add_argument('--load_model', type=str, default='saved_models/ppo.pth', help='Saved model to load. Default: "saved_models/ppo.pth"')
    #parser.add_argument('--inference', default=False, action='store_true', help='Run in inference mode. Default: False')
    #parser.add_argument("--world", type=int, default=1)
    #parser.add_argument("--stage", type=int, default=1)
    #parser.add_argument("--action_type", type=str, default="complex")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.9, help='discount factor for rewards')
    parser.add_argument('--tau', type=float, default=1.0, help='parameter for GAE')
    parser.add_argument('--beta', type=float, default=0.01, help='entropy coefficient')
    parser.add_argument("--num_local_steps", type=int, default=50)
    parser.add_argument("--num_global_steps", type=int, default=5e6)
    parser.add_argument("--num_processes", type=int, default=4)
    parser.add_argument("--save_interval", type=int, default=500, help="Number of steps between savings")
    parser.add_argument("--max_actions", type=int, default=200, help="Maximum repetition steps in test phase")
    parser.add_argument("--log_path", type=str, default="tensorboard/a3c_super_mario_bros")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--saved_filepath", type=str, default="a3c_animalai")
    parser.add_argument("--load_model", type=str, default="")
    #parser.add_argument("--load_from_previous_stage", type=bool, default=False,
    #                    help="Load weight from previous trained stage")
    parser.add_argument("--use_gpu", type=bool, default=False)
    args = parser.parse_args()
    return args


def train(opt):
    torch.manual_seed(123)
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    if not os.path.isdir(opt.saved_path):
        os.makedirs(opt.saved_path)
    mp = _mp.get_context("spawn")
    #mp = _mp.get_context("fork")

    num_states = 3
    num_actions = 5
    global_model = ActorCritic(num_states, num_actions)

    if opt.use_gpu:
        global_model.cuda()
    global_model.share_memory()

    if os.path.isfile("{}/{}".format(opt.saved_path, opt.load_model)):
        print("loaded global model from {}/{}".format(opt.saved_path, opt.load_model))
        global_model.load_state_dict(torch.load("{}/{}".format(opt.saved_path, opt.load_model)))

    optimizer = GlobalAdam(global_model.parameters(), lr=opt.lr)
    processes = []
    for index in range(opt.num_processes):
        print("local train {}".format(index))
        if index == 0:
            process = mp.Process(target=local_train, args=(index, opt, global_model, optimizer, True))
        else:
            process = mp.Process(target=local_train, args=(index, opt, global_model, optimizer))
        process.start()
        processes.append(process)
    #process = mp.Process(target=local_test, args=(opt.num_processes, opt, global_model))
    #process.start()
    #processes.append(process)
    for process in processes:
        process.join()


if __name__ == "__main__":
    opt = get_args()
    train(opt)
