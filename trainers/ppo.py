import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import time
import argparse
import os
import datetime

from animalai.envs import UnityEnvironment
from animalai.envs.arena_config import ArenaConfig


from env_utils import *

parser = argparse.ArgumentParser(description="Train ppo agent for AnimalAI.")
parser.add_argument('--train_name', type=str, help='Will save model with this name. Default: random')
parser.add_argument('--config', type=str, default='configs/1-Food.yaml', help='Environment config file. Default: "configs/1-Food.yaml"')
parser.add_argument('--load_model', type=str, default='saved_models/ppo.pth', help='Saved model to load. Default: "saved_models/ppo.pth"')
parser.add_argument('--inference', default=False, action='store_true', help='Run in inference mode. Default: False')

args = parser.parse_args()


if not args.inference:
    if args.train_name is not None:
        train_filename = '{}.pth'.format(args.train_name)
    else:
        train_filename = 'ppo_{}.pth'.format(np.random.randint(100000,999999))
# my params
env_path = '../env/AnimalAI'
brain_name = 'Learner'
train_mode = True
num_actions = 9
color_channels = 3
env_field = args.config
n_episodes = 20000
#max_t = 100
actions_array = np.array([[0,0],[0,1],[0,2],[1,0], [1,1],[1,2], [2,0],[2,1],[2,2]])
n_arenas = 3
print_interval = 1
save_interval = 10
save_path = 'saved_models/'

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#Hyperparameters
learning_rate = 0.0005
gamma         = 0.98
lmbda         = 0.95
eps_clip      = 0.1
K_epoch       = 4
T_horizon     = 500



class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []

        self.conv1 = nn.Conv2d(color_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc_pi = nn.Linear(512, num_actions)
        self.fc_v = nn.Linear(512, 1)

        #self.fc1   = nn.Linear(4,256)
        #self.fc_pi = nn.Linear(256,2)
        #self.fc_v  = nn.Linear(256,1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, softmax_dim = 1):
        #x = x.permute(2,0,1)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        #x = x.transpose(1,3)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        x = self.fc_pi(x)

        #x = F.relu(self.fc1(x))
        #x = self.fc_pi(x)

        prob = F.softmax(x, dim=softmax_dim)


        return prob

    def v(self, x):

        #x = x.transpose(1,3)
        #print(x.shape)
        #x = x.permute(2,0,1)
        #x = x.unsqueeze(0)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))

        #x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v

    def put_data(self, transition):

        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition


            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        s,a,r,s_prime,done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                          torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                          torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)

        prob_a = prob_a.to(device)
        a = a.to(device)
        s_prime = s_prime.to(device)
        r = r.to(device)
        done_mask = done_mask.to(device)
        s = s.to(device)

        self.data = []
        return s, a, r, s_prime, done_mask, prob_a

    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()


        for i in range(K_epoch):
            td_target = r + gamma * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
            delta = delta.detach().cpu().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float).to(device)

            pi = self.pi(s, softmax_dim=1)


            pi_a = pi.gather(1,a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s) , td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

def train():
    env=UnityEnvironment(file_name=env_path, n_arenas=n_arenas, worker_id=np.random.randint(1,100), inference=args.inference)
    arena_config_in = ArenaConfig(env_field)
    #print(arena_config_in.arenas)


    model = PPO()
    if os.path.exists(args.load_model):
        model.load_state_dict(torch.load(args.load_model))
        print("Successfully loaded saved model from {}".format(args.load_model))

    model = model.to(device)


    total_obs = 0

    for n_epi in range(1, n_episodes+1):
        action_info = env.reset(arenas_configurations=arena_config_in, train_mode=train_mode)
        state = action_info[brain_name].visual_observations[0]

        #state = np.moveaxis(state, -1, 0)
        state = np.moveaxis(state, -1, 1)
        done = False
        score = 0.0
        scores = []

        start_episode = time.time()
        n_obs = 0
        while not done:
            for t in range(T_horizon):
                n_obs += n_arenas

                prob = model.pi(torch.from_numpy(state).float().to(device))
                m = Categorical(prob)

                #a = m.sample().item()
                a = m.sample()
                action = actions_array[a.cpu().numpy().astype(int)]
                #s_prime, reward, done, info =
                action_info = env.step(vector_action=action)
                next_state = action_info[brain_name].visual_observations[0]
                velocity_obs = action_info[brain_name].vector_observations
                print(velocity_obs)
                asdf
                #next_state = np.moveaxis(next_state, -1, 0)
                next_state = np.moveaxis(next_state, -1, 1) # next state shape = [n_arenas, 3, 84, 84]
                reward     = action_info[brain_name].rewards # list of rewards len = n_arenas
                arenas_done       = action_info[brain_name].local_done
                done = any(arenas_done)

                prob_a = prob[np.arange(prob.shape[0])[:,None], a.cpu().numpy().astype(int)[:,None]]

                for (s, a, r, n_s, p_a, d) in zip (state, a, reward, next_state, prob_a, arenas_done):
                    model.put_data((s, a, r, n_s, p_a, d))
                    scores.append(r)
                #model.put_data((state, a, reward, next_state, prob[0][a].item(), done))
                #model.put_data((state, a, reward, next_state, prob_a, done))
                state = next_state

                #score += reward
                if done:
                    break

            start_train = time.time()
            model.train_net()
            end_train = time.time()
            #print('time to train: ',end_train - start_train)

        end_episode = time.time()

        #print('{} observations/second'.format(n_obs/(end_episode - start_episode)))

        #scores.append(score)

        if n_epi%print_interval==0 and n_epi!=0:
            print("Episode: {}, avg score: {:.4f}, [{:.0f}] observations/second".format(n_epi, np.mean(scores)/n_arenas, n_obs/(end_episode - start_episode)))

        if n_epi%save_interval==0 and n_epi!=0:
            print("Saving model to {}ppo.pth at {}".format(save_path, datetime.datetime.now()))
            torch.save(model.state_dict(), save_path+train_filename)


    env.close()

def env_info(env_config):

    for i, arena in env_config.arenas.items():
        print("Arena Config #{}".format(i))
        print("max time steps = {}".format(arena.t))
        for j, item in enumerate(arena.items):
            print("Item name: {}".format(item.name))
            print("Item positions: {}".format(item.positions))
            print("Item rotations: {}".format(item.rotations))
            print("Item sizes: {}".format(item.sizes))
            print("Item colors: {}".format(item.colors))

def inference():
    env=UnityEnvironment(file_name=env_path, n_arenas=n_arenas, worker_id=np.random.randint(1,100), play=False,inference=args.inference)
    #arena_config_in = ArenaConfig(env_field)


    b_env = better_env(n_arenas = 1)
    arena_config_in = b_env.env_config
    start_positions, start_rotations = b_env.get_start_positions()
    ps = position_tracker(start_positions, start_rotations)


    model = PPO()
    if os.path.exists(args.load_model):
        model.load_state_dict(torch.load(args.load_model))
        print("Successfully loaded saved model from {}".format(args.load_model))

    model = model.to(device)


    total_obs = 0

    for n_epi in range(1, n_episodes+1):

        action_info = env.reset(arenas_configurations=arena_config_in, train_mode=False)
        state = action_info[brain_name].visual_observations[0]

        state = np.moveaxis(state, -1, 1)
        done = False
        score = 0.0

        start_episode = time.time()
        n_obs = 0
        action = [[0,1]]
        action_info = env.step(vector_action=action)
        velocity_obs = action_info[brain_name].vector_observations
        ps.position_step(velocity_obs, action)
        action_info = env.step(vector_action=action)
        velocity_obs = action_info[brain_name].vector_observations
        ps.position_step(velocity_obs, action)
        action_info = env.step(vector_action=action)
        velocity_obs = action_info[brain_name].vector_observations
        ps.position_step(velocity_obs, action)
        action_info = env.step(vector_action=action)
        velocity_obs = action_info[brain_name].vector_observations
        ps.position_step(velocity_obs, action)

        while not done:
            for t in range(T_horizon):
                n_obs += n_arenas

                prob = model.pi(torch.from_numpy(state).float().to(device))
                m = Categorical(prob)

                a = m.sample()
                #action = actions_array[a.cpu().numpy().astype(int)]
                #if np.random.randint(0,2):
                #    action = [0,1]
                #else:
                #    action = [0,2]
                action_info = env.step(vector_action=action)
                action = [[1,0]]
                next_state = action_info[brain_name].visual_observations[0]
                velocity_obs = action_info[brain_name].vector_observations

                ps.position_step(velocity_obs, action)
                print('Current position = {}, velocity = {}'.format(ps.current_position, velocity_obs))

                next_state = np.moveaxis(next_state, -1, 1) # next state shape = [n_arenas, 3, 84, 84]
                reward     = action_info[brain_name].rewards # list of rewards len = n_arenas
                arenas_done       = action_info[brain_name].local_done
                done = any(arenas_done)


                state = next_state

                score += reward[0]
                if done:
                    break


        end_episode = time.time()


        if n_epi%print_interval==0 and n_epi!=0:
            print("Episode: {}, avg score: {:.4f}, [{:.0f}] observations/second".format(n_epi, score/n_obs, n_obs/(end_episode - start_episode)))



    env.close()
if __name__ == '__main__':

    if not args.inference:
        print("Starting agent in train mode...")
        train()
    else:
        print("Starting agent in inference mode...")
        inference()
