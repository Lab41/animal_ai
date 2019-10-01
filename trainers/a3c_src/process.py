"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""

import torch
#from src.env import create_train_env
from a3c_src.model import ActorCritic
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque
from tensorboardX import SummaryWriter
import timeit

from animalai.envs import UnityEnvironment
from animalai.envs.arena_config import ArenaConfig

from env_utils import *

#device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cpu")
actions_array = np.array([[0,0],[0,1],[0,2],[1,0],[2,0]])
brain_name = 'Learner'

def local_train(index, opt, global_model, optimizer, save=False):
    torch.manual_seed(123 + index)
    if save:
        start_time = timeit.default_timer()
    writer = SummaryWriter(opt.log_path)


    # Unity
    #env_path = '../env/AnimalAI'
    #n_arenas=1
    #env=UnityEnvironment(file_name=env_path, n_arenas=n_arenas, worker_id=np.random.randint(1,100), play=False,inference=False)



    b_env = better_env(n_arenas = 1)
    env = b_env.env
    #arena_config_in = b_env.env_config
    #start_positions, start_rotations = b_env.get_start_positions()
    #ps = position_tracker(start_positions, start_rotations)
    # end unity
    num_states = 3
    num_actions = 5

    #env, num_states, num_actions = create_train_env(opt.world, opt.stage, opt.action_type)
    local_model = ActorCritic(num_states, num_actions).to(device)
    local_model.train()



    action_info = env.reset(arenas_configurations=b_env.env_config, train_mode=True)
    state = action_info[brain_name].visual_observations[0]
    state = torch.from_numpy(np.moveaxis(state, -1, 1)).float().to(device)



    done = True
    curr_step = 0
    curr_episode = 0
    while True:
        if save:
            if curr_episode % opt.save_interval == 0 and curr_episode > 0:
                torch.save(global_model.state_dict(),
                           "{}/{}_{}".format(opt.saved_path, opt.saved_filepath, curr_episode))
            #print("Process {}. Episode {}".format(index, curr_episode))
        if curr_episode > 0:
            print("Process {}. Episode {}, total_loss = {}".format(index, curr_episode, total_loss.item()))
        curr_episode += 1
        local_model.load_state_dict(global_model.state_dict())
        if done:
            h_0 = torch.zeros((1, 512), dtype=torch.float)
            c_0 = torch.zeros((1, 512), dtype=torch.float)
        else:
            h_0 = h_0.detach()
            c_0 = c_0.detach()

        h_0 = h_0.to(device)
        c_0 = c_0.to(device)
        #if opt.use_gpu:
        #    h_0 = h_0.cuda()
        #    c_0 = c_0.cuda()

        log_policies = []
        values = []
        rewards = []
        entropies = []

        for _ in range(opt.num_local_steps):
            curr_step += 1
            logits, value, h_0, c_0 = local_model(state, h_0, c_0)
            policy = F.softmax(logits, dim=1)
            log_policy = F.log_softmax(logits, dim=1)
            entropy = -(policy * log_policy).sum(1, keepdim=True)

            m = Categorical(policy)
            action_idx = m.sample().item()

            action = actions_array[action_idx]
            #action = actions_array[action.cpu().numpy().astype(int)]
            #state, reward, done, _ = env.step(action)
            action_info = env.step(vector_action=action)


            state = action_info[brain_name].visual_observations[0]
            state = torch.from_numpy(np.moveaxis(state, -1, 1)).float().to(device)
            velocity_obs = action_info[brain_name].vector_observations
            b_env.position_tracker.position_step(velocity_obs, action)
            #print("Distance to goal = {}".format(ps.distance_to_goal()))
            #print('Current position = {}, velocity = {}'.format(ps.current_position, velocity_obs))
            reward     = action_info[brain_name].rewards # list of rewards len = n_arenas
            reward = reward[0]

            # reward based on visiting squares
            total_unvisited = np.sum(b_env.position_tracker.visited)
            reward -= total_unvisited/10000
            reward -= b_env.position_tracker.distance_to_goal()/500
            reward -= b_env.position_tracker.angle_to_goal()/1000
            #print("{} reward = {}".format(index, reward))

            arenas_done       = action_info[brain_name].local_done
            done = any(arenas_done)





            #state = torch.from_numpy(state)

            if opt.use_gpu:
                state = state.cuda()
            if curr_step > opt.num_global_steps:
                done = True

            #if curr_step > 500:
            #    done = True

            if done:
                curr_step = 0

                #b_env = better_env(n_arenas = 1)
                #arena_config_in = b_env.env_config
                #start_positions, start_rotations = b_env.get_start_positions()
                #ps = position_tracker(start_positions, start_rotations)
                b_env.generate_new_config()
                action_info = env.reset(arenas_configurations=b_env.env_config, train_mode=True)
                state = action_info[brain_name].visual_observations[0]
                state = torch.from_numpy(np.moveaxis(state, -1, 1)).float().to(device)
                #state = torch.from_numpy(env.reset())
                #if opt.use_gpu:
                #    state = state.cuda()

            values.append(value)
            log_policies.append(log_policy[0, action_idx])
            rewards.append(reward)
            entropies.append(entropy)

            if done:
                break


        R = torch.zeros((1, 1), dtype=torch.float)
        if opt.use_gpu:
            R = R.cuda()
        if not done:
            _, R, _, _ = local_model(state, h_0, c_0)

        gae = torch.zeros((1, 1), dtype=torch.float)
        if opt.use_gpu:
            gae = gae.cuda()
        actor_loss = 0
        critic_loss = 0
        entropy_loss = 0
        next_value = R

        for value, log_policy, reward, entropy in list(zip(values, log_policies, rewards, entropies))[::-1]:
            gae = gae * opt.gamma * opt.tau
            gae = gae + reward + opt.gamma * next_value.detach() - value.detach()
            next_value = value
            actor_loss = actor_loss + log_policy * gae
            R = R * opt.gamma + reward
            critic_loss = critic_loss + (R - value) ** 2 / 2
            entropy_loss = entropy_loss + entropy

        total_loss = -actor_loss + critic_loss - opt.beta * entropy_loss
        #print("Loss = {}".format(total_loss))
        writer.add_scalar("Train_{}/Loss".format(index), total_loss, curr_episode)
        optimizer.zero_grad()
        total_loss.backward()

        for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
            if global_param.grad is not None:
                break
            global_param._grad = local_param.grad

        optimizer.step()

        if curr_episode == int(opt.num_global_steps / opt.num_local_steps):
            print("Training process {} terminated".format(index))
            if save:
                end_time = timeit.default_timer()
                print('The code runs for %.2f s ' % (end_time - start_time))
            return


def local_test(index, opt, global_model):
    torch.manual_seed(123 + index)
    env, num_states, num_actions = create_train_env(opt.world, opt.stage, opt.action_type)
    local_model = ActorCritic(num_states, num_actions)
    local_model.eval()
    state = torch.from_numpy(env.reset())
    done = True
    curr_step = 0
    actions = deque(maxlen=opt.max_actions)
    while True:
        curr_step += 1
        if done:
            local_model.load_state_dict(global_model.state_dict())
        with torch.no_grad():
            if done:
                h_0 = torch.zeros((1, 512), dtype=torch.float)
                c_0 = torch.zeros((1, 512), dtype=torch.float)
            else:
                h_0 = h_0.detach()
                c_0 = c_0.detach()

        logits, value, h_0, c_0 = local_model(state, h_0, c_0)
        policy = F.softmax(logits, dim=1)
        action = torch.argmax(policy).item()
        state, reward, done, _ = env.step(action)
        env.render()
        actions.append(action)
        if curr_step > opt.num_global_steps or actions.count(actions[0]) == actions.maxlen:
            done = True
        if done:
            curr_step = 0
            actions.clear()
            state = env.reset()
        state = torch.from_numpy(state)
