"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""

import torch
#from src.env import create_train_env
from a3c_src.model import ActorCriticAutoencoder, Mapper
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque
from tensorboardX import SummaryWriter
import timeit
import matplotlib.pyplot as plt

from animalai.envs import UnityEnvironment
from animalai.envs.arena_config import ArenaConfig

from env_utils import *

from conv_autoencoder import autoencoder

#device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cpu")
actions_array = np.array([[0,0],[0,1],[0,2],[1,0],[2,0]])
brain_name = 'Learner'
map_loss_func = torch.nn.MSELoss()

def local_train(index, opt, global_model, optimizer, save=False, observation_collector=None, testing=0):
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


    autoenc_model = autoencoder().to(device)
    autoenc_model.load_state_dict(torch.load('conv_autoencoder.pth'))



    for param in autoenc_model.parameters():
        param.requires_grad = False

    #env, num_states, num_actions = create_train_env(opt.world, opt.stage, opt.action_type)
    local_model = ActorCriticAutoencoder(num_states, num_actions).to(device)
    local_model.train()

    local_mapper = Mapper(num_states).to(device)
    local_mapper.train()



    action_info = env.reset(arenas_configurations=b_env.env_config, train_mode=True)
    state = action_info[brain_name].visual_observations[0]
    state = torch.from_numpy(np.moveaxis(state, -1, 1)).float().to(device)

    print('here')


    ax1 = plt.subplot(111)
    plt.ion()

    done = True
    curr_step = 0
    curr_episode = 0
    while True:
        if save:
            if curr_episode % opt.save_interval == 0 and curr_episode > 0:
                torch.save(global_model.state_dict(),
                           "{}/{}_{}".format(opt.saved_path, opt.saved_filepath, curr_episode))
            #print("Process {}. Episode {}".format(index, curr_episode))
        if curr_episode > 0 and curr_episode % 10 == 0:
            print("Process {}. Episode {}, total_loss = {:10.6f}, map_loss = {:10.6f}".format(index, curr_episode, total_loss.item(), map_loss.item()))
        curr_episode += 1
        local_model.load_state_dict(global_model.state_dict())
        if done:
            h_0 = torch.zeros((1, 512), dtype=torch.float)
            c_0 = torch.zeros((1, 512), dtype=torch.float)

            h_0_mapper = torch.zeros((1, 512), dtype=torch.float)
            c_0_mapper = torch.zeros((1, 512), dtype=torch.float)

        else:
            h_0 = h_0.detach()
            c_0 = c_0.detach()

            h_0_mapper = h_0_mapper.detach()
            c_0_mapper = c_0_mapper.detach()

        h_0 = h_0.to(device)
        c_0 = c_0.to(device)

        h_0_mapper = h_0_mapper.to(device)
        c_0_mapper = c_0_mapper.to(device)


        log_policies = []
        values = []
        rewards = []
        entropies = []
        map_losses = []

        testing = index

        for _ in range(opt.num_local_steps):

            if opt.collect_observations and _ % 20 == 0:
                observation_collector.append(state.cpu().numpy()[0])
                if len(observation_collector) == 1000:
                    o = np.array(observation_collector)
                    np.save('observations/collected_obs_{}_{}.npy'.format(index,curr_episode),o)
                    print("saving npy in index ",index)
                    observation_collector = []

            curr_step += 1
            logits, value, h_0, c_0, mapper = local_model(autoenc_model.encoder(state), h_0, c_0)
            #pred_map, h_0_mapper, c_0_mapper = local_mapper(state, h_0_mapper, c_0_mapper)

            #target_map = b_env.position_tracker.get_map()


            # compare mapper and b_env.position_tracker.get_map()

            # grid mapper (otuput is [40,40])
            #map_loss_gradient = 0.2 * map_loss_func(mapper.detach(), torch.tensor(b_env.position_tracker.get_map(),requires_grad=True))

            # coordinate mapper (output is [2])
            map_prediction = torch.squeeze(mapper.detach())

            # coordinate gaussian (output is [2,2])
            map_prediction = map_prediction.view(2, 2)
            x_dist = torch.distributions.normal.Normal(map_prediction[0][0], map_prediction[0][1])
            y_dist = torch.distributions.normal.Normal(map_prediction[1][0], map_prediction[1][1])
            map_prediction = torch.tensor([x_dist.sample(), y_dist.sample()])
            #print(torch.tensor(b_env.position_tracker.good_goal_start[0][[0,2]]/40 - 0.5, requires_grad=True),',',map_prediction)


            #map_loss_gradient = 0.01 * map_loss_func(map_prediction, torch.tensor(b_env.position_tracker.good_goal_start[0][[0,2]]/40 - 0.5, requires_grad=True))
            map_loss_gradient = map_loss_func(map_prediction, torch.tensor(b_env.position_tracker.agent_goal_vec()[0][[0,2]]/40 - 0.5, requires_grad=True))

            map_losses.append(map_loss_gradient)
            #map_loss_gradient.backward()




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
            #total_unvisited = np.sum(b_env.position_tracker.visited)
            #reward -= total_unvisited/10000
            reward -= b_env.position_tracker.distance_to_goal()/500
            #reward -= b_env.position_tracker.angle_to_goal()/1000
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
            _, R, _, _, _ = local_model(autoenc_model.encoder(state), h_0, c_0)

        gae = torch.zeros((1, 1), dtype=torch.float)
        if opt.use_gpu:
            gae = gae.cuda()
        actor_loss = 0
        critic_loss = 0
        entropy_loss = 0
        map_loss = 0
        next_value = R

        for value, log_policy, reward, entropy, map_l in list(zip(values, log_policies, rewards, entropies, map_losses))[::-1]:
            gae = gae * opt.gamma * opt.tau
            gae = gae + reward + opt.gamma * next_value.detach() - value.detach()
            next_value = value
            actor_loss = actor_loss + log_policy * gae
            R = R * opt.gamma + reward
            critic_loss = critic_loss + (R - value) ** 2 / 2
            entropy_loss = entropy_loss + entropy
            map_loss = map_loss * opt.gamma + map_l

        value_weight = 1
        total_loss = -actor_loss + value_weight * critic_loss - opt.beta * entropy_loss
        #print("Loss = {}".format(total_loss))
        writer.add_scalar("Train_{}/Loss".format(index), total_loss, curr_episode)
        optimizer.zero_grad()
        total_loss.backward()
        map_loss.backward()

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
    local_model = ActorCriticAutoencoder(num_states, num_actions)
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
