from animalai.envs.arena_config import Vector3, RGB, Item, Arena, ArenaConfig
from collections import defaultdict
import numpy as np
import pprint


pp = pprint.PrettyPrinter(indent=4)



class position_tracker():

    def __init__(self, starting_positions, starting_rotations):

        self.agent_start = starting_positions['Agent']
        self.good_goal_start = starting_positions['GoodGoal']


        self.current_position = np.array(self.agent_start).astype('float64')

        self.current_rotation = np.array(starting_rotations['Agent']).astype('float64')


    def position_step(self, velocity_obs, action):


        action = np.array(action)
        self.current_rotation[np.where(action[:,1] == 1)] -= 6
        self.current_rotation[np.where(action[:,1] == 2)] += 6

        rot_mat = get_rot_mat(deg_to_rad(self.current_rotation[0][0]))

        velocity_obs = np.dot(rot_mat, np.array(velocity_obs).T).T
        delta_distance = 0.0595 * velocity_obs

        self.current_position += delta_distance


def deg_to_rad(deg):
    return deg * (np.pi/180)

def get_rot_mat(rad):
    return np.array([[np.cos(rad), 0, -np.sin(rad)],[0, 1, 0],[np.sin(rad), 0, np.cos(rad)]])



class better_env():

    def __init__(self, n_arenas=3):

        self.n_arenas = n_arenas
        #self.details = {}
        self.env_config = self.create_env(n_arenas=n_arenas)
        self.details = self.get_details()


    def create_env(self, n_arenas=3):

        include_items = {'Agent':1}#, 'GoodGoal':1, 'Wall':2}


        env_config = ArenaConfig()

        # Loop over arenas
        for i in range(n_arenas):
            env_config.arenas[i] = Arena()

            #self.details[i] = {}


            item_list = []
            # Loop over item types in each arena
            for item_type, item_count in include_items.items():

                #self.details[i][item_type] = defaultdict(list)

                name = item_type
                colors = []
                positions = []
                rotations = []

                # Loop over item counts
                for j in range(item_count):
                    if item_type == 'Wall':
                        colors.append(RGB(r=153, g=153, b=153))
                        #self.details[i][item_type]['colors'].append((153,153,153))


                    elif item_type == 'GoodGoal':
                        x = np.random.randint(1,39)
                        y = np.random.randint(1,39)
                        z = np.random.randint(1,39)
                        #self.details[i][item_type]['positions'].append((x,y,z))

                        positions.append(Vector3(x=x, y=y, z=z))

                    elif item_type == 'Agent':
                        #x = np.random.randint(1,39)
                        #y = np.random.randint(1,39)
                        #z = np.random.randint(1,39)
                        x = 0.5
                        y = 0.5
                        z = 0.5
                        #self.details[i][item_type]['positions'].append((x,y,z))

                        positions.append(Vector3(x=x, y=y, z=z))
                        rotations.append(0)

                item_list.append(Item(name=name, positions=positions, rotations=rotations, colors=colors))
            env_config.arenas[i].items = item_list

        return env_config

    def get_details(self):

        details = {}

        for i, arena in self.env_config.arenas.items():
            details[i] = {}

            for j, item in enumerate(arena.items):
                details[i][item.name] = {}
                details[i][item.name]['positions'] = []
                details[i][item.name]['rotations'] = []
                details[i][item.name]['sizes'] = []
                details[i][item.name]['colors'] = []

                for position in item.positions:
                    details[i][item.name]['positions'].append((position.x, position.y, position.z))
                for rotation in item.rotations:
                    details[i][item.name]['rotations'].append(rotation)
                for size in item.sizes:
                    details[i][item.name]['sizes'].append((size.x, size.y, size.z))
                for color in item.colors:
                    details[i][item.name]['colors'].append((color.r, color.g, color.b))

        return details

    def get_start_positions(self):

        start_positions = {'Agent': [], 'GoodGoal': []}
        start_rotations = {'Agent':[]}

        for arena_idx, arena in self.env_config.arenas.items():

            for item_idx, item in enumerate(arena.items):
                if item.name == 'Agent' or item.name == 'GoodGoal':
                    for position in item.positions:
                        start_positions[item.name].append([position.x, position.y, position.z])
                if item.name == 'Agent':
                    for rotation in item.rotations:
                        start_rotations[item.name].append([rotation])

        return start_positions, start_rotations





def env_info(env_config):

    for i, arena in env_config.arenas.items():
        print("Arena Config #{}".format(i))
        print("max time steps = {}".format(arena.t))
        for j, item in enumerate(arena.items):
            print("{:4s}Item name: {}".format('',item.name))
            print("{:8s}Item positions: {}".format('',item.positions))
            print("{:8s}Item rotations: {}".format('',item.rotations))
            print("{:8s}Item sizes: {}".format('',item.sizes))
            print("{:8s}Item colors: {}".format('',item.colors))

env = better_env()
env_config = env.env_config
env_info(env_config)
pp.pprint(env.details)
#pp.pprint(env.details2)

pp.pprint(env.get_start_positions())

start_pos, start_rot = env.get_start_positions()
ps = position_tracker(start_pos, start_rot)
print(ps.current_position)
