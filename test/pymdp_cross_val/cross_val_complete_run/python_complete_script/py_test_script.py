import os
import sys
import pathlib
import numpy as np

from pymdp.agent import Agent
from pymdp import utils, maths

grid_dims = [5, 7] # dimensions of the grid (number of rows, number of columns)
num_grid_points = np.prod(grid_dims) # total number of grid locations (rows X columns)

# create a look-up table `loc_list` that maps linear indices to tuples of (y, x) coordinates 
grid = np.arange(num_grid_points).reshape(grid_dims)
it = np.nditer(grid, flags=["multi_index"])

loc_list = []
while not it.finished:
    loc_list.append(it.multi_index)
    it.iternext()

# (y, x) coordinate of the first cue's location, and then a list of the (y, x) coordinates of the possible locations of the second cue, and their labels (`L1`, `L2`, ...)
cue1_location = (2, 0)

cue2_loc_names = ['L1', 'L2', 'L3', 'L4']
cue2_locations = [(0, 2), (1, 3), (3, 3), (4, 2)]

# names of the reward conditions and their locations
reward_conditions = ["TOP", "BOTTOM"]
reward_locations = [(1, 5), (3, 5)]

# list of dimensionalities of the hidden states -- useful for creating generative model later on
num_states = [num_grid_points, len(cue2_locations), len(reward_conditions)]

# Names of the cue1 observation levels, the cue2 observation levels, and the reward observation levels
cue1_names = ['Null'] + cue2_loc_names # signals for the possible Cue 2 locations, that only are seen when agent is visiting Cue 1
cue2_names = ['Null', 'reward_on_top', 'reward_on_bottom']
reward_names = ['Null', 'Cheese', 'Shock']

num_obs = [num_grid_points, len(cue1_names), len(cue2_names), len(reward_names)]

A_m_shapes = [ [o_dim] + num_states for o_dim in num_obs] # list of shapes of modality-specific A[m] arrays
A = utils.obj_array_zeros(A_m_shapes) # initialize A array to an object array of all-zero subarrays

# make the location observation only depend on the location state (proprioceptive observation modality)
A[0] = np.tile(np.expand_dims(np.eye(num_grid_points), (-2, -1)), (1, 1, num_states[1], num_states[2]))

# make the cue1 observation depend on the location (being at cue1_location) and the true location of cue2
A[1][0,:,:,:] = 1.0 # default makes Null the most likely observation everywhere

# Make the Cue 1 signal depend on 1) being at the Cue 1 location and 2) the location of Cue 2
for i, cue_loc2_i in enumerate(cue2_locations):
    A[1][0,loc_list.index(cue1_location),i,:] = 0.0
    A[1][i+1,loc_list.index(cue1_location),i,:] = 1.0
    
# make the cue2 observation depend on the location (being at the correct cue2_location) and the reward condition
A[2][0,:,:,:] = 1.0 # default makes Null the most likely observation everywhere

for i, cue_loc2_i in enumerate(cue2_locations):

    # if the cue2-location is the one you're currently at, then you get a signal about where the reward is
    A[2][0,loc_list.index(cue_loc2_i),i,:] = 0.0 
    A[2][1,loc_list.index(cue_loc2_i),i,0] = 1.0
    A[2][2,loc_list.index(cue_loc2_i),i,1] = 1.0

# make the reward observation depend on the location (being at reward location) and the reward condition
A[3][0,:,:,:] = 1.0 # default makes Null the most likely observation everywhere

rew_top_idx = loc_list.index(reward_locations[0]) # linear index of the location of the "TOP" reward location
rew_bott_idx = loc_list.index(reward_locations[1]) # linear index of the location of the "BOTTOM" reward location

# fill out the contingencies when the agent is in the "TOP" reward location
A[3][0,rew_top_idx,:,:] = 0.0
A[3][1,rew_top_idx,:,0] = 1.0
A[3][2,rew_top_idx,:,1] = 1.0

# fill out the contingencies when the agent is in the "BOTTOM" reward location
A[3][0,rew_bott_idx,:,:] = 0.0
A[3][1,rew_bott_idx,:,1] = 1.0
A[3][2,rew_bott_idx,:,0] = 1.0

# initialize `num_controls`
num_controls = [5, 1, 1]

# initialize the shapes of each sub-array `B[f]`
B_f_shapes = [ [ns, ns, num_controls[f]] for f, ns in enumerate(num_states)]

# create the `B` array and fill it out
B = utils.obj_array_zeros(B_f_shapes)

actions = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]

# fill out `B[0]` using the 
for action_id, action_label in enumerate(actions):

  for curr_state, grid_location in enumerate(loc_list):

    y, x = grid_location

    if action_label == "UP":
      next_y = y - 1 if y > 0 else y 
      next_x = x
    elif action_label == "DOWN":
      next_y = y + 1 if y < (grid_dims[0]-1) else y 
      next_x = x
    elif action_label == "LEFT":
      next_x = x - 1 if x > 0 else x 
      next_y = y
    elif action_label == "RIGHT":
      next_x = x + 1 if x < (grid_dims[1]-1) else x 
      next_y = y
    elif action_label == "STAY":
      next_x = x
      next_y = y

    new_location = (next_y, next_x)
    next_state = loc_list.index(new_location)
    B[0][next_state, curr_state, action_id] = 1.0

B[1][:,:,0] = np.eye(num_states[1])
B[2][:,:,0] = np.eye(num_states[2])

C = utils.obj_array_zeros(num_obs)

C[3][1] = 2.0 # make the agent want to encounter the "Cheese" observation level
C[3][2] = -4.0 # make the agent not want to encounter the "Shock" observation level

D = utils.obj_array_uniform(num_states)
D[0] = utils.onehot(loc_list.index((0,0)), num_grid_points)

class GridWorldEnv():
    
    def __init__(self,starting_loc = (0,0), cue1_loc = (2, 0), cue2 = 'L1', reward_condition = 'TOP'):

        self.init_loc = starting_loc
        self.current_location = self.init_loc

        self.cue1_loc = cue1_loc
        self.cue2_name = cue2
        self.cue2_loc_names = ['L1', 'L2', 'L3', 'L4']
        self.cue2_loc = cue2_locations[self.cue2_loc_names.index(self.cue2_name)]

        self.reward_condition = reward_condition
        print(f'Starting location is {self.init_loc}, Reward condition is {self.reward_condition}, cue is located in {self.cue2_name}')
    
    def step(self,action_label):

        (Y, X) = self.current_location

        if action_label == "UP": 
          
          Y_new = Y - 1 if Y > 0 else Y
          X_new = X

        elif action_label == "DOWN": 

          Y_new = Y + 1 if Y < (grid_dims[0]-1) else Y
          X_new = X

        elif action_label == "LEFT": 
          Y_new = Y
          X_new = X - 1 if X > 0 else X

        elif action_label == "RIGHT": 
          Y_new = Y
          X_new = X +1 if X < (grid_dims[1]-1) else X

        elif action_label == "STAY":
          Y_new, X_new = Y, X 
        
        self.current_location = (Y_new, X_new) # store the new grid location

        loc_obs = self.current_location # agent always directly observes the grid location they're in 

        if self.current_location == self.cue1_loc:
          cue1_obs = self.cue2_name
        else:
          cue1_obs = 'Null'

        if self.current_location == self.cue2_loc:
          cue2_obs = cue2_names[reward_conditions.index(self.reward_condition)+1]
        else:
          cue2_obs = 'Null'
        
        # @NOTE: here we use the same variable `reward_locations` to create both the agent's generative model (the `A` matrix) as well as the generative process. 
        # This is just for simplicity, but it's not necessary -  you could have the agent believe that the Cheese/Shock are actually stored in arbitrary, incorrect locations.

        if self.current_location == reward_locations[0]:
          if self.reward_condition == 'TOP':
            reward_obs = 'Cheese'
          else:
            reward_obs = 'Shock'
        elif self.current_location == reward_locations[1]:
          if self.reward_condition == 'BOTTOM':
            reward_obs = 'Cheese'
          else:
            reward_obs = 'Shock'
        else:
          reward_obs = 'Null'

        return loc_obs, cue1_obs, cue2_obs, reward_obs

    def reset(self):
        self.current_location = self.init_loc
        print(f'Re-initialized location to {self.init_loc}')
        loc_obs = self.current_location
        cue1_obs = 'Null'
        cue2_obs = 'Null'
        reward_obs = 'Null'

        return loc_obs, cue1_obs, cue2_obs, reward_obs

my_agent = Agent(A = A, B = B, C = C, D = D, policy_len = 4, use_param_info_gain = True)

my_env = GridWorldEnv(starting_loc = (0,0), cue1_loc = (2, 0), cue2 = 'L4', reward_condition = 'BOTTOM')

loc_obs, cue1_obs, cue2_obs, reward_obs = my_env.reset()

history_of_locs = [loc_obs]
obs = [loc_list.index(loc_obs), cue1_names.index(cue1_obs), cue2_names.index(cue2_obs), reward_names.index(reward_obs)]

T = 10 # number of total timesteps

for t in range(T):

    qs = my_agent.infer_states(obs)
    
    my_agent.infer_policies()
    chosen_action_id = my_agent.sample_action()

    movement_id = int(chosen_action_id[0])

    choice_action = actions[movement_id]

    print(f'Action at time {t}: {choice_action}')

    loc_obs, cue1_obs, cue2_obs, reward_obs = my_env.step(choice_action)

    obs = [loc_list.index(loc_obs), cue1_names.index(cue1_obs), cue2_names.index(cue2_obs), reward_names.index(reward_obs)]

    history_of_locs.append(loc_obs)

    print(f'Grid location at time {t}: {loc_obs}')

    print(f'Reward at time {t}: {reward_obs}')


