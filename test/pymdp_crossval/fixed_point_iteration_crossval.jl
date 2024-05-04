using Pkg
using PyCall
using LinearAlgebra
using ActiveInference

#Pkg.develop(path = raw"C:\Users\jonat\Desktop\University\Exam\4_Semester\Continued_ActiveInference\Dev_Branch\ActiveInference.jl")

ENV["PYTHON"] = raw"C:\Users\jonat\miniconda3\envs\ActiveInferenceEnv\python.exe"
Pkg.build("PyCall")

pyversion = pyimport("sys").version
pyexecutable = pyimport("sys").executable
println("Using Python version: $pyversion at $pyexecutable")

pymdp = PyCall.pyimport("pymdp")


pymdp.utils.onehot(1, 5)

########################################################## Julia Results ##########################################################

@pyimport numpy as np
@pyimport pymdp.utils as utils
@pyimport pymdp.maths as maths
@pyimport pymdp.learning as learning
@pyimport pymdp.control as control
@pyimport pymdp.inference as inference

@pyimport pymdp.agent as pymdp_agent
Agent_py = pymdp_agent.Agent



# Copying the A-matrix
A_py = utils.obj_array(length(A))

for i in 1:length(A)
    A_py[i] = np.array(A[i])
end
A_py

# Copying the B-matrix

B_py = utils.obj_array(length(B)) 
for i in 1:length(B)
    B_py[i] = np.array(B[i])
end

B_py

# Checking if the arrays are as they are supposed to be
pyeval("A_py[0]", A_py=A_py)
pyeval("B_py[0]", B_py=B_py)

# Convert lists to a structured format that pymdp can understand
A_structured = pyeval("np.array(obj)", obj=A_py)
B_structured = pyeval("np.array(obj)", obj=B_py)

typeof(A_py)

first_A = A_py[1]

println(pytypeof(first_A))


my_py_agent = Agent_py(A_py, B_py)

EA = np.empty(4)


A_py = np.array(A_py)













########################################################## Julia Results ##########################################################

##################################
### Creating a Random A-Matrix ###
##################################

grid_locations = collect(Iterators.product(1:5, 1:7))

grid_dims = size(grid_locations)

#plot_gridworld(grid_locations)

n_grid_points = prod(grid_dims)
location_to_index = Dict(loc => idx for (idx, loc) in enumerate(grid_locations))

cue1_location = (3,1)

cue2_loc_names = ["L1","L2","L3","L4"]
cue2_locations = [(1, 3), (2, 4), (4, 4), (5, 3)]

reward_conditions = ["TOP", "BOTTOM"]
reward_locations = [(2,6), (4,6)]

#------------------------------------------------------------------------------------------------------------------
#= ----Generative Model---- =#

#Hidden States "s" = 3-state factors 
n_states = [n_grid_points, length(cue2_locations), length(reward_conditions)]

#Observations "o" = 4-observation modalities
cue1_names = ["Null";cue2_loc_names]
cue2_names = ["Null", "reward_on_top", "reward_on_bottom"]
reward_names = ["Null", "Cheese", "Shock"]
n_obs = [n_grid_points, length(cue1_names), length(cue2_names), length(reward_names)]

#------------------------------------------------------------------------------------------------------------------
# --Observation Model A-matrix--

A_m_shapes = [[o_dim; n_states] for o_dim in n_obs]
display(A_m_shapes)
A = array_of_any_zeros(A_m_shapes)

### A[1] = First modality = Location = [35,35,4,2] 
show(A_m_shapes[1])

# 35x35 identity matrix 
identity_matrix = Matrix{Float64}(I, n_grid_points,n_grid_points)
# Keep it as 35x35 identity matrix but adds two more dimensions :,:,1,1 
expanded_matrix = reshape(identity_matrix, size(identity_matrix, 1), size(identity_matrix, 2), 1, 1)
# repeat function is like a copy machine. It takes expanded_matrix and makes multiple copies of it along specified dimensions.
tiled_matrix = repeat(expanded_matrix, outer=(1, 1, n_states[2], n_states[3]))

A[1] = tiled_matrix

#### A[2] = Second Modality = Cue-1 Observation Modality = [5, 35, 4, 2]
show(A_m_shapes[2])

#making the cue1 observation depend on "being at cue1_location and the true location of cue2
A[2][1,:,:,:] .= 1.0

A[2][1,:,:,:]

# making the Cue 1 signal depend on 1) being at the cue 1 location and 2) the location of cue 2 
for (i, cue_loc2_i) in enumerate(cue2_locations)
    A[2][1, location_to_index[cue1_location], i, :] .= 0.0
    A[2][i+1, location_to_index[cue1_location], i, :] .= 1.0
end 

#### A[3] = Third Modality = Cue-2 Observation Modality = [3, 35, 4, 2]
A[3][1,:,:,:] .= 1.0

for (i, cue_loc2_i) in enumerate(cue2_locations)
    A[3][1,location_to_index[cue_loc2_i],i,:] .= 0.0
    A[3][2,location_to_index[cue_loc2_i],i,1]  = 1.0
    A[3][3,location_to_index[cue_loc2_i],i,2]  = 1.0

end

#### A[4] = Fourth Modality = Reward Observation Modality = [3, 35, 4, 2]

A[4][1,:,:,:] .= 1.0 #Null is the most likely observation everywhere

rew_top_idx = location_to_index[reward_locations[1]]
rew_bott_idx = location_to_index[reward_locations[2]]

#agent is in the TOP reward condition
A[4][1,rew_top_idx,:,:] .= 0.0
A[4][2,rew_top_idx,:,1] .= 1.0
A[4][3,rew_top_idx,:,2] .= 1.0

# agent is in the BOTTOM reward condition
A[4][1,rew_bott_idx,:,:] .= 0.0
A[4][2,rew_bott_idx,:,2] .= 1.0
A[4][3,rew_bott_idx,:,1] .= 1.0 


#----------------- The transitional model: B array ------------------------

n_controls = [5, 1, 1]

B_f_shapes = [[ns, ns, n_controls[f]] for (f, ns) in enumerate(n_states)]

B = array_of_any_zeros(B_f_shapes)

# Populating the first dimension of the B-Matrix
actions = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]

len_y, len_x = size(grid_locations)

for (action_id, action_label) in enumerate(actions)

    for (curr_state, grid_locations) in enumerate(grid_locations)

        y, x = grid_locations

        next_y, next_x = y, x
        if action_label == "DOWN"
            next_y = y < len_y ? y + 1 : y
        elseif action_label == "UP"
            next_y = y > 1 ? y - 1 : y
        elseif action_label == "LEFT"
            next_x = x > 1 ? x - 1 : x
        elseif action_label == "RIGHT"
            next_x = x < len_x ? x + 1 : x
        elseif action_label == "STAY"    
        end

        new_location = (next_y, next_x)
        next_state = location_to_index[new_location]

        B[1][next_state, curr_state, action_id] = 1.0

    end

end
# Populating the 2nd and 3rd layer of the B-matrix with identity-matrices
B[2][:,:,1] = Matrix{Float64}(I, n_states[2], n_states[2])
B[3][:,:,1] = Matrix{Float64}(I, n_states[3], n_states[3])

# Populating the C-Matrix
C = array_of_any_zeros(n_obs)
display(C)

C[4][2] = 4.0
C[4][3] = -2.0


# Populating the D-Matrix
D = array_of_any_uniform(n_states) # improved: new function included

D[1] = onehot(location_to_index[(1, 1)], n_grid_points)


##########################
### Creating the Agent ###
##########################

settings = Dict(
    "policy_len" => 1
)

my_agent_test = init_aif(A, B, C = C, D = D; settings = settings);

###########################
### Creating random obs ###
###########################

obs = [2, 1, 1, 1]
# for i in 1:length(A)
#     rand_obs = Int64.(rand(1:n_obs[i]))
#     push!(obs, rand_obs)
# end

obs = Int64.(obs)

######################################
### Testing infer_states! function ###
######################################
my_agent_test.action = [2.0, 1.0, 1.0]
my_agent_test.qs_current = D


my_agent_test.action

qs = infer_states!(my_agent_test, obs)







