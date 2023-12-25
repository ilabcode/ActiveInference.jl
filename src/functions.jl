using LinearAlgebra
using Plots
using IterTools



"""Creates an array of "Any" with the desired number of sub-arrays"""
function array_of_any(num_arr::Int) 
    return Array{Any}(undef, num_arr) #saves it as {Any} e.g. can be any kind of data type.
end

"""Creates an array of "Any" with the desired number of sub-arrays filled with zeros"""
function array_of_any_zeros(shape_list)
    arr = Array{Any}(undef, length(shape_list))
    for (i, shape) in enumerate(shape_list)
        arr[i] = zeros(Float64, shape...)
    end
    return arr
end

"""Function for Plotting Beliefs"""
function plot_beliefs(belief_dist, title_str="")
    if abs(sum(belief_dist) - 1.0) > 1e-6
        throw(ArgumentError("Distribution not normalized"))
    end

    bar(belief_dist,alpha=0.6, xlabel="Categories", ylabel="Probabilities", title=title_str, xticks=1:length(belief_dist), legend = false)
end


"""Function for Plotting Grid World"""
function plot_gridworld(grid_locations)
    # Determine the size of the grid
    max_x = maximum(x -> x[2], grid_locations)
    max_y = maximum(y -> y[1], grid_locations)

    # Initialize a matrix for the heatmap
    heatmap_matrix = zeros(max_y, max_x)

    # Fill the matrix with state ids
    for (index, (y, x)) in enumerate(grid_locations)
        heatmap_matrix[y, x] = index
    end

    # Create the heatmap
    heatmap_plot = heatmap(1:max_x, 1:max_y, heatmap_matrix, 
                           aspect_ratio=:equal, 
                           xticks=1:max_x,
                           yticks=1:max_y, 
                           legend=false, 
                           color=:viridis,
                           yflip=true
                           )


    max_row, max_col = size(grid_locations)

    index_matrix = zeros(Int, max_row, max_col)

    for (index, (x, y)) in enumerate(grid_locations)
    index_matrix[x, y] = index
    annotate!(y, x, text(string(index), :center, 8, :white))
    end

    return heatmap_plot
end


"""Function for Plotting Likelihood"""
function plot_likelihood(A)
    # Create the heatmap
    heatmap(A, color=:greys, clim=(0, 1),aspect_ratio = :equal, yflip=true, legend=false,xticks=1:9, yticks=1:9, xlabel="STATES", ylabel="OBSERVATIONS" )
end


"""Function for creating the B-Matrix"""
function create_B_matrix(grid_locations, actions)
    num_states = length(grid_locations)
    num_actions = length(actions)
    B = zeros(num_states, num_states, num_actions)
    
    len_y, len_x = size(grid_locations)

    # Create a map from grid locations to the index 
    location_to_index = Dict(loc => idx for (idx, loc) in enumerate(grid_locations))

    for (action_id, action_label) in enumerate(actions)
        for (curr_state, grid_location) in enumerate(grid_locations)
            y, x = grid_location

            # Compute next location
            next_y, next_x = y, x
            if action_label == "DOWN" # UP and DOWN is reversed
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

            # Populating the B matrix
            B[next_state, curr_state, action_id] = 1
        end
    end

    return B
end 


"""Function for Creating onehots"""
# Creates a vector filled with 0's and a 1 in a given location
function onehot(value, num_values)
    arr = zeros(Float64, num_values)
    arr[value] = 1.0
    return arr
end


"""Function for Plotting point on Grid"""
# Be aware that an emoticon is also needed as input. For instance 🐭
function plot_point_on_grid(starting_state, grid_locations, emoticon)

    # Determine the size of the grid
    max_x = maximum(x -> x[1], grid_locations)
    max_y = maximum(y -> y[2], grid_locations)

    # Create an empty grid
    heatmap_matrix = zeros(max_x, max_y)

    # Find the location corresponding to the active state in the one-hot vector
    # -> findfirst finds the index of the first element in starting_state that equals to 1.0
    # than the index of this onehot encoded element is assigned a variable active_location
    active_location_idx = findfirst(x -> x == 1.0, starting_state)
    active_location = grid_locations[active_location_idx]

    # Set the value of the corresponding grid cell
    # here we impute our heatmap_matrix full of zeros with 1 to the location which
    #   corresponds to the active_location 
    heatmap_matrix[active_location...] = 1

    # Plot the heatmap
    p = heatmap(heatmap_matrix, aspect_ratio=1, legend=false, xticks = 1:max_x, yflip=true, yticks=1:max_y) # flipped y axis

    # Add emoticon at the active location
    annotate!(p, active_location[2], active_location[1], text(emoticon, 10, :center))

    return p
end


"""Function for Inferring States"""
function infer_states(observation_index, A, prior)

    log_likelihood = spm_log_single(A[observation_index,:])

    log_prior = spm_log_single(prior)

    qs = softmax(log_likelihood .+ log_prior)

    return qs
end


"""Function for Getting Expected States"""
# Note that this takes the action variable as a string and not an index 
function get_expected_states(B, qs_current, action, actions)
    
    action_id = findfirst(isequal(action), actions)

    qs_u = B[:,:,action_id] * qs_current

    return qs_u
end


"""Function for Getting Expected Observation"""
function get_expected_observations(A, qs_u)
    
    qo_u = A * qs_u

    return qo_u
end


"""Function Calculating the Expectped Free Energy"""
function calculate_G(A, B, C, qs_current, action_list)
    G = zeros(length(action_list))

    H_A = entropy(A)

    for (idx_i, action_i) in enumerate(action_list)

        qs_u = get_expected_states(B, qs_current, action_i, action_list)
        qo_u = get_expected_observations(A, qs_u)

        pred_uncertainty = H_A * qs_u
        pred_divergence = kl_divergence(qo_u, C)
        G[idx_i] = sum(pred_uncertainty .+ pred_divergence)
    end

    return G
end


"""Function That does Active Inference Loop"""
function run_active_inference_loop(A, B, C, D, actions, env, T)
    prior = copy(D)  # Initialize the prior
    obs = reset!(env)  # Reset the environment and get the initial observation

    for t in 1:T
        println("Time $t: Agent observes itself in location: $obs")

        # Convert observation to index
        obs_idx = findfirst(isequal(obs), grid_locations)

        # Perform inference over hidden states (our infer_states function)
        qs_current = infer_states(obs_idx, A, prior)

        # Plot beliefs (it will create plots separately, careful here xD)
        plot = plot_beliefs(qs_current, "Beliefs about location at time $t")
        display(plot)

        # Calculate expected free energy (our calculate_G function)
        G = calculate_G(A, B, C, qs_current, actions)

        # Compute posterior over action 
        Q_u = softmax(-G)

        # Sample action from Q_u using the function that we have predefined in the beginning of the document
        chosen_action_idx = sample_category(Q_u)

        # Update prior for next timestep
        prior = B[:, :, chosen_action_idx] * qs_current

        # Update generative process
        action_label = actions[chosen_action_idx]
        obs = step!(env, action_label)
    end

    return qs_current
end


"""Function for Constructing Policies"""
function construct_policies(num_states; num_controls=nothing, policy_len=1, control_fac_idx=nothing)

    # Number of factors (states) we are dealing with, remember that n_states must be converted to an array!
   num_factors = length(num_states)
   
   # Loops for controllable factors
   if isnothing(control_fac_idx)
       if !isnothing(num_controls)
           # If specific controls are given, find which factors have more than one control option
           control_fac_idx = findall(x -> x > 1, num_controls)
       else
           # If no controls are specified, assume all factors are controllable
           control_fac_idx = 1:num_factors
       end
   end

   # Determine the number of controls for each factor
   if isnothing(num_controls)
       num_controls = [in(c_idx, control_fac_idx) ? num_states[c_idx] : 1 for c_idx in 1:num_factors]
   end

   # Create a list of possible actions for each time step
   x = repeat(num_controls, policy_len)
   # Generate all combinations of actions across all time steps
   policies = collect(product([1:i for i in x]...))

   # Reshape each policy combination into a matrix of (policy_len x num_factors)
   reshaped_policies = [reshape(collect(policy), policy_len, num_factors) for policy in policies]

   return reshaped_policies
end


"""Function that Calculate the Expected Free Energy for Policies"""
function calculate_G_policies(A, B, C, qs_current, policies, actions)
    G = zeros(length(policies))
    H_A = entropy(A)

    for (policy_id, policy) in enumerate(policies)
        t_horizon = size(policy, 1)
        G_pi = 0.0
        qs_pi_t = copy(qs_current)

        for t in 1:t_horizon
            # Assuming policy[t, 1] gives an action index, convert it to the action label
            action_label = actions[policy[t, 1]]  # This one might be problematic 

            qs_prev = t == 1 ? qs_current : qs_pi_t
            qs_pi_t = get_expected_states(B, qs_prev, action_label, actions)
            qo_pi_t = get_expected_observations(A, qs_pi_t)
            kld = kl_divergence(qo_pi_t, C)
            G_pi_t = dot(H_A, qs_pi_t) + kld
            G_pi += G_pi_t
        end

        G[policy_id] += G_pi
    end

    return G
end


"""Function for Computing Probability for Actions"""
function compute_prob_actions(actions, policies, Q_pi)
    P_u = zeros(length(actions)) # initialize the vector of probabilities of each action

    for (policy_id, policy) in enumerate(policies)
        # Assuming the first action of the policy is an index into the actions array
        action_index = policy[1, 1]
        if action_index >= 1 && action_index <= length(actions)
            P_u[action_index] += Q_pi[policy_id]
        else
            error("Invalid action index: $action_index")
        end
    end

    # Normalize the action probabilities
    P_u = norm_dist(P_u)
  
    return P_u
end


"""Function for Active Inference with Planning"""
function active_inference_with_planning(A, B, C, D, actions, env, policy_len , T, grid_locations)
    # Initialize prior, first observation, and policies
    prior = D
    obs = reset!(env) 

    x_grid, y_grid = size(grid_locations)
    n_states = x_grid*y_grid

    n_actions = length(actions)

    policies = construct_policies([n_states],num_controls=[n_actions],  policy_len = policy_len)

    for t in 1:T
        println("Time $t: Agent observes itself in location: $obs")

        # Convert observation to index
        obs_idx = findfirst(isequal(obs), grid_locations)
        obs_idx = findfirst(isequal(obs), grid_locations)

        # Perform inference over hidden states
        qs_current = infer_states(obs_idx, A, prior)

        #plot beliefs
        plot = plot_beliefs(qs_current, "Beliefs about location at time $t")
        display(plot)

        # Calculate expected free energy of policies-actions
        G = calculate_G_policies(A, B, C, qs_current, policies, actions)

        # Marginalize P(u|pi) with the probabilities of each policy Q(pi)
        Q_pi = softmax(-G)

        # Compute the probability of each action
        P_u = compute_prob_actions(actions, policies, Q_pi)

        # Sample action from probability distribution over actions
        chosen_action = sample_category(P_u)

        # Compute prior for next timestep of inference
        prior = B[:, :, chosen_action] * qs_current

        # Step the generative process and get new observation
        action_label = actions[chosen_action]
        obs = step!(env, action_label)

    end

    return qs_current

end

"""Grid World for Epistemic Chaining"""

mutable struct GridWorldEnv
    init_loc::Tuple{Int, Int}
    current_loc::Tuple{Int, Int}
    cue1_loc::Tuple{Int, Int}
    cue2::String
    reward_condition::String
    len_y::Int
    len_x::Int
    grid_locations::Matrix

    function GridWorldEnv(starting_loc::Tuple{Int, Int}, cue1_loc::Tuple{Int, Int}, cue2::String, reward_condition::String, grid_locations::Matrix)
        len_y, len_x = size(grid_locations)
        new(starting_loc, starting_loc, cue1_loc, cue2, reward_condition)
    end
end


function step!(env::GridWorldEnv, action_label::String)
    y, x = env.current_state
    next_y, next_x = y, x

    if action_label == "DOWN"
        next_y = y < env.len_y ? y + 1 : y
    elseif action_label == "UP"
        next_y = y > 1 ? y - 1 : y
    elseif action_label == "LEFT"
        next_x = x > 1 ? x - 1 : x
    elseif action_label == "RIGHT"
        next_x = x < env.len_x ? x + 1 : x
    elseif action_label == "STAY" 
    end

    env.current_loc = (next_y, next_x)

    loc_obs = env.current_loc

    cue2_loc_names = ["L1","L2","L3","L4"]
    cue2_locs = [(1, 3), (2, 4), (4, 4), (5, 3)]

    cue2_loc_idx = Dict(cue2_loc_names[1] => 1, cue2_loc_names[2] => 2, cue2_loc_names[3] => 3, cue2_loc_names[4] => 4)

    cue2_loc = cue2_locs[cue2_loc_idx[env.cue2]]

    if env.current_loc == cue1_loc
        cue1_obs = env.cue2
    else
        cue1_obs = "Null"
    end

    reward_conditions = ["TOP", "BOTTOM"]
    rew_cond_idx = Dict(reward_conditions[1] => 1, reward_conditions[2] => 2)


    if env.current_loc == cue2_loc
        cue2_obs = cue2_names[rew_cond_idx[env.reward_condition] + 1]
    else
        cue2_obs = "Null"
    end


    if env.current_loc == reward_locations[1]
        if env.reward_condition == "TOP"
            reward_obs = "Cheese"
        else
            reward_obs = "Shock"
        end
    elseif env.current_loc == reward_locations[2]
        if env.reward_condition == "BOTTOM"
            reward_obs = "Cheese"
        else
            reward_obs = "Shock"
        end
    else
        reward_obs = "Null"
    end


    return loc_obs, cue1_obs, cue2_obs, reward_obs
end

function reset!(env::GridWorldEnv)
    env.current_loc = env.init_loc

    return env.current_loc
end






