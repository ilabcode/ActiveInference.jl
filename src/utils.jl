""" -------- Utility Functions -------- """

using LinearAlgebra
using Plots
using IterTools
using Random
using Distributions

""" Creates an array of "Any" with the desired number of sub-arrays"""
function array_of_any(num_arr::Int) 
    return Array{Any}(undef, num_arr) #saves it as {Any} e.g. can be any kind of data type.
end

""" Creates an array of "Any" with the desired number of sub-arrays filled with zeros"""
function array_of_any_zeros(shape_list)
    arr = Array{Any}(undef, length(shape_list))
    for (i, shape) in enumerate(shape_list)
        arr[i] = zeros(Float64, shape...)
    end
    return arr
end

""" Creates an array of "Any" as a uniform categorical distribution"""
function array_of_any_uniform(shape_list)
    arr = Array{Any}(undef, length(shape_list))  
    for i in eachindex(shape_list)
        shape = shape_list[i]
        arr[i] = norm_dist(ones(shape))  
    end
    return arr
end

""" Function for Creating onehots"""
# Creates a vector filled with 0's and a 1 in a given location
function onehot(value, num_values)
    arr = zeros(Float64, num_values)
    arr[value] = 1.0
    return arr
end

""" Construct Policies """
function construct_policies_full(num_states; num_controls=nothing, policy_len=1, control_fac_idx=nothing)
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
    policies = collect(Iterators.product([1:i for i in x]...))

    transformed_policies = []

    for policy_tuple in policies
        # Convert tuple to an array
        policy_array = collect(policy_tuple)
        
        policy_matrix = reshape(policy_array, (length(policy_array) ÷ policy_len, policy_len))' 
        
        # Push the reshaped matrix to the list of transformed policies
        push!(transformed_policies, policy_matrix)
    end

    return transformed_policies
end

""" Process Observation to the Correct Format """
function process_observation(obs, num_modalities, num_observations)
    processed_obs = []

    # Check if obs is an integer, and num_modalities is 1, then it's a single modality observation
    if isa(obs, Int) && num_modalities == 1
        one_hot = zeros(Float64, num_observations[1])
        one_hot[obs] = 1.0
        push!(processed_obs, one_hot)
    elseif (isa(obs, Array) || isa(obs, Tuple)) && length(obs) == num_modalities
        # If obs is an array or tuple, and its length matches num_modalities, process each modality
        for (m, o) in enumerate(obs)
            one_hot = zeros(Float64, num_observations[m])
            one_hot[o] = 1.0
            push!(processed_obs, one_hot)
        end
    else
        throw(ArgumentError("Observation does not match expected modalities or format"))
    end

    return processed_obs
end

""" Get Model Dimensions from either A or B Matrix """
function get_model_dimensions(A = nothing, B = nothing)
    if A === nothing && B === nothing
        throw(ArgumentError("Must provide either `A` or `B`"))
    end
    num_obs, num_modalities, num_states, num_factors = nothing, nothing, nothing, nothing

    if A !== nothing
        num_obs = [size(a, 1) for a in A]
        num_modalities = length(num_obs)
    end

    if B !== nothing
        num_states = [size(b, 1) for b in B]
        num_factors = length(num_states)
    elseif A !== nothing
        num_states = [size(A[1], i) for i in 2:ndims(A[1])]
        num_factors = length(num_states)
    end

    return num_obs, num_states, num_modalities, num_factors
end


""" Equivalent to pymdp's "to_obj_array" [Recommendation to avoid using at as much as possible] """
function to_array_of_any(arr::Array)
    # Check if arr is already an array of arrays
    if typeof(arr) == Array{Array,1}
        return arr
    end
    # Create an array_out and assign squeezed array to the first element
    obj_array_out = Array{Any,1}(undef, 1)
    obj_array_out[1] = dropdims(arr, dims = tuple(findall(size(arr) .== 1)...))  
    return obj_array_out
end


""" Selects the highest value from Array -- used for deterministic action sampling """
function select_highest(options_array::Array{Float64})
    options_with_idx = [(i, option) for (i, option) in enumerate(options_array)]
    max_value = maximum(value for (idx, value) in options_with_idx)
    same_prob = [idx for (idx, value) in options_with_idx if abs(value - max_value) <= 1e-8]

    if length(same_prob) > 1
        return same_prob[rand(1:length(same_prob))]
    else
        return same_prob[1]
    end
end


""" Selects action from computed actions probabilities -- used for stochastic action sampling """
function action_select(probabilities)
    sample_onehot = rand(Multinomial(1, probabilities))
    return findfirst(sample_onehot .== 1)
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


#=
"""Function for creating the B-Matrix || Needs to be made generic! """
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
=# 