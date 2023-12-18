using LinearAlgebra


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
    max_x = maximum(x -> x[1], grid_locations)
    max_y = maximum(y -> y[2], grid_locations)

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
function get_expected_states(B, qs_current, action)
    
    action_id = findfirst(isequal(action), actions)

    qs_u = B[:,:,action_id] * qs_current

    return qs_u
end


"""Function for Getting Expected Observation"""
function get_expected_observations(A, qs_u)
    
    qo_u = A * qs_u

    return qo_u
end




