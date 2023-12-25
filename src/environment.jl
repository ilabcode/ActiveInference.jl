using LinearAlgebra
using Plots
using IterTools


"""Grid World for Epistemic Chaining"""

mutable struct EpistChainEnv
    init_loc::Tuple{Int, Int}
    current_loc::Tuple{Int, Int}
    cue1_loc::Tuple{Int, Int}
    cue2::String
    reward_condition::String
    len_y::Int
    len_x::Int
    grid_locations::Matrix{Tuple{Int, Int}}

    function EpistChainEnv(starting_loc::Tuple{Int, Int}, cue1_loc::Tuple{Int, Int}, cue2::String, reward_condition::String, grid_locations::Matrix{Tuple{Int, Int}})
        len_y, len_x = size(grid_locations)
        new(starting_loc, starting_loc, cue1_loc, cue2, reward_condition)
    end
end


function step!(env::EpistChainEnv, action_label::String)
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

function reset!(env::EpistChainEnv)
    env.current_loc = env.init_loc

    return env.current_loc
end


"""Gridworld Simple"""
mutable struct GridWorldEnv
    init_state::Tuple{Int, Int}
    current_state::Tuple{Int, Int}
    len_y::Int
    len_x::Int

    function GridWorldEnv(starting_state::Tuple{Int, Int}, grid_locations)
        len_y, len_x = maximum(first.(grid_locations)), maximum(last.(grid_locations))
        new(starting_state, starting_state, len_y, len_x)
    end
end

"""Function for how to "step" in the Grid World"""
function step_GWE!(env::GridWorldEnv, action_label::String)
    y, x = env.current_state
    next_y, next_x = y, x

    if action_label == "DOWN" # Y-axis reversed
        next_y = y < env.len_y ? y + 1 : y
    elseif action_label == "UP"
        next_y = y > 1 ? y - 1 : y
    elseif action_label == "LEFT"
        next_x = x > 1 ? x - 1 : x
    elseif action_label == "RIGHT"
        next_x = x < env.len_x ? x + 1 : x
    elseif action_label == "STAY" 
    end

    env.current_state = (next_y, next_x)

    return env.current_state
end

"""Reset function"""

function reset_GWE!(env::GridWorldEnv)
    env.current_state = env.init_state
    println("Re-initialized location to ", env.init_state)
    return env.current_state
end

