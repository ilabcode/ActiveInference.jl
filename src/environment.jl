using LinearAlgebra

# """Mutable structure creating the environment"""
# mutable struct GridWorldEnv
#     init_state::Tuple{Int, Int}
#     current_state::Tuple{Int, Int}
#     len_y::Int
#     len_x::Int

#     function GridWorldEnv(starting_state::Tuple{Int, Int}, grid_locations)
#         len_y, len_x = maximum(first.(grid_locations)), maximum(last.(grid_locations))
#         new(starting_state, starting_state, len_y, len_x)
#     end
# end

# """Function for how to "step" in the Grid World"""
# function step!(env::GridWorldEnv, action_label::String)
#     y, x = env.current_state
#     next_y, next_x = y, x

#     if action_label == "DOWN" # Y-axis reversed
#         next_y = y < env.len_y ? y + 1 : y
#     elseif action_label == "UP"
#         next_y = y > 1 ? y - 1 : y
#     elseif action_label == "LEFT"
#         next_x = x > 1 ? x - 1 : x
#     elseif action_label == "RIGHT"
#         next_x = x < env.len_x ? x + 1 : x
#     elseif action_label == "STAY" 
#     end

#     env.current_state = (next_y, next_x)

#     return env.current_state
# end

# """Reset function"""

# function reset!(env::GridWorldEnv)
#     env.current_state = env.init_state
#     println("Re-initialized location to ", env.init_state)
#     return env.current_state
# end

