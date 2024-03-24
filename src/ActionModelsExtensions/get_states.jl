"""
This module extends the "get_states" functionality of the ActionModels package to work specifically with instances of the AIF type.

    get_states(aif::AIF, target_states::Vector{String})
Retrieves multiple states from an AIF agent. 

    get_states(aif::AIF, target_state::String)
Retrieves a single target state from an AIF agent.

    get_states(aif::AIF)
Retrieves all states from an AIF agent.
"""

using ActionModels


# Retrieve multiple states
function ActionModels.get_states(aif::AIF, target_states::Vector{String})
    states = Dict()

    for target_state in target_states
        try
            states[target_state] = get_states(aif, target_state)
        catch e
            # Catch the error if a specific state does not exist
            if isa(e, ArgumentError)
                throw(ArgumentError("The specified state $target_state does not exist"))
            else
                rethrow(e) 
            end
        end
    end

    return states
end

# Retrieve a single state
function ActionModels.get_states(aif::AIF, target_state::String)
    # Check if the state is in the agent's states
    if haskey(aif.states, target_state)
        #  Directly store the constructed policies
        if target_state == "policies"
            return aif.states[target_state]
        else
            # Retrieve the latest value of the target state
            state_history = aif.states[target_state]
            return state_history isa AbstractVector ? last(state_history) : state_history
        end
    else
        # If the target state is not found, throw an ArgumentError
        throw(ArgumentError("The specified state $target_state does not exist"))
    end
end


# Retrieve all states
function ActionModels.get_states(aif::AIF)
    all_states = Dict()
    for (key, state_history) in aif.states
        #  Directly store the constructed policies
        if key == "policies"
            all_states[key] = state_history
        else
            # For other keys, store the latest value of each state
            all_states[key] = state_history isa AbstractVector ? last(state_history) : state_history
        end
    end
    return all_states
end



