"""
This extends the "get_history" function of the ActionModels package to work specifically with instances of the AIF type.

    get_history(aif::AIF, target_states::Vector{String})
Retrieves a history for multiple states of an AIF agent. 

    get_history(aif::AIF, target_state::String)
Retrieves a single target state history from an AIF agent.

    get_history(aif::AIF)
Retrieves history of all states from an AIF agent.
"""

using ActionModels


# Retrieve multiple states history
function ActionModels.get_history(aif::AIF, target_states::Vector{String})
    history = Dict()

    for target_state in target_states
        try
            history[target_state] = get_history(aif, target_state)
        catch e
            # Catch the error if a specific state does not exist
            if isa(e, ArgumentError)
                throw(ArgumentError("The specified state $target_state does not exist"))
            else
                rethrow(e) 
            end
        end
    end

    return history
end

# Retrieve a history from a single state
function ActionModels.get_history(aif::AIF, target_state::String)
    # Check if the state is in the AIF's states
    if haskey(aif.states, target_state)

        return aif.states[target_state]
    else
        # If the target state is not found, throw an ArgumentError
        throw(ArgumentError("The specified state $target_state does not exist"))
    end
end


# Retrieve all states history
function ActionModels.get_history(aif::AIF)
    return aif.states
end