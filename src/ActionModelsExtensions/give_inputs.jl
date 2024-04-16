"""

This is an experimental module to extend the give_inputs! functionality of ActionsModels.jl to work with instances of the AIF type.

    single_input!(aif::AIF, obs)
Give a single observation to an AIF agent. 


"""

using ActionModels

### Give single observation to the agent
function ActionModels.single_input!(aif::AIF, obs::Vector)

    # Running the action model to retrieve the action distributions
    action_distributions = action_pomdp!(aif, obs)

    # Get number of factors from the action distributions
    num_factors = length(action_distributions)

    # if there is only one factor
    if num_factors == 1
        aif.action = rand(action_distributions)
        push!(aif.states["action"], aif.action)


    # if there are multiple factors
    else
        # Initialize a vector for sampled actions 
        sampled_actions = zeros(num_factors)

        # Sample action per factor
        for factor in eachindex(action_distributions)
            sampled_actions[factor] = rand(action_distributions[factor])
        end

        aif.action = sampled_actions
    end

    push!(aif.states["action"], aif.action)

    return aif.action
end

function ActionModels.give_inputs!(aif::AIF, observations::Vector{Any})

    for observation in observations

        ActionModels.single_input!(aif, observation)

    end


end