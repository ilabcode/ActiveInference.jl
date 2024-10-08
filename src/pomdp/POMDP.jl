"""
This module contains models of Partially Observable Markov Decision Processes under Active Inference

"""

### Action Model:  Returns probability distributions for actions per factor

function action_pomdp!(agent::Agent, obs::Vector{Int64})

    aif = agent.substruct

    ### Get parameters 
    alpha = agent.parameters["alpha"]
    n_factors = length(aif.settings["num_controls"])

    # Initialize empty arrays for action distribution per factor
    action_p = array_of_any(n_factors)
    action_distribution = Vector(undef, n_factors)

    #If there was a previous action
    if !ismissing(agent.states["action"])

        #Extract it
        previous_action = agent.states["action"]

        #If it is not a vector, make it one
        if !(previous_action isa Vector)
            previous_action = [previous_action]
        end

        #Store the action in the AIF substruct
        agent.substruct.action = previous_action
    end

    ### Infer states & policies

    # Run state inference 
    infer_states!(aif, obs)

    # Run policy inference 
    infer_policies!(aif)

    ### Retrieve log marginal probabilities of actions
    log_action_marginals = get_log_action_marginals(aif)

    ### Pass action marginals through softmax function to get action probabilities
    for factor in 1:n_factors
        action_p[factor] = softmax(log_action_marginals[factor] * alpha)
        action_distribution[factor] = Distributions.Categorical(action_p[factor])
    end
    
    return n_factors == 1 ? action_distribution[1] : action_distribution
end

function action_pomdp!(aif::AIF, obs::Vector{Int64})

    ### Get parameters 
    alpha = aif.parameters["alpha"]
    n_factors = length(aif.settings["num_controls"])

    # Initialize an empty arrays for action distribution per factor
    action_p = array_of_any(n_factors)
    action_distribution = Vector(undef, n_factors)

    ### Infer states & policies

    # Run state inference 
    infer_states!(aif, obs)

    # Run policy inference 
    infer_policies!(aif)

    ### Retrieve log marginal probabilities of actions
    log_action_marginals = get_log_action_marginals(aif)
    
    ### Pass action marginals through softmax function to get action probabilities
    for factor in 1:n_factors
        action_p[factor] = softmax(log_action_marginals[factor] * alpha)
        action_distribution[factor] = Distributions.Categorical(action_p[factor])
    end

    return action_distribution
end

function action_pomdp!(agent::Agent, obs::Int64)
    action_pomdp!(agent::Agent, [obs])
end