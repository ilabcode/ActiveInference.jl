"""
This module contains models of Partially Observable Markov Decision Processes under Active Inference


"""

### Action Model:  Returns probability distributions for actions per factor

function action_pomdp!(aif, obs)

    ### Get parameters 
    alpha = get_parameters(aif, "alpha")
    n_factors = length(aif.num_controls)

    # Initialize an empty arrays for action distribution per factor
    action_p = array_of_any(n_factors)
    action_distribution = array_of_any(n_factors)

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