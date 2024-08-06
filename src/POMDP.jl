"""
This module contains models of Partially Observable Markov Decision Processes under Active Inference
"""
##PTW_CR: This is not a module, just a function
##PTW_CR: We discussed this: Here is my suggestions for a general structure: 
##PTW_CR: There should be an AbstractGenerativeModel typeof
##PTW_CR: Then there can be different subtypes of that - the currently implemented one is POMDPActiveInference
##PTW_CR: There should be three functions that dispatches differently on different AbstractGenerativeModel subtypes:
##PTW_CR: namely, perception!(), learning!() and action!()
##PTW_CR: could also be called state_inference!(), parameter_inference!(), and action_inference!()
##PTW_CR: Then there can be this function, called active_inference!()
##PTW_CR: This function runs on an AbstractGenerativeModel, and calls the three functions above in sequence
##PTW_CR: action!() should return the action probability distribution
##PTW_CR: Apart from that, neither function should return anything. They should modify the input AbstractGenerativeModel
##PTW_CR: There can be a function which stores the sampled action in the AbstractGenerativeModel.
##PTW_CR: Users can use this function to store the actions they sample if they run each aprt separately
##PTW_CR: But it's also used when runnign active_inference on an ActionModels Agent, as below
##PTW_CR: When running active_inference!() on an agent, it should just store the previous action in the substruct with the AbstractGenerativeModel
##PTW_CR: And then run active_inference!() on the AbstractGenerativeModel
##PTW_CR: Then learning!(), for example, will look in the settings part of the AbstractGenerativeModel subtype to see which matrices should be updated, etc
##PTW_CR: Ultimately, perception!() and learning!() (and perhaps structure_learning!()) can be combined as one function, but that can wait
##PTW_CR: I would also call this file active_inference!()
##PTW_CR: And then I would have different folders for each AbstractGenerativeModel subtype we'll create - for now there is just one folders
##PTW_CR: (We might make these into separate modules or even packages one day, but no rush)
##PTW_CR: One other thing: we should consider in which order those three are called. Why does learning!() come after perception!()? Let me know if there is a good reason for this, otherwise I think it should be the other way around.

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
    
    return action_distribution
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




##PTW_CR: !! -- I'll use this place for other kinds of comments -- !!

##PTW_CR: As we discussed, I think there should be a syntax for specifying a stransformation function
##PTW_CR: Which goes from a template matrices to the actual matrices
##PTW_CR: One of these being a softmax transformed
##PTW_CR: Where they can specify parameters for the transform
##PTW_CR: Which are then parameters that can be estimated form data with Turing
##PTW_CR: Stuff like that - the precision of the A matrix - are often the questions asked
##PTW_CR: For some transforms, specific parameter learning functions can also be implemented: 
##PTW_CR: There is a way for the agent to learn the precision of the softmax ofn the A, for example
##PTW_CR: Which is a much more data-effective parameter learning approach
##PTW_CR: So: People put in a template A; 
##PTW_CR: then optionally they put in a transform function (with it's own parameters); 
##PTW_CR: then optionally again, they put in a learning function for (some of) those parameters (with it's own hyperparameters)