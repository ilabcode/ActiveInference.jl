""" Update obs likelihood matrix """
##PTW_CR: the modalities = "all" shouldn't e the way to specify it
##PTW_CR: Hard ot type (not it can be an int _and_ a string)
##PTW_CR: And people might write all sorts of strings
##PTW_CR: I suggest that a 0 makes it all.
##PTW_CR: Or even better: when the modalities to learn field is constructed in the init function, then if an 'all' flag has been set, 
##PTW_CR: there is just a list of all the modalities stored in the struct
##PTW_CR: So that this function _always_ receives a vector of Integers
function update_obs_likelihood_dirichlet(pA, A, obs, qs; lr = 1.0, fr = 1.0, modalities = "all")

    ##PTW_CR: Again, if these were just stored in the struct, they wouldn't have to be constructed again on every timestep
    ##PTW_CR: Basically, anything which doesn't jave to be re-created on every timesteo (i.e. everything which stays the same)
    ##PTW_CR: Should be stored in the struct and not created in these functions
    ##PTW_CR: The update functions are where most of the time is spent, so they should be as fast as possible
    # Extracting the number of modalities and observations from the dirichlet: pA
    num_modalities = length(pA)
    num_observations = [size(pA[modality + 1], 1) for modality in 0:(num_modalities - 1)]

    ##PTW_CR: I think I would let this happen outside the learning function, but happen in superfunction. Something like 'store_observation!()' 
    ##PTW_CR: which takes the observation and saves it in the struct in the right format
    ##PTW_CR: In onehot.
    ##PTW_CR: Also save the observations in the history.
    obs = process_observation(obs, num_modalities, num_observations)

    if modalities === "all"
        modalities = collect(1:num_modalities)
    end

    ##PTW_CR: deepcopy is usually slow in many languages. Is this really necessary ? 
    qA = deepcopy(pA)

    ##PTW_CR: If it is just a cross with itself, then input the qs twice (instead of having a version of the function which figures that out itself) I think
    # Important! Takes first the cross product of the qs itself, so that it matches dimensions with the A and pA matrices
    qs_cross = spm_cross(qs)

    ##PTW_CR: You cna probably use a map() to do this instead of a deepcopy and a for loop
    ##PTW_CR: Just need a function which just runs those calculations there
    for modality in modalities
        ##PTW_CR: spell these names out.. these are the FE gradients
        dfda = spm_cross(obs[modality], qs_cross)
        ##PTW_CR: Why do you multiply with whether A is positive here? I have a sense that this and the spm_cross can be made a bit more elegant together
        dfda = dfda .* (A[modality] .> 0)
        qA[modality] = (fr * qA[modality]) + (lr * dfda)
    end

    return qA
end


##PTW_CR: I think I would in general never just write 'factors', but stick with 'state_factors'.
##PTW_CR: Otherwise, people have to remember this rather arbitrary thing, that states have 'factors' and observations have 'modalities'
##PTW_CR: At the end of the day, the two are not actually that distinguishable. 
""" Update state likelihood matrix """
function update_state_likelihood_dirichlet(pB, B, actions, qs, qs_prev; lr = 1.0, fr = 1.0, factors = "all")

    ##PTW_CR: I think most of the above comments apply here as well

    ##PTW_CR: Explain to me why there is not a corss with itself here ?

    ##PTW_CR: I really think there needs to be acertain consistency with the matrices dimensions here
    ##PTW_CR: They should always o/s times s times modality/factor times action
    ##PTW_CR: So the action dependnecy is always the fourth dimension
    ##PTW_CR: (optionally, it can be a vector with a three-dimensional matrix for each action)
    ##PTW_CR: It strikes me that letting the A-matrix be action-dependent becomes very easy
    ##PTW_CR: And that this function and the function above essentially become equivalent if that is done?
    ##PTW_CR: Essentially, making an observation is identical to having a qs that is fully certain



    num_factors = length(pB)

    qB = deepcopy(pB)


    if factors === "all"
        factors = collect(1:num_factors)
    end

    for factor in factors
        dfdb = spm_cross(qs[factor], qs_prev[factor])
        dfdb .*= (B[factor][:,:,Int(actions[factor])] .> 0)
        qB[factor][:,:,Int(actions[factor])] = qB[factor][:,:,Int(actions[factor])]*fr .+ (lr .* dfdb)
    end

    return qB
end

""" Update prior D matrix """
function update_state_prior_dirichlet(pD, qs; lr = 1.0, fr = 1.0, factors = "all")

    ##PTW_CR: If I understand correctly, the update equations are identical, except that what goes into them is different. 
    ##PTW_CR: Might make the update function a function which takes as inputs the things that vary.

    ##PTW_CR: Also: is the D matrix only updated once series of inputs? I guess there has to be multiple lets say experiments or sessions for updating this to be meaningful.
    
    ##PTW_CR: Finally... We should converge a bit on the namign perhaps... ABCDE can either be vectors, matrices or tensors, and it varies depending on whether they can be controlled etc.
    ##PTW_CR: maybe call them tensors in general, when needed, and when not needed just call them ABCDE ? 

    num_factors = length(pD)

    qD = deepcopy(pD)

    if factors == "all"
        factors = collect(1:num_factors)
    end

    for factor in factors
        idx = pD[factor] .> 0
        qD[factor][idx] = (fr * qD[factor][idx]) .+ (lr * qs[factor][idx])
    end  
    
    return qD
end