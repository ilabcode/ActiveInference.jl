""" -------- Utility Functions -------- """

using LinearAlgebra
using Plots
##PTW_CR: Consider if IterTools is a necessry dependency
using IterTools
using Random
using Distributions

##PTW_CR: I really think this is a rather inelegant and unnecessary function :) 
##PTW_CR: We just want proper typing on the inputs to the functions instead. I'll comment on that there.
""" Creates an array of "Any" with the desired number of sub-arrays"""
function array_of_any(num_arr::Int) 
    return Array{Any}(undef, num_arr) #saves it as {Any} e.g. can be any kind of data type.
end

##PTW_CR: These two are also a bit weird. I think instead there should be one function called "create_matrix_templates"
##PTW_CR: This takes the n_obs n_states n_controls and returns all 5 matrices. 
##PTW_CR: Then I would just let them return uniform probability distributions
""" Creates an array of "Any" with the desired number of sub-arrays filled with zeros"""
function array_of_any_zeros(shape_list)
    arr = Array{Any}(undef, length(shape_list))
    for (i, shape) in enumerate(shape_list)
        arr[i] = zeros(Real, shape...)
    end
    return arr
end

""" Creates an array of "Any" as a uniform categorical distribution"""
function array_of_any_uniform(shape_list)
    arr = Array{Any}(undef, length(shape_list))  
    for i in eachindex(shape_list)
        shape = shape_list[i]
        arr[i] = norm_dist(ones(Real, shape))  
    end
    return arr
end

##PTW_CR: Is there really not a native ulia function that can do exactly this, but faster ? 
""" Creates a onehot encoded vector """
function onehot(value, num_values)
    arr = zeros(Real, num_values)
    arr[value] = 1.0
    return arr
end

##PTW_CR: Here and in general, when using the num_states, num_controls and num_obs, make it possible for people to just put in a vector for a single dimension
##PTW_CR: So do a typing thing, where there is multiple dispatch: if people put in a Vector{T} {where T<:Real} then put that into another vector (this is the format you usually use), and run the function with that instead
##PTW_CR: Also, call them n_factors etc I'd say
##PTW_CR: I would put this function either in an action file, or in the create_agent file, depending on whether this is run only once when the agent is created, or if it is run every time the agent selects actions
##PTW_CR: If it is run every time actiosn are selected, make sure it is speed-optimized
""" Construct Policies """
function construct_policies_full(num_states; num_controls=nothing, policy_len=1, control_fac_idx=nothing)
    num_factors = length(num_states)

    ##PTW_CR: Careful with comments saying you loop when you dont
    # Loops for controllable factors
    if isnothing(control_fac_idx)
        if !isnothing(num_controls)
            # If specific controls are given, find which factors have more than one control option
            control_fac_idx = findall(x -> x > 1, num_controls)
        else
            # If no controls are specified, assume all factors are controllable
            control_fac_idx = 1:num_factors
        end
    end

    # Determine the number of controls for each factor
    if isnothing(num_controls)
        num_controls = [in(c_idx, control_fac_idx) ? num_states[c_idx] : 1 for c_idx in 1:num_factors]
    end

    # Create a list of possible actions for each time step
    x = repeat(num_controls, policy_len)

    # Generate all combinations of actions across all time steps
    policies = collect(Iterators.product([1:i for i in x]...))

    ##PTW_CR: Type this vector when you create it
    transformed_policies = []

    for policy_tuple in policies
        # Convert tuple to an array
        policy_array = collect(policy_tuple)

        ##PTW_CR: Make sure to comment and explain what these are doign exactly - to make it easier to read for others  
        policy_matrix = reshape(policy_array, (length(policy_array) ÷ policy_len, policy_len))' 
        
        # Push the reshaped matrix to the list of transformed policies
        push!(transformed_policies, policy_matrix)
    end

    return transformed_policies
end

##PTW_CR: Perhaps this shoudl be part of the perception file?
""" Process Observation to the Correct Format """
function process_observation(obs, num_modalities, num_observations)
    ##PTW_CR: WHen initializing empty vectors, type them - for speed. I wont repet this in other places
    processed_obs = []

    ##PTW_CR: This ifelse you can do with multiple dispatch - one function that takes a single integer, one that takes a vetors of integers
    # Check if obs is an integer, and num_modalities is 1, then it's a single modality observation
    if isa(obs, Int) && num_modalities == 1
        one_hot = zeros(Real, num_observations[1])
        one_hot[obs] = 1.0
        push!(processed_obs, one_hot)
        ##PTW_CR: In this version of the function, I would probably first create the onehot, and then put Vector around it
    elseif (isa(obs, Array) || isa(obs, Tuple)) && length(obs) == num_modalities
        ##PTW_CR: I generally think variable names should be spelled out - not 0 but observation of whatever it stands for
        # If obs is an array or tuple, and its length matches num_modalities, process each modality
        for (m, o) in enumerate(obs)
            one_hot = zeros(Real, num_observations[m])
            one_hot[o] = 1.0
            push!(processed_obs, one_hot)
            ##PTW_CR: Do this with a list comprehension - if there isn't some in-built way of creating a series of one-hot vectors
            ##PTW_CR: Also, perhaps make a helper function that creates the onehot.
        end
    else
        ##PTW_CR: Not needed if you have multiple dispatch.
        throw(ArgumentError("Observation does not match expected modalities or format"))
    end

    return processed_obs
end

""" Get Model Dimensions from either A or B Matrix """
##PTW_CR: In general, make docstring for all functions. Check online how to format docstrings - there is a format (whcih is not this). These will show up in the documentation, os it matters. 
function get_model_dimensions(A = nothing, B = nothing)
    ##PTW_CR: I think I would, when the agent is created, have a field where this information is stored from the beginning, so it can just be accessed.
    ##PTW_CR: Like POMDP.dimensions.n_states etc
    ##PTW_CR: Then it doest have to be run on the go, and that information is used then anway as far as I can tell
    if A === nothing && B === nothing
        throw(ArgumentError("Must provide either `A` or `B`"))
    end
    num_obs, num_modalities, num_states, num_factors = nothing, nothing, nothing, nothing

    if A !== nothing
        num_obs = [size(a, 1) for a in A]
        num_modalities = length(num_obs)
    end

    if B !== nothing
        num_states = [size(b, 1) for b in B]
        num_factors = length(num_states)
    elseif A !== nothing
        num_states = [size(A[1], i) for i in 2:ndims(A[1])]
        num_factors = length(num_states)
    end

    return num_obs, num_states, num_modalities, num_factors
end


""" Equivalent to pymdp's "to_obj_array" """
##PTW_CR: Unsure what this function is used for exactly - explain it in the docstring
##PTW_CR: However, it seems inelegant-ish, and a consequence of copying from Conor's work.
##PTW_CR: So my gut feelign is this can be improved
function to_array_of_any(arr::Array)
    # Check if arr is already an array of arrays
    if typeof(arr) == Array{Array,1}
        return arr
    end
    # Create an array_out and assign squeezed array to the first element
    obj_array_out = Array{Any,1}(undef, 1)
    obj_array_out[1] = dropdims(arr, dims = tuple(findall(size(arr) .== 1)...))  
    return obj_array_out
end


##PTW_CR: I actually don't think we should have deterministic action sampling at all. One can just have a high alpha. Just because it is applicable whne fitting to data.
##PTW_CR: Just use this: getindex.(findall(A .== maximum(A)),2)   (This finds all the maximum values - round it if you want 1e-8 buffer)
##PTW_CR: And sample a random value from the resulting vector
""" Selects the highest value from Array -- used for deterministic action sampling """
function select_highest(options_array::Array{Float64})
    options_with_idx = [(i, option) for (i, option) in enumerate(options_array)]
    max_value = maximum(value for (idx, value) in options_with_idx)
    same_prob = [idx for (idx, value) in options_with_idx if abs(value - max_value) <= 1e-8]

    if length(same_prob) > 1
        return same_prob[rand(1:length(same_prob))]
    else
        return same_prob[1]
    end
end

##PTW_CR: Suggestion: make this functiona nd the one above the same. But in the one above, still make a Multinomial that just is uniform only for the maximum values. 
##PTW_CR: Then there is just a flag (or even better: multiple dispatch) that chooses between the two versions. 
##PTW_CR: Why not just use a Categorical btw...? 
##PTW_CR: I would also rather have a function that ouitputs the action probability distribution here. Then it interfaces well with ActionModels.
##PTW_CR: Also include alpha here.
##PTW_CR: And put this function and the above one in the action file
""" Selects action from computed actions probabilities -- used for stochastic action sampling """
function action_select(probabilities)
    sample_onehot = rand(Multinomial(1, probabilities))
    return findfirst(sample_onehot .== 1)
end

##PTW_CR: Might combine this function with the above, just call it calculate_action_probabilities or sometthing
##PTW_CR: Unless the log actio marginals are needed for other things too
##PTW_CR: Also, if save_history is on, I think the action probabilities should be saved at each timestep (along with the rest of the important things)
""" Function to get log marginal probabilities of actions """
function get_log_action_marginals(aif)
    num_factors = length(aif.num_controls)
    action_marginals = array_of_any_zeros(aif.num_controls)
    log_action_marginals = array_of_any(num_factors)
    ##PTW_CR: states are things that change. def the case for q_pi - is it for policies? 
    ##PTW_CR: I guess since the time horizon never changes, the policy list is contant. I would save it somewhere in the active inference object.
    ##PTW_CR: Also, at least my policy is to not write the mathematical notation, but write its meaning. So write policy_posterior instead of q_pi
    ##PTW_CR: Also, I think it should be called the 'policy_posterior' not the 'posterior_policies'
    q_pi = get_states(aif, "posterior_policies")
    policies = get_states(aif, "policies")
    
    ##PTW_CR: This is the actual marginalization. Might want there to be a helper function which just marginalizes q_pi (or any matrix) and call it here.
    ##PTW_CR: I also feel like there should be a more elegant way of doing this. I am a bit uncertain about the format of the policies object here, so it's a bit hard for me to say. 
    for (pol_idx, policy) in enumerate(policies)
        for (factor_i, action_i) in enumerate(policy[1,:])
            action_marginals[factor_i][action_i] += q_pi[pol_idx]
        end
    end

    ##PTW_CR: Make sure to comment the code
    action_marginals = norm_dist_array(action_marginals)

    ##PTW_CR: You can just use broadcasting to get this: 
    ##PTW_CR: log_action_marginals = spm_log_single.(action_marginals)
    ##PTW_CR: This will run that function on all entries of the array. You cna probably use that other places. It's fast.
    for factor_i in 1:num_factors
        log_marginal_f = spm_log_single(action_marginals[factor_i])
        log_action_marginals[factor_i] = log_marginal_f
    end

    return log_action_marginals
end

##PTW_CR: rand(m,n) creates a m by n matrix of random numbers, just use that and normalize it.
##PTW_CR: Also, why would people every want to create a random generative model?
##PTW_CR: Even then, I think it should be part of the 'create_matrix_templates' function, to gather it all. That's a nice helper function then.
""" Generate Random Generative Model as A and B matrices """
function generate_random_GM(n_states::Vector{Int64}, n_obs::Vector{Int64}, n_controls::Vector{Int64})

    # Initialize A matrices:
    A_shapes = [[o_dim; n_states] for o_dim in n_obs]
    A = array_of_any_zeros(A_shapes)

    # Fill A matrices with random probabilities
    for (i, matrix) in enumerate(A)
        for idx in CartesianIndices(matrix)
            matrix[idx] = rand()
        end
        A[i] = norm_dist(matrix)
    end

    # Initialize B matrices
    B_shapes = [[ns, ns, n_controls[f]] for (f, ns) in enumerate(n_states)]
    B = array_of_any_zeros(B_shapes)

    # Fill B matrices with random probabilities
    for (i, matrix) in enumerate(B)
        for idx in CartesianIndices(matrix)
            matrix[idx] = rand()
        end
        B[i] = norm_dist(matrix)
    end

    return A, B
end


##PTW_CR: You have to check whether they sum to one _and_ whether they are all positive.    
##PTW_CR: I think the utils and the maths files could be combined
##PTW_CR: I'd change the name here to something like 'check_probability_distribution'
""" Check if the array is a proper probability distribution """
function check_normalization(arr)
    return all(tensor -> all(isapprox.(sum(tensor, dims=1), 1.0, rtol=1e-5, atol=1e-8)), arr)
end