""" -------- AIF Mutable Struct -------- """

using LinearAlgebra

##PTW_CR: Like I said before, I think this should be called POMDPActiveInference
##PTW_CR: Write out the proper typings here. A is a Vector{Array{T}} where T<:Real
##PTW_CR: Note: you can do multiple dispatch on the dimensions of the Array as well, if you want a function to act differently with Vectors and Arrays. But that will come below.
##PTW_CR: I would have some structs that are inside this one to sort the types of fields
##PTW_CR: So for example a POMDPActiveInferenceSettings struct that has all the settings, a POMDPActiveInferenceParameters struct that has all the parameters, a POMDPActiveInferenceStates that has the current belief state, and perhaps a POMDPActiveInferenceHistory that has the history
##PTW_CR: All the matrices, and all the hyperparameters, should be in the POMDPActiveInferenceParameters struct. Perhaps there can be two different ones for the matrices and the hyperparameters
##PTW_CR: The modalities_to_learn etc should be in the POMDPActiveInferenceSettings struct
##PTW_CR: The states should be everything that changes.. G, qs. q_pi etc
##PTW_CR: Also spell these variable names out, and def be consistens (small or large Q for example)

##PTW_CR: A different thing: don't call pA a 'prior' - it isn't. It's the current belief (i.e. posteiror) over the Dirichlet distributed beleif about the matrix. 
##PTW_CR: On every timestep, when running learning!(), they should be updated. Then, the matrices themselves should be changed. Technically, they are the mean of the DIrichlet distribution described by the dirichlet parameters
mutable struct AIF
    A::Array{Any,1} # A-matrix
    B::Array{Any,1} # B-matrix
    C::Array{Any,1} # C-vectors
    D::Array{Any,1} # D-vectors
    E::Union{Array{Any, 1}, Nothing} # E - vector (Habits)
    pA::Union{Array{Any,1}, Nothing} # Dirichlet priors for A-matrix
    pB::Union{Array{Any,1}, Nothing} # Dirichlet priors for B-matrix
    pD::Union{Array{Any,1}, Nothing} # Dirichlet priors for D-vector
    lr_pA::Real # pA Learning Parameter
    fr_pA::Real # pA Forgetting Parameter,  1.0 for no forgetting
    lr_pB::Real # pB learning Parameter
    fr_pB::Real # pB Forgetting Parameter
    lr_pD::Real # pD Learning parameter
    fr_pD::Real # PD Forgetting parameter
    modalities_to_learn::Union{String, Vector{Int64}} # Modalities can be eithe "all" or "# modality"
    factors_to_learn::Union{String, Vector{Int64}} # Modalities can be eithe "all" or "# factor"
    gamma::Real # Gamma parameter
    alpha::Real # Alpha parameter
    ##PTW_CR: Is this indeces for the policies, or the full list of them ? Probably belongs in settings
    policies::Array # Inferred from the B matrix
    num_controls::Array{Int,1} # Number of actions per factor
    control_fac_idx::Array{Int,1} # Indices of controllable factors
    policy_len::Int  # Policy length
    qs_current::Array{Any,1} # Current beliefs about states
    ##PTW_CR: Uncertain how this is used exactly, but I wouldn't have a separate prior field. There is just the current belief, and perhaps the previous belief if you need both at the same time. The prior is just the starting value.
    prior::Array{Any,1} # Prior beliefs about states
    Q_pi::Array{Real,1} # Posterior beliefs over policies
    G::Array{Real,1} # Expected free energy of policy
    ##PTW_CR: Call this previous_action
    action::Vector{Any} # Last action
    ##PTW_CR:To settings with these
    use_utility::Bool # Utility Boolean Flag
    use_states_info_gain::Bool # States Information Gain Boolean Flag
    use_param_info_gain::Bool # Include the novelty value in the learning parameters
    action_selection::String # Action selection: can be either "deterministic" or "stochastic"
    FPI_num_iter::Int # Number of iterations stopping condition in the FPI algorithm
    FPI_dF_tol::Float64 # Free energy difference stopping condition in the FPI algorithm
    states::Dict{String,Array{Any,1}} # States Dictionary
    parameters::Dict{String,Real} # Parameters Dictionary
    settings::Dict{String,Any} # Settings Dictionary
    ##PTW_CR: Also in settings
    save_history::Bool # Save history boolean flag
end

##PTW_CR: Perhaps call this function init_generative_model(), and let it dispatch on the types of inputs it gets
##PTW_CR: Why are there two functions here? I will look properly
##PTW_CR: Having looked, I think it's best just ot have one function instead of two here. 
##PTW_CR: Also, I really think: People should only ever input a matrix (like A) and not have a spearate field for inputting pA. 
##PTW_CR: If learning is not activated for the given matrix, then that matrix is just used and not updated. Then there should be a check for whether it is a proper probability distirbution.
##PTW_CR: If learning is activated, then the matrix is just used as the Dirchlet parameters. During initialization, the mean of the Dirhclet distirbution is then just taken right away to set the matrix.
##PTW_CR: I would have a little warning letting the user know which defaults are being used (unless vernose is set to false)

##PTW_CR: Different thing: A and B matrices can also have defaults where they are uniform. 
##PTW_CR: Maybe the workflow should be: 
##PTW_CR: 1) use the generate_matrix_templates function to create uniform versions of all matrices. Alternatively, specify them from the bottom.
##PTW_CR: 2) Input all these matrices into the init_generative_model function below
##PTW_CR: You might even create a struct that has all the matrices in it, and then pass that struct. I don't know.
##PTW_CR: Internally in the POMDPActiveInference there might be a struct with the matrices too. 

##PTW_CR: Make sure to type this function and the next.
# Create ActiveInference Agent 
function create_aif(A, B;
                    C = nothing,
                    D = nothing,
                    E = nothing,
                    pA = nothing, 
                    pB = nothing, 
                    pD = nothing, 
                    lr_pA = 1.0, 
                    fr_pA = 1.0, 
                    lr_pB = 1.0, 
                    fr_pB = 1.0, ##PTW_CR: I would call these 'learning_rate' and 'forgetting_rate'
                    lr_pD = 1.0, ##PTW_CR: A note: in time, we will treat these 'parameters' as states themselves, and not have these somewhat artificial learning and forgetting rates
                    fr_pD = 1.0, 
                    modalities_to_learn = "all",
                    factors_to_learn = "all", 
                    gamma=16.0,                 ##PTW_CR: Perhaps call these what they are: policy precision and action precision?
                    alpha=16.0, 
                    policy_len=1,               ##PTW_CR: policy_length
                    num_controls=nothing, 
                    control_fac_idx=nothing, 
                    use_utility=true,            ##PTW_CR: use_pragmatic_value
                    use_states_info_gain=true,   ##PTW_CR: Call it 'use_state_information_gain' or 'use_state_epistemic_value'
                    use_param_info_gain = false, ##PTW_CR: 'use_parameter_information_gain'
                    action_selection="stochastic",
                    FPI_num_iter=10,             ##PTW_CR: In settings, of course
                    FPI_dF_tol=0.001,
                    save_history=true
    )
    ##PTW_CR: I would store num_states etc in a field in the POMDPActiveInference somewhere (perhaps in the settings) - so that you can just access them 
    num_states = [size(B[f], 1) for f in eachindex(B)]
    num_obs = [size(A[f], 1) for f in eachindex(A)]

    # If C-vectors are not provided
    if isnothing(C)
        C = array_of_any_zeros(num_obs)
    end

    ##PTW_CR: These can be specified directly as defaults in the function signature
    # If D-vectors are not provided
    if isnothing(D)
        D = array_of_any_uniform(num_states)
    end


    ##PTW_CR: I don't think people should be able to both give num_control and also give the matrices. They might be inconsistent.
    ##PTW_CR: See above: I think there should be one function for generating the matrices based on num_blabla, and then another function that takes the matrices.
    # if num_controls are not given, they are inferred from the B matrix
    if isnothing(num_controls)
        num_controls = [size(B[f], 3) for f in eachindex(B)]  
    end

    ##PTW_CR: And same: I think the 'which factors are controllable' belongs when creating the matrices originally.
    ##PTW_CR: Basically, I think it's dagerous to have users input the same infomation twice in different formats.
    # Determine which factors are controllable
    if isnothing(control_fac_idx)
        control_fac_idx = [f for f in eachindex(num_controls) if num_controls[f] > 1]
    end

    policies = construct_policies_full(num_states, num_controls=num_controls, policy_len=policy_len, control_fac_idx=control_fac_idx)


    ##PTW_CR: It strikes me that setting the E matrix is going to be hard in general for people. They don't know how the policy list is constructed.
    ##PTW_CR: I think the default approach should be that people set an E matrix with an entry per _action_.
    ##PTW_CR: Then in the init function, this is transformed into a distribution over policies.
    ##PTW_CR: Optionally, people can set the E matrix with an entry per policy.
    ##PTW_CR: But in this case, you should make some helper function that shows them exactly which policies have which indices.
    # Throw error if the E-vector does not match the length of policies
    if !isnothing(E) && length(E) != length(policies)
        error("Length of E-vector must match the number of policies.")
    end

    ##PTW_CR: qs_current should be set to D. And that's really the only time D should be used.
    ##PTW_CR: I think these all should be part of an POMDPActiveInferenceStates struct
    qs_current = array_of_any_uniform(num_states)
    prior = D
    Q_pi = ones(Real,length(policies)) / length(policies)  
    G = zeros(Real,length(policies))
    action = []

    ##PTW_CR: Dictionaries are slow, but fully flexible (i.e. the user can add any field they want).
    ##PTW_CR: We don't need that here though, since there is a fixed set of fields. 
    ##PTW_CR: That's a good reason for using structs instead. No weird errors because of typos in the field names - and they're faster.
    ##PTW_CR: You can use the Base.@kwdef macro to make the struct have keyword arguments
    states = Dict(
        "action" => Vector{Real}[],
        "posterior_states" => Vector{Any}[],
        "prior" => Vector{Any}[],
        "posterior_policies" => Vector{Any}[],
        "expected_free_energies" => Vector{Any}[],
        "policies" => policies
    )

    ##PTW_CR: These are technically 'hyperparameters' - while the matrices are parameters.
    # initialize parameters dictionary
    parameters = Dict(
        "gamma" => gamma,
        "alpha" => alpha,
        "lr_pA" => lr_pA,
        "fr_pA" => fr_pA,
        "lr_pB" => lr_pB,
        "fr_pB" => fr_pB,
        "lr_pD" => lr_pD,
        "fr_pD" => fr_pD
    )

    # initialize settings dictionary
    settings = Dict(
        "policy_len" => policy_len,
        "num_controls" => num_controls,
        "control_fac_idx" => control_fac_idx,
        "use_utility" => use_utility,
        "use_states_info_gain" => use_states_info_gain,
        "use_param_info_gain" => use_param_info_gain,
        "action_selection" => action_selection,
        "modalities_to_learn" => modalities_to_learn,
        "factors_to_learn" => factors_to_learn,
        "FPI_num_iter" => FPI_num_iter,
        "FPI_dF_tol" => FPI_dF_tol
    )

    return AIF( A,
                B,
                C, 
                D, 
                E,
                pA, 
                pB, 
                pD,
                lr_pA, 
                fr_pA, 
                lr_pB, 
                fr_pB, 
                lr_pD, 
                fr_pD, 
                modalities_to_learn, 
                factors_to_learn, 
                gamma, 
                alpha, 
                policies, 
                num_controls, 
                control_fac_idx, 
                policy_len, 
                qs_current, 
                prior, 
                Q_pi, 
                G, 
                action, 
                use_utility,
                use_states_info_gain, 
                use_param_info_gain, 
                action_selection, 
                FPI_num_iter, 
                FPI_dF_tol,
                states,
                parameters, 
                settings, 
                save_history)
end



##PTW_CR: I think I would make none of the matrices optional here - but rather have that function which constructs all of them before, so they can be passed along here.
"""
Initialize Active Inference Agent
function init_aif(
        A,
        B;
        C=nothing,
        D=nothing,
        E = nothing,
        pA = nothing,
        pB = nothing, 
        pD = nothing,
        parameters::Union{Nothing, Dict{String,Real}} = nothing,
        settings::Union{Nothing, Dict} = nothing,
        save_history::Bool = true)

# Arguments
- 'A': Relationship between hidden states and observations.
- 'B': Transition probabilities.
- 'C = nothing': Prior preferences over observations.
- 'D = nothing': Prior over initial hidden states.
- 'E = nothing': Prior over policies. (habits)
- 'pA = nothing':
- 'pB = nothing':
- 'pD = nothing':
- 'parameters::Union{Nothing, Dict{String,Real}} = nothing':
- 'settings::Union{Nothing, Dict} = nothing':
- 'settings::Union{Nothing, Dict} = nothing':

"""
function init_aif(A, B; C=nothing, D=nothing, E = nothing, pA = nothing, pB = nothing, pD = nothing,
                  parameters::Union{Nothing, Dict{String, T}} where T<:Real = nothing,
                  settings::Union{Nothing, Dict} = nothing,
                  save_history::Bool = true, verbose::Bool = true)

    # Throw error if A, B or D is not a proper probability distribution  
    if !check_normalization(A)
        error("The A matrix is not normalized.")
    end

    if !check_normalization(B)
        error("The B matrix is not normalized.")
    end

    ##PTW_CR: Check the normalization for all the matrices. 
    ##PTW_CR: Can do that after initializing the unspecified matrices (if some of them are optional)
    ##PTW_CR: But I don't think they should be optional - or if they are, they should be specified in the function signature.
    
    if !isnothing(D) && !check_normalization(D)
        error("The D matrix is not normalized.")
    end

    # Throw warning if no D-vector is provided. 
    if verbose == true && isnothing(C)
        @warn "No C-vector provided, no prior preferences will be used."
    end 

    # Throw warning if no D-vector is provided. 
    if verbose == true && isnothing(D)
        @warn "No D-vector provided, a uniform distribution will be used."
    end 

    # Throw warning if no E-vector is provided. 
    if verbose == true && isnothing(E)
        @warn "No E-vector provided, a uniform distribution will be used."
    end           
    
    # Check if settings are provided or use defaults
    if verbose == true && isnothing(settings)
        ##PTW_CR: I think it might be nice to say which defaults are being used
        ##PTW_CR: Check the warn_premade_defaults or whatever it's called in ActionModels. You can take that if you want. 
        ##PTW_CR: I let people use a 'config' dict, which is later merged with the default one, and the unspecified settings are warned. 
        @warn "No settings provided, default settings will be used."
        settings = Dict(
            "policy_len" => 1, 
            "num_controls" => nothing, 
            "control_fac_idx" => nothing, 
            "use_utility" => true, 
            "use_states_info_gain" => true, 
            ##PTW_CR: It's kinda weird to have the option for enabling parameter learning, but not have the parameter info gain active
            ##PTW_CR: Also, I think in the code, but def in the docstrings and documentation, 
            ##PTW_CR: I think it's a good idea to be clear which things are part of the generative model
            ##PTW_CR: And which things are part of the action selection (like the use_ settings)
            "use_param_info_gain" => false,
            "action_selection" => "stochastic", 
            "modalities_to_learn" => "all",
            "factors_to_learn" => "all",
            "FPI_num_iter" => 10,
            "FPI_dF_tol" => 0.001
        )
    end

    ##PTW_CR: Same as above. Could all be in a config dict. Or could be a struct which the user creates and inputs (instead of a dict) - might be preferable.
    # Check if parameters are provided or use defaults
    if verbose == true && isnothing(parameters)
        @warn "No parameters provided, default parameters will be used."
        parameters = Dict("gamma" => 16.0,
                          "alpha" => 16.0,
                          "lr_pA" => 1.0,
                          "fr_pA" => 1.0,
                          "lr_pB" => 1.0,
                          "fr_pB" => 1.0,
                          "lr_pD" => 1.0,
                          "fr_pD" => 1.0
                          )
    end

    # Extract parameters and settings from the dictionaries or use defaults
    gamma = get(parameters, "gamma", 16.0)  
    alpha = get(parameters, "alpha", 16.0)
    lr_pA = get(parameters, "lr_pA", 1.0)
    fr_pA = get(parameters, "fr_pA", 1.0)
    lr_pB = get(parameters, "lr_pB", 1.0)
    fr_pB = get(parameters, "fr_pB", 1.0)
    lr_pD = get(parameters, "lr_pD", 1.0)
    fr_pD = get(parameters, "fr_pD", 1.0)

    
    policy_len = get(settings, "policy_len", 1)
    num_controls = get(settings, "num_controls", nothing)
    control_fac_idx = get(settings, "control_fac_idx", nothing)
    use_utility = get(settings, "use_utility", true)
    use_states_info_gain = get(settings, "use_states_info_gain", true)
    use_param_info_gain = get(settings, "use_param_info_gain", false)
    action_selection = get(settings, "action_selection", "stochastic")
    modalities_to_learn = get(settings, "modalities_to_learn", "all" )
    factors_to_learn = get(settings, "factors_to_learn", "all" )
    FPI_num_iter = get(settings, "FPI_num_iter", 10 )
    FPI_dF_tol = get(settings, "FPI_dF_tol", 0.001 )

    # Call create_aif 
    aif = create_aif(A, B,
                    C=C,
                    D=D,
                    E=E,
                    pA=pA,
                    pB=pB,
                    pD=pD,
                    lr_pA = lr_pA, 
                    fr_pA = fr_pA, 
                    lr_pB = lr_pB, 
                    fr_pB = fr_pB, 
                    lr_pD = lr_pD, 
                    fr_pD = fr_pD,
                    modalities_to_learn=modalities_to_learn,
                    factors_to_learn=factors_to_learn,
                    gamma=gamma,
                    alpha=alpha, 
                    policy_len=policy_len,
                    num_controls=num_controls,
                    control_fac_idx=control_fac_idx, 
                    use_utility=use_utility, 
                    use_states_info_gain=use_states_info_gain, 
                    use_param_info_gain=use_param_info_gain,
                    action_selection=action_selection,
                    FPI_num_iter=FPI_num_iter,
                    FPI_dF_tol=FPI_dF_tol,
                    save_history=save_history
                    )

    ##PTW_CR: Do this with a pretty-print for the POMDPActiveInference struct. https://docs.julialang.org/en/v1/manual/types/#man-custom-pretty-printing
    #Print out agent settings
    if verbose == true
        settings_summary = 
        """
        AIF Agent initialized successfully with the following settings and parameters:
        - Gamma (γ): $(aif.gamma)
        - Alpha (α): $(aif.alpha)
        - Policy Length: $(aif.policy_len)
        - Number of Controls: $(aif.num_controls)
        - Controllable Factors Indices: $(aif.control_fac_idx)
        - Use Utility: $(aif.use_utility)
        - Use States Information Gain: $(aif.use_states_info_gain)
        - Use Parameter Information Gain: $(aif.use_param_info_gain)
        - Action Selection: $(aif.action_selection)
        - Modalities to Learn = $(aif.modalities_to_learn)
        - Factors to Learn = $(aif.factors_to_learn)
        """
        println(settings_summary)
    end
    
    return aif
end


##PTW_CR: I would put the below in files for perception!(), learning!() and action!(), as discussed in the POMDP-file comments
""" Update the agents's beliefs over states """
function infer_states!(aif::AIF, obs::Vector{Int64})
    
    if !isempty(aif.action)
        
        ##PTW_CR: Why round them? Just type the action field to only have integers. 
        ##PTW_CR: That might clash with Turing (it needs fields to be <:Real, so that they can both include the original values and the autodiff stuff)
        ##PTW_CR: Outcomes of the sample from a Categorical should always be Integers though, so that'd be fine here. 
        ##PTW_CR: Use Integer types except when it breaks Turing - just test it out.
        int_action = round.(Int, aif.action)
        ##PTW_CR: This is the belief - not the prior. (that is, the belief is a prior on the nect timestep, and a posterior after the update)
        ##PTW_CR: So call it 'belief' or 'state_posterior' or something like that
        aif.prior = get_expected_states(aif.qs_current, aif.B, reshape(int_action, 1, length(int_action)))[1]
        ##PTW_CR: I'll look at this better: but I guess 'get_expected_states' is really 'get_prediction' or 'update prediction'. That would be a terminology that fits better with the HGF etc.

        ##PTW_CR: In general, there is first a prediction: then an update (sometimes explicilty including a prediciton error): and then a belief update
    else
        aif.prior = aif.D
    end

    ##PTW_CR: This would be the belief update
    # Update posterior over states
    aif.qs_current = update_posterior_states(aif.A, obs, prior=aif.prior, num_iter=aif.FPI_num_iter, dF_tol=aif.FPI_dF_tol)

    ##PTW_CR: Firstly, have a .history field which is separate fom the .states field
    ##PTW_CR: Secondly, make the .history field a mutable struct instead of a dict
    ##PTW_CR: Thirdly, only push to the .history field if save_history=true
    ##PTW_CR: This saves _a lot_ of runtime.
    ##PTW_CR: Also: are you sure that the copy is necessary here? I don't use it in the HGF code. 
    # Push changes to agent's history
    push!(aif.states["prior"], copy(aif.prior))
    push!(aif.states["posterior_states"], copy(aif.qs_current))

end

""" Update the agents's beliefs over policies """
function infer_policies!(aif::AIF)
    # Update posterior over policies and expected free energies of policies
    ##PTW_CR: Up to you: but you mgiht consider just passing the AIF struct to the function, and then unpacking inside it. This is something I'm debating with myself currently though, so I'm not sure what the best way is yet.
    q_pi, G = update_posterior_policies(aif.qs_current, aif.A, aif.B, aif.C, aif.policies, aif.use_utility, aif.use_states_info_gain, aif.use_param_info_gain, aif.pA, aif.pB, aif.E, aif.gamma)

    ##PTW_CR: In ActionModels, there is an update_state!() function, which sets the state to the new state, and updates the history if save_history is true. 
    ##PTW_CR: May be convenient to use that here.
    ##PTW_CR: Perhaps more or just as convenient is to have one block in the end where all the history is updated, i save_history is true.
    aif.Q_pi = q_pi
    aif.G = G  

    # Push changes to agent's history
    push!(aif.states["posterior_policies"], copy(aif.Q_pi))
    push!(aif.states["expected_free_energies"], copy(aif.G))

    return q_pi, G
end

##PTW_CR: I amde suggestions regarding this part in the POMDP-file comments.
""" Sample action from the beliefs over policies """
function sample_action!(aif::AIF)
    action = sample_action(aif.Q_pi, aif.policies, aif.num_controls; action_selection=aif.action_selection, alpha=aif.alpha)

    aif.action = action 

    # Push action to agent's history
    push!(aif.states["action"], copy(aif.action))


    return action
end

##PTW_CR: I think these should all be part of a parameter_learning!() function
##PTW_CR: And then there should be settings which decide which of these are being called.

""" Update A-matrix """
function update_A!(aif::AIF, obs::Vector{Int64})

    qA = update_obs_likelihood_dirichlet(aif.pA, aif.A, obs, aif.qs_current, lr = aif.lr_pA, fr = aif.fr_pA, modalities = aif.modalities_to_learn)
    
    ##PTW_CR: This is a good example of where it is a bit weird to call it 'pA' for 'priorA', since it here is also the posterior.
    ##PTW_CR: If anything, I would call it posterior (since that it what it will be normally used as)
    ##PTW_CR: And then, of course, a posteiror is a prior on the next timestep
    ##PTW_CR: And again, I prefer them spelled out to using p and q. Short parameter name conventions are a remnant from old times without autocompletion
    aif.pA = qA
    aif.A = norm_dist_array(qA)

    return qA
end

##PTW_CR: Remember to type these
""" Update B-matrix """
function update_B!(aif::AIF, qs_prev)

    ##PTW_CR: Why not just call this function update_B() ? Same point above. More consistent.
    qB = update_state_likelihood_dirichlet(aif.pB, aif.B, aif.action, aif.qs_current, qs_prev, lr = aif.lr_pB, fr = aif.fr_pB, factors = aif.factors_to_learn)

    aif.pB = qB
    aif.B = norm_dist_array(qB)

    return qB
end

##PTW_CR: I like the solution of having versions with and without the '!' in the end. Might steal that for the HGF package.
""" Update D-matrix """
function update_D!(aif::AIF, qs_t1)

    qD = update_state_prior_dirichlet(aif.pD, qs_t1; lr = aif.lr_pD, fr = aif.fr_pD, factors = aif.factors_to_learn)

    aif.pD = qD
    aif.D = norm_dist_array(qD)

    return qD
end