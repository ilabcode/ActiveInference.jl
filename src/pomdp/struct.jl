""" -------- AIF Mutable Struct -------- """

mutable struct AIF
    A::Vector{Array{<:Real}} # A-matrix
    B::Vector{Array{<:Real}} # B-matrix
    C::Vector{Array{<:Real}} # C-vectors
    D::Vector{Array{<:Real}} # D-vectors
    E::Union{Vector{<:Real}, Nothing}  # E-vector (Habits)
    pA::Union{Vector{Array{<:Real}}, Nothing} # Dirichlet priors for A-matrix
    pB::Union{Vector{Array{<:Real}}, Nothing} # Dirichlet priors for B-matrix
    pD::Union{Vector{Array{<:Real}}, Nothing} # Dirichlet priors for D-vector
    lr_pA::Real # pA Learning Parameter
    fr_pA::Real # pA Forgetting Parameter,  1.0 for no forgetting
    lr_pB::Real # pB learning Parameter
    fr_pB::Real # pB Forgetting Parameter
    lr_pD::Real # pD Learning parameter
    fr_pD::Real # PD Forgetting parameter
    modalities_to_learn::Union{String, Vector{Int64}} # Modalities can be eithe "all" or "# modality"
    factors_to_learn::Union{String, Vector{Int64}} # Modalities can be either "all" or "# factor"
    gamma::Real # Gamma parameter
    alpha::Real # Alpha parameter
    policies::Array # Inferred from the B matrix
    num_controls::Array{Int,1} # Number of actions per factor
    control_fac_idx::Array{Int,1} # Indices of controllable factors
    policy_len::Int  # Policy length
    qs_current::Array{Any,1} # Current beliefs about states
    prior::Array{Any,1} # Prior beliefs about states
    Q_pi::Array{Real,1} # Posterior beliefs over policies
    G::Array{Real,1} # Expected free energy of policy
    action::Vector{Any} # Last action
    use_utility::Bool # Utility Boolean Flag
    use_states_info_gain::Bool # States Information Gain Boolean Flag
    use_param_info_gain::Bool # Include the novelty value in the learning parameters
    action_selection::String # Action selection: can be either "deterministic" or "stochastic"
    FPI_num_iter::Int # Number of iterations stopping condition in the FPI algorithm
    FPI_dF_tol::Float64 # Free energy difference stopping condition in the FPI algorithm
    states::Dict{String,Array{Any,1}} # States Dictionary
    parameters::Dict{String,Real} # Parameters Dictionary
    settings::Dict{String,Any} # Settings Dictionary
    save_history::Bool # Save history boolean flag
end

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
                    fr_pB = 1.0, 
                    lr_pD = 1.0, 
                    fr_pD = 1.0, 
                    modalities_to_learn = "all", 
                    factors_to_learn = "all", 
                    gamma=16.0, 
                    alpha=16.0, 
                    policy_len=1, 
                    num_controls=nothing, 
                    control_fac_idx=nothing, 
                    use_utility=true, 
                    use_states_info_gain=true, 
                    use_param_info_gain = false, 
                    action_selection="stochastic",
                    FPI_num_iter=10,
                    FPI_dF_tol=0.001,
                    save_history=true
    )

    num_states = [size(B[f], 1) for f in eachindex(B)]
    num_obs = [size(A[f], 1) for f in eachindex(A)]

    # If C-vectors are not provided
    if isnothing(C)
        C = create_matrix_templates(num_obs, "zeros")
    end

    # If D-vectors are not provided
    if isnothing(D)
        D = create_matrix_templates(num_states)
    end

    # if num_controls are not given, they are inferred from the B matrix
    if isnothing(num_controls)
        num_controls = [size(B[f], 3) for f in eachindex(B)]  
    end

    # Determine which factors are controllable
    if isnothing(control_fac_idx)
        control_fac_idx = [f for f in eachindex(num_controls) if num_controls[f] > 1]
    end

    policies = construct_policies(num_states, n_controls=num_controls, policy_length=policy_len, controllable_factors_indices=control_fac_idx)

    # Throw error if the E-vector does not match the length of policies
    if !isnothing(E) && length(E) != length(policies)
        error("Length of E-vector must match the number of policies.")
    end

    qs_current = create_matrix_templates(num_states)
    prior = D
    Q_pi = ones(Real,length(policies)) / length(policies)  
    G = zeros(Real,length(policies))
    action = []

    # initialize states dictionary
    states = Dict(
        "action" => Vector{Real}[],
        "posterior_states" => Vector{Any}[],
        "prior" => Vector{Any}[],
        "posterior_policies" => Vector{Any}[],
        "expected_free_energies" => Vector{Any}[],
        "policies" => policies
    )

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
function init_aif(A, B; C=nothing, D=nothing, E=nothing, pA=nothing, pB=nothing, pD=nothing,
                  parameters::Union{Nothing, Dict{String, T}} where T<:Real = nothing,
                  settings::Union{Nothing, Dict} = nothing,
                  save_history::Bool = true, verbose::Bool = true)

    # Catch error if A, B or D is not a proper probability distribution  
    # Check A matrix
    try
        if !check_probability_distribution(A)
            error("The A matrix is not a proper probability distribution.")
        end
    catch e
        # Add context and rethrow the error
        error("The A matrix is not a proper probability distribution. Details: $(e)")
    end

    # Check B matrix
    try
        if !check_probability_distribution(B)
            error("The B matrix is not a proper probability distribution.")
        end
    catch e
        # Add context and rethrow the error
        error("The B matrix is not a proper probability distribution. Details: $(e)")
    end

    # Check D matrix (if it's not nothing)
    try
        if !isnothing(D) && !check_probability_distribution(D)
            error("The D matrix is not a proper probability distribution.")
        end
    catch e
        # Add context and rethrow the error
        error("The D matrix is not a proper probability distribution. Details: $(e)")
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
    if isnothing(settings)

        if verbose == true
            @warn "No settings provided, default settings will be used."
        end

        settings = Dict(
            "policy_len" => 1, 
            "num_controls" => nothing, 
            "control_fac_idx" => nothing, 
            "use_utility" => true, 
            "use_states_info_gain" => true, 
            "use_param_info_gain" => false,
            "action_selection" => "stochastic", 
            "modalities_to_learn" => "all",
            "factors_to_learn" => "all",
            "FPI_num_iter" => 10,
            "FPI_dF_tol" => 0.001
        )
    end

    # Check if parameters are provided or use defaults
    if isnothing(parameters)

        if verbose == true
            @warn "No parameters provided, default parameters will be used."
        end
        
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

### Struct related functions ###

"""
    construct_policies(n_states::Vector{T} where T <: Real; n_controls::Union{Vector{T}, Nothing} where T <: Real=nothing, 
                       policy_length::Int=1, controllable_factors_indices::Union{Vector{Int}, Nothing}=nothing)

Construct policies based on the number of states, controls, policy length, and indices of controllable state factors.

# Arguments
- `n_states::Vector{T} where T <: Real`: A vector containing the number of  states for each factor.
- `n_controls::Union{Vector{T}, Nothing} where T <: Real=nothing`: A vector specifying the number of allowable actions for each state factor. 
- `policy_length::Int=1`: The length of policies. (planning horizon)
- `controllable_factors_indices::Union{Vector{Int}, Nothing}=nothing`: A vector of indices identifying which state factors are controllable.

"""
function construct_policies(
    n_states::Vector{T} where T <: Real; 
    n_controls::Union{Vector{T}, Nothing} where T <: Real=nothing, 
    policy_length::Int=1, 
    controllable_factors_indices::Union{Vector{Int}, Nothing}=nothing
    )

    # Determine the number of state factors
    n_factors = length(n_states)

    # If indices of controllable factors are not given 
    if isnothing(controllable_factors_indices)
        if !isnothing(n_controls)
            # Determine controllable factors based on which factors have more than one control
            controllable_factors_indices = findall(x -> x > 1, n_controls)
        else
            # If no controls are given, assume all factors are controllable
            controllable_factors_indices = 1:n_factors
        end
    end

    # if number of controls is not given, determine it based n_states and controllable_factors_indices
    if isnothing(n_controls)
        n_controls = [in(factor_index, controllable_factors_indices) ? n_states[factor_index] : 1 for factor_index in 1:n_factors]
    end

    # Create a vector of possible actions for each time step
    x = repeat(n_controls, policy_length)

    # Generate all combinations of actions across all time steps
    policies = collect(Iterators.product([1:i for i in x]...))

    # Initialize an empty vector to store transformed policies
    transformed_policies = Vector{Matrix{Int64}}()

    for policy_tuple in policies
        # Convert tuple into a vector
        policy_vector = collect(policy_tuple)
        
        # Reshape the policy vector into a matrix and transpose it
        policy_matrix = reshape(policy_vector, (length(policy_vector) ÷ policy_length, policy_length))'
        
        # Push the reshaped matrix to the vector of transformed policies
        push!(transformed_policies, policy_matrix)
    end

    return transformed_policies
end

""" Update the agents's beliefs over states """
function infer_states!(aif::AIF, obs::Vector{Int64})
    if !isempty(aif.action)
        int_action = round.(Int, aif.action)
        aif.prior = get_expected_states(aif.qs_current, aif.B, reshape(int_action, 1, length(int_action)))[1]
    else
        aif.prior = aif.D
    end

    # Update posterior over states
    aif.qs_current = update_posterior_states(aif.A, obs, prior=aif.prior, num_iter=aif.FPI_num_iter, dF_tol=aif.FPI_dF_tol)

    # Push changes to agent's history
    push!(aif.states["prior"], copy(aif.prior))
    push!(aif.states["posterior_states"], copy(aif.qs_current))

end

""" Update the agents's beliefs over policies """
function infer_policies!(aif::AIF)
    # Update posterior over policies and expected free energies of policies
    q_pi, G = update_posterior_policies(aif.qs_current, aif.A, aif.B, aif.C, aif.policies, aif.use_utility, aif.use_states_info_gain, aif.use_param_info_gain, aif.pA, aif.pB, aif.E, aif.gamma)

    aif.Q_pi = q_pi
    aif.G = G  

    # Push changes to agent's history
    push!(aif.states["posterior_policies"], copy(aif.Q_pi))
    push!(aif.states["expected_free_energies"], copy(aif.G))

    return q_pi, G
end

""" Sample action from the beliefs over policies """
function sample_action!(aif::AIF)
    action = sample_action(aif.Q_pi, aif.policies, aif.num_controls; action_selection=aif.action_selection, alpha=aif.alpha)

    aif.action = action 

    # Push action to agent's history
    push!(aif.states["action"], copy(aif.action))


    return action
end

""" Update A-matrix """
function update_A!(aif::AIF, obs::Vector{Int64})

    qA = update_obs_likelihood_dirichlet(aif.pA, aif.A, obs, aif.qs_current, lr = aif.lr_pA, fr = aif.fr_pA, modalities = aif.modalities_to_learn)
    
    aif.pA = qA
    aif.A = normalize_arrays(qA)

    return qA
end

""" Update B-matrix """
function update_B!(aif::AIF, qs_prev)

    qB = update_state_likelihood_dirichlet(aif.pB, aif.B, aif.action, aif.qs_current, qs_prev, lr = aif.lr_pB, fr = aif.fr_pB, factors = aif.factors_to_learn)

    aif.pB = qB
    aif.B = normalize_arrays(qB)

    return qB
end

""" Update D-matrix """
function update_D!(aif::AIF, qs_t1)

    qD = update_state_prior_dirichlet(aif.pD, qs_t1; lr = aif.lr_pD, fr = aif.fr_pD, factors = aif.factors_to_learn)

    aif.pD = qD
    aif.D = normalize_arrays(qD)

    return qD
end