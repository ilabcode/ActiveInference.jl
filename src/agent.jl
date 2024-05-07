""" -------- AIF Mutable Struct -------- """

using LinearAlgebra

mutable struct AIF
    A::Array{Any,1} # A-matrix
    B::Array{Any,1} # B-matrix
    C::Array{Any,1} # C-vectors
    D::Array{Any,1} # D-vectors
    E::Union{Array{Any, 1}, Nothing} # E - vector (Habits)
    pA::Union{Array{Any,1}, Nothing}
    pB::Union{Array{Any,1}, Nothing}
    pD::Union{Array{Any,1}, Nothing}
    lr_pA::Float64 # pA Learning Parameter
    fr_pA::Float64 # pA Forgetting Parameter,  1.0 for no forgetting
    lr_pB::Float64 # pB learning Parameter
    fr_pB::Float64 # pB Forgetting Parameter
    lr_pD::Float64 # pD Learning parameter
    fr_pD::Float64 # PD Forgetting parameter
    modalities_to_learn::Union{String, Vector{Int64}} # Modalities can be eithe "all" or "# modality"
    factors_to_learn::Union{String, Vector{Int64}} # Modalities can be eithe "all" or "# factor"
    gamma::Float64 # Gamma parameter
    alpha::Float64 # Alpha parameter
    policies::Array  # Inferred from the B matrix
    num_controls::Array{Int,1}  # Number of actions per factor
    control_fac_idx::Array{Int,1}  # Indices of controllable factors
    policy_len::Int  # Policy length
    qs_current::Array{Any,1}  # Current beliefs about states
    prior::Array{Any,1}  # Prior beliefs about states
    Q_pi::Array{Float64,1} # Posterior beliefs over policies
    G::Array{Float64,1} # Expected free energy of policy
    action::Vector{Any} # Last action
    use_utility::Bool # Utility Boolean Flag
    use_states_info_gain::Bool # States Information Gain Boolean Flag
    use_param_info_gain::Bool # Include the novelty value in the learning parameters
    action_selection::String # Action selection: can be either "deterministic" or "stochastic"   
    states::Dict{String,Array{Any,1}} # States Dictionary
    parameters::Dict{String,Float64} # Parameters Dictionary
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
                    save_history=true
    )

    num_states = [size(B[f], 1) for f in eachindex(B)]
    num_obs = [size(A[f], 1) for f in eachindex(A)]

    # If C-vectors are not provided
    if isnothing(C)
        C = array_of_any_zeros(num_obs)
    end

    # If D-vectors are not provided
    if isnothing(D)
        D = array_of_any_uniform(num_states)
    end

    # if num_controls are not given, they are inferred from the B matrix
    if isnothing(num_controls)
        num_controls = [size(B[f], 3) for f in eachindex(B)]  
    end

    # Determine which factors are controllable
    if isnothing(control_fac_idx)
        control_fac_idx = [f for f in eachindex(num_controls) if num_controls[f] > 1]
    end

    policies = construct_policies_full(num_states, num_controls=num_controls, policy_len=policy_len, control_fac_idx=control_fac_idx)
    qs_current = array_of_any_uniform(num_states)
    prior = D
    Q_pi = ones(length(policies)) / length(policies)  
    G = zeros(length(policies))
    action = []

    # initialize states dictionary
    states = Dict(
        "action" => Vector{Float64}[],
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
        "factors_to_learn" => factors_to_learn
    )

    return AIF(A, B, C, D, E, pA, pB, pD, lr_pA, fr_pA, lr_pB, fr_pB, lr_pD, fr_pD, modalities_to_learn, factors_to_learn, gamma, alpha, policies, num_controls, control_fac_idx, policy_len, qs_current, prior, Q_pi, G, action, use_utility, use_states_info_gain, use_param_info_gain, action_selection, states, parameters, settings, save_history)
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
        parameters::Union{Nothing, Dict{String,Float64}} = nothing,
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
- 'parameters::Union{Nothing, Dict{String,Float64}} = nothing':
- 'settings::Union{Nothing, Dict} = nothing':
- 'settings::Union{Nothing, Dict} = nothing':

"""
function init_aif(A, B; C=nothing, D=nothing, E = nothing, pA = nothing, pB = nothing, pD = nothing,
                  parameters::Union{Nothing, Dict{String,Float64}} = nothing,
                  settings::Union{Nothing, Dict} = nothing,
                  save_history::Bool = true)

    # Throw warning if no D-vector is provided. 
    if isnothing(C)
        @warn "No C-vector provided, no prior preferences will be used."
    end 

    # Throw warning if no D-vector is provided. 
    if isnothing(D)
        @warn "No D-vector provided, a uniform distribution will be used."
    end 

    # Throw warning if no E-vector is provided. 
    if isnothing(E)
        @warn "No E-vector provided, a uniform distribution will be used."
    end           
    
    # Check if settings are provided or use defaults
    if isnothing(settings)
        @warn "No settings provided, default settings will be used."
        settings = Dict(
            "policy_len" => 1, 
            "num_controls" => nothing, 
            "control_fac_idx" => nothing, 
            "use_utility" => true, 
            "use_states_info_gain" => true, 
            "use_param_info_gain" => false,
            "action_selection" => "stochastic", 
            "modalities_to_learn" => "all",
            "factors_to_learn" => "all"
        )
    end

    # Check if parameters are provided or use defaults
    if isnothing(parameters)
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
                    save_history=save_history
                    )

    #Print out agent settings
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
    
    return aif
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
    aif.qs_current = update_posterior_states(aif.A, obs, prior=aif.prior) 

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
    aif.A = norm_dist_array(qA)

    return qA
end

""" Update B-matrix """
function update_B!(aif::AIF, qs_prev)

    qB = update_state_likelihood_dirichlet(aif.pB, aif.B, aif.action, aif.qs_current, qs_prev, lr = aif.lr_pB, fr = aif.fr_pB, factors = aif.factors_to_learn)

    aif.pB = qB
    aif.B = norm_dist_array(qB)

    return qB
end

""" Update D-matrix """
function update_D!(aif::AIF, qs_t1)

    qD = update_state_prior_dirichlet(aif.pD, qs_t1; lr = aif.lr_pD, fr = aif.fr_pD, factors = aif.factors_to_learn)

    aif.pD = qD
    aif.D = norm_dist_array(qD)

    return qD
end