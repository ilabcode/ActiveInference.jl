""" -------- Agent Mutable Struct -------- """

using LinearAlgebra

mutable struct Agent
    A::Array{Any,1}
    B::Array{Any,1}
    C::Array{Any,1}  
    D::Array{Any,1}
    E::Any # E - vector (Habits)
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
    action::Vector{Float64} # Last action
    use_utility::Bool # Utility Boolean Flag
    use_states_info_gain::Bool # States Information Gain Boolean Flag
    action_selection::String # Action selection: can be either "deterministic" or "stochastic"   
    states::Dict{String,Array{Any,1}} # States Dictionary
    parameters::Dict{String,Float64} # Parameters Dictionary
    settings::Dict{String,Any} # Settings Dictionary
end

# Create ActiveInference Agent 
function create_agent(A, B, C, D, E=nothing; gamma=16.0, alpha=16.0, policy_len=1, num_controls=nothing, control_fac_idx=nothing, use_utility=true, use_states_info_gain=true, action_selection="stochastic")
    num_states = [size(B[f], 1) for f in eachindex(B)]

    # if num_controls are not given, they are inferred from the B matrix
    if isnothing(num_controls)
        num_controls = [size(B[f], 3) for f in eachindex(B)]  
    end

    # Determine which factors are controllable
    if isnothing(control_fac_idx)
        control_fac_idx = [f for f in eachindex(num_controls) if num_controls[f] > 1]
    end

    policies = construct_policies_full(num_states, num_controls=num_controls, policy_len=policy_len, control_fac_idx=control_fac_idx)
    qs_current = onehot(1, size(B, 1))
    prior = D
    Q_pi = ones(length(policies)) / length(policies)  
    G = zeros(length(policies))
    action = Float64[]

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
        "alpha" => alpha
    )

    # initialize settings dictionary
    settings = Dict(
        "policy_len" => policy_len,
        "num_controls" => num_controls,
        "control_fac_idx" => control_fac_idx,
        "use_utility" => use_utility,
        "use_states_info_gain" => use_states_info_gain,
        "action_selection" => action_selection
    )

    return Agent(A, B, C, D, E, gamma, alpha, policies, num_controls, control_fac_idx, policy_len, qs_current, prior, Q_pi, G, action, use_utility, use_states_info_gain, action_selection, states, parameters, settings)
end

# Initialize active inference agent 
function init_aif(A, B, C, D; E = nothing,
                  parameters::Union{Nothing, Dict{String,Float64}} = nothing,
                  settings::Union{Nothing, Dict} = nothing)

    # Throw warning if no E-vector is provided. 
    if isnothing(E)
        @warn "No E vector provided, a uniform distribution will be used."
    end           
    
    # Check if settings and parameters are provided or use defaults
    if isnothing(settings)
        @warn "No settings provided, default settings will be used."
        settings = Dict(
            "policy_len" => 1, 
            "num_controls" => nothing, 
            "control_fac_idx" => nothing, 
            "use_utility" => true, 
            "use_states_info_gain" => true, 
            "action_selection" => "stochastic"
        )
    end

    if isnothing(parameters)
        @warn "No parameters provided, default parameters will be used."
        parameters = Dict("gamma" => 16.0,
                          "alpha" => 16.0)
    end

    # Extract parameters and settings from the dictionaries or use defaults
    gamma = get(parameters, "gamma", 16.0)  
    alpha = get(parameters, "alpha", 16.0)
    
    policy_len = get(settings, "policy_len", 1)
    num_controls = get(settings, "num_controls", nothing)
    control_fac_idx = get(settings, "control_fac_idx", nothing)
    use_utility = get(settings, "use_utility", true)
    use_states_info_gain = get(settings, "use_states_info_gain", true)
    action_selection = get(settings, "action_selection", "stochastic")

    # Call create_agent 
    agent = create_agent(A, B, C, D, E; 
                         gamma=gamma,
                         alpha=alpha, 
                         policy_len=policy_len,
                         num_controls=num_controls,
                         control_fac_idx=control_fac_idx, 
                         use_utility=use_utility, 
                         use_states_info_gain=use_states_info_gain, 
                         action_selection=action_selection)
    
    println("Agent initialized successfully!")
    return agent
end

# Update the agent's beliefs over states
function infer_states!(agent::Agent, obs)
    if !isempty(agent.action)
        int_action = round.(Int, agent.action)
        agent.prior = get_expected_states(agent.qs_current, agent.B, reshape(int_action, 1, length(int_action)))[1]
    else
        agent.prior = agent.D
    end
    agent.qs_current = update_posterior_states(agent.A, obs, prior=agent.prior) 
end

# Update the agent's beliefs over policies
function infer_policies!(agent::Agent)
    q_pi, G = update_posterior_policies(agent.qs_current, agent.A, agent.B, agent.C, agent.policies, agent.use_utility, agent.use_states_info_gain,agent.E, agent.gamma)

    agent.Q_pi = q_pi
    agent.G = G  
    return q_pi, G
end

# Sample action from the beliefs over policies
function sample_action!(agent::Agent)
    action = sample_action(agent.Q_pi, agent.policies, agent.num_controls; action_selection=agent.action_selection, alpha=agent.alpha)

    # Agent sample action
    agent.action = action 
    return action
end