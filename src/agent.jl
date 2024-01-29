""" -------- Agent Mutable Struct -------- """

using LinearAlgebra


mutable struct Agent
    A::Array{Any,1}
    B::Array{Any,1}
    C::Array{Any,1}  
    D::Array{Any,1}
    policies::Array  # Inferred from the B matrix
    num_controls::Array{Int,1}  # Number of actions per factor
    control_fac_idx::Array{Int,1}  # Indices of controllable factors
    policy_len::Int  # Policy length
    qs_current::Array{Any,1}  # Current beliefs about states
    prior::Array{Any,1}  # Prior beliefs about states
    Q_pi::Array{Float64,1} # Posterior beliefs over policies
    G::Array{Float64,1} # Expected free energy of policy
    action::Vector{Float64} # Last action
    E::Any # E - vector (Habits)
    gamma::Float64 # Gamma parameter
    alpha::Float64 # Alpha parameter
    use_utility::Bool # Utility Boolean Flag
    use_states_info_gain::Bool # States Information Gain Boolean Flag
    action_selection::String # Action selection: can be either "deterministic" or "stochastic"   

end

function initialize_agent(A, B, C, D; num_controls=nothing, control_fac_idx=nothing, policy_len=1, E=nothing, gamma=16.0, alpha=16.0, use_utility=true, use_states_info_gain=true, action_selection="stochastic")
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
    return Agent(A, B, C, D, policies, num_controls, control_fac_idx, policy_len, qs_current, prior, Q_pi, G, action, E, gamma, alpha, use_utility, use_states_info_gain, action_selection)
end

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

# Agent sample action
function sample_action!(agent::Agent)
    action = sample_action(agent.Q_pi, agent.policies, agent.num_controls; action_selection=agent.action_selection, alpha=agent.alpha)

    # Agent sample action
    agent.action = action 
    return action
end