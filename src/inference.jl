""" -------- Inference Functions -------- """

using LinearAlgebra
using IterTools

#### State Inference #### 

""" Get Expected States """
function get_expected_states(qs, B, policy)
    n_steps, n_factors = size(policy)

    # initializing posterior predictive density as a list of beliefs over time
    qs_pi = [deepcopy(qs) for _ in 1:n_steps+1]

    # expected states over time
    for t in 1:n_steps
        for control_factor in 1:n_factors
            action = policy[t, control_factor]
            
            qs_pi[t+1][control_factor] = B[control_factor][:, :, action] * qs_pi[t][control_factor]
        end
    end

    return qs_pi[2:end]
end

""" Update Posterior States """
function update_posterior_states(A::Vector{Any}, obs::Vector{Int64}; prior::Union{Nothing, Vector{Any}}=nothing, kwargs...)
    num_obs, num_states, num_modalities, num_factors = get_model_dimensions(A)

    obs_processed = process_observation(obs, num_modalities, num_obs)
    return fixed_point_iteration(A, obs_processed, num_obs, num_states, prior=prior)
end


""" Run State Inference via Fixed-Point Iteration """
function fixed_point_iteration(A::Vector{Any}, obs::Vector{Any}, num_obs::Vector{Int64}, num_states::Vector{Int64}; prior::Union{Nothing, Vector{Any}}=nothing, num_iter=10, dF=1.0, dF_tol=0.001)
    n_modalities = length(num_obs)
    n_factors = length(num_states)

    # Get joint likelihood
    likelihood = get_joint_likelihood(A, obs, num_states)
    likelihood = spm_log_single(likelihood)

    # Initialize posterior and prior
    qs = Array{Any}(undef, n_factors)
    for factor in 1:n_factors
        qs[factor] = ones(num_states[factor]) / num_states[factor]
    end

    if prior === nothing
        prior = array_of_any_uniform(num_states)
    end
    
    prior = spm_log_array_any(prior) 

    # Initialize free energy
    prev_vfe = calc_free_energy(qs, prior, n_factors)

    # Single factor condition
    if n_factors == 1
        qL = dot_likelihood(likelihood, qs[1])  
        return to_array_of_any(softmax(qL .+ prior[1]))
    else
        # Run Iteration 
        curr_iter = 0
        while curr_iter < num_iter && dF >= dF_tol
            qs_all = qs[1]
            for factor in 2:n_factors
                qs_all = qs_all .* reshape(qs[factor], tuple(ones(Int, factor - 1)..., :, 1))
            end
            LL_tensor = likelihood .* qs_all

            for factor in 1:n_factors
                qL = zeros(size(qs[factor]))
                for i in 1:size(qs[factor], 1)
                    qL[i] = sum([LL_tensor[indices...] / qs[factor][i] for indices in Iterators.product([1:size(LL_tensor, dim) for dim in 1:n_factors]...) if indices[factor] == i])
                end
                qs[factor] = softmax(qL + prior[factor])
            end

            # Recompute free energy
            vfe = calc_free_energy(qs, prior, n_factors, likelihood)

            # Update stopping condition
            dF = abs(prev_vfe - vfe)
            prev_vfe = vfe

            curr_iter += 1
        end

        return qs
    end
end



""" Calculate Accuracy Term """
function compute_accuracy(log_likelihood, qs)
    n_factors = length(qs)
    ndims_ll = ndims(log_likelihood)
    dims = (ndims_ll - n_factors + 1) : ndims_ll

    # Calculate the accuracy term
    accuracy = sum(
        log_likelihood[indices...] * prod(qs[factor][indices[dims[factor]]] for factor in 1:n_factors)
        for indices in Iterators.product((1:size(log_likelihood, i) for i in 1:ndims_ll)...)
    )

    return accuracy
end


""" Calculate Free Energy """
function calc_free_energy(qs, prior, n_factors, likelihood=nothing)
    # Initialize free energy
    free_energy = 0.0
    
    # Calculate free energy for each factor
    for factor in 1:n_factors
        # Neg-entropy of posterior marginal
        negH_qs = dot(qs[factor], log.(qs[factor] .+ 1e-16))
        # Cross entropy of posterior marginal with prior marginal
        xH_qp = -dot(qs[factor], prior[factor])
        # Add to total free energy
        free_energy += negH_qs + xH_qp
    end
    
    # Subtract accuracy
    if likelihood !== nothing
        free_energy -= compute_accuracy(likelihood, qs)
    end
    
    return free_energy
end

#### Policy Inference #### 
""" Update Posterior over Policies """
function update_posterior_policies(
    qs::Vector{Any},
    A::Vector{Any},
    B::Vector{Any},
    C::Vector{Any},
    policies::Vector{Any},
    use_utility::Bool=true,
    use_states_info_gain::Bool=true,
    use_param_info_gain::Bool=false,
    pA = nothing,
    pB = nothing,
    E = nothing,
    gamma::Float64=16.0
)
    n_policies = length(policies)
    G = zeros(n_policies)
    q_pi = zeros(n_policies, 1)
    qs_pi = Vector{Float64}[]
    qo_pi = Vector{Float64}[]
  
    if isnothing(E)
        lnE = spm_log_single(ones(n_policies) / n_policies)
    else
        lnE = spm_log_single(E)
    end

    for (idx, policy) in enumerate(policies)
        qs_pi = get_expected_states(qs, B, policy)
        qo_pi = get_expected_obs(qs_pi, A)

        if use_utility
            G[idx] += calc_expected_utility(qo_pi, C)
        end

        if use_states_info_gain
            G[idx] += calc_states_info_gain(A, qs_pi)
        end

        if use_param_info_gain
            if pA !== nothing
                G[idx] += calc_pA_info_gain(pA, qo_pi, qs_pi)
            end
            if pB !== nothing
                G[idx] += calc_pB_info_gain(pB, qs_pi, qs, policy)
            end
        end

    end

    q_pi = softmax(G * gamma + lnE)
    return q_pi, G
end

""" Get Expected Observations """
function get_expected_obs(qs_pi, A::Vector{Any})
    n_steps = length(qs_pi)
    qo_pi = []

    for t in 1:n_steps
        qo_pi_t = array_of_any(length(A))
        qo_pi = push!(qo_pi, qo_pi_t)
    end

    for t in 1:n_steps
        for (modality, A_m) in enumerate(A)
            qo_pi[t][modality] = spm_dot(A_m, qs_pi[t])
        end
    end

    return qo_pi
end

""" Calculate Expected Utility """
function calc_expected_utility(qo_pi, C)
    n_steps = length(qo_pi)
    expected_utility = 0.0
    num_modalities = length(C)

    modalities_to_tile = [modality_i for modality_i in 1:num_modalities if ndims(C[modality_i]) == 1]

    C_tiled = deepcopy(C)
    for modality in modalities_to_tile
        modality_data = reshape(C_tiled[modality], :, 1)
        C_tiled[modality] = repeat(modality_data, 1, n_steps)
    end
    
    C_prob = softmax_array(C_tiled)
    lnC =[]
    for t in 1:n_steps
        for modality in 1:num_modalities
            lnC = spm_log_single(C_prob[modality][:, t])
            expected_utility += dot(qo_pi[t][modality], lnC) 
        end
    end

    return expected_utility
end

""" Calculate States Information Gain """
function calc_states_info_gain(A, qs_pi)
    n_steps = length(qs_pi)
    states_surprise = 0.0

    for t in 1:n_steps
        states_surprise += spm_MDP_G(A, qs_pi[t])
    end

    return states_surprise
end

""" Calculate observation to state info Gain """
function calc_pA_info_gain(pA, qo_pi, qs_pi)

    n_steps = length(qo_pi)
    num_modalities = length(pA)

    wA = array_of_any(num_modalities)
    for (modality, pA_m) in enumerate(pA)
        wA[modality] = spm_wnorm(pA[modality])
    end

    pA_info_gain = 0

    for modality in 1:num_modalities
        wA_modality = wA[modality] .* pA[modality]

        for t in 1:n_steps
            pA_info_gain -= dot(qo_pi[t][modality], spm_dot(wA_modality, qs_pi[t]))
        end
    end
    return pA_info_gain
end

""" Calculate state to state info Gain """
function calc_pB_info_gain(pB, qs_pi, qs_prev, policy)
    n_steps = length(qs_pi)
    num_factors = length(pB)

    wB = array_of_any(num_factors)
    for (factor, pB_f) in enumerate(pB)
        wB[factor] = spm_wnorm(pB_f)
    end

    pB_info_gain = 0

    for t in 1:n_steps
        if t == 1
            previous_qs = qs_prev
        else
            previous_qs = qs_pi[t-1]
        end

        policy_t = policy[t, :]

        for (factor, a_i) in enumerate(policy_t)
            wB_factor_t = wB[factor][:,:,Int(a_i)] .* pB[factor][:,:,Int(a_i)]
            pB_info_gain -= dot(qs_pi[t][factor], wB_factor_t * previous_qs[factor])
        end
    end
    return pB_info_gain
end

### Action Sampling ###
""" Sample Action [Stochastic or Deterministic] """
function sample_action(q_pi, policies, num_controls; action_selection="stochastic", alpha=16.0)
    num_factors = length(num_controls)
    action_marginals = array_of_any_zeros(num_controls)
    selected_policy = zeros(num_factors)
    
    for (pol_idx, policy) in enumerate(policies)
        for (factor_i, action_i) in enumerate(policy[1,:])
            action_marginals[factor_i][action_i] += q_pi[pol_idx]
        end
    end

    action_marginals = norm_dist_array(action_marginals)

    for factor_i in 1:num_factors
        if action_selection == "deterministic"
            selected_policy[factor_i] = select_highest(action_marginals[factor_i])
        elseif action_selection == "stochastic"
            log_marginal_f = spm_log_single(action_marginals[factor_i])
            p_actions = softmax(log_marginal_f * alpha)
            selected_policy[factor_i] = action_select(p_actions)
        end
    end
    return selected_policy
end

""" Edited Compute Accuracy [Still needs to be nested within Fixed-Point Iteration] """
function compute_accuracy_new(log_likelihood, qs)
    n_factors = length(qs)
    ndims_ll = ndims(log_likelihood)
    dims = (ndims_ll - n_factors + 1) : ndims_ll

    result_size = size(log_likelihood, 1) 
    results = zeros(result_size)

    for indices in Iterators.product((1:size(log_likelihood, i) for i in 1:ndims_ll)...)
        product = log_likelihood[indices...] * prod(qs[factor][indices[dims[factor]]] for factor in 1:n_factors)
        results[indices[1]] += product
    end

    return results
end