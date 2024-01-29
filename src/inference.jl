""" -------- Inference Functions -------- """




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
function update_posterior_states(A, obs; prior=nothing, kwargs...)
    num_obs, num_states, num_modalities, num_factors = get_model_dimensions(A)

    obs_processed = process_observation(obs, num_modalities, num_obs)
    return fixed_point_iteration(A, obs_processed, num_obs, num_states, prior=prior)
end


""" Run State Inference via Fixed-Point Iteration """
function fixed_point_iteration(A, obs, num_obs, num_states; prior=nothing, num_iter=10, dF=1.0, dF_tol=0.001)
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
        return to_array_of_any(softmax(qL + prior[1]))
    else
        # Run Iteration [NOTE: Tensor operations can be made more efficient perhaps?]
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

    # Calculate the accuracy term using array comprehension and reduction [NOTE: Tensor operations can be made more efficient perhaps?]
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
        # Neg-entropy of posterior marginal H(q[f])
        negH_qs = dot(qs[factor], log.(qs[factor] .+ 1e-16))
        # Cross entropy of posterior marginal with prior marginal H(q[f],p[f])
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