using LinearAlgebra
using IterTools

"""Normalizes a Categorical probability distribution"""
function norm_dist(dist)
    return dist ./ sum(dist, dims=1)
end


"""Sampling Function"""
function sample_category(probabilities)
    rand_num = rand()
    cum_probabilities = cumsum(probabilities)
    category = findfirst(x -> x > rand_num, cum_probabilities)
    return category
end

"""Softmax Function"""
function softmax(dist)

    output = dist .- maximum(dist, dims = 1)
    output = exp.(output)
    output = output ./ sum(output, dims = 1)

    return output
end


"""Function for Taking Natural logarithm"""
# This is the one named log_simple in the active inference tutorial
# Should maybe be renamed??
function spm_log_single(arr)
    EPS_VAL = 1e-16
    return log.(ℯ, (arr .+ EPS_VAL))
end


"""Function for Calculating Entropy of A-Matrix"""
function entropy(A)
    
    H_A = .- sum((A .* spm_log_single(A)), dims = 1)

    return H_A
end


"""Function for getting KL-divergence"""
function kl_divergence(qo_u, C)

    #dist = (spm_log_single(qo_u) .- spm_log_single(C))
    #kl_div = dot(dist, qo_u)

    return dot((spm_log_single(qo_u) .- spm_log_single(C)), qo_u)
end

""" Get Joint Likelihood """
function get_joint_likelihood(A, obs_processed, num_states)
    ll = ones(num_states...)
    for modality in eachindex(A)
        ll .*= dot_likelihood(A[modality], obs_processed[modality])
    end

    return ll
end


function dot_likelihood(A, obs)
    # Adjust the shape of obs to match A
    reshaped_obs = reshape(obs, (length(obs), 1, 1, 1))  
    # Element-wise multiplication and sum over the first axis
    LL = sum(A .* reshaped_obs, dims=1)
    # Remove singleton dimensions
    LL = dropdims(LL, dims= tuple(findall(size(LL) .== 1)...))
    if prod(size(LL)) == 1
        LL = [LL[]]  
    end
    return LL
end

function spm_log_array_any(arr)
    # Initialize an empty array
    arr_logged = Any[nothing for _ in 1:length(arr)]
    # Apply spm_log_single to each element of arr
    for (idx, sub_arr) in enumerate(arr)
        arr_logged[idx] = spm_log_single(sub_arr)
    end

    return arr_logged
end