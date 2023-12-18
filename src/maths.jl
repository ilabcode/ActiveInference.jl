using LinearAlgebra

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



