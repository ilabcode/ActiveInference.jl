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
# This is the one named after log_single in SPM 
# Should maybe be renamed??
function spm_log_single(arr)
    EPS_VAL = 1e-16
    return log.(ℯ, (arr .+ EPS_VAL))
end


"""Function for Calculating Entropy of A-Matrix"""
function entropy_A(A)
    
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

""" Dot-Product Function """
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

""" Apply spm_log to array of arrays """
function spm_log_array_any(arr)
    # Initialize an empty array
    arr_logged = Any[nothing for _ in 1:length(arr)]
    # Apply spm_log_single to each element of arr
    for (idx, sub_arr) in enumerate(arr)
        arr_logged[idx] = spm_log_single(sub_arr)
    end

    return arr_logged
end

""" Softmax Function for array of arrays """
function softmax_array(arr)
    output = Array{Any}(undef, length(arr))
    
    # Iterate through each index in arr and apply softmax
    for idx in eachindex(arr)
        output[idx] = softmax(arr[idx])
    end
    
    return output
end


""" Multi-dimensional outer product """
function spm_cross(x, y=nothing; remove_singleton_dims=true, args...)
    # If only x is provided and it is a vector of arrays, recursively call spm_cross on its elements.
    if y === nothing && isempty(args)
        if x isa AbstractVector
            return reduce((a, b) -> spm_cross(a, b), x)
        elseif typeof(x) <: Number || typeof(x) <: AbstractArray
            return x
        else
            throw(ArgumentError("Invalid input to spm_cross (\$x)"))
        end
    end

    # If y is provided, perform the cross multiplication.
    if y !== nothing
        reshape_dims_x = tuple(size(x)..., ones(Int, ndims(y))...)
        A = reshape(x, reshape_dims_x)

        reshape_dims_y = tuple(ones(Int, ndims(x))..., size(y)...)
        B = reshape(y, reshape_dims_y)

        z = A .* B
    else
        z = x
    end

    # Recursively call spm_cross for additional arguments
    for arg in args
        z = spm_cross(z, arg; remove_singleton_dims=remove_singleton_dims)
    end

    # remove singleton dimension if true--
    if remove_singleton_dims
        z = dropdims(z, dims = tuple(findall(size(z) .== 1)...))
    end

    return z
end

#Multi-dimensional inner product
#= Instead of summing over all indices, the function sums over only the last three
dimensions of X while keeping the first dimension separate, creating a sum for each "layer" of X. =#
    """
function spm_dot(X, x)

    if all(isa.(x, AbstractArray))  
        n_factors = length(x)
    else
        x = [x]  
        n_factors = length(x)
    end

    ndims_X = ndims(X)
    dims = collect(ndims_X - n_factors + 1 : ndims_X)
    Y = zeros(size(X, 1))

    for indices in Iterators.product((1:size(X, i) for i in 1:ndims_X)...)
        product = X[indices...] * prod(x[factor][indices[dims[factor]]] for factor in 1:n_factors)
        Y[indices[1]] += product
    end

    if prod(size(Y)) <= 1
        Y = only(Y)
        Y = [float(Y)]  
    end

    return Y
end
"""
""" Multi-dimensional inner product """
function spm_dot(X, x)
    if all(isa.(x, AbstractArray))
        n_factors = length(x)
    else
        x = [x]
        n_factors = length(x)
    end

    ndims_X = ndims(X)
    dims = collect(ndims_X - n_factors + 1 : ndims_X)
    Y = zeros(size(X, 1))

    # thread-local storage for accumulators
    Y_local = [zeros(size(X, 1)) for _ in 1:Threads.nthreads()]

    all_indices = collect(Iterators.product([1:size(X, i) for i in 1:ndims_X]...))

    Threads.@threads for idx_tuple in all_indices
        tid = Threads.threadid()  
        indices = Tuple(idx_tuple)
        product = X[indices...] * prod(x[factor][indices[dims[factor]]] for factor in 1:n_factors)
        Y_local[tid][indices[1]] += product 
    end

    Y .= Y_local[1]
    for i in eachindex(Y_local)
        if i != 1
            Y .+= Y_local[i]
        end
    end

    if prod(size(Y)) <= 1
        Y = only(Y)
        Y = [float(Y)]
    end

    return Y
end


""" Calculate Bayesian Surprise """
function spm_MDP_G(A, x)
    qx = spm_cross(x)
    G = 0.0
    qo = Float64[]
    idx = [collect(Tuple(indices)) for indices in findall(qx .> exp(-16))]
    index_vector = []

    for i in idx   
        po = ones(1)
        for (_, A_m) in enumerate(A)
            index_vector = (1:size(A_m, 1),)  
            for additional_index in i  
                index_vector = (index_vector..., additional_index)  
            end
            po = spm_cross(po, A_m[index_vector...])
        end
        po = vec(po) 
        if isempty(qo)
            resize!(qo, length(po))
            fill!(qo, 0.0)
        end
        qo += qx[i...] * po
        G += qx[i...] * dot(po, log.(po .+ exp(-16)))
    end
    G = G - dot(qo, spm_log_single(qo))
    return G
end

""" Normalizes muliple arrays """
function norm_dist_array(obj_arr::Array{Any})
    normed_obj_array = Array{Any}(undef, length(obj_arr))
    for i in 1:length(obj_arr)
        normed_obj_array[i] = norm_dist(obj_arr[i])  
    end
    return normed_obj_array
end

""" SPM_wnorm """
function spm_wnorm(A)
    EPS_VAL = 1e-16

    A .+= EPS_VAL
    norm = 1.0 ./ sum(A, dims = 1)
    avg = 1 ./ A
    wA = norm .- avg

    return wA
end