using LinearAlgebra
using IterTools

##PTW_CR: I in general prefer spelled out names: normalize_distribution
"""Normalizes a Categorical probability distribution"""
function norm_dist(dist)
    return dist ./ sum(dist, dims=1)
end


##PTW_CR: This seems a bit weird - is this just sampling from a categorical distribution ? 
##PTW_CR: If so, why not just use the inbuilt functions for sampling from a categorical distribution?
##PTW_CR: I think we can in generally rely a bit more on Distributions, because it gives a lot of utility for the distributions. 
##PTW_CR: If you want to keep this, do it in one line: 
##PTW_CR: findfirst(x -> x > rand(), cumsum(probabilities))
##PTW_CR: The sample function in StatsBase can also do weighted sampling. I'm unsure which one is fastest. 
##PTW_CR: Spend energy on optimizing this for speed if it is run often.
"""Sampling Function"""
function sample_category(probabilities)
    rand_num = rand()
    cum_probabilities = cumsum(probabilities)
    category = findfirst(x -> x > rand_num, cum_probabilities)
    return category
end

##PTW_CR: LogExpFunctions.jl has a softmax function (it's a fine dependency because it's part of Julia natively)
##PTW_CR: Then you just multiply with the temperature parameter before inputting it.
##PTW_CR: Should be faster
"""Softmax Function"""
function softmax(dist)

    output = dist .- maximum(dist, dims = 1)
    output = exp.(output)
    output = output ./ sum(output, dims = 1)

    return output
end

##PTW_CR: A note: Might want to check this one: https://docs.julialang.org/en/v1/manual/performance-tips/
##PTW_CR: You might want to try and run a profiler on your package to see which parts of it contribute to the processing time: https://docs.julialang.org/en/v1/manual/profile/
##PTW_CR: You might also want to run some stuff to see where Julia can and can't infer the types - this is important for speed. I think JET or Aqua does that.


##PTW_CR: The reason this epsilon is added is to avoid getting minus infinite from log(0)
##PTW_CR: This is a normal trick to use. I don't know if it's important here, though.
##PTW_CR: There's no reason to have a function that runs log on an array when you can just broadcast it outside of the function anyway.
##PTW_CR: Try just using the normal log. throughout. It does the same.
##PTW_CR: If you do want a capped log (we use a capped exp in the hgf for example) then I have created a more natural function below.
##PTW_CR: Here I use Julia's inbuilt eps() function to get the smallest possible number. Also, log uses the natural logarithm by default.
"""Function for Taking Natural logarithm"""
# This is the one named after log_single in SPM 
# Should maybe be renamed??
function spm_log_single(arr)
    EPS_VAL = 1e-16
    return log.(ℯ, (arr .+ EPS_VAL))
    #return log.(2.7182818284590, (arr .+ EPS_VAL))
end
#Version of the natural logarithm that is capped at the smallest possible number
function capped_log(x::T) {where T <: Real}
    return log(max(x, eps(T)))
end

##PTW_CR: Also - do not call the functions 'spm' something. This shouldn't be a copy of a copy of spm. It should just do the same tihng.
##PTW_CR: Name the functions based on what they do, and use native Julia functions when possible. 


##PTW_CR: StatsBase.jl has a function that calculates the entropy of a matrix. Use that instead - should be faster. https://juliastats.org/StatsBase.jl/v0.19/scalarstats.html#StatsBase.entropy
##PTW_CR: Also, this function should be the same for any probability distribution, not just the A matrix.
"""Function for Calculating Entropy of A-Matrix"""
function entropy_A(A)
    
    H_A = .- sum((A .* spm_log_single(A)), dims = 1)

    return H_A
end


##PTW_CR: StatsBase.jl also has a KL-divergence function, which should be faster. Just use that.
##PTW_CR: Also, its easy to be unsure what qo_u is - give them proper names
##PTW_CR: I'm assuming both inputs here are categorical probability distirbutions though. 
"""Function for getting KL-divergence"""
function kl_divergence(qo_u, C)

    #dist = (spm_log_single(qo_u) .- spm_log_single(C))
    #kl_div = dot(dist, qo_u)

    return dot((spm_log_single(qo_u) .- spm_log_single(C)), qo_u)
end


##PTW_CR: Spell out the ll
##PTW_CR: Also, I dont think you need to first crearte a matrix of ones and then multiply it with the likelihoods.
##PTW_CR: I think you can just broadcast, or use map()
""" Get Joint Likelihood """
function get_joint_likelihood(A, obs_processed, num_states)
    ll = ones(Real, num_states...)
    for modality in eachindex(A)
        ll .*= dot_likelihood(A[modality], obs_processed[modality])
    end

    return ll
end

##PTW_CR: Why capital LL here? Also, spell it out.
##PTW_CR: Also, there simply must be a julia function for dot products. Just use that instead. https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.dot
##PTW_CR: Let me know if there is some reason you need to do anything else. In that case, make sure the description of the function makes clear how this is different form a normal dot product
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

##PTW_CR: Just use map to apply log.() to all of the arrays in the list of arrays
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

##PTW_CR: Again, just use map or broadcast. Should be much much faster
""" Softmax Function for array of arrays """
function softmax_array(arr)
    output = Array{Any}(undef, length(arr))
    
    # Iterate through each index in arr and apply softmax
    for idx in eachindex(arr)
        output[idx] = softmax(arr[idx])
    end
    
    return output
end


##PTW_CR: Again, LinearAlgebra has a function for the cross-product. Use that at least. 
##PTW_CR: Also, this function is a bit hard to understand. I think you should try to make it more clear what it does.
##PTW_CR: Haha, my autopilot suggested the above line - it's true though. Some comments would be nice.
##PTW_CR: Do these if things with multiple dispatch - faster, more readable, more Julia 
##PTW_CR: And again, I think you can just use map here to run it on all subarrays.
##PTW_CR: In general, I think that the matrices should _always_ be in the same shape (i.e. a vector of matrices which can be just one long if there is only one modality etc.)
##PTW_CR: The only place it should be otherwise, should be in the API for the user
##PTW_CR: Then you don't need all these if statements like this
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
        reshape_dims_x = tuple(size(x)..., ones(Real, ndims(y))...)
        A = reshape(x, reshape_dims_x)

        reshape_dims_y = tuple(ones(Real, ndims(x))..., size(y)...)
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

#Multidimensional inner product
# Instead of summing over all indices, the function sums over only the last three
# dimensions of X while keeping the first dimension separate, creating a sum for each "layer" of X.

##PTW_CR: I think most of the comments for the above function applies here as well.
function spm_dot(X, x)

    if all(isa.(x, AbstractArray))  
        n_factors = length(x)
    else
        x = [x]  
        n_factors = length(x)
    end

    ndims_X = ndims(X)
    dims = collect(ndims_X - n_factors + 1 : ndims_X)
    Y = zeros(Real, size(X, 1))

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


##PTW_CR: If you're not using it, delete this function.
##PTW_CR: I didn't read through this because its commented out.
# """ Multi-dimensional inner product """
# function spm_dot(X, x)
#     if all(isa.(x, AbstractArray))
#         n_factors = length(x)
#     else
#         x = [x]
#         n_factors = length(x)
#     end

#     ndims_X = ndims(X)
#     dims = collect(ndims_X - n_factors + 1 : ndims_X)
#     Y = zeros(size(X, 1))

#     thread-local storage for accumulators
#     Y_local = [zeros(size(X, 1)) for _ in 1:Threads.nthreads()]

#     all_indices = collect(Iterators.product([1:size(X, i) for i in 1:ndims_X]...))

#     #Threads.@threads for idx_tuple in all_indices
#     for idx_tuple in all_indices
#         tid = Threads.threadid()  
#         indices = Tuple(idx_tuple)
#         product = X[indices...] * prod(x[factor][indices[dims[factor]]] for factor in 1:n_factors)
#         Y_local[tid][indices[1]] += product 
#     end

#     Y .= Y_local[1]
#     for i in eachindex(Y_local)
#         if i != 1
#             Y .+= Y_local[i]
#         end
#     end

#     if prod(size(Y)) <= 1
#         Y = only(Y)
#         Y = [float(Y)]
#     end

#     return Y
# end


##PTW_CR: Give this a proper title which can be understood, and which doesn't refer to SPM
##PTW_CR: Spell out the variables
##PTW_CR: What exactly does the cross function do when you only give it x? Unclear
##PTW_CR: I think it's a really good idea here to generally not just call them x
##PTW_CR: Also, use docstrings to expain these properly
##PTW_CR: Also comment it so I can understand it
##PTW_CR: If this essentially calculates the surprise at receiving a specific observation, given the generative model and current beliefs
##PTW_CR: Then Distributions can create a Categorical distribution, and you can use the logpdf function to get the log likelihood of an observation
""" Calculate Bayesian Surprise """
function spm_MDP_G(A, x)
    qx = spm_cross(x)
    G = 0.0
    qo = Real[]
    idx = [collect(Tuple(indices)) for indices in findall(qx .> exp(-16))]
    index_vector = []

    for i in idx   
        po = ones(Real, 1)
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

##PTW_CR: Just use mapm or broadcast instead of this
""" Normalizes muliple arrays """
function norm_dist_array(obj_arr::Array{Any})
    normed_obj_array = Array{Any}(undef, length(obj_arr))
    for i in 1:length(obj_arr)
        normed_obj_array[i] = norm_dist(obj_arr[i])  
    end
    return normed_obj_array
end

##PTW_CR: I dont know what wnorm is supposed to mean
##PTW_CR: weighted norm?
##PTW_CR: You can do A::Array{T} where T <: Real 
##PTW_CR: And then eps(T) to find the smallest possible number in Julia
##PTW_CR: I don't really know if you need it here though
##PTW_CR: If all you do is subtract the average from normalized A, then I would do that outside the function in the appropriate places
""" SPM_wnorm """
function spm_wnorm(A)
    EPS_VAL = 1e-16

    A .+= EPS_VAL
    norm = 1.0 ./ sum(A, dims = 1)
    avg = 1 ./ A
    wA = norm .- avg

    return wA
end