using LinearAlgebra


"""Normalizes a Categorical probability distribution"""
function norm_dist(dist)
    return dist ./ sum(dist, dims=1)
end