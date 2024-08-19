"""
    create_matrix_templates_custom(n_states::Vector{Int64}, n_observations::Vector{Int64}, n_controls::Vector{Int64}, policy_length::Int64, template_type::String = "uniform")

Creates templates for the A, B, C, D, and E matrices based on the specified parameters.

# Arguments
- `n_states::Vector{Int64}`: A vector specifying the dimensions and number of states.
- `n_observations::Vector{Int64}`: A vector specifying the dimensions and number of observations.
- `n_controls::Vector{Int64}`: A vector specifying the number of controls per factor.
- `policy_length::Int64`: The length of the policy sequence. 
- `template_type::String`: The type of templates to create. Can be "uniform", "random", or "zeros". Defaults to "uniform".

# Returns
- `A, B, C, D, E`: The generative model as matrices and vectors.

"""

### Function for creating uniform matrices
function create_matrix_templates(n_states::Vector{Int64}, n_observations::Vector{Int64}, n_controls::Vector{Int64}, policy_length::Int64)
    
    # Calculate the number of policies based on the policy length
    n_policies = prod(n_controls) ^ policy_length

    # Uniform A matrices
    A = [norm_dist(ones(vcat(observation_dimension, n_states)...)) for observation_dimension in n_observations]

    # Uniform B matrices
    B = [norm_dist(ones(state_dimension, state_dimension, n_controls[index])) for (index, state_dimension) in enumerate(n_states)]

    # C vectors as zero vectors
    C = [zeros(observation_dimension) for observation_dimension in n_observations]

    # Uniform D vectors
    D = [fill(1.0 / state_dimension, state_dimension) for state_dimension in n_states]

    # Uniform E vector
    E = fill(1.0 / n_policies, n_policies)

    return A, B, C, D, E
end

### Function for specific template type, can be either 'random' or 'zeros'
function create_matrix_templates(n_states::Vector{Int64}, n_observations::Vector{Int64}, n_controls::Vector{Int64}, policy_length::Int64, template_type::String)
    
    # If the template_type is uniform
    if template_type == "uniform"
        return create_matrix_templates(n_states, n_observations, n_controls, policy_length)
    end

    # Calculate the number of policies based on the policy length
    n_policies = prod(n_controls) ^ policy_length

    # If the template type is random, populate the matrices with random values
    if template_type == "random"
        # Random A matrices
        A = [norm_dist(rand(vcat(observation_dimension, n_states)...)) for observation_dimension in n_observations]

        # Random B matrices
        B = [norm_dist(rand(state_dimension, state_dimension, n_controls[index])) for (index, state_dimension) in enumerate(n_states)]

        # C vectors populated with random integers between -4 and 4
        C = [rand(-4:4, observation_dimension) for observation_dimension in n_observations]

        # Random D vectors
        D = [norm_dist(rand(state_dimension)) for state_dimension in n_states]

        # Random E vector
        E = norm_dist(rand(n_policies))
    
    # If the template type is zeros, populate the matrices with zeros
    elseif template_type == "zeros"

        A = [zeros(vcat(observation_dimension, n_states)...) for observation_dimension in n_observations]
        B = [zeros(state_dimension, state_dimension, n_controls[index]) for (index, state_dimension) in enumerate(n_states)]
        C = [zeros(observation_dimension) for observation_dimension in n_observations]
        D = [zeros(state_dimension) for state_dimension in n_states]
        E = zeros(n_policies)
    
    else
        # Throw error for invalid template type
        throw(ArgumentError("Invalid type: $template_type. Choose either 'uniform', 'random' or 'zeros'."))
    end

    return A, B, C, D, E
end