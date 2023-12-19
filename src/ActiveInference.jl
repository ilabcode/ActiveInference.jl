module ActiveInference

include("functions.jl")

# From functions.jl
export plot_beliefs, plot_gridworld, plot_likelihood, create_B_matrix, onehot, plot_point_on_grid, infer_states, get_expected_states, get_expected_observations, calculate_G, run_active_inference_loop, construct_policies, calculate_G_policies, compute_prob_actions,active_inference_with_planning

    # export plot_likelihood
    # export create_B_matrix
    # export onehot
    # export plot_point_on_grid
    # export infer_states
    # export get_expected_states
    # export get_expected_observations
    # export calculate_G
    # export run_active_inference_loop
    # export construct_policies
    # export calculate_G_policies
    # export compute_prob_actions
    # export active_inference_with_planning

end


module maths
    
include("maths.jl")

#export norm_dist, sample_category, softmax, spm_log_single, entropy, kl_divergence
    
    # export sample_category
    # export softmax
    # export spm_log_single
    # export entropy
    # export kl_divergence

end


module environment

include("environment.jl")

#export GridWorldEnv, step!, reset!
    # export step!
    # export reset!

end




