using LinearAlgebra
using Plots

module ActiveInference

include("functions.jl")

# From functions.jl
export plot_beliefs, plot_gridworld, plot_likelihood, create_B_matrix, onehot, plot_point_on_grid, infer_states, get_expected_states, get_expected_observations, calculate_G, run_active_inference_loop, construct_policies, calculate_G_policies, compute_prob_actions,active_inference_with_planning

    # From maths.jl
    module Maths
    
    include("maths.jl")
    
    export norm_dist, sample_category, softmax, spm_log_single, entropy, kl_divergence
    
    end;

    # From environment.jl
    module Environment

    include("environment.jl")
    
    export GridWorldEnv, step!, reset!
       
    end;


end;






