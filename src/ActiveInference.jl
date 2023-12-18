module ActiveInference

    include("functions.jl")
    include("maths.jl")
    include("environment.jl")

    # From functions.jl
    export plot_beliefs
    export plot_gridworld
    export plot_likelihood
    export create_B_matrix
    export onehot
    export plot_point_on_grid
    export infer_states
    export get_expected_states
    export get_expected_observations
    export calculate_G
    export run_active_inference_loop
    export construct_policies
    export calculate_G_policies
    export compute_prob_actions
    export active_inference_with_planning


    # From maths.jl
    export norm_dist
    export sample_category
    export softmax
    export spm_log_single
    export entropy
    export kl_divergence

    # From environment.jl
    export GridWorldEnv
    export step!
    export reset!



    end

