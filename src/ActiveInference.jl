module ActiveInference



include("maths.jl")
include("Environments\\EpistChainEnv.jl")
include("agent.jl")
include("utils.jl")
include("inference.jl")
include("ActionModelsExtensions/get_states.jl")
include("ActionModelsExtensions/get_parameters.jl")
include("ActionModelsExtensions/get_history.jl")


export # maths.jl
       norm_dist,
       sample_category,
       softmax,
       spm_log_single,
       entropy_A,
       kl_divergence,
       get_joint_likelihood,
       dot_likelihood,
       spm_log_array_any,
       softmax_array,
       spm_cross,
       spm_dot,
       spm_MDP_G,
       norm_dist_array,


       # utils.jl
       array_of_any, 
       array_of_any_zeros, 
       array_of_any_uniform, 
       onehot,
       construct_policies_full,
       plot_gridworld,
       process_observation,
       get_model_dimensions,
       to_array_of_any,
       select_highest,
       action_select,


       # agent.jl
       init_aif,
       infer_states!,
       infer_policies!,
       sample_action!,

       # inference.jl
       get_expected_states,
       update_posterior_states,
       fixed_point_iteration,
       compute_accuracy,
       calc_free_energy,
       update_posterior_policies,
       get_expected_obs,
       calc_expected_utility,
       calc_states_info_gain,
       sample_action,

       # ActionModelsExtensions
       get_states,
       get_parameters,
       get_history


    # From Environments\\EpistChainEnv.jl
    module Environments

    include("Environments\\EpistChainEnv.jl")
    
    export EpistChainEnv, step!, reset!
       
    end
end






