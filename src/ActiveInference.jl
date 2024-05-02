module ActiveInference

using ActionModels

include("maths.jl")
include("agent.jl")
include("learning.jl")
include("utils.jl")
include("inference.jl")
include("ActionModelsExtensions/get_states.jl")
include("ActionModelsExtensions/get_parameters.jl")
include("ActionModelsExtensions/get_history.jl")
include("ActionModelsExtensions/set_parameters.jl")
include("ActionModelsExtensions/reset.jl")
include("ActionModelsExtensions/give_inputs.jl")
include("ActionModelsExtensions/set_save_history.jl")
include("POMDP.jl")

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
       generate_random_GM,     

       # agent.jl
       init_aif,
       infer_states!,
       infer_policies!,
       sample_action!,
       update_A!,
       update_B!,
       update_D!,

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

       # POMDP.jl
       action_pomdp!,

       # ActionModelsExtensions
       get_states,
       get_parameters,
       get_history,
       set_parameters!,
       reset!,
       single_input!,
       set_save_history!


    # From Environments\\EpistChainEnv.jl
    module Environments

    include("Environments/EpistChainEnv.jl")
    
    export EpistChainEnv, step!, reset_env!
       
    end
end






