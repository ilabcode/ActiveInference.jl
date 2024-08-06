module ActiveInference

##PTW_CR: Put all the dependencies you use here at the beginning, not in the other files
##PTW_CR: We want to have as few dependencies as possible - and to make sure that the dependencies we do have are well-supported packages. Consider if there are packages you don't need
using ActionModels

##PTW_CR: Conors original work is great. But we don't have to copy everything. I think I would have a slightly different structure to the files.
##PTW_CR: I would have one folder with helper functions (including maths and utils - consider if they should be restructured)
##PTW_CR: And one folder with the POMDP generative model (four files: the struct, inference, learning and action, perhaps)
##PTW_CR: I would also add a folder with premade generative model, like a premade T-mase actinf, a premade Multiple-Armed Bandit agetn, etc
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

##PTW_CR: Export anything that people shoudl be able to use, and nothing else
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
       give_inputs!,
       set_save_history!


    # From Environments\\EpistChainEnv.jl
    module Environments
    ##PTW_CR: I really think we should have an Envionrmnet type in ActionModels, and then have premade environments that can be added to it like premade agents. Then ActiveINference.jl can add some environments if they are fairly actinf-specific - or they can just be natively in ActionModels. Then you'd be contributors there and can perhaps join that paper too. 
    ##PTW_CR: If we have a standard Environment type, we can also have standard functiosn for it (like step! and reset!
    ##PTW_CR: Note that you can actually make perfectly fine environments as ActionModels Agent objets- We can talk about this.

    include("Environments/EpistChainEnv.jl")
    
    export EpistChainEnv, step!, reset_env!

    include("Environments/TMazeEnv.jl")

    export TMazeEnv, step_TMaze!, reset_TMaze!, initialize_gp
       
    end
end






