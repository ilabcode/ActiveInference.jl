module ActiveInference

include("maths.jl")
include("Environments\\EpistChainEnv.jl")
include("agent.jl")
include("utils.jl")
# From functions.jl
export array_of_any, 
       array_of_any_zeros, 
       array_of_any_uniform, 
       onehot,
       construct_policies_full,






       plot_gridworld

    # From maths.jl
    module Maths
    
    include("maths.jl")
    
    export norm_dist, sample_category, softmax, spm_log_single, entropy, kl_divergence
    
    end

    # From Environments\\EpistChainEnv.jl
    module Environments

    include("Environments\\EpistChainEnv.jl")
    
    export EpistChainEnv, step!, reset!
       
    end
end






