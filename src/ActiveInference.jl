module ActiveInference

include("functions.jl")
include("maths.jl")

#from functions.jl
export plot_beliefs
export plot_gridworld
export plot_likelihood
export create_B_matrix
export onehot
export plot_point_on_grid
export infer_states
export get_expected_states
export get_expected_observations


#from maths.jl
export norm_dist
export sample_category
export softmax
export spm_log_single
export entropy



end

