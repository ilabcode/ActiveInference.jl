var documenterSearchIndex = {"docs":
[{"location":"GenerativeModelCreation/#Creating-the-Generative-Model","page":"Creation of the Generative Model","title":"Creating the Generative Model","text":"","category":"section"},{"location":"GenerativeModelCreation/","page":"Creation of the Generative Model","title":"Creation of the Generative Model","text":"In this section we will go through the process of creating a generative model and how it should be structured.","category":"page"},{"location":"GenerativeModelCreation/#The-Generative-Model-Conceptually","page":"Creation of the Generative Model","title":"The Generative Model Conceptually","text":"","category":"section"},{"location":"GenerativeModelCreation/","page":"Creation of the Generative Model","title":"Creation of the Generative Model","text":"The generative model is the parameters that constitute the agent's beliefs on how the hidden states of the environment generates observations based on states, and how hidden underlying states changes over time. In the generative model is also the beliefs of how the agent through actions can influence the states of the environment. Together this holds the buidling blocks that allows for the perception-action loop.","category":"page"},{"location":"GenerativeModelCreation/","page":"Creation of the Generative Model","title":"Creation of the Generative Model","text":"There are five main buidling blocks of the generative model which are; A, B, C, D, and E. Each of these contain parameters that describe the agent's beliefs about the environment. We will now go through these conecptually one at a time.","category":"page"},{"location":"GenerativeModelCreation/#A","page":"Creation of the Generative Model","title":"A","text":"","category":"section"},{"location":"GenerativeModelCreation/","page":"Creation of the Generative Model","title":"Creation of the Generative Model","text":"A is the observation likelihood model, and describes the agent's beliefs about how the hidden states of the environment generates observations. Practically in this package, and other POMDP implemantations as well, this is described through a series of categorical distributions, meaning that for each observation, there is a categorical probability distribution over how likely each hidden state is to generate that observation. Let us for example imagine a simple case, where the agent is in a four location state environment, could be a 2x2 gridworld. In this case, there would be one obseration linked to each hidden state, and A then maps the agent's belief of how likely each hidden location state is to generate each observation. The agent can then use this belief to infer what state it is in based on the observation it receives. Let's look at an example A, which in this case would be a 4x4 matrix:","category":"page"},{"location":"GenerativeModelCreation/","page":"Creation of the Generative Model","title":"Creation of the Generative Model","text":"A =\noversettextnormalsize Statesvphantombeginarrayc 0  0 endarray\n    beginarraycccc\n        1  0  0  0 \n        0  1  0  0 \n        0  0  1  0 \n        0  0  0  1\n    endarray\n\nquad\ntextnormalsize Observations","category":"page"},{"location":"GenerativeModelCreation/","page":"Creation of the Generative Model","title":"Creation of the Generative Model","text":"In this case, the agent is quite certain about which states produces which observations. This matrix could be made more uncertain to the point of complete uniformity and it could be made certain in the sense of each column being a one-hot vector. In the case of a certain A, the generative model stops being a \"partially observable\" Markov decision process, and becomes a fully observable one, making it a Markov decision process (MDP). For a more technical and mathematical definition of the observation likelihood model.","category":"page"},{"location":"GenerativeModelCreation/#B","page":"Creation of the Generative Model","title":"B","text":"","category":"section"},{"location":"GenerativeModelCreation/","page":"Creation of the Generative Model","title":"Creation of the Generative Model","text":"B is the transition likelihood model that encodes the agent's beliefs about how the hidden states of the environment changes over time. This is also made up of categorical distributions, though instead of observations to states, it maps states to states. If we take the same case again, a 2x2 gridworld, we would have a 4x4 matrix that describes how the agent believes the states evolve over time. An extra addition to B, is that it can depend on actions, meaning that it can believe that the hidden states of the environment change differently depending on the action taken by the agent. Due to this fact, we would the have a matrix for each action, making B a 3 dimensional tensor, with 2 dimensions for the \"from\" state and the \"to\" state, and then an action dimension. Let's look at an example of a slice of B for the action \"down\" in the grid world, which in this case would be a 4x4 matrix:","category":"page"},{"location":"GenerativeModelCreation/","page":"Creation of the Generative Model","title":"Creation of the Generative Model","text":"B(down) =\noversettextnormalsize Previous Statevphantombeginarrayc 0  0 endarray\n    beginarraycccc\n        0  0  0  0 \n        1  1  0  0 \n        0  0  0  0 \n        0  0  1  1\n    endarray\n\nquad\ntextnormalsize Current State","category":"page"},{"location":"GenerativeModelCreation/","page":"Creation of the Generative Model","title":"Creation of the Generative Model","text":"We could make 3 more similar matrices for the actions \"up\", \"left\", and \"right\", and then we would have the full B tensor for the gridworld. But here, the main point is that B decsribes the agent's belief of how hidden states change over time, and this can be dependent on actions, but might also be independent of actions, and thus the agent believes that the changes are out of its control.","category":"page"},{"location":"GenerativeModelCreation/#C","page":"Creation of the Generative Model","title":"C","text":"","category":"section"},{"location":"GenerativeModelCreation/","page":"Creation of the Generative Model","title":"Creation of the Generative Model","text":"C is the prior over observations, also called preferences over observations. This is an integral part of the utility of certain observations, i.e. it encodes how much the agent prefers or dislikes certain observations. C is a simple vector over observations, where each entry is a value that describes the utility or preference of that specific observation. If we continue with the simple 2x2 gridworld example, we would have 4 observations, one for each location state (same amount of observations as in A). Let's say that we would like for the agent to dislike observing the top left location (indexed as 1), and prefer the bottom right location (indexed as 4). We would then create C in the following way:","category":"page"},{"location":"GenerativeModelCreation/","page":"Creation of the Generative Model","title":"Creation of the Generative Model","text":"C =\nbeginarraycccc\n    -2  0  0  2 \nendarray","category":"page"},{"location":"GenerativeModelCreation/","page":"Creation of the Generative Model","title":"Creation of the Generative Model","text":"The magnitude of the values in C is arbitrary, and denotes a ratio and amount of dislike/preference. Here, we have chosen the value of -2 and 2 to encode that the agent dislikes the top left location just as much as it likes the bottom right location. The zeros in between just means that the agent has not preference or dislike for these locatin observations. Note that since C is not a categorical distribution, it does not need to sum to 1, and the values can be any real number.","category":"page"},{"location":"GenerativeModelCreation/#D","page":"Creation of the Generative Model","title":"D","text":"","category":"section"},{"location":"GenerativeModelCreation/","page":"Creation of the Generative Model","title":"Creation of the Generative Model","text":"D is the prior over states, and is the agent's beliefs about the initial state of the environment. This is also a simple vector that is a categorical distribution. Note that if A is certain, then D does not matter a lot for the inference process, as the agent can infer the state from the observation. However, if A is uncertain, then D becomes very important, as it serves as the agent's anchor point of where it is initially in the environment. In the case of out 2x2 gridworld, we would have a vector with 4 entries, one for each location state. If we assume that the agent correctly infers it's initial location as upper left corner, D would look like:","category":"page"},{"location":"GenerativeModelCreation/","page":"Creation of the Generative Model","title":"Creation of the Generative Model","text":"D =\nbeginarraycccc\n    1  0  0  0 \nendarray","category":"page"},{"location":"GenerativeModelCreation/#E","page":"Creation of the Generative Model","title":"E","text":"","category":"section"},{"location":"GenerativeModelCreation/","page":"Creation of the Generative Model","title":"Creation of the Generative Model","text":"E is the prior over policies, and can be described as the agent's habits. Policies in Active Inference vernacular are sets of actions, with an action for each step in the future, specified by a policy length. It is a categorical distribution over policies, with a probability for each policy. This will have an effect on the agent posterior over policies, which is the probability of taking a certain action at a time step. This will often be set to a uniform distribution, if we are not interested in giving the agent habits. Let us assume that we will give our agent a uniform E for a policy length of 2, this mean that we will have a uniform categorical distribution over 16 possible policies (4 (actions) ^ 2 (policy length)):","category":"page"},{"location":"GenerativeModelCreation/","page":"Creation of the Generative Model","title":"Creation of the Generative Model","text":"E =\nbeginarraycccc\n00625  00625  00625  00625  00625  00625  00625  00625  00625  00625  00625  00625  00625  00625  00625  00625 \nendarray","category":"page"},{"location":"GenerativeModelCreation/#Creating-the-Generative-Model-using-a-Helper-Function","page":"Creation of the Generative Model","title":"Creating the Generative Model using a Helper Function","text":"","category":"section"},{"location":"GenerativeModelCreation/","page":"Creation of the Generative Model","title":"Creation of the Generative Model","text":"","category":"page"},{"location":"GenerativeModelCreation/","page":"Creation of the Generative Model","title":"Creation of the Generative Model","text":"This page was generated using Literate.jl.","category":"page"},{"location":"Introduction/#Introduction-to-the-ActiveInference.jl-package","page":"Introduction","title":"Introduction to the ActiveInference.jl package","text":"","category":"section"},{"location":"Introduction/","page":"Introduction","title":"Introduction","text":"This package is a Julia implementation of the Active Inference framework, with a specific focus on cognitive modelling. In its current implementation, the package is designed to handle scenarios that can be modelled as discrete state spaces, with 'partially observable Markov decision process' (POMDP). In this documentation we will go through the basic concepts of how to use the package for different purposes; simulation and model inversion with Active Inference, also known as parameter estimation.","category":"page"},{"location":"Introduction/#Installing-Package","page":"Introduction","title":"Installing Package","text":"","category":"section"},{"location":"Introduction/","page":"Introduction","title":"Introduction","text":"Installing the package is done by adding the package from the julia official package registry in the following way:","category":"page"},{"location":"Introduction/","page":"Introduction","title":"Introduction","text":"using Pkg\nPkg.add(\"ActiveInference\")","category":"page"},{"location":"Introduction/","page":"Introduction","title":"Introduction","text":"Now, having added the package, we simply import the package to start using it:","category":"page"},{"location":"Introduction/","page":"Introduction","title":"Introduction","text":"using ActiveInference","category":"page"},{"location":"Introduction/","page":"Introduction","title":"Introduction","text":"In the next section we will go over the basic concepts of how to start using the package. We do this by providing instructions on how to create and design a generative model, that can be used for both simulation and parameter estimation.","category":"page"},{"location":"Introduction/","page":"Introduction","title":"Introduction","text":"","category":"page"},{"location":"Introduction/","page":"Introduction","title":"Introduction","text":"This page was generated using Literate.jl.","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = ActiveInference","category":"page"},{"location":"#ActiveInference","page":"Home","title":"ActiveInference","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for ActiveInference.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [ActiveInference, ActiveInference.Environments]","category":"page"},{"location":"#ActiveInference.action_select-Tuple{Any}","page":"Home","title":"ActiveInference.action_select","text":"Selects action from computed actions probabilities – used for stochastic action sampling \n\n\n\n\n\n","category":"method"},{"location":"#ActiveInference.array_of_any_zeros-Tuple{Any}","page":"Home","title":"ActiveInference.array_of_any_zeros","text":"Creates an array of \"Any\" with the desired number of sub-arrays filled with zeros\n\n\n\n\n\n","category":"method"},{"location":"#ActiveInference.bayesian_model_average-Tuple{Any, Any}","page":"Home","title":"ActiveInference.bayesian_model_average","text":"Calculate Bayesian Model Average (BMA)\n\nCalculates the Bayesian Model Average (BMA) which is used for the State Action Prediction Error (SAPE). It is a weighted average of the expected states for all policies weighted by the posterior over policies. The qs_pi_all should be the collection of expected states given all policies. Can be retrieved with the get_expected_states function.\n\nqs_pi_all: Vector{Any} \n\nq_pi: Vector{Float64}\n\n\n\n\n\n","category":"method"},{"location":"#ActiveInference.calc_expected_utility-Tuple{Any, Any}","page":"Home","title":"ActiveInference.calc_expected_utility","text":"Calculate Expected Utility \n\n\n\n\n\n","category":"method"},{"location":"#ActiveInference.calc_free_energy","page":"Home","title":"ActiveInference.calc_free_energy","text":"Calculate Free Energy \n\n\n\n\n\n","category":"function"},{"location":"#ActiveInference.calc_pA_info_gain-Tuple{Any, Any, Any}","page":"Home","title":"ActiveInference.calc_pA_info_gain","text":"Calculate observation to state info Gain \n\n\n\n\n\n","category":"method"},{"location":"#ActiveInference.calc_pB_info_gain-NTuple{4, Any}","page":"Home","title":"ActiveInference.calc_pB_info_gain","text":"Calculate state to state info Gain \n\n\n\n\n\n","category":"method"},{"location":"#ActiveInference.calc_states_info_gain-Tuple{Any, Any}","page":"Home","title":"ActiveInference.calc_states_info_gain","text":"Calculate States Information Gain \n\n\n\n\n\n","category":"method"},{"location":"#ActiveInference.calculate_SAPE-Tuple{ActiveInference.AIF}","page":"Home","title":"ActiveInference.calculate_SAPE","text":"Calculate State-Action Prediction Error \n\n\n\n\n\n","category":"method"},{"location":"#ActiveInference.calculate_bayesian_surprise-Tuple{Any, Any}","page":"Home","title":"ActiveInference.calculate_bayesian_surprise","text":"Calculate Bayesian Surprise \n\n\n\n\n\n","category":"method"},{"location":"#ActiveInference.capped_log-Tuple{Array{Float64}}","page":"Home","title":"ActiveInference.capped_log","text":"capped_log(array::Array{Float64})\n\n\n\n\n\n","category":"method"},{"location":"#ActiveInference.capped_log-Tuple{Real}","page":"Home","title":"ActiveInference.capped_log","text":"capped_log(x::Real)\n\nArguments\n\nx::Real: A real number.\n\nReturn the natural logarithm of x, capped at the machine epsilon value of x.\n\n\n\n\n\n","category":"method"},{"location":"#ActiveInference.capped_log-Tuple{Vector{Real}}","page":"Home","title":"ActiveInference.capped_log","text":"capped_log(array::Vector{Real})\n\n\n\n\n\n","category":"method"},{"location":"#ActiveInference.capped_log-Union{Tuple{Array{T}}, Tuple{T}} where T<:Real","page":"Home","title":"ActiveInference.capped_log","text":"capped_log(array::Array{T}) where T <: Real\n\n\n\n\n\n","category":"method"},{"location":"#ActiveInference.capped_log_array-Tuple{Any}","page":"Home","title":"ActiveInference.capped_log_array","text":"Apply capped_log to array of arrays \n\n\n\n\n\n","category":"method"},{"location":"#ActiveInference.check_probability_distribution-Union{Tuple{Array{Vector{T}, 1}}, Tuple{T}} where T<:Real","page":"Home","title":"ActiveInference.check_probability_distribution","text":"Check if the vector of vectors is a proper probability distribution.\n\nArguments\n\n(Array::Vector{Vector{T}}) where T<:Real\n\nThrows an error if the array is not a valid probability distribution:\n\nThe values must be non-negative.\nThe sum of the values must be approximately 1.\n\n\n\n\n\n","category":"method"},{"location":"#ActiveInference.check_probability_distribution-Union{Tuple{Vector{<:Array{T}}}, Tuple{T}} where T<:Real","page":"Home","title":"ActiveInference.check_probability_distribution","text":"Check if the vector of arrays is a proper probability distribution.\n\nArguments\n\n(Array::Vector{<:Array{T}}) where T<:Real\n\nThrows an error if the array is not a valid probability distribution:\n\nThe values must be non-negative.\nThe sum of the values must be approximately 1.\n\n\n\n\n\n","category":"method"},{"location":"#ActiveInference.check_probability_distribution-Union{Tuple{Vector{T}}, Tuple{T}} where T<:Real","page":"Home","title":"ActiveInference.check_probability_distribution","text":"Check if the vector is a proper probability distribution.\n\nArguments\n\n(Vector::Vector{T}) where T<:Real : The vector to be checked.\n\nThrows an error if the array is not a valid probability distribution:\n\nThe values must be non-negative.\nThe sum of the values must be approximately 1.\n\n\n\n\n\n","category":"method"},{"location":"#ActiveInference.compute_accuracy-Tuple{Any, Array{Vector{T}, 1} where T<:Real}","page":"Home","title":"ActiveInference.compute_accuracy","text":"Calculate Accuracy Term \n\n\n\n\n\n","category":"method"},{"location":"#ActiveInference.compute_accuracy_new-Tuple{Any, Vector{Vector{Real}}}","page":"Home","title":"ActiveInference.compute_accuracy_new","text":"Edited Compute Accuracy [Still needs to be nested within Fixed-Point Iteration] \n\n\n\n\n\n","category":"method"},{"location":"#ActiveInference.construct_policies-Tuple{Vector{T} where T<:Real}","page":"Home","title":"ActiveInference.construct_policies","text":"construct_policies(n_states::Vector{T} where T <: Real; n_controls::Union{Vector{T}, Nothing} where T <: Real=nothing, \n                   policy_length::Int=1, controllable_factors_indices::Union{Vector{Int}, Nothing}=nothing)\n\nConstruct policies based on the number of states, controls, policy length, and indices of controllable state factors.\n\nArguments\n\nn_states::Vector{T} where T <: Real: A vector containing the number of  states for each factor.\nn_controls::Union{Vector{T}, Nothing} where T <: Real=nothing: A vector specifying the number of allowable actions for each state factor. \npolicy_length::Int=1: The length of policies. (planning horizon)\ncontrollable_factors_indices::Union{Vector{Int}, Nothing}=nothing: A vector of indices identifying which state factors are controllable.\n\n\n\n\n\n","category":"method"},{"location":"#ActiveInference.create_matrix_templates","page":"Home","title":"ActiveInference.create_matrix_templates","text":"create_matrix_templates(shapes::Vector{Int64}, template_type::String)\n\nCreates templates based on the specified shapes vector and template type. Templates can be uniform, random, or filled with zeros.\n\nArguments\n\nshapes::Vector{Int64}: A vector specifying the dimensions of each template to create.\ntemplate_type::String: The type of templates to create. Can be \"uniform\" (default), \"random\", or \"zeros\".\n\nReturns\n\nA vector of arrays, each corresponding to the shape given by the input vector.\n\n\n\n\n\n","category":"function"},{"location":"#ActiveInference.create_matrix_templates-Tuple{Vector{Int64}, Vector{Int64}, Vector{Int64}, Int64}","page":"Home","title":"ActiveInference.create_matrix_templates","text":"create_matrix_templates(n_states::Vector{Int64}, n_observations::Vector{Int64}, n_controls::Vector{Int64}, policy_length::Int64, template_type::String = \"uniform\")\n\nCreates templates for the A, B, C, D, and E matrices based on the specified parameters.\n\nArguments\n\nn_states::Vector{Int64}: A vector specifying the dimensions and number of states.\nn_observations::Vector{Int64}: A vector specifying the dimensions and number of observations.\nn_controls::Vector{Int64}: A vector specifying the number of controls per factor.\npolicy_length::Int64: The length of the policy sequence. \ntemplate_type::String: The type of templates to create. Can be \"uniform\", \"random\", or \"zeros\". Defaults to \"uniform\".\n\nReturns\n\nA, B, C, D, E: The generative model as matrices and vectors.\n\n\n\n\n\n","category":"method"},{"location":"#ActiveInference.create_matrix_templates-Tuple{Vector{Int64}}","page":"Home","title":"ActiveInference.create_matrix_templates","text":"create_matrix_templates(shapes::Vector{Int64})\n\nCreates uniform templates based on the specified shapes vector.\n\nArguments\n\nshapes::Vector{Int64}: A vector specifying the dimensions of each template to create.\n\nReturns\n\nA vector of normalized arrays.\n\n\n\n\n\n","category":"method"},{"location":"#ActiveInference.create_matrix_templates-Tuple{Vector{Vector{Int64}}, String}","page":"Home","title":"ActiveInference.create_matrix_templates","text":"create_matrix_templates(shapes::Vector{Vector{Int64}}, template_type::String)\n\nCreates a multidimensional template based on the specified vector of shape vectors and template type. Templates can be uniform, random, or filled with zeros.\n\nArguments\n\nshapes::Vector{Vector{Int64}}: A vector of vectors, where each vector represent a dimension of the template to create.\ntemplate_type::String: The type of templates to create. Can be \"uniform\" (default), \"random\", or \"zeros\".\n\nReturns\n\nA vector of arrays, each having the multi-dimensional shape specified in the input vector.\n\n\n\n\n\n","category":"method"},{"location":"#ActiveInference.create_matrix_templates-Tuple{Vector{Vector{Int64}}}","page":"Home","title":"ActiveInference.create_matrix_templates","text":"create_matrix_templates(shapes::Vector{Vector{Int64}})\n\nCreates a uniform, multidimensional template based on the specified shapes vector.\n\nArguments\n\nshapes::Vector{Vector{Int64}}: A vector of vectors, where each vector represent a dimension of the template to create.\n\nReturns\n\nA vector of normalized arrays (uniform distributions), each having the multi-dimensional shape specified in the input vector.\n\n\n\n\n\n","category":"method"},{"location":"#ActiveInference.dot_likelihood-Tuple{Any, Any}","page":"Home","title":"ActiveInference.dot_likelihood","text":"Dot-Product Function \n\n\n\n\n\n","category":"method"},{"location":"#ActiveInference.fixed_point_iteration-Tuple{Array{Array{T, N}, 1} where {T<:Real, N}, Vector{Vector{Float64}}, Vector{Int64}, Vector{Int64}}","page":"Home","title":"ActiveInference.fixed_point_iteration","text":"Run State Inference via Fixed-Point Iteration \n\n\n\n\n\n","category":"method"},{"location":"#ActiveInference.get_expected_obs-Tuple{Any, Array{Array{T, N}, 1} where {T<:Real, N}}","page":"Home","title":"ActiveInference.get_expected_obs","text":"Get Expected Observations \n\n\n\n\n\n","category":"method"},{"location":"#ActiveInference.get_expected_states-Tuple{Array{Vector{T}, 1} where T<:Real, Any, Matrix{Int64}}","page":"Home","title":"ActiveInference.get_expected_states","text":"Get Expected States \n\n\n\n\n\n","category":"method"},{"location":"#ActiveInference.get_expected_states-Tuple{Vector{Vector{Float64}}, Any, Vector{Matrix{Int64}}}","page":"Home","title":"ActiveInference.get_expected_states","text":"Multiple dispatch for all expected states given all policies\n\nMultiple dispatch for getting expected states for all policies based on the agents currently inferred states and the transition matrices for each factor and action in the policy.\n\nqs::Vector{Vector{Real}} \n\nB: Vector{Array{<:Real}} \n\npolicy: Vector{Matrix{Int64}}\n\n\n\n\n\n","category":"method"},{"location":"#ActiveInference.get_joint_likelihood-Tuple{Any, Any, Any}","page":"Home","title":"ActiveInference.get_joint_likelihood","text":"Get Joint Likelihood \n\n\n\n\n\n","category":"method"},{"location":"#ActiveInference.get_log_action_marginals-Tuple{Any}","page":"Home","title":"ActiveInference.get_log_action_marginals","text":"Function to get log marginal probabilities of actions \n\n\n\n\n\n","category":"method"},{"location":"#ActiveInference.get_model_dimensions","page":"Home","title":"ActiveInference.get_model_dimensions","text":"Get Model Dimensions from either A or B Matrix \n\n\n\n\n\n","category":"function"},{"location":"#ActiveInference.infer_policies!-Tuple{ActiveInference.AIF}","page":"Home","title":"ActiveInference.infer_policies!","text":"Update the agents's beliefs over policies \n\n\n\n\n\n","category":"method"},{"location":"#ActiveInference.infer_states!-Tuple{ActiveInference.AIF, Vector{Int64}}","page":"Home","title":"ActiveInference.infer_states!","text":"Update the agents's beliefs over states \n\n\n\n\n\n","category":"method"},{"location":"#ActiveInference.init_aif-Tuple{Any, Any}","page":"Home","title":"ActiveInference.init_aif","text":"Initialize Active Inference Agent function initaif(         A,         B;         C=nothing,         D=nothing,         E = nothing,         pA = nothing,         pB = nothing,          pD = nothing,         parameters::Union{Nothing, Dict{String,Real}} = nothing,         settings::Union{Nothing, Dict} = nothing,         savehistory::Bool = true)\n\nArguments\n\n'A': Relationship between hidden states and observations.\n'B': Transition probabilities.\n'C = nothing': Prior preferences over observations.\n'D = nothing': Prior over initial hidden states.\n'E = nothing': Prior over policies. (habits)\n'pA = nothing':\n'pB = nothing':\n'pD = nothing':\n'parameters::Union{Nothing, Dict{String,Real}} = nothing':\n'settings::Union{Nothing, Dict} = nothing':\n'settings::Union{Nothing, Dict} = nothing':\n\n\n\n\n\n","category":"method"},{"location":"#ActiveInference.kl_divergence-Tuple{Vector{Vector{Vector{Real}}}, Vector{Vector{Vector{Real}}}}","page":"Home","title":"ActiveInference.kl_divergence","text":"kl_divergence(P::Vector{Vector{Vector{Float64}}}, Q::Vector{Vector{Vector{Float64}}})\n\nArguments\n\nP::Vector{Vector{Vector{Real}}}\nQ::Vector{Vector{Vector{Real}}}\n\nReturn the Kullback-Leibler (KL) divergence between two probability distributions.\n\n\n\n\n\n","category":"method"},{"location":"#ActiveInference.normalize_arrays-Tuple{Vector{<:Array{<:Real}}}","page":"Home","title":"ActiveInference.normalize_arrays","text":"Normalizes multiple arrays \n\n\n\n\n\n","category":"method"},{"location":"#ActiveInference.normalize_arrays-Tuple{Vector{Any}}","page":"Home","title":"ActiveInference.normalize_arrays","text":"Normalizes multiple arrays \n\n\n\n\n\n","category":"method"},{"location":"#ActiveInference.normalize_distribution-Tuple{Any}","page":"Home","title":"ActiveInference.normalize_distribution","text":"Normalizes a Categorical probability distribution\n\n\n\n\n\n","category":"method"},{"location":"#ActiveInference.onehot-Tuple{Int64, Int64}","page":"Home","title":"ActiveInference.onehot","text":"Creates a onehot encoded vector \n\n\n\n\n\n","category":"method"},{"location":"#ActiveInference.outer_product","page":"Home","title":"ActiveInference.outer_product","text":"Multi-dimensional outer product \n\n\n\n\n\n","category":"function"},{"location":"#ActiveInference.process_observation-Tuple{Int64, Int64, Vector{Int64}}","page":"Home","title":"ActiveInference.process_observation","text":"process_observation(observation::Int, n_modalities::Int, n_observations::Vector{Int})\n\nProcess a single modality observation. Returns a one-hot encoded vector. \n\nArguments\n\nobservation::Int: The index of the observed state with a single observation modality.\nn_modalities::Int: The number of observation modalities in the observation. \nn_observations::Vector{Int}: A vector containing the number of observations for each modality.\n\nReturns\n\nVector{Vector{Real}}: A vector containing a single one-hot encoded observation.\n\n\n\n\n\n","category":"method"},{"location":"#ActiveInference.process_observation-Tuple{Union{Tuple{Vararg{Int64}}, Array{Int64}}, Int64, Vector{Int64}}","page":"Home","title":"ActiveInference.process_observation","text":"process_observation(observation::Union{Array{Int}, Tuple{Vararg{Int}}}, n_modalities::Int, n_observations::Vector{Int})\n\nProcess observation with multiple modalities and return them in a one-hot encoded format \n\nArguments\n\nobservation::Union{Array{Int}, Tuple{Vararg{Int}}}: A collection of indices of the observed states for each modality.\nn_modalities::Int: The number of observation modalities in the observation. \nn_observations::Vector{Int}: A vector containing the number of observations for each modality.\n\nReturns\n\nVector{Vector{Real}}: A vector containing one-hot encoded vectors for each modality.\n\n\n\n\n\n","category":"method"},{"location":"#ActiveInference.sample_action!-Tuple{ActiveInference.AIF}","page":"Home","title":"ActiveInference.sample_action!","text":"Sample action from the beliefs over policies \n\n\n\n\n\n","category":"method"},{"location":"#ActiveInference.sample_action-Tuple{Any, Vector{Matrix{Int64}}, Any}","page":"Home","title":"ActiveInference.sample_action","text":"Sample Action [Stochastic or Deterministic] \n\n\n\n\n\n","category":"method"},{"location":"#ActiveInference.select_highest-Union{Tuple{Vector{T}}, Tuple{T}} where T<:Real","page":"Home","title":"ActiveInference.select_highest","text":"Selects the highest value from Array – used for deterministic action sampling \n\n\n\n\n\n","category":"method"},{"location":"#ActiveInference.softmax_array-Tuple{Any}","page":"Home","title":"ActiveInference.softmax_array","text":"Softmax Function for array of arrays \n\n\n\n\n\n","category":"method"},{"location":"#ActiveInference.spm_wnorm-Tuple{Any}","page":"Home","title":"ActiveInference.spm_wnorm","text":"SPM_wnorm \n\n\n\n\n\n","category":"method"},{"location":"#ActiveInference.update_A!-Tuple{ActiveInference.AIF, Vector{Int64}}","page":"Home","title":"ActiveInference.update_A!","text":"Update A-matrix \n\n\n\n\n\n","category":"method"},{"location":"#ActiveInference.update_B!-Tuple{ActiveInference.AIF, Any}","page":"Home","title":"ActiveInference.update_B!","text":"Update B-matrix \n\n\n\n\n\n","category":"method"},{"location":"#ActiveInference.update_D!-Tuple{ActiveInference.AIF, Any}","page":"Home","title":"ActiveInference.update_D!","text":"Update D-matrix \n\n\n\n\n\n","category":"method"},{"location":"#ActiveInference.update_obs_likelihood_dirichlet-NTuple{4, Any}","page":"Home","title":"ActiveInference.update_obs_likelihood_dirichlet","text":"Update obs likelihood matrix \n\n\n\n\n\n","category":"method"},{"location":"#ActiveInference.update_posterior_policies","page":"Home","title":"ActiveInference.update_posterior_policies","text":"Update Posterior over Policies \n\n\n\n\n\n","category":"function"},{"location":"#ActiveInference.update_posterior_states-Tuple{Array{Array{T, N}, 1} where {T<:Real, N}, Vector{Int64}}","page":"Home","title":"ActiveInference.update_posterior_states","text":"Update Posterior States \n\n\n\n\n\n","category":"method"},{"location":"#ActiveInference.update_state_likelihood_dirichlet-Tuple{Any, Any, Any, Array{Vector{T}, 1} where T<:Real, Any}","page":"Home","title":"ActiveInference.update_state_likelihood_dirichlet","text":"Update state likelihood matrix \n\n\n\n\n\n","category":"method"},{"location":"#ActiveInference.update_state_prior_dirichlet-Tuple{Any, Array{Vector{T}, 1} where T<:Real}","page":"Home","title":"ActiveInference.update_state_prior_dirichlet","text":"Update prior D matrix \n\n\n\n\n\n","category":"method"},{"location":"#ActiveInference.Environments.bayesian_model_average-Tuple{Any, Any}","page":"Home","title":"ActiveInference.Environments.bayesian_model_average","text":"Calculate Bayesian Model Average (BMA)\n\nCalculates the Bayesian Model Average (BMA) which is used for the State Action Prediction Error (SAPE). It is a weighted average of the expected states for all policies weighted by the posterior over policies. The qs_pi_all should be the collection of expected states given all policies. Can be retrieved with the get_expected_states function.\n\nqs_pi_all: Vector{Any} \n\nq_pi: Vector{Float64}\n\n\n\n\n\n","category":"method"},{"location":"#ActiveInference.Environments.calculate_bayesian_surprise-Tuple{Any, Any}","page":"Home","title":"ActiveInference.Environments.calculate_bayesian_surprise","text":"Calculate Bayesian Surprise \n\n\n\n\n\n","category":"method"},{"location":"#ActiveInference.Environments.capped_log-Tuple{Array{Float64}}","page":"Home","title":"ActiveInference.Environments.capped_log","text":"capped_log(array::Array{Float64})\n\n\n\n\n\n","category":"method"},{"location":"#ActiveInference.Environments.capped_log-Tuple{Real}","page":"Home","title":"ActiveInference.Environments.capped_log","text":"capped_log(x::Real)\n\nArguments\n\nx::Real: A real number.\n\nReturn the natural logarithm of x, capped at the machine epsilon value of x.\n\n\n\n\n\n","category":"method"},{"location":"#ActiveInference.Environments.capped_log-Tuple{Vector{Real}}","page":"Home","title":"ActiveInference.Environments.capped_log","text":"capped_log(array::Vector{Real})\n\n\n\n\n\n","category":"method"},{"location":"#ActiveInference.Environments.capped_log-Union{Tuple{Array{T}}, Tuple{T}} where T<:Real","page":"Home","title":"ActiveInference.Environments.capped_log","text":"capped_log(array::Array{T}) where T <: Real\n\n\n\n\n\n","category":"method"},{"location":"#ActiveInference.Environments.capped_log_array-Tuple{Any}","page":"Home","title":"ActiveInference.Environments.capped_log_array","text":"Apply capped_log to array of arrays \n\n\n\n\n\n","category":"method"},{"location":"#ActiveInference.Environments.dot_likelihood-Tuple{Any, Any}","page":"Home","title":"ActiveInference.Environments.dot_likelihood","text":"Dot-Product Function \n\n\n\n\n\n","category":"method"},{"location":"#ActiveInference.Environments.get_joint_likelihood-Tuple{Any, Any, Any}","page":"Home","title":"ActiveInference.Environments.get_joint_likelihood","text":"Get Joint Likelihood \n\n\n\n\n\n","category":"method"},{"location":"#ActiveInference.Environments.kl_divergence-Tuple{Vector{Vector{Vector{Real}}}, Vector{Vector{Vector{Real}}}}","page":"Home","title":"ActiveInference.Environments.kl_divergence","text":"kl_divergence(P::Vector{Vector{Vector{Float64}}}, Q::Vector{Vector{Vector{Float64}}})\n\nArguments\n\nP::Vector{Vector{Vector{Real}}}\nQ::Vector{Vector{Vector{Real}}}\n\nReturn the Kullback-Leibler (KL) divergence between two probability distributions.\n\n\n\n\n\n","category":"method"},{"location":"#ActiveInference.Environments.normalize_arrays-Tuple{Vector{<:Array{<:Real}}}","page":"Home","title":"ActiveInference.Environments.normalize_arrays","text":"Normalizes multiple arrays \n\n\n\n\n\n","category":"method"},{"location":"#ActiveInference.Environments.normalize_arrays-Tuple{Vector{Any}}","page":"Home","title":"ActiveInference.Environments.normalize_arrays","text":"Normalizes multiple arrays \n\n\n\n\n\n","category":"method"},{"location":"#ActiveInference.Environments.normalize_distribution-Tuple{Any}","page":"Home","title":"ActiveInference.Environments.normalize_distribution","text":"Normalizes a Categorical probability distribution\n\n\n\n\n\n","category":"method"},{"location":"#ActiveInference.Environments.outer_product","page":"Home","title":"ActiveInference.Environments.outer_product","text":"Multi-dimensional outer product \n\n\n\n\n\n","category":"function"},{"location":"#ActiveInference.Environments.softmax_array-Tuple{Any}","page":"Home","title":"ActiveInference.Environments.softmax_array","text":"Softmax Function for array of arrays \n\n\n\n\n\n","category":"method"},{"location":"#ActiveInference.Environments.spm_wnorm-Tuple{Any}","page":"Home","title":"ActiveInference.Environments.spm_wnorm","text":"SPM_wnorm \n\n\n\n\n\n","category":"method"}]
}
