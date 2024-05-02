using IterTools
using LinearAlgebra
using ActiveInference
using Tests

""" Test Agent """

@testset "initialize AIF agent and run default inference loop" begin

    # Initializse States, Observations, and Controls
    states = [64,2]
    observations = [64,2]
    controls = [5,1]

    # Generate random Generative Model 
    A_matrix, B_matrix = generate_random_GM(states, observations, controls)

    # Initialize agent with default settings/parameters
    aif = init_aif(A_matrix, B_matrix);

    # Give observation to agent and run state inference
    # observation = [rand(1:n_obs[i]) for i in axes(n_obs, 1)]
    observation = [1,1]
    QS = infer_states!(aif, observation)

    # Run policy inference
    Q_pi, G = infer_policies!(aif)

    # Sample action
    action = sample_action!(aif)
end