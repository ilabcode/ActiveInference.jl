using IterTools
using LinearAlgebra
using ActiveInference
using Test

""" Quick tests """

@testset "Multiple Factors/Modalities Default Settings" begin

    # Initializse States, Observations, and Controls
    states = [5,2]
    observations = [5, 4, 2]
    controls = [2,1]
    policy_length = 1

    # Generate random Generative Model 
    A,B = create_matrix_templates(states, observations, controls, policy_length, "random");

    # Initialize agent with default settings/parameters
    aif = init_aif(A,B);

    # Give observation to agent and run state inference
    observation = [rand(1:observations[i]) for i in axes(observations, 1)]
    infer_states!(aif, observation)

    # Run policy inference
    infer_policies!(aif)

    # Sample action
    sample_action!(aif)
end