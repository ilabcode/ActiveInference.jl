using IterTools
using LinearAlgebra
using ActiveInference
using Test

""" Test Utils & ActionModels.jl Extensions """

@testset "ActionModels Utils" begin

    # Initializse States, Observations, and Controls
    states = [25]
    observations = [25]
    controls = [2]

    # Generate random Generative Model 
    A_matrix, B_matrix = generate_random_GM(states, observations, controls)

    # Initialize agent with default settings/parameters
    aif = init_aif(A_matrix, B_matrix);

    # Set Parameters as dictionary
    params=Dict("lr_pA" => 1.0,
                "fr_pA" => 1.0,
                "lr_pB" => 1.0,
                "lr_pD" => 1.0,
                "alpha" => 1.0,
                "gamma" => 1.0,
                "fr_pB" => 1.0,
                "fr_pD" => 1.0)

    set_parameters!(aif, params)

    # Get states
    get_states(aif)
    get_states(aif, "action")
    get_states(aif, ["action", "prior"])

    # Test get_history
    set_save_history!(aif, true)
    get_history(aif)
    get_history(aif,"action")
    get_history(aif,["action", "prior"])


    # Give individual parameters
    set_parameters!(aif, "lr_pA", 0.5)
    set_parameters!(aif, "fr_pA", 0.5)
    set_parameters!(aif, "lr_pB", 0.5)
    set_parameters!(aif, "lr_pD", 0.5)
    set_parameters!(aif, "gamma", 10.0)
    set_parameters!(aif, "alpha", 10.0)
    set_parameters!(aif, "fr_pB", 0.5)
    set_parameters!(aif, "fr_pD", 0.5)

    # test get_parameters
    get_parameters(aif)
    get_parameters(aif, ["alpha", "gamma"])

end

@testset "Give Inputs and Reset" begin

    # Initializse States, Observations, and Controls
    states = [25]
    observations = [25]
    controls = [2]

    # Generate random Generative Model 
    A_matrix, B_matrix = generate_random_GM(states, observations, controls)

    # Initialize agent with default settings/parameters
    aif = init_aif(A_matrix, B_matrix);


    observation = [rand(1:observations[i]) for i in axes(observations, 1)]

    single_input!(aif, observation)

    reset!(aif)


end