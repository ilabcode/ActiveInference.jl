using IterTools
using LinearAlgebra
using ActiveInference
using Test

""" Test Agent """

@testset "Single Factor Condition - Default Settings" begin

    # Initializse States, Observations, and Controls
    states = [25]
    observations = [25]
    controls = [2]
    policy_length = 1

    # Generate random Generative Model 
    A, B = create_matrix_templates(states, observations, controls, policy_length, "random");

    # Initialize agent with default settings/parameters
    aif = init_aif(A, B);

    # Give observation to agent and run state inference
    observation = [rand(1:observations[i]) for i in axes(observations, 1)]
    QS = infer_states!(aif, observation)

    # Run policy inference
    Q_pi, G = infer_policies!(aif)

    # Sample action
    action = sample_action!(aif)
end


@testset "If There are more factors - Default Settings" begin

    # Initializse States, Observations, and Controls
    states = [64,2]
    observations = [64,2]
    controls = [5,1]
    policy_length = 1

    # Generate random Generative Model 
    A, B = create_matrix_templates(states, observations, controls, policy_length, "random");

    # Initialize agent with default settings/parameters
    aif = init_aif(A, B);

    # Give observation to agent and run state inference
    observation = [rand(1:observations[i]) for i in axes(observations, 1)]
    QS = infer_states!(aif, observation)

    # Run policy inference
    Q_pi, G = infer_policies!(aif)

    # Sample action
    action = sample_action!(aif)
end


@testset "Provide custom settings" begin

    # Initializse States, Observations, and Controls
    states = [64,2]
    observations = [64,2]
    controls = [5,2]
    policy_length = 3

    # Generate random Generative Model 
    A, B, C, D = create_matrix_templates(states, observations, controls, policy_length, "random");

    settings = Dict(
    "policy_len"           => 3,
    "use_states_info_gain" => true,
    "action_selection"     => "deterministic",
    "use_utility"          => true)

    # Initialize agent with custom settings
    aif = init_aif(A, B; C=C, D=D, settings = settings);

    # Give observation to agent and run state inference
    observation = [rand(1:observations[i]) for i in axes(observations, 1)]
    QS = infer_states!(aif, observation)

    # Run policy inference
    Q_pi, G = infer_policies!(aif)

    # Sample action deterministically 
    action = sample_action!(aif)

    # And infer new state
    observation = [rand(1:observations[i]) for i in axes(observations, 1)]
    QS_2 = infer_states!(aif, observation)
end


@testset "Learning with custom parameters" begin

    # Initializse States, Observations, and Controls
    states = [64,2]
    observations = [64,2]
    controls = [5,1]
    policy_length = 2

    # Generate random Generative Model 
    A, B, C, D = create_matrix_templates(states, observations, controls, policy_length, "random");

    # pA concentration parameter
    pA = deepcopy(A)
    for i in eachindex(pA)
        pA[i] .= 1.0
    end

    # pB concentration parameter
    pB = deepcopy(B)
    for i in eachindex(pB)
        pB[i] .= 1.0
    end

    # pD concentration parameter
    pD = deepcopy(D)
    for i in 1:length(D)
        pD[i] .= 1.0
    end

    # Give some settings to agent
    settings = Dict(
        "use_param_info_gain" => true,
        "policy_len" => 2
        )

    # Give custom parameters to agent
    parameters = Dict{String, Real}(
        "lr_pA" => 0.5,
        "fr_pA" => 1.0,
        "lr_pB" => 0.6,
        "lr_pD" => 0.7,
        "alpha" => 2.0,
        "gamma" => 2.0,
        "fr_pB" => 1.0,
        "fr_pD" => 1.0,
        )
    # initialize ageent
    aif = init_aif(A,
                   B; 
                   D = D,
                   pA = pA,
                   pB = pB,
                   pD = pD,
                   settings = settings,
                   parameters = parameters);

    ## Run inference with Learning
    for t in 1:2
        # Give observation to agent and run state inference
        observation = [rand(1:observations[i]) for i in axes(observations, 1)]
        QS = infer_states!(aif, observation)
    
        # # If action is empty, update D vectors
        # if ismissing(get_states(aif)["action"])
        #     QS_t1 = get_history(aif)["posterior_states"][1]
        #     update_D!(aif, QS_t1)
        # end

        # # If agent has taken action, update transition matrices
        # if get_states(aif)["action"] !== missing
        #     QS_prev = get_history(aif)["posterior_states"][end-1]
        #     update_B!(aif, QS_prev)
        # end
        # # Update A matrix
        # update_A!(aif, observation)

        update_parameters!(aif)
    
        # Run policy inference
        Q_pi, G = infer_policies!(aif)
    
        # Sample action
        action = sample_action!(aif)
    end
end
