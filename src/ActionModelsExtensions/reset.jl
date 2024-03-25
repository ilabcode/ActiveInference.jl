"""
Resets an AIF type agent to its initial state

    reset!(aif::AIF)

"""

using ActionModels

function ActionModels.reset!(aif::AIF)
    # Reset the agent's state fields to initial conditions
    aif.qs_current = onehot(1, size(aif.B, 1))
    aif.prior = aif.D
    aif.Q_pi = ones(length(aif.policies)) / length(aif.policies)
    aif.G = zeros(length(aif.policies))
    aif.action = Float64[]

    # Clear the history in the states dictionary
    for key in keys(aif.states)

        if key != "policies"
            aif.states[key] = []
        end
    end
    return nothing
end