"""
Resets an AIF type agent to its initial state

    reset!(aif::AIF)

"""

using ActionModels

function ActionModels.reset!(aif::AIF)
    # Reset the agent's state fields to initial conditions
    aif.qs_current = array_of_any_uniform([size(aif.B[f], 1) for f in eachindex(aif.B)])
    aif.prior = aif.D
    aif.Q_pi = ones(Real,length(aif.policies)) / length(aif.policies)
    aif.G = zeros(Real,length(aif.policies))
    aif.action = Real[]

    # Clear the history in the states dictionary
    for key in keys(aif.states)

        if key != "policies"
            aif.states[key] = []
        end
    end
    return nothing
end