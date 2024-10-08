"""
This module extends the "set_parameters!" functionality of the ActionModels package to work with instances of the AIF type.

    set_parameters!(aif::AIF, target_param::String, param_value::Real)
Set a single parameter in the AIF agent

    set_parameters!(aif::AIF, parameters::Dict{String, Real})
Set multiple parameters in the AIF agent

"""

using ActionModels

# Setting a single parameter
function ActionModels.set_parameters!(aif::AIF, target_param::String, param_value::Real)
    # Update the parameters dictionary
    aif.parameters[target_param] = param_value

    # Update the struct's field based on the target_param
    if target_param == "alpha"
        aif.alpha = param_value
    elseif target_param == "gamma"
        aif.gamma = param_value
    elseif target_param == "lr_pA"
        aif.lr_pA = param_value
    elseif target_param == "fr_pA"
        aif.fr_pA = param_value
    elseif target_param == "lr_pB"
        aif.lr_pB = param_value
    elseif target_param == "fr_pB"
        aif.fr_pB = param_value
    elseif target_param == "lr_pD"
        aif.lr_pD = param_value
    elseif target_param == "fr_pD"
        aif.fr_pD = param_value
    else
        throw(ArgumentError("The parameter $target_param is not recognized."))
    end
end

# Setting multiple parameters
function ActionModels.set_parameters!(aif::AIF, parameters::Dict)
    # For each parameter in the input dictionary
    for (target_param, param_value) in parameters
        # Directly set each parameter
        set_parameters!(aif, target_param, param_value)
    end
end