"""
This module extends the "set_parameters!" functionality of the ActionModels package to work with instances of the AIF type.

    set_parameters!(aif::AIF, target_param::String, param_value::Any)
Set a single parameter in the AIF agent

    set_parameters!(aif::AIF, parameters::Dict{String, Any})
Set multiple parameters in the AIF agent

"""

using ActionModels

# Setting a single parameter
function ActionModels.set_parameters!(aif::AIF, target_param::String, param_value::Any)
    # Check if the target parameter exists within the AIF's parameters
    if haskey(aif.parameters, target_param)
        # Set the parameter
        aif.parameters[target_param] = param_value
    else
        # If the target parameter does not exist
        throw(ArgumentError("The parameter $target_param does not exist within the AIF's parameters."))
    end
end

# Setting multiple parameters
function ActionModels.set_parameters!(aif::AIF, parameters::Dict{String, Float64})
    # For each parameter in the input dictionary
    for (target_param, param_value) in parameters
        # Directly set each parameter
        set_parameters!(aif, target_param, param_value)
    end
end