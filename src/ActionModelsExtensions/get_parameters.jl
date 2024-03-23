"""
This module extends the "get_parameters" functionality of the ActionModels package to work specifically with instances of the AIF type.


    get_parameters(aif::AIF, target_parameters::Vector{String})
Retrieves multiple target parameters from an AIF agent. 

    get_parameters(aif::AIF, target_parameter::String)
Retrieves a single target parameter from an AIF agent.

    get_parameters(aif::AIF)
Retrieves all parameters from an AIF agent.

"""

using ActionModels

# Retrieves multiple target parameters
function ActionModels.get_parameters(aif::AIF, target_parameters::Vector{String})
    parameters = Dict()

    for target_parameter in target_parameters
        try
            parameters[target_parameter] = get_parameters(aif, target_parameter)
        catch e
            if isa(e, ArgumentError)
                throw(ArgumentError("The specified parameter $target_parameter does not exist"))
            else
                rethrow(e)
            end
        end
    end

    return parameters
end

# Retrieves a single parameter
function ActionModels.get_parameters(aif::AIF, target_parameter::String)
    if haskey(aif.parameters, target_parameter)
        return aif.parameters[target_parameter]
    else
        throw(ArgumentError("The specified parameter $target_parameter does not exist"))
    end
end


# Retrieves all parameters 
function ActionModels.get_parameters(aif::AIF)
    return aif.parameters
end