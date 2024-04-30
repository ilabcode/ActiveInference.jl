"""
ActionModels - set save history
"""

using ActionModels

function ActionModels.set_save_history!(aif::AIF, save_history::Bool)
    aif.save_history = save_history
end