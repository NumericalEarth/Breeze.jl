"""
Extension module for integrating CloudMicrophysics.jl schemes with Breeze.jl.

This extension provides integration between CloudMicrophysics.jl microphysics schemes
and Breeze.jl's microphysics interface, allowing CloudMicrophysics schemes to be used
with AtmosphereModel.

The extension is automatically loaded when CloudMicrophysics is available in the environment.
"""
module BreezeCloudMicrophysicsExt

using CloudMicrophysics

# Import Breeze modules needed for integration
using ..Breeze
using ..Breeze.AtmosphereModels
using ..Breeze.Thermodynamics
using ..Breeze.Microphysics

# TODO: Add integration code here
# This will include:
# - Extending compute_thermodynamic_state for CloudMicrophysics schemes
# - Extending moisture_mass_fractions for CloudMicrophysics schemes
# - Extending prognostic_field_names, materialize_microphysical_fields, 
#   and update_microphysical_fields! for CloudMicrophysics schemes
# - Wrapper types to bridge CloudMicrophysics schemes with Breeze interface

end # module BreezeCloudMicrophysicsExt

