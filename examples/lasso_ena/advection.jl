#####
##### Advection schemes per prognostic scalar.
#####

"""
    scalar_advection_schemes(order, microphysics, moisture_name; bounded_condensates=true, positive_moments=true,
                             energy_name=:ρθ, microphysics_order=order)

Advection scheme per prognostic scalar, following Breeze's `examples/rico.jl`: the potential
temperature (`energy_name`, `ρθ`) uses plain WENO; every microphysical *water-mass* tracer (the vapor / equilibrium
moisture and all condensate masses `ρq*`) uses bounds-preserving WENO with bounds `(0, 1)`;
dimensional number and volume moments (`ρn*`, `ρb*`), whose magnitudes are not bounded by
one, use the same limiter with bounds `(0, ∞)`, i.e. positivity only. Plain WENO for the
moments (`positive_moments = false`, with P3's `SpeciesBorrowing` clamping the negative
undershoots afterwards) let the rain number in the 10-m surface cells of the Covert grid
grow by ~8 % per step once P3 had diagnosed micron drops there (see the README); the
positivity limiter holds it at physical values.

Known issue (documented, tracked): the bounds-preserving WENO of the pinned Oceananigans
limits only the upwind reconstruction of the evaluating cell, so the two cells sharing a
face can apply different fluxes when its limiter fires and mass is not conserved there. With
P3's fast sedimentation this piled phantom rain into the surface cell of the 45-minute
GPU probes (surface `qʳ` reached 9.5 g kg⁻¹, versus < 0.03 g kg⁻¹ with plain WENO), and the
forcing-free rain-shaft budget in `test/diagnostics/mass_budget_probe.jl` records the residual.
Oceananigans `main` (pinned since the `glw/weno-z-float32-overflow` revision) stores the limiter as a cell field and rescales every face
reconstruction with its own cell's factor, which restores conservation (validated by
`test/diagnostics/mass_budget_probe.jl` and the GPU smokes). `bounded_condensates = false` (plain WENO for the
condensate masses) is retained as a diagnostic sensitivity only. `microphysics_order` sets the WENO order of
every microphysical prognostic (all names other than `energy_name` and the moisture variable).
"""
function scalar_advection_schemes(order, microphysics, moisture_name; bounded_condensates=true, positive_moments=true,
                                  energy_name=:ρθ, microphysics_order=order)
    weno = WENO(; order)
    bounded = WENO(; order, bounds=(0, 1))
    # The microphysical prognostics (condensate masses and moments) may use their own order
    micro_weno = WENO(; order = microphysics_order)
    micro_bounded = WENO(; order = microphysics_order, bounds=(0, 1))
    # Number and volume moments are not bounded by one, so their limiter only enforces
    # positivity (an infinite upper bound disables the maximum side of the limiter).
    moments = positive_moments ? WENO(; order = microphysics_order, bounds=(0.0, Inf)) : micro_weno
    moisture = Symbol("ρ", moisture_name)
    names = (energy_name, moisture, Breeze.AtmosphereModels.prognostic_field_names(microphysics)...)
    schemes = map(names) do name
        name === energy_name ? weno :
        name === moisture ? bounded :
        occursin("ρq", string(name)) ? (bounded_condensates ? micro_bounded : micro_weno) : moments
    end
    return NamedTuple{names}(schemes)
end
