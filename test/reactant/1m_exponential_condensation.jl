include(joinpath(@__DIR__, "microphysics_compilation_setup.jl"))

using Breeze.Microphysics: NonEquilibriumCloudFormation
using CloudMicrophysics

BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
using .BreezeCloudMicrophysicsExt: OneMomentCloudMicrophysics

formation = ExponentialRateCondensateFormation(1 / 120, 0.02)
microphysics = OneMomentCloudMicrophysics(; cloud_formation = NonEquilibriumCloudFormation(formation, formation))

initial_state = (; ρqᵛ=0.01, ρqᶜˡ=1e-4, ρqᶜⁱ=1e-5, ρqʳ=1e-5, ρqˢⁿ=1e-6)

run_microphysics_tests("OneMomentCloudMicrophysics (MPNE1M, ExponentialRateCondensateFormation)", microphysics, initial_state)
