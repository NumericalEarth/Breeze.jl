include("microphysics_compilation_setup.jl")

using Breeze.Microphysics: NonEquilibriumCloudFormation
using CloudMicrophysics

BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
using .BreezeCloudMicrophysicsExt: OneMomentCloudMicrophysics

microphysics = OneMomentCloudMicrophysics(;
    cloud_formation = NonEquilibriumCloudFormation(nothing, :ice))

initial_state = (; ρqᵛ=0.01, ρqᶜˡ=1e-4, ρqᶜⁱ=1e-5, ρqʳ=1e-5, ρqˢⁿ=1e-6)

run_microphysics_tests("OneMomentCloudMicrophysics (MPNE1M)", microphysics, initial_state)
