include(joinpath(@__DIR__, "setup.jl"))

#####
##### Tests for `DYCOMSRadiation`, the idealized longwave parameterization of
##### Stevens et al. (2005) for nocturnal marine stratocumulus.
#####
##### The tests are posed on the DYCOMS-II RF01 initial state, whose two-layer
##### structure makes every quantity below checkable by hand:
#####
##### - T1 — the diagnosed inversion height sits at the top of the mixed layer.
#####        This is the regression for a tie-breaking bug: the initial qᵗ is
#####        *uniform* below the inversion, so every level of the mixed layer is
#####        exactly equidistant from the qᵗ = 8 g/kg contour. Keeping the first
#####        such level instead of the last put zᵢ on the surface, which silently
#####        inflated the free-tropospheric term.
##### - T2 — the density at zᵢ recovers the ρᵢ ≈ 1.13 kg/m³ quoted by the case.
##### - T3 — ℐ at the surface reduces to the cloud-base amplitude ℐ₁.
##### - T4 — ℐ at the top of the column matches the closed-form three-term flux.
##### - T5 — the stored energy source (-∂z ℐ) is the exact discrete divergence of ℐ,
#####        so that ∫ (-∂z ℐ) dz telescopes to -(ℐ(H) - ℐ(0)).
##### - T6 — the radiative heating is a *cooling* peaked at cloud top.
#####

using Breeze
using Breeze: total_density
using Oceananigans
using Oceananigans.Grids: znodes, Center, Face
using Statistics
using Test

# DYCOMS-II RF01 initial profiles (Stevens et al. 2005, Eqs. 1-2)
const zᵢ_spec = 840.0

dycoms_θˡⁱ(z) = z <= zᵢ_spec ? 289.0 : 297.5 + (z - zᵢ_spec)^(1/3)
dycoms_qᵗ(z) = z <= zᵢ_spec ? 9.0e-3 : 1.5e-3

function dycoms_rf01_model(arch, Nz)
    grid = RectilinearGrid(arch; x=(0, 3360), y=(0, 3360), z=(0, 1500),
                           size = (8, 8, Nz), halo = (5, 5, 5),
                           topology = (Periodic, Periodic, Bounded))

    constants = ThermodynamicConstants(dry_air_heat_capacity=1015)

    reference_state = ReferenceState(grid, constants,
                                     base_pressure = 101780,
                                     potential_temperature = 289)

    radiation = DYCOMSRadiation(grid)

    model = AtmosphereModel(grid;
                            dynamics = AnelasticDynamics(reference_state),
                            thermodynamic_constants = constants,
                            microphysics = SaturationAdjustment(equilibrium=WarmPhaseEquilibrium()),
                            radiation)

    set!(model, θ=(x, y, z) -> dycoms_θˡⁱ(z), qᵗ=(x, y, z) -> dycoms_qᵗ(z))

    return model, radiation, grid
end

@testset "DYCOMSRadiation" begin
    @testset "Construction" begin
        grid = RectilinearGrid(default_arch; size=(4, 4, 8), x=(0, 1), y=(0, 1), z=(0, 1500))
        radiation = DYCOMSRadiation(grid)

        @test radiation.cloud_top_cooling == 70
        @test radiation.cloud_base_warming == 22
        @test radiation.absorption_coefficient == 85
        @test radiation.divergence == 3.75e-6
        @test radiation.inversion_moisture == 8e-3

        @test size(radiation.net_upward_flux) == (4, 4, 9)
        @test size(radiation.flux_divergence) == (4, 4, 8)

        # `show` is exercised by the docstring's doctest; check it runs here too
        @test occursin("DYCOMSRadiation", sprint(show, radiation))
    end

    for Nz in (150, 300)
        Δz = 1500 / Nz

        @testset "DYCOMS-II RF01 initial state (Nz = $Nz)" begin
            model, radiation, grid = dycoms_rf01_model(default_arch, Nz)

            zᶜ = znodes(grid, Center())
            ℐ = Field(Average(radiation.net_upward_flux, dims=(1, 2)))
            radiative_energy_source = Field(Average(radiation.flux_divergence, dims=(1, 2)))
            compute!(ℐ)
            compute!(radiative_energy_source)

            ℐᶻ = Array(interior(ℐ, 1, 1, :))
            Qᴿ = Array(interior(radiative_energy_source, 1, 1, :))  # -∂z ℐ, W/m³

            zᵢ = mean(Array(interior(radiation.inversion_height)))

            # T1: the mixed layer is uniform in qᵗ, so every level ties. The inversion
            # must land on the *topmost* tied level -- the last center below 840 m --
            # not on the first (which would put zᵢ at the surface).
            zᵢ_expected = zᶜ[searchsortedlast(zᶜ, zᵢ_spec)]
            @test zᵢ ≈ zᵢ_expected
            @test zᵢ > zᵢ_spec - 2Δz   # emphatically not the surface

            # T2: the case quotes an air density just below cloud top of ρᵢ ≈ 1.13 kg/m³
            ρ = total_density(model.dynamics)
            kᵢ = searchsortedlast(zᶜ, zᵢ_spec)
            ρᵢ = Array(interior(ρ, 1, 1, kᵢ:kᵢ))[1]
            @test ρᵢ ≈ 1.13 atol=0.02

            # T3: at the surface the optical thickness below vanishes, so ℐ(0) → ℐ₁,
            # plus the exponentially small remnant of the cloud-top term.
            @test ℐᶻ[1] ≈ radiation.cloud_base_warming rtol=0.05

            # T4: at the top of the column the closed form is
            #     ℐ(H) = ℐ₀ + ℐ₁ e^{-τ₀ᴴ} + ρᵢ cᵖ D (H - zᵢ)^{1/3} ((H - zᵢ)/4 + zᵢ)
            qˡ = model.microphysical_fields.qˡ
            LWP = Field(Integral(ρ * qˡ, dims=3))
            compute!(LWP)
            τ₀ᴴ = radiation.absorption_coefficient * mean(Array(interior(LWP)))

            H = znodes(grid, Face())[end]
            ζ = H - zᵢ
            free_troposphere = ρᵢ * radiation.heat_capacity * radiation.divergence *
                               cbrt(ζ) * (ζ / 4 + zᵢ)

            ℐᴴ_expected = radiation.cloud_top_cooling +
                          radiation.cloud_base_warming * exp(-τ₀ᴴ) +
                          free_troposphere

            @test ℐᶻ[end] ≈ ℐᴴ_expected rtol=0.02

            # T5: the stored divergence must telescope exactly
            @test sum(Qᴿ) * Δz ≈ -(ℐᶻ[end] - ℐᶻ[1]) rtol=1e-6

            # T6: radiative heating is a cooling, peaked at cloud top
            kmax = argmax(abs.(Qᴿ))
            @test Qᴿ[kmax] < 0
            @test zᶜ[kmax] ≈ zᵢ_spec atol=2Δz
        end
    end
end
