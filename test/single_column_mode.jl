include(joinpath(@__DIR__, "setup.jl"))

#####
##### Tests for single-column mode: an AtmosphereModel on a (Flat, Flat, Bounded) grid runs as a
##### single column (or, via ColumnEnsembleSize, a forest of independent columns) with the anelastic
##### pressure solve and vertical-velocity stepping both omitted (w ≡ 0).
#####

using Breeze
using Breeze.AtmosphereModels: SingleColumnGrid

using Oceananigans
using Oceananigans.Grids: ColumnEnsembleSize
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Architectures: on_architecture
using Oceananigans.Coriolis: FPlane
using Oceananigans.Forcings: Forcing
using Oceananigans: prognostic_fields
using GPUArraysCore: @allowscalar
using Test

# Discrete per-column forcing: heat each column by its own rate, read from a per-column array.
@inline column_heating(i, j, k, grid, clock, fields, Q) = @inbounds Q[i, j]

const scm_constants = ThermodynamicConstants()

function single_column_model(grid; closure = VerticalScalarDiffusivity(ν=1.0, κ=1.0))
    reference_state = ReferenceState(grid, scm_constants; surface_pressure=101325, potential_temperature=290)
    return AtmosphereModel(grid; dynamics=AnelasticDynamics(reference_state), closure)
end

# Prognostic thermodynamic density name (:ρθ for potential-temperature formulations)
thermodynamic_name(model) = :ρθ ∈ keys(prognostic_fields(model)) ? :ρθ : :ρe

@testset "Single column mode [$FT]" for FT in test_float_types()
    arch = default_arch
    Nz = 16

    @testset "Single column construction and w ≡ 0" begin
        grid = RectilinearGrid(arch, FT; size=Nz, z=(0, 2000), topology=(Flat, Flat, Bounded))
        @test grid isa SingleColumnGrid

        model = single_column_model(grid)

        # No pressure solver is built in single-column mode, and a 1×1 column keeps the
        # memory-efficient reduced (Nothing, Nothing, Center) reference profile.
        @test model.pressure_solver === nothing
        @test size(interior(model.dynamics.reference_state.density)) == (1, 1, Nz)

        set!(model, θ = z -> 290 + FT(0.01) * z)
        T_before = Array(interior(model.temperature))

        for n in 1:20
            time_step!(model, 1)
        end

        # Vertical velocity is identically zero: no pressure projection, no z-momentum stepping.
        @test maximum(abs, interior(model.velocities.w)) == 0
        @test maximum(abs, interior(model.momentum.ρw)) == 0
        # The column still evolves: vertical diffusion mixes the imposed θ gradient.
        @test !any(isnan, interior(model.temperature))
        @test Array(interior(model.temperature)) != T_before
    end

    @testset "Column ensemble builds with full reference fields" begin
        N₁, N₂ = 3, 2
        grid = RectilinearGrid(arch, FT; size = ColumnEnsembleSize(Nz=Nz, ensemble=(N₁, N₂), Hz=3),
                               z = (0, 2000), topology = (Flat, Flat, Bounded))
        @test grid isa SingleColumnGrid
        @test size(grid) == (N₁, N₂, Nz)

        model = single_column_model(grid)
        @test model.pressure_solver === nothing
        # On an ensemble grid the reference profile is a full field (one profile per column).
        @test size(interior(model.dynamics.reference_state.density)) == (N₁, N₂, Nz)
    end

    @testset "Column ensemble reproduces independent single columns" begin
        N₁, N₂ = 3, 2
        amp(i, j) = FT(0.002) * (i + N₁ * (j - 1))  # distinct θ-gradient amplitude per column

        # Ensemble of N₁×N₂ columns, each initialized with a distinct θ gradient.
        ensemble_grid = RectilinearGrid(arch, FT; size = ColumnEnsembleSize(Nz=Nz, ensemble=(N₁, N₂), Hz=3),
                                        z = (0, 2000), topology = (Flat, Flat, Bounded))
        ensemble = single_column_model(ensemble_grid)
        name = thermodynamic_name(ensemble)

        set!(ensemble, θ = 290)
        zc = Array(znodes(ensemble_grid, Center()))
        @allowscalar begin
            ρθ = prognostic_fields(ensemble)[name]
            ρᵣ = interior(ensemble.dynamics.reference_state.density)
            ρθi = interior(ρθ)
            for j in 1:N₂, i in 1:N₁, k in 1:Nz
                ρθi[i, j, k] += amp(i, j) * zc[k] * ρᵣ[i, j, k]
            end
            fill_halo_regions!(ρθ)
        end

        # The corresponding standalone single columns, initialized identically.
        standalone = [single_column_model(RectilinearGrid(arch, FT; size=Nz, z=(0, 2000),
                                                          topology=(Flat, Flat, Bounded)))
                      for i in 1:N₁, j in 1:N₂]
        for j in 1:N₂, i in 1:N₁
            m = standalone[i, j]
            set!(m, θ = 290)
            zcm = Array(znodes(m.grid, Center()))
            @allowscalar begin
                ρθm = prognostic_fields(m)[name]
                ρᵣm = interior(m.dynamics.reference_state.density)
                im = interior(ρθm)
                for k in 1:Nz
                    im[1, 1, k] += amp(i, j) * zcm[k] * ρᵣm[1, 1, k]
                end
                fill_halo_regions!(ρθm)
            end
        end

        for n in 1:15
            time_step!(ensemble, 1)
            for m in standalone
                time_step!(m, 1)
            end
        end

        # Each ensemble column must match its standalone counterpart bit-for-bit: the columns
        # are genuinely independent (no horizontal coupling, no halo exchange between columns).
        for j in 1:N₂, i in 1:N₁
            ensemble_column = Array(interior(prognostic_fields(ensemble)[name]))[i, j, :]
            standalone_column = Array(interior(prognostic_fields(standalone[i, j])[name]))[1, 1, :]
            @test ensemble_column == standalone_column
        end

        @test maximum(abs, interior(ensemble.velocities.w)) == 0
    end

    @testset "Per-column closure array" begin
        N₁, N₂ = 3, 2
        κ(i, j) = FT(0.5) * (i + N₁ * (j - 1))  # distinct vertical diffusivity per column
        grid = RectilinearGrid(arch, FT; size = ColumnEnsembleSize(Nz=Nz, ensemble=(N₁, N₂), Hz=3),
                               z = (0, 2000), topology = (Flat, Flat, Bounded))
        rs = ReferenceState(grid, scm_constants; surface_pressure=101325, potential_temperature=290)
        closures = [VerticalScalarDiffusivity(ν=κ(i, j), κ=κ(i, j)) for i in 1:N₁, j in 1:N₂]
        ensemble = AtmosphereModel(grid; dynamics=AnelasticDynamics(rs), closure=closures)
        @test ensemble.closure isa AbstractArray

        name = thermodynamic_name(ensemble)
        set!(ensemble, θ = z -> 290 + FT(0.02) * z)  # same IC everywhere; only κ differs per column
        for n in 1:20
            time_step!(ensemble, 1)
        end

        # Each column matches a standalone single column with the same κ, bit-for-bit.
        for j in 1:N₂, i in 1:N₁
            m = single_column_model(RectilinearGrid(arch, FT; size=Nz, z=(0, 2000),
                                                    topology=(Flat, Flat, Bounded));
                                    closure = VerticalScalarDiffusivity(ν=κ(i, j), κ=κ(i, j)))
            set!(m, θ = z -> 290 + FT(0.02) * z)
            for n in 1:20
                time_step!(m, 1)
            end
            ensemble_column = Array(interior(prognostic_fields(ensemble)[name]))[i, j, :]
            standalone_column = Array(interior(prognostic_fields(m)[name]))[1, 1, :]
            @test ensemble_column == standalone_column
        end
    end

    @testset "Per-column Coriolis array" begin
        N₁, N₂ = 3, 2
        grid = RectilinearGrid(arch, FT; size = ColumnEnsembleSize(Nz=Nz, ensemble=(N₁, N₂), Hz=3),
                               z = (0, 2000), topology = (Flat, Flat, Bounded))
        rs = ReferenceState(grid, scm_constants; surface_pressure=101325, potential_temperature=290)
        planes = [FPlane(FT; f = FT(1e-4) * (i + N₁ * (j - 1))) for i in 1:N₁, j in 1:N₂]
        ensemble = AtmosphereModel(grid; dynamics=AnelasticDynamics(rs), coriolis=planes,
                                   closure=VerticalScalarDiffusivity(ν=1.0, κ=1.0))
        @test ensemble.coriolis isa AbstractArray

        fill!(parent(prognostic_fields(ensemble).ρu), 1)  # uniform ρu; Coriolis turns it into ρv per f
        set!(ensemble, θ = 290)
        for n in 1:20
            time_step!(ensemble, 10)
        end
        ρv = Array(interior(prognostic_fields(ensemble).ρv))[:, :, Nz ÷ 2]
        # Distinct f per column ⇒ distinct ρv per column.
        @test allunique(round.(vec(ρv); sigdigits=8))
    end

    @testset "Per-column forcing (discrete, array parameters)" begin
        N₁, N₂ = 3, 2
        grid = RectilinearGrid(arch, FT; size = ColumnEnsembleSize(Nz=Nz, ensemble=(N₁, N₂), Hz=3),
                               z = (0, 2000), topology = (Flat, Flat, Bounded))
        rs = ReferenceState(grid, scm_constants; surface_pressure=101325, potential_temperature=290)
        Q = on_architecture(arch, [FT(1e-3) * (i + N₁ * (j - 1)) for i in 1:N₁, j in 1:N₂])
        θ_forcing = Forcing(column_heating, discrete_form=true, parameters=Q)
        ensemble = AtmosphereModel(grid; dynamics=AnelasticDynamics(rs),
                                   closure=VerticalScalarDiffusivity(ν=1.0, κ=1.0), forcing=(; θ=θ_forcing))

        name = thermodynamic_name(ensemble)
        set!(ensemble, θ = 290)
        for n in 1:20
            time_step!(ensemble, 1)
        end
        ρθ_top = Array(interior(prognostic_fields(ensemble)[name]))[:, :, Nz]
        # Distinct heating per column ⇒ distinct warming per column.
        @test allunique(round.(vec(ρθ_top); sigdigits=8))
    end

    @testset "Heterogeneous per-column reference state" begin
        N₁, N₂ = 3, 2
        θ₀ = [FT(285 + 5 * (i - 1) + 2 * (j - 1)) for i in 1:N₁, j in 1:N₂]
        p₀ = [FT(101325 - 200 * (i - 1)) for i in 1:N₁, j in 1:N₂]
        grid = RectilinearGrid(arch, FT; size = ColumnEnsembleSize(Nz=Nz, ensemble=(N₁, N₂), Hz=3),
                               z = (0, 2000), topology = (Flat, Flat, Bounded))
        rs = ReferenceState(grid, scm_constants; surface_pressure=p₀, potential_temperature=θ₀)
        @test size(interior(rs.density)) == (N₁, N₂, Nz)

        # Each column's reference density matches a standalone reference built with that column's p₀, θ₀.
        for j in 1:N₂, i in 1:N₁
            g = RectilinearGrid(arch, FT; size=Nz, z=(0, 2000), topology=(Flat, Flat, Bounded))
            r = ReferenceState(g, scm_constants; surface_pressure=p₀[i, j], potential_temperature=θ₀[i, j])
            @test Array(interior(rs.density))[i, j, :] ≈ Array(interior(r.density))[1, 1, :]
        end

        # Array-valued parameters require a column-ensemble grid.
        single = RectilinearGrid(arch, FT; size=Nz, z=(0, 2000), topology=(Flat, Flat, Bounded))
        @test_throws ArgumentError ReferenceState(single, scm_constants; potential_temperature=fill(FT(290), 1, 1))
    end

    @testset "Three-dimensional model keeps the reduced reference profile" begin
        grid = RectilinearGrid(arch, FT; size=(4, 4, Nz), extent=(1000, 1000, 2000))
        reference_state = ReferenceState(grid, scm_constants; surface_pressure=101325, potential_temperature=290)
        # No regression: on a non-ensemble grid the reference profile stays reduced (1×1×Nz).
        @test size(interior(reference_state.density)) == (1, 1, Nz)
    end
end
