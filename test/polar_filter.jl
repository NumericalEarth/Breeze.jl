using Breeze
using Breeze.PolarFilters: materialize_polar_filter, apply_polar_filter_field!, apply_polar_filter_intensive!
using Oceananigans
using Oceananigans.Grids: φnodes
using Test

# Note: When run through the test runner, test_float_types and default_arch
# are defined in the init_code. When run directly, we need fallbacks.
if !@isdefined(test_float_types)
    test_float_types() = (Float64,)
end
if !@isdefined(default_arch)
    const default_arch = CPU()
end

function build_polar_filter_test_grid(arch; FT=Float64, Nx=36, Ny=34, Nz=4)
    Oceananigans.defaults.FloatType = FT
    LatitudeLongitudeGrid(arch;
                          size = (Nx, Ny, Nz),
                          halo = (3, 3, 3),
                          longitude = (0, 360),
                          latitude = (-85, 85),
                          z = (0, 30000))
end

#####
##### Constructor and materialization tests
#####

@testset "PolarFilter constructor [$(FT)]" for FT in test_float_types()
    grid = build_polar_filter_test_grid(CPU(); FT)
    φ = φnodes(grid, Center())

    pf_skeleton = PolarFilter(threshold_latitude=60)
    @test pf_skeleton.threshold_latitude == 60.0
    @test pf_skeleton.filtered_indices === nothing
    @test pf_skeleton.passes_per_row === nothing

    pf = materialize_polar_filter(grid, pf_skeleton)

    ## filtered_indices should cover both hemispheres
    @test length(pf.filtered_indices) > 0
    @test all(abs(φ[j]) > 60 for j in pf.filtered_indices)

    ## Both hemispheres represented
    @test any(φ[j] < 0 for j in pf.filtered_indices)
    @test any(φ[j] > 0 for j in pf.filtered_indices)

    ## Symmetric: same number of filtered rows per hemisphere
    n_south = count(φ[j] < 0 for j in pf.filtered_indices)
    n_north = count(φ[j] > 0 for j in pf.filtered_indices)
    @test n_south == n_north

    ## passes_per_row has the right length
    @test length(pf.passes_per_row) == length(pf.filtered_indices)

    ## All passes are non-negative
    @test all(pf.passes_per_row .>= 0)

    ## threshold_latitude stored correctly
    @test pf.threshold_latitude == FT(60)

    ## Grid is stored
    @test pf.grid === grid
end

@testset "materialize_polar_filter returns nothing for no filter" begin
    grid = build_polar_filter_test_grid(CPU())
    @test materialize_polar_filter(grid, nothing) === nothing
end

#####
##### Smoother tests: low wavenumber preserved
#####

@testset "Low wavenumber preserved [$(FT)]" for FT in test_float_types()
    grid = build_polar_filter_test_grid(CPU(); FT)
    pf = materialize_polar_filter(grid, PolarFilter(threshold_latitude=60))

    ## A k=1 zonal wave should be nearly untouched
    field = CenterField(grid)
    Nλ = grid.Nx
    λ_nodes = 2π * (0:Nλ-1) / Nλ
    for k in 1:grid.Nz, j in 1:grid.Ny, i in 1:Nλ
        field[i, j, k] = FT(cos(λ_nodes[i]))
    end

    field_before = deepcopy(interior(field))
    apply_polar_filter_field!(pf, field)

    ## k=1 wave should be preserved: relative error < 5%
    max_change = maximum(abs, interior(field) .- field_before)
    max_val = maximum(abs, field_before)
    @test max_change / max_val < 0.05
end

#####
##### Smoother tests: high wavenumber removed at high latitudes
#####

@testset "High wavenumber removed at high latitudes [$(FT)]" for FT in test_float_types()
    grid = build_polar_filter_test_grid(CPU(); FT)
    pf = materialize_polar_filter(grid, PolarFilter(threshold_latitude=60))

    ## A high-k wave at filtered latitudes should be heavily damped
    field = CenterField(grid)
    Nλ = grid.Nx
    k_high = Nλ ÷ 2  # Nyquist-like
    λ_nodes = 2π * (0:Nλ-1) / Nλ
    for k in 1:grid.Nz, j in 1:grid.Ny, i in 1:Nλ
        field[i, j, k] = FT(cos(k_high * λ_nodes[i]))
    end

    field_before = deepcopy(interior(field))
    apply_polar_filter_field!(pf, field)

    ## At rows with nonzero passes, the high-k wave should be significantly damped
    φ = φnodes(grid, Center())
    for j_local in 1:length(pf.filtered_indices)
        pf.passes_per_row[j_local] == 0 && continue
        j = pf.filtered_indices[j_local]
        amplitude_before = maximum(abs, field_before[:, j, 1])
        amplitude_after = maximum(abs, interior(field)[:, j, 1])
        @test amplitude_after < 0.5 * amplitude_before
    end
end

#####
##### Equatorial latitudes unchanged
#####

@testset "Equatorial latitudes unchanged [$(FT)]" for FT in test_float_types()
    grid = build_polar_filter_test_grid(CPU(); FT)
    pf = materialize_polar_filter(grid, PolarFilter(threshold_latitude=60))

    ## Fill with a high-k wave
    field = CenterField(grid)
    Nλ = grid.Nx
    k_high = Nλ ÷ 2
    λ_nodes = 2π * (0:Nλ-1) / Nλ
    for k in 1:grid.Nz, j in 1:grid.Ny, i in 1:Nλ
        field[i, j, k] = FT(cos(k_high * λ_nodes[i]))
    end

    field_before = deepcopy(interior(field))
    apply_polar_filter_field!(pf, field)

    ## Equatorial rows (below threshold) should be completely unchanged
    φ = φnodes(grid, Center())
    filtered_set = Set(pf.filtered_indices)
    for j in 1:grid.Ny
        if !(j in filtered_set)
            @test interior(field)[:, j, :] == field_before[:, j, :]
        end
    end
end

#####
##### Architecture test (CPU or GPU)
#####

@testset "Low wavenumber preserved on $(default_arch) [$(FT)]" for FT in test_float_types()
    grid = build_polar_filter_test_grid(default_arch; FT)
    pf = materialize_polar_filter(grid, PolarFilter(threshold_latitude=60))

    field = CenterField(grid)
    Nλ = grid.Nx
    λ_nodes = 2π * (0:Nλ-1) / Nλ
    for k in 1:grid.Nz, j in 1:grid.Ny, i in 1:Nλ
        field[i, j, k] = FT(cos(λ_nodes[i]))
    end

    field_before = Array(interior(field))
    apply_polar_filter_field!(pf, field)

    max_change = maximum(abs, Array(interior(field)) .- field_before)
    max_val = maximum(abs, field_before)
    @test max_change / max_val < 0.05
end

@testset "High wavenumber removed on $(default_arch) [$(FT)]" for FT in test_float_types()
    grid = build_polar_filter_test_grid(default_arch; FT)
    pf = materialize_polar_filter(grid, PolarFilter(threshold_latitude=60))

    field = CenterField(grid)
    Nλ = grid.Nx
    k_high = Nλ ÷ 2
    λ_nodes = 2π * (0:Nλ-1) / Nλ
    for k in 1:grid.Nz, j in 1:grid.Ny, i in 1:Nλ
        field[i, j, k] = FT(cos(k_high * λ_nodes[i]))
    end

    field_before = Array(interior(field))
    apply_polar_filter_field!(pf, field)

    φ = φnodes(grid, Center())
    passes = Array(pf.passes_per_row)
    for j_local in 1:length(pf.filtered_indices)
        passes[j_local] == 0 && continue
        j = pf.filtered_indices[j_local]
        amplitude_before = maximum(abs, field_before[:, j, 1])
        amplitude_after = maximum(abs, Array(interior(field))[:, j, 1])
        @test amplitude_after < 0.5 * amplitude_before
    end
end

#####
##### Integration test: full model run
#####

@testset "PolarFilter integration [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = build_polar_filter_test_grid(default_arch; FT, Nx=36, Ny=34, Nz=4)

    pf = PolarFilter(threshold_latitude=60)
    dynamics = CompressibleDynamics(ExplicitTimeStepping();
                                    surface_pressure = 100000,
                                    reference_potential_temperature = 300,
                                    polar_filter = pf)
    coriolis = HydrostaticSphericalCoriolis()
    model = AtmosphereModel(grid; dynamics, coriolis, advection=WENO())
    set!(model; θ=300, ρ=1.2)

    @test model.dynamics.polar_filter isa PolarFilter

    simulation = Simulation(model; Δt=0.1, stop_iteration=3, verbose=false)
    run!(simulation)

    @test model.clock.iteration == 3
    @test !any(isnan, parent(model.dynamics.density))
    @test !any(isnan, parent(model.momentum.ρu))
    @test !any(isnan, parent(model.momentum.ρv))
end
