#####
##### validation/substepping/04_bomex.jl
#####
##### BOMEX shallow-cumulus intercomparison mirrored from examples/bomex.jl.
##### Reduced runtime: stop_time = 30 min (original 6h) to keep timing tractable.
#####

include("common.jl")

using Breeze
using Oceananigans
using Oceananigans.Units
using AtmosphericProfilesLibrary
using CUDA
using Printf
using Random
using JLD2

const CASE = "bomex"
const OUTDIR = joinpath(@__DIR__, "out", CASE)
isdir(OUTDIR) || mkpath(OUTDIR)

const arch = CUDA.functional() ? GPU() : CPU()
const Δt     = 10.0
const STOP_T = 30minutes  # original: 6hour
Oceananigans.defaults.FloatType = Float32

const Nx = Ny = 64
const Nz = 75
const x  = y = (0, 6400)
const z  = (0, 3000)

function build_grid()
    RectilinearGrid(arch; x, y, z, size = (Nx, Ny, Nz), halo = (5, 5, 5),
                    topology = (Periodic, Periodic, Bounded))
end

function _shared_setup(grid, constants, ρᵣ)
    FT = eltype(grid)
    p₀ = 101500
    θ₀ = 299.1
    q₀ = Breeze.Thermodynamics.MoistureMassFractions{FT} |> zero
    ρ₀ = Breeze.Thermodynamics.density(θ₀, p₀, q₀, constants)

    w′θ′  = 8e-3
    w′qᵗ′ = 5.2e-5

    ρθ_bcs  = FieldBoundaryConditions(bottom = FluxBoundaryCondition(ρ₀ * w′θ′))
    ρqᵉ_bcs = FieldBoundaryConditions(bottom = FluxBoundaryCondition(ρ₀ * w′qᵗ′))

    u★ = 0.28
    @inline ρu_drag(x, y, t, ρu, ρv, p) = - p.ρ₀ * p.u★^2 * ρu / sqrt(ρu^2 + ρv^2)
    @inline ρv_drag(x, y, t, ρu, ρv, p) = - p.ρ₀ * p.u★^2 * ρv / sqrt(ρu^2 + ρv^2)
    ρu_bcs = FieldBoundaryConditions(bottom = FluxBoundaryCondition(ρu_drag,
                     field_dependencies = (:ρu, :ρv), parameters = (; ρ₀, u★)))
    ρv_bcs = FieldBoundaryConditions(bottom = FluxBoundaryCondition(ρv_drag,
                     field_dependencies = (:ρu, :ρv), parameters = (; ρ₀, u★)))

    wˢ = Field{Nothing, Nothing, Face}(grid)
    wˢ_profile = AtmosphericProfilesLibrary.Bomex_subsidence(FT)
    set!(wˢ, z -> wˢ_profile(z))
    subsidence = SubsidenceForcing(wˢ)

    coriolis = FPlane(f = 3.76e-5)
    uᵍ = AtmosphericProfilesLibrary.Bomex_geostrophic_u(FT)
    vᵍ = AtmosphericProfilesLibrary.Bomex_geostrophic_v(FT)
    geostrophic = geostrophic_forcings(z -> uᵍ(z), z -> vᵍ(z))

    drying = Field{Nothing, Nothing, Center}(grid)
    dqdt_profile = AtmosphericProfilesLibrary.Bomex_dqtdt(FT)
    set!(drying, z -> dqdt_profile(z))
    set!(drying, ρᵣ * drying)
    ρqᵉ_drying_forcing = Forcing(drying)

    Fρe_field = Field{Nothing, Nothing, Center}(grid)
    cᵖᵈ = constants.dry_air.heat_capacity
    dTdt_bomex = AtmosphericProfilesLibrary.Bomex_dTdt(FT)
    set!(Fρe_field, z -> dTdt_bomex(1, z))
    set!(Fρe_field, ρᵣ * cᵖᵈ * Fρe_field)
    ρe_radiation_forcing = Forcing(Fρe_field)

    forcing = (; ρu = (subsidence, geostrophic.ρu),
                 ρv = (subsidence, geostrophic.ρv),
                 ρθ = subsidence,
                 ρe = ρe_radiation_forcing,
                 ρqᵉ = (subsidence, ρqᵉ_drying_forcing))

    bcs = (; ρθ = ρθ_bcs, ρqᵉ = ρqᵉ_bcs, ρu = ρu_bcs, ρv = ρv_bcs)

    return (; forcing, bcs, coriolis)
end

function build_anelastic(grid)
    Random.seed!(938); CUDA.functional() && CUDA.seed!(938)
    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants,
                                     surface_pressure = 101500,
                                     potential_temperature = 299.1)
    dynamics = AnelasticDynamics(reference_state)
    S = _shared_setup(grid, constants, reference_state.density)
    microphysics = SaturationAdjustment(equilibrium = WarmPhaseEquilibrium())
    AtmosphereModel(grid; dynamics, coriolis = S.coriolis, microphysics,
                    advection = WENO(order=9), forcing = S.forcing,
                    boundary_conditions = S.bcs)
end

function build_compressible(grid; damping = PressureProjectionDamping(coefficient = 0.1))
    Random.seed!(938); CUDA.functional() && CUDA.seed!(938)
    constants = ThermodynamicConstants()
    td = SplitExplicitTimeDiscretization(; damping)
    dynamics = CompressibleDynamics(td;
                                    surface_pressure = 101500,
                                    reference_potential_temperature = 299.1)
    # Need reference_state built already to scale forcing fields — build once to extract.
    tmp = CompressibleDynamics(td; surface_pressure = 101500,
                               reference_potential_temperature = 299.1)
    ref = Breeze.CompressibleEquations.CompressibleDynamics # placeholder for the type
    # Use Breeze helper to materialize a ref density on this grid:
    ref_state = Breeze.Thermodynamics.ExnerReferenceState(grid, constants;
                 surface_pressure = 101500, potential_temperature = 299.1)
    S = _shared_setup(grid, constants, ref_state.density)
    microphysics = SaturationAdjustment(equilibrium = WarmPhaseEquilibrium())
    AtmosphereModel(grid; dynamics, coriolis = S.coriolis, microphysics,
                    advection = WENO(order=9), forcing = S.forcing,
                    boundary_conditions = S.bcs,
                    timestepper = :AcousticRungeKutta3)
end

function _ic!(model)
    FT = eltype(model.grid)
    θˡⁱ₀ = AtmosphericProfilesLibrary.Bomex_θ_liq_ice(FT)
    qᵗ₀  = AtmosphericProfilesLibrary.Bomex_q_tot(FT)
    u₀   = AtmosphericProfilesLibrary.Bomex_u(FT)

    δθ = 0.1f0
    δqᵗ = 2.5f-5
    zδ = 1600
    ϵ() = rand() - 0.5
    θᵢ(x, y, z) = θˡⁱ₀(z) + δθ * ϵ() * (z < zδ)
    qᵢ(x, y, z) = qᵗ₀(z) + δqᵗ * ϵ() * (z < zδ)
    uᵢ(x, y, z) = u₀(z)
    if model.dynamics isa Breeze.CompressibleEquations.CompressibleDynamics
        ref = model.dynamics.reference_state
        set!(model; θ = θᵢ, qᵗ = qᵢ, u = uᵢ, ρ = ref.density)
    else
        set!(model; θ = θᵢ, qᵗ = qᵢ, u = uᵢ)
    end
    return model
end

function run_case(label, builder)
    grid = build_grid()
    model = builder(grid)
    _ic!(model)
    sim = Simulation(model; Δt, stop_time = STOP_T, verbose = false)
    outputs = (; w = model.velocities.w, T = model.temperature)
    sim.output_writers[:jld2] = JLD2Writer(model, outputs;
                                           filename = joinpath(OUTDIR, "$(label).jld2"),
                                           schedule = TimeInterval(2minutes),
                                           overwrite_existing = true)
    res = timed_run!(sim; label)
    return summarize_result(label, res, model)
end

@info "[$CASE] Anelastic run…"
a = run_case("anelastic", build_anelastic)
@info "[$CASE] Compressible run…"
c = run_case("compressible", build_compressible)

wa = try; FieldTimeSeries(joinpath(OUTDIR, "anelastic.jld2"), "w")[end]; catch; nothing; end
wc = try; FieldTimeSeries(joinpath(OUTDIR, "compressible.jld2"), "w")[end]; catch; nothing; end
try
    if wa !== nothing && wc !== nothing
        Nz_ = size(wa)[3]; k_mid = div(Nz_, 2)
        wa_arr = Array(interior(wa, :, :, k_mid))
        wc_arr = Array(interior(wc, :, :, k_mid))
        any(!isfinite, wa_arr) && (wa_arr[.!isfinite.(wa_arr)] .= 0)
        any(!isfinite, wc_arr) && (wc_arr[.!isfinite.(wc_arr)] .= 0)
        fig = Figure(size = (1200, 500))
        ax1 = Axis(fig[1, 1]; title = "anelastic w (z-slice)")
        ax2 = Axis(fig[1, 2]; title = "compressible w (z-slice)")
        vmax = max(maximum(abs, wa_arr), maximum(abs, wc_arr))
        vmax = isfinite(vmax) && vmax > 0 ? vmax : 1
        hm1 = heatmap!(ax1, wa_arr; colormap = :balance, colorrange = (-vmax, vmax))
        heatmap!(ax2, wc_arr; colormap = :balance, colorrange = (-vmax, vmax))
        Colorbar(fig[1, 3], hm1; label = "w (m/s)")
        save(joinpath(OUTDIR, "summary.png"), fig)
    end
catch e
    @warn "plot failed" exception=e
end

jldsave(joinpath(OUTDIR, "result.jld2"); anelastic = a, compressible = c, case = CASE, Δt, stop_time = STOP_T)
io = IOBuffer()
report_case(io, CASE,
            "BOMEX shallow cumulus, 64×64×75, Δt=$(Δt)s, stop=$(Int(STOP_T))s (shortened from 6h), GPU, Float32, WENO(9), SatAdj, surface fluxes + subsidence + geostrophic + radiation.",
            a, c)
write(joinpath(OUTDIR, "report.md"), take!(io))
@info "[$CASE] done"
