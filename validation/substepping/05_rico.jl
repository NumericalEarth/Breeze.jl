#####
##### validation/substepping/05_rico.jl
#####
##### RICO precipitating shallow cumulus. Reduced to 20 min (original 8h).
#####

include("common.jl")

using Breeze
using Breeze: BulkDrag, BulkSensibleHeatFlux, BulkVaporFlux
using Oceananigans
using Oceananigans.Units
using AtmosphericProfilesLibrary
using CloudMicrophysics
using CUDA
using Random
using JLD2

const CASE = "rico"
const OUTDIR = joinpath(@__DIR__, "out", CASE)
isdir(OUTDIR) || mkpath(OUTDIR)

const arch = CUDA.functional() ? GPU() : CPU()
const Δt     = 2.0
const STOP_T = 20minutes
Oceananigans.defaults.FloatType = Float32

const Nx = Ny = 128
const Nz    = 100
const x     = y = (0, 12800)
const z     = (0, 4000)

function build_grid()
    RectilinearGrid(arch; x, y, z, size = (Nx, Ny, Nz), halo = (5, 5, 5),
                    topology = (Periodic, Periodic, Bounded))
end

const Cᴰ = 1.229e-3
const Cᵀ = 1.094e-3
const Cᵛ = 1.133e-3
const T₀ = 299.8

function _shared(grid, ρᵣ)
    FT = eltype(grid)
    ρe_bcs  = FieldBoundaryConditions(bottom = BulkSensibleHeatFlux(coefficient = Cᵀ, surface_temperature = T₀))
    ρqᵉ_bcs = FieldBoundaryConditions(bottom = BulkVaporFlux(coefficient = Cᵛ, surface_temperature = T₀))
    ρu_bcs  = FieldBoundaryConditions(bottom = BulkDrag(coefficient = Cᴰ))
    ρv_bcs  = FieldBoundaryConditions(bottom = BulkDrag(coefficient = Cᴰ))

    wˢ = Field{Nothing, Nothing, Face}(grid)
    set!(wˢ, z -> AtmosphericProfilesLibrary.Rico_subsidence(FT)(z))
    subsidence = SubsidenceForcing(wˢ)

    coriolis = FPlane(f = 4.5e-5)
    geostrophic = geostrophic_forcings(z -> AtmosphericProfilesLibrary.Rico_geostrophic_ug(FT)(z),
                                       z -> AtmosphericProfilesLibrary.Rico_geostrophic_vg(FT)(z))

    ∂t_ρqᵉ = Field{Nothing, Nothing, Center}(grid)
    set!(∂t_ρqᵉ, z -> AtmosphericProfilesLibrary.Rico_dqtdt(FT)(z))
    set!(∂t_ρqᵉ, ρᵣ * ∂t_ρqᵉ)
    dq_forcing = Forcing(∂t_ρqᵉ)

    ∂t_ρθ = Field{Nothing, Nothing, Center}(grid)
    set!(∂t_ρθ, ρᵣ * (- 2.5 / day))
    rad_forcing = Forcing(∂t_ρθ)

    sponge = Relaxation(rate = 1/8, mask = GaussianMask{:z}(center = 3500, width = 500))

    forcing = (ρu = (subsidence, geostrophic.ρu),
               ρv = (subsidence, geostrophic.ρv),
               ρw = sponge,
               ρqᵉ = (subsidence, dq_forcing),
               ρθ = (subsidence, rad_forcing))

    bcs = (ρe = ρe_bcs, ρqᵉ = ρqᵉ_bcs, ρu = ρu_bcs, ρv = ρv_bcs)
    return (; forcing, bcs, coriolis)
end

function build_micro()
    ext = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
    ext.OneMomentCloudMicrophysics(; cloud_formation = SaturationAdjustment(equilibrium = WarmPhaseEquilibrium()))
end

function build_anelastic(grid)
    Random.seed!(42); CUDA.functional() && CUDA.seed!(42)
    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants,
                                     surface_pressure = 101540, potential_temperature = 297.9)
    dynamics = AnelasticDynamics(reference_state)
    S = _shared(grid, reference_state.density)
    weno = WENO(order = 5)
    bp_weno = WENO(order = 5, bounds = (0, 1))
    AtmosphereModel(grid; dynamics, coriolis = S.coriolis, microphysics = build_micro(),
                    momentum_advection = weno,
                    scalar_advection = (ρθ = weno, ρqᵉ = bp_weno, ρqᶜˡ = bp_weno, ρqʳ = bp_weno),
                    forcing = S.forcing, boundary_conditions = S.bcs)
end

function build_compressible(grid; damping = PressureProjectionDamping(coefficient = 0.1))
    Random.seed!(42); CUDA.functional() && CUDA.seed!(42)
    constants = ThermodynamicConstants()
    td = SplitExplicitTimeDiscretization(; damping)
    dynamics = CompressibleDynamics(td;
                                    surface_pressure = 101540,
                                    reference_potential_temperature = 297.9)
    # Pre-build reference density for forcing field scaling.
    ref_state = Breeze.Thermodynamics.ExnerReferenceState(grid, constants;
                 surface_pressure = 101540, potential_temperature = 297.9)
    S = _shared(grid, ref_state.density)
    weno = WENO(order = 5)
    bp_weno = WENO(order = 5, bounds = (0, 1))
    AtmosphereModel(grid; dynamics, coriolis = S.coriolis, microphysics = build_micro(),
                    momentum_advection = weno,
                    scalar_advection = (ρθ = weno, ρqᵉ = bp_weno, ρqᶜˡ = bp_weno, ρqʳ = bp_weno),
                    forcing = S.forcing, boundary_conditions = S.bcs,
                    timestepper = :AcousticRungeKutta3)
end

function _ic!(model)
    FT = eltype(model.grid)
    θˡⁱ₀_p = AtmosphericProfilesLibrary.Rico_θ_liq_ice(FT)
    qᵗ₀_p  = AtmosphericProfilesLibrary.Rico_q_tot(FT)
    u₀_p   = AtmosphericProfilesLibrary.Rico_u(FT)
    v₀_p   = AtmosphericProfilesLibrary.Rico_v(FT)
    zϵ = 1500
    θᵢ(x, y, z) = θˡⁱ₀_p(z) + 1e-2 * (rand() - 0.5) * (z < zϵ)
    qᵢ(x, y, z) = qᵗ₀_p(z)
    uᵢ(x, y, z) = u₀_p(z)
    vᵢ(x, y, z) = v₀_p(z)
    if model.dynamics isa Breeze.CompressibleEquations.CompressibleDynamics
        ref = model.dynamics.reference_state
        set!(model; θ = θᵢ, qᵗ = qᵢ, u = uᵢ, v = vᵢ, ρ = ref.density)
    else
        set!(model; θ = θᵢ, qᵗ = qᵢ, u = uᵢ, v = vᵢ)
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
        Nz_ = size(wa)[3]; k_mid = div(Nz_, 4)  # near cloud base
        wa_arr = Array(interior(wa, :, :, k_mid))
        wc_arr = Array(interior(wc, :, :, k_mid))
        any(!isfinite, wa_arr) && (wa_arr[.!isfinite.(wa_arr)] .= 0)
        any(!isfinite, wc_arr) && (wc_arr[.!isfinite.(wc_arr)] .= 0)
        fig = Figure(size = (1200, 500))
        ax1 = Axis(fig[1, 1]; title = "anelastic w (k=$k_mid)")
        ax2 = Axis(fig[1, 2]; title = "compressible w (k=$k_mid)")
        vmax = max(maximum(abs, wa_arr), maximum(abs, wc_arr)); vmax = isfinite(vmax) && vmax > 0 ? vmax : 1
        hm1 = heatmap!(ax1, wa_arr; colormap = :balance, colorrange = (-vmax, vmax))
        heatmap!(ax2, wc_arr; colormap = :balance, colorrange = (-vmax, vmax))
        Colorbar(fig[1, 3], hm1; label = "w (m/s)")
        save(joinpath(OUTDIR, "summary.png"), fig)
    end
catch e
    @warn "plot failed" exception = e
end

jldsave(joinpath(OUTDIR, "result.jld2"); anelastic = a, compressible = c, case = CASE, Δt, stop_time = STOP_T)
io = IOBuffer()
report_case(io, CASE,
            "RICO precipitating shallow cumulus, 128×128×100, Δt=$(Δt)s, stop=$(Int(STOP_T))s (shortened from 8h), GPU, Float32, WENO(5), OneMomentCloudMicrophysics, surface fluxes, subsidence, geostrophic, radiation, sponge.",
            a, c)
write(joinpath(OUTDIR, "report.md"), take!(io))
@info "[$CASE] done"
