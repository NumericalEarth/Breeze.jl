#####
##### validation/substepping/06_neutral_abl.jl
#####
##### Moeng & Sullivan 1994 shear-driven neutral ABL. Reduced to 10 min (original 5h).
#####

include("common.jl")

using Breeze
using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids: znode
using CUDA
using Random
using JLD2

const CASE = "neutral_abl"
const OUTDIR = joinpath(@__DIR__, "out", CASE)
isdir(OUTDIR) || mkpath(OUTDIR)

const arch = CUDA.functional() ? GPU() : CPU()
const Δt     = 0.5
const STOP_T = 10minutes
Oceananigans.defaults.FloatType = Float32

const Nx = Ny = Nz = 96
const x  = y = (0, 3000)
const z  = (0, 1000)

const p₀ = 1e5
const θ₀ = 300
const uᵍ = 15
const vᵍ = 0

function build_grid()
    RectilinearGrid(arch; x, y, z, size = (Nx, Ny, Nz), halo = (5, 5, 5),
                    topology = (Periodic, Periodic, Bounded))
end

function θᵣ_factory(grid)
    Δz  = first(zspacings(grid))
    zᵢ₁ = 468
    zᵢ₂ = zᵢ₁ + 6Δz
    Γᵢ  = 8 / 6Δz
    Γᵗᵒᵖ = 0.003
    (z) -> z < zᵢ₁ ? θ₀ :
           z < zᵢ₂ ? θ₀ + Γᵢ * (z - zᵢ₁) :
           θ₀ + Γᵢ * (zᵢ₂ - zᵢ₁) + Γᵗᵒᵖ * (z - zᵢ₂)
end

function _shared(grid, ρ_for_sponge)
    FT = eltype(grid)
    q₀ = Breeze.Thermodynamics.MoistureMassFractions{FT} |> zero
    ρ₀ = Breeze.Thermodynamics.density(θ₀, p₀, q₀, ThermodynamicConstants())
    u★ = 0.5

    @inline ρu_drag(x, y, t, ρu, ρv, param) = - param.ρ₀ * param.u★^2 * ρu / max(sqrt(ρu^2 + ρv^2), 1e-6)
    @inline ρv_drag(x, y, t, ρu, ρv, param) = - param.ρ₀ * param.u★^2 * ρv / max(sqrt(ρu^2 + ρv^2), 1e-6)
    ρu_bcs = FieldBoundaryConditions(bottom = FluxBoundaryCondition(ρu_drag, field_dependencies = (:ρu, :ρv), parameters = (; ρ₀, u★)))
    ρv_bcs = FieldBoundaryConditions(bottom = FluxBoundaryCondition(ρv_drag, field_dependencies = (:ρu, :ρv), parameters = (; ρ₀, u★)))

    sponge_mask = GaussianMask{:z}(center = last(z), width = 200)
    θᵣ = θᵣ_factory(grid)
    ρθᵣ = Field{Nothing, Nothing, Center}(grid)
    set!(ρθᵣ, (z_val) -> θᵣ(z_val))
    set!(ρθᵣ, ρ_for_sponge * ρθᵣ)
    ρθᵣ_data = interior(ρθᵣ, 1, 1, :)

    @inline function ρθ_sponge_fun(i, j, k, grid, clock, model_fields, p)
        zₖ = znode(k, grid, Center())
        return @inbounds p.rate * p.mask(0, 0, zₖ) * (p.target[k] - model_fields.ρθ[i, j, k])
    end
    ρθ_sponge = Forcing(ρθ_sponge_fun; discrete_form = true,
                        parameters = (rate = 0.01, mask = sponge_mask, target = ρθᵣ_data))
    ρw_sponge = Relaxation(rate = 0.01, mask = sponge_mask)

    coriolis = FPlane(f = 1e-4)
    geostrophic = geostrophic_forcings(uᵍ, vᵍ)

    forcing = (ρu = geostrophic.ρu, ρv = geostrophic.ρv, ρw = ρw_sponge, ρθ = ρθ_sponge)
    bcs = (ρu = ρu_bcs, ρv = ρv_bcs)
    return (; forcing, bcs, coriolis, θᵣ)
end

function build_anelastic(grid)
    Random.seed!(1994); CUDA.functional() && CUDA.seed!(1994)
    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants; surface_pressure = p₀, potential_temperature = θ₀)
    dynamics = AnelasticDynamics(reference_state)
    S = _shared(grid, reference_state.density)
    AtmosphereModel(grid; dynamics, coriolis = S.coriolis, advection = WENO(order = 9),
                    forcing = S.forcing, closure = SmagorinskyLilly(),
                    boundary_conditions = S.bcs), S.θᵣ
end

function build_compressible(grid; damping = PressureProjectionDamping(coefficient = 0.1))
    Random.seed!(1994); CUDA.functional() && CUDA.seed!(1994)
    constants = ThermodynamicConstants()
    td = SplitExplicitTimeDiscretization(; damping)
    dynamics = CompressibleDynamics(td; surface_pressure = p₀, reference_potential_temperature = θ₀)
    ref_state = Breeze.Thermodynamics.ExnerReferenceState(grid, constants;
                 surface_pressure = p₀, potential_temperature = θ₀)
    S = _shared(grid, ref_state.density)
    AtmosphereModel(grid; dynamics, coriolis = S.coriolis, advection = WENO(order = 9),
                    forcing = S.forcing, closure = SmagorinskyLilly(),
                    boundary_conditions = S.bcs,
                    timestepper = :AcousticRungeKutta3), S.θᵣ
end

function _ic!(model, θᵣ)
    δu = δv = 0.01; δθ = 0.1; zδ = 400
    ϵ() = rand() - 0.5
    uᵢ(x, y, z) = uᵍ + δu * ϵ() * (z < zδ)
    vᵢ(x, y, z) = vᵍ + δv * ϵ() * (z < zδ)
    θᵢ(x, y, z) = θᵣ(z) + δθ * ϵ() * (z < zδ)
    if model.dynamics isa Breeze.CompressibleEquations.CompressibleDynamics
        ref = model.dynamics.reference_state
        set!(model; θ = θᵢ, u = uᵢ, v = vᵢ, ρ = ref.density)
    else
        set!(model; θ = θᵢ, u = uᵢ, v = vᵢ)
    end
end

function run_case(label, builder)
    grid = build_grid()
    model, θᵣ = builder(grid)
    _ic!(model, θᵣ)
    sim = Simulation(model; Δt, stop_time = STOP_T, verbose = false)
    outputs = (; w = model.velocities.w)
    sim.output_writers[:jld2] = JLD2Writer(model, outputs;
                                           filename = joinpath(OUTDIR, "$(label).jld2"),
                                           schedule = TimeInterval(1minute),
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
        wa_a = Array(interior(wa, :, :, k_mid)); wc_a = Array(interior(wc, :, :, k_mid))
        any(!isfinite, wa_a) && (wa_a[.!isfinite.(wa_a)] .= 0)
        any(!isfinite, wc_a) && (wc_a[.!isfinite.(wc_a)] .= 0)
        fig = Figure(size = (1200, 500))
        ax1 = Axis(fig[1, 1]; title = "anelastic w (z-slice)")
        ax2 = Axis(fig[1, 2]; title = "compressible w (z-slice)")
        vmax = max(maximum(abs, wa_a), maximum(abs, wc_a)); vmax = isfinite(vmax) && vmax > 0 ? vmax : 1
        hm1 = heatmap!(ax1, wa_a; colormap = :balance, colorrange = (-vmax, vmax))
        heatmap!(ax2, wc_a; colormap = :balance, colorrange = (-vmax, vmax))
        Colorbar(fig[1, 3], hm1; label = "w (m/s)")
        save(joinpath(OUTDIR, "summary.png"), fig)
    end
catch e
    @warn "plot failed" exception = e
end

jldsave(joinpath(OUTDIR, "result.jld2"); anelastic = a, compressible = c, case = CASE, Δt, stop_time = STOP_T)
io = IOBuffer()
report_case(io, CASE,
            "Moeng-Sullivan shear-driven neutral ABL, 96³, Δt=$(Δt)s, stop=$(Int(STOP_T))s (shortened from 5h), GPU, Float32, WENO(9), SmagorinskyLilly, capping inversion + geostrophic.",
            a, c)
write(joinpath(OUTDIR, "report.md"), take!(io))
@info "[$CASE] done"
