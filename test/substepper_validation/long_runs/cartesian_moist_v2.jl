#####
##### Cartesian moist anel-vs-comp v2 — based on validated 05_prescribed_sst.
#####
##### Differences from 05_prescribed_sst:
#####   - microphysics: SaturationAdjustment → NonEquilibriumCloudFormation
#####     (τ_relax=200s) + OneMomentCloudMicrophysics (the moist BW physics)
#####   - longer integration (1 hour) to give convection time to develop
#####   - adaptive Δt with cfl=0.7 to find the binding constraint for each
#####
##### If anelastic completes 1h cleanly with this microphysics path and
##### compressible-substepper crashes earlier, the substepper is still the
##### binding factor for moist physics. If both crash, microphysics path
##### is the issue.
#####

using Oceananigans
using Oceananigans.Units
using Printf
using CUDA
using JLD2
using CloudMicrophysics
using Breeze
using Breeze: BulkDrag, BulkSensibleHeatFlux, BulkVaporFlux

const arch = CUDA.functional() ? GPU() : CPU()
const OUTDIR = @__DIR__
const RUN_LABEL = "cartesian_moist_v2"

Oceananigans.defaults.FloatType = Float32

const constants = ThermodynamicConstants()
const p₀ = 101325.0
const θ₀ = 285.0
const ΔT = 4.0
const Uᵍ = 1e-2

# Same SST front as 05_prescribed_sst
T₀_func(x) = θ₀ + ΔT / 2 * sign(cos(2π * x / 20kilometers))

function build_grid()
    RectilinearGrid(arch; size = (128, 128), halo = (5, 5),
                    x = (-10kilometers, 10kilometers),
                    z = (0, 10kilometers),
                    topology = (Periodic, Flat, Bounded))
end

function make_bcs()
    coef = 1e-3
    ρu_flux  = BulkDrag(coefficient=coef; gustiness=Uᵍ, surface_temperature=T₀_func)
    ρθ_flux  = BulkSensibleHeatFlux(coefficient=coef; gustiness=Uᵍ, surface_temperature=T₀_func)
    ρqᵛ_flux = BulkVaporFlux(coefficient=coef; gustiness=Uᵍ, surface_temperature=T₀_func)
    return (
        ρu  = FieldBoundaryConditions(bottom = ρu_flux),
        ρθ  = FieldBoundaryConditions(bottom = ρθ_flux),
        ρqᵛ = FieldBoundaryConditions(bottom = ρqᵛ_flux),
    )
end

# NonEquilibriumCloudFormation microphysics matching the moist BW
function build_microphysics()
    BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
    τ_relax = 200.0
    relaxation = ConstantRateCondensateFormation(1/τ_relax)
    cloud_formation = NonEquilibriumCloudFormation(relaxation, relaxation)
    return BreezeCloudMicrophysicsExt.OneMomentCloudMicrophysics(; cloud_formation)
end

function build_anelastic()
    grid = build_grid()
    reference_state = ReferenceState(grid, constants;
                                     surface_pressure = p₀,
                                     potential_temperature = θ₀)
    dyn = AnelasticDynamics(reference_state)

    weno = WENO(order = 5)
    bp_weno = WENO(order = 5, bounds = (0, 1))

    bcs = make_bcs()
    model = AtmosphereModel(grid;
                            momentum_advection = weno,
                            scalar_advection   = (ρθ = weno, ρqᵛ = bp_weno,
                                                  ρqᶜˡ = bp_weno, ρqᶜⁱ = bp_weno,
                                                  ρqʳ = bp_weno, ρqˢ = bp_weno),
                            microphysics = build_microphysics(),
                            thermodynamic_constants = constants,
                            dynamics = dyn,
                            boundary_conditions = bcs)
    set!(model; θ = θ₀, u = 1.0, qᵛ = 0.005)   # 5 g/kg uniform initial vapor
    return model
end

function build_compressible()
    grid = build_grid()
    td = SplitExplicitTimeDiscretization()
    dyn = CompressibleDynamics(td;
                               surface_pressure = p₀,
                               reference_potential_temperature = θ₀)
    weno = WENO(order = 5)
    bp_weno = WENO(order = 5, bounds = (0, 1))
    bcs = make_bcs()
    model = AtmosphereModel(grid;
                            momentum_advection = weno,
                            scalar_advection   = (ρθ = weno, ρqᵛ = bp_weno,
                                                  ρqᶜˡ = bp_weno, ρqᶜⁱ = bp_weno,
                                                  ρqʳ = bp_weno, ρqˢ = bp_weno),
                            microphysics = build_microphysics(),
                            thermodynamic_constants = constants,
                            dynamics = dyn,
                            boundary_conditions = bcs,
                            timestepper = :AcousticRungeKutta3)
    ref = model.dynamics.reference_state
    set!(model; θ = θ₀, u = 1.0, qᵛ = 0.005, ρ = ref.density)
    return model
end

function run_with_diagnostics(label, model;
                               Δt_init, max_Δt, cfl, stop_time, sample_every = 100)
    sim = Simulation(model; Δt = Δt_init, stop_time, verbose = false)
    conjure_time_step_wizard!(sim; cfl, max_Δt, max_change = 1.05)

    iters = Int[]; ts = Float64[]; dts = Float64[]
    wmax = Float64[]; umax = Float64[]
    qcl_max = Float64[]; walls = Float64[]
    wall0 = Ref(time_ns())

    function diag_cb(s)
        m = s.model
        u, w = m.velocities.u, m.velocities.w
        wm = Float64(maximum(abs, interior(w)))
        um = Float64(maximum(abs, interior(u)))
        ρqᶜˡ_max = if hasproperty(m, :microphysical_fields)
            Float64(maximum(interior(m.microphysical_fields.ρqᶜˡ)))
        else
            0.0
        end
        push!(iters, iteration(s)); push!(ts, time(s)); push!(dts, s.Δt)
        push!(wmax, wm); push!(umax, um); push!(qcl_max, ρqᶜˡ_max)
        push!(walls, (time_ns() - wall0[]) / 1e9)
        @info @sprintf("[%s] iter=%6d t=%6.2fmin Δt=%5.2fs max|u|=%.2f max|w|=%.3e max(ρqcl)=%.2e wall=%.0fs",
                       label, iteration(s), time(s)/60, s.Δt, um, wm, ρqᶜˡ_max, walls[end])
        flush(stdout)
        return nothing
    end
    add_callback!(sim, diag_cb, IterationInterval(sample_every))

    wall0[] = time_ns()
    crashed = false
    try
        run!(sim)
    catch err
        crashed = true
        @error "[$label] CRASHED" err
    end

    return (; label, iters, ts, dts, wmax, umax, qcl_max, walls,
             crashed, final_t = time(sim), final_iter = iteration(sim))
end

stop_time = 1hour
cfl       = 0.7

@info "===== Compressible-substepper run ====="
result_comp = run_with_diagnostics("comp", build_compressible();
    Δt_init = 0.2, max_Δt = 30.0, cfl, stop_time, sample_every = 100)

@info "===== Anelastic run ====="
result_anel = run_with_diagnostics("anel", build_anelastic();
    Δt_init = 1.0, max_Δt = 60.0, cfl, stop_time, sample_every = 100)

jldsave(joinpath(OUTDIR, RUN_LABEL * "_results.jld2");
        comp = result_comp, anel = result_anel,
        config = (; stop_time, cfl))

@info "Done. Output: $(joinpath(OUTDIR, RUN_LABEL))_results.jld2"
@info "Compressible: final_t=$(result_comp.final_t)s, final_iter=$(result_comp.final_iter), crashed=$(result_comp.crashed)"
@info "Anelastic:    final_t=$(result_anel.final_t)s, final_iter=$(result_anel.final_iter), crashed=$(result_anel.crashed)"
