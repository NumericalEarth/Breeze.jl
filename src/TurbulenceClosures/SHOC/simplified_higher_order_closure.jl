#####
##### `SimplifiedHigherOrderClosure`: SHOC as a turbulence closure of `AtmosphereModel`
#####
##### SHOC is operator split from the dynamics, as in EAMxx: during the Runge–Kutta stages the
##### closure contributes no flux and no tendency, and once per time step, in
##### `step_closure_prognostics!`, every column takes one whole SHOC step (`shoc_column!`) over
##### the full `Δt`. That step diffuses θˡ, qᵗ, u, v and the turbulent kinetic energy implicitly,
##### applies the surface fluxes, and diagnoses the second and third moments and the subgrid
##### cloud of the assumed PDF.
#####
##### Turbulent kinetic energy is the tracer `ρe`, which the closure adds to the model, so that in
##### three dimensions it is advected by the resolved flow during the stages.
#####

"""The name of the prognostic turbulent kinetic energy tracer of SHOC, which holds the density ``ρ e``."""
const TKE_NAME = :ρe

"""
$(TYPEDEF)

The Simplified Higher-Order Closure of [Bogenschutz and Krueger (2013)](@cite Bogenschutz2013),
as implemented in E3SM (`shoc.F90`).

$(TYPEDFIELDS)
"""
struct SimplifiedHigherOrderClosure{P, FT} <: AbstractTurbulenceClosure{ExplicitTimeDiscretization, 1}
    "The [`SHOCParameters`](@ref): tunable coefficients, physical constants and limits"
    parameters :: P
    "Default kinematic surface flux of liquid water potential temperature, K m s⁻¹"
    surface_temperature_flux :: FT
    "Default kinematic surface flux of total water, kg kg⁻¹ m s⁻¹"
    surface_moisture_flux :: FT
    "Default kinematic surface flux of x-momentum, m² s⁻²"
    surface_x_momentum_flux :: FT
    "Default kinematic surface flux of y-momentum, m² s⁻²"
    surface_y_momentum_flux :: FT
    "Default friction velocity, m s⁻¹; where positive, the surface momentum flux is −u★² u / |u| at the lowest level"
    friction_velocity :: FT
    "Horizontal mesh size that caps the mixing length along `Flat` directions, m"
    host_grid_spacing :: FT
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return a `SimplifiedHigherOrderClosure` (SHOC) for an `AtmosphereModel`.

SHOC is operator split from the dynamics: it contributes nothing during the Runge–Kutta stages,
and once per time step every column takes one SHOC step over the whole time step. The closure
adds the prognostic tracer `ρe` (turbulent kinetic energy) to the model, and keeps the state that
SHOC carries between steps — the subgrid cloud liquid `shoc_ql`, the cloud fraction
`shoc_cldfrac`, the buoyancy flux `wthv_sec`, the diffusivities `tkh` and `tk` — together with
every diagnosed moment in `model.closure_fields.diagnostics`.

SHOC applies the surface fluxes of heat, moisture and momentum inside its implicit solve. They are
kinematic fluxes held in the two-dimensional fields of `model.closure_fields.surface_specification`
(`wthl_sfc`, `wqw_sfc`, `uw_sfc`, `vw_sfc`, `friction_velocity`), which are initialized uniformly
from the keyword arguments below and may be `set!` afterwards, column by column. The model
fields should keep their default (no-flux) bottom boundary conditions, or the fluxes are applied
twice.

The model must use the liquid-ice potential temperature formulation; `ρθ / ρ` is SHOC's liquid
water potential temperature and the model's moisture density over `ρ` its total water. SHOC
owns the subgrid cloud: the liquid water it diagnoses is not fed back to the thermodynamics of the
model.

Keyword arguments
=================

- `parameters`: the [`SHOCParameters`](@ref); the defaults are those of E3SM `shoc.F90`.
- `surface_temperature_flux`, `surface_moisture_flux`, `surface_x_momentum_flux`,
  `surface_y_momentum_flux`: uniform kinematic surface fluxes.
- `friction_velocity`: where positive, the surface momentum flux is diagnosed every step as
  `−u★² u / |u|` from the wind of the lowest level, and the prescribed momentum fluxes are ignored.
- `host_grid_spacing`: the horizontal mesh size that caps the mixing length along `Flat`
  directions (elsewhere the grid spacing is used).

```jldoctest
using Breeze

closure = SimplifiedHigherOrderClosure(surface_temperature_flux=0.1)

# output
SimplifiedHigherOrderClosure{Float64}
├── surface fluxes: wthl_sfc = 0.1, wqw_sfc = 0.0, uw_sfc = 0.0, vw_sfc = 0.0, friction_velocity = 0.0
├── host_grid_spacing: 100000.0
└── parameters: SHOCParameters{Float64}
```
"""
function SimplifiedHigherOrderClosure(FT::DataType = Oceananigans.defaults.FloatType;
                                      parameters = SHOCParameters(FT),
                                      surface_temperature_flux = 0,
                                      surface_moisture_flux = 0,
                                      surface_x_momentum_flux = 0,
                                      surface_y_momentum_flux = 0,
                                      friction_velocity = 0,
                                      host_grid_spacing = 100000)

    return SimplifiedHigherOrderClosure(parameters,
                                        convert(FT, surface_temperature_flux),
                                        convert(FT, surface_moisture_flux),
                                        convert(FT, surface_x_momentum_flux),
                                        convert(FT, surface_y_momentum_flux),
                                        convert(FT, friction_velocity),
                                        convert(FT, host_grid_spacing))
end

const SHOCClosureArray = AbstractArray{<:SimplifiedHigherOrderClosure}

"""Either a single `SimplifiedHigherOrderClosure` or an ensemble array of them."""
const FlavorOfSHOC = Union{SimplifiedHigherOrderClosure, SHOCClosureArray}

Oceananigans.TurbulenceClosures.closure_required_tracers(::FlavorOfSHOC) = tuple(TKE_NAME)

function Utils.with_tracers(tracer_names, closure::FlavorOfSHOC)
    TKE_NAME ∈ tracer_names ||
        throw(ArgumentError("Tracers must contain :$(TKE_NAME) to represent turbulent kinetic " *
                            "energy for `SimplifiedHigherOrderClosure`."))
    return closure
end

Base.summary(::SimplifiedHigherOrderClosure{P, FT}) where {P, FT} = "SimplifiedHigherOrderClosure{$FT}"

function Base.show(io::IO, closure::SimplifiedHigherOrderClosure)
    print(io, summary(closure), '\n',
          "├── surface fluxes: wthl_sfc = ", prettysummary(closure.surface_temperature_flux),
          ", wqw_sfc = ", prettysummary(closure.surface_moisture_flux),
          ", uw_sfc = ", prettysummary(closure.surface_x_momentum_flux),
          ", vw_sfc = ", prettysummary(closure.surface_y_momentum_flux),
          ", friction_velocity = ", prettysummary(closure.friction_velocity), '\n',
          "├── host_grid_spacing: ", prettysummary(closure.host_grid_spacing), '\n',
          "└── parameters: ")
    print(io, summary(closure.parameters))
end

#####
##### No fluxes during the Runge–Kutta stages
#####

@inline AtmosphereModels.∂ⱼ_𝒯₁ⱼ(i, j, k, grid, ρ, ::SimplifiedHigherOrderClosure, closure_fields,
                                clock, model_fields, buoyancy) = zero(grid)

@inline AtmosphereModels.∂ⱼ_𝒯₂ⱼ(i, j, k, grid, ρ, ::SimplifiedHigherOrderClosure, closure_fields,
                                clock, model_fields, buoyancy) = zero(grid)

@inline AtmosphereModels.∂ⱼ_𝒯₃ⱼ(i, j, k, grid, ρ, ::SimplifiedHigherOrderClosure, closure_fields,
                                clock, model_fields, buoyancy) = zero(grid)

@inline AtmosphereModels.∇_dot_Jᶜ(i, j, k, grid, ρ, ::SimplifiedHigherOrderClosure, closure_fields,
                                  id, c, clock, model_fields, buoyancy) = zero(grid)

#####
##### Closure fields
#####

"""
$(TYPEDEF)

The fields of [`SimplifiedHigherOrderClosure`](@ref), named as in E3SM `shoc.F90`. Level fields
are at `(Center, Center, Center)`, interface fields at `(Center, Center, Face)`.

$(TYPEDFIELDS)
"""
struct SHOCFields{S, H, F, D, W, I}
    "SHOC's copy of the specific prognostic state: `thetal`, `qw`, `u_wind`, `v_wind`, `tke`"
    state :: S
    "Host inputs, rebuilt every step: `zt_grid`, `zi_grid`, `pres`, `pdel`, `inv_exner`, `thv`, `w_field`, `host_dx`, `host_dy`"
    host :: H
    "The kinematic surface fluxes SHOC used in the last step: `wthl_sfc`, `wqw_sfc`, `uw_sfc`, `vw_sfc`"
    surface :: F
    "The user-facing surface specification: `wthl_sfc`, `wqw_sfc`, `uw_sfc`, `vw_sfc`, `friction_velocity`"
    surface_specification :: F
    "State carried between steps (`shoc_ql`, `shoc_cldfrac`, `wthv_sec`, `tkh`, `tk`) and every diagnosed moment"
    diagnostics :: D
    "The tridiagonal factorization: `du`, `dl`, `d`"
    scratch :: W
    "The increments of the level winds over the last step: `u`, `v`"
    increments :: I
end

Adapt.adapt_structure(to, fields::SHOCFields) =
    SHOCFields(adapt(to, fields.state),
               adapt(to, fields.host),
               adapt(to, fields.surface),
               adapt(to, fields.surface_specification),
               adapt(to, fields.diagnostics),
               adapt(to, fields.scratch),
               adapt(to, fields.increments))

# Nothing here is read across a horizontal boundary by the dynamics.
BoundaryConditions.fill_halo_regions!(::SHOCFields, args...; kw...) = nothing

const LEVEL_DIAGNOSTICS = (:wthv_sec, :tkh, :tk, :shoc_ql, :shoc_cldfrac, :shoc_mix, :isotropy,
                           :w_sec, :wqls_sec, :brunt, :shoc_ql2)

const INTERFACE_DIAGNOSTICS = (:thl_sec, :qw_sec, :qwthl_sec, :wthl_sec, :wqw_sec, :wtke_sec,
                               :uw_sec, :vw_sec, :w3)

surface_field(grid) = Field{Center, Center, Nothing}(grid)

function surface_field(grid, value)
    field = surface_field(grid)
    fill!(parent(field), value)
    return field
end

uniform_surface_value(closure::SimplifiedHigherOrderClosure, name) = getproperty(closure, name)

# An ensemble array of closures initializes the surface specification from its first member; set
# the fields of `closure_fields.surface_specification` for column-by-column values.
uniform_surface_value(closures::SHOCClosureArray, name) = getproperty(@allowscalar(first(closures)), name)

function Oceananigans.TurbulenceClosures.build_closure_fields(grid, clock, tracer_names, bcs, closure::FlavorOfSHOC)
    state = (thetal = CenterField(grid),
             qw = CenterField(grid),
             u_wind = CenterField(grid),
             v_wind = CenterField(grid),
             tke = CenterField(grid))

    host = (zt_grid = CenterField(grid),
            zi_grid = ZFaceField(grid),
            pres = CenterField(grid),
            pdel = CenterField(grid),
            inv_exner = CenterField(grid),
            thv = CenterField(grid),
            w_field = CenterField(grid),
            host_dx = surface_field(grid),
            host_dy = surface_field(grid))

    surface = (wthl_sfc = surface_field(grid),
               wqw_sfc = surface_field(grid),
               uw_sfc = surface_field(grid),
               vw_sfc = surface_field(grid),
               friction_velocity = surface_field(grid))

    surface_specification =
        (wthl_sfc = surface_field(grid, uniform_surface_value(closure, :surface_temperature_flux)),
         wqw_sfc = surface_field(grid, uniform_surface_value(closure, :surface_moisture_flux)),
         uw_sfc = surface_field(grid, uniform_surface_value(closure, :surface_x_momentum_flux)),
         vw_sfc = surface_field(grid, uniform_surface_value(closure, :surface_y_momentum_flux)),
         friction_velocity = surface_field(grid, uniform_surface_value(closure, :friction_velocity)))

    level_diagnostics = NamedTuple{LEVEL_DIAGNOSTICS}(map(name -> CenterField(grid), LEVEL_DIAGNOSTICS))
    interface_diagnostics = NamedTuple{INTERFACE_DIAGNOSTICS}(map(name -> ZFaceField(grid), INTERFACE_DIAGNOSTICS))
    diagnostics = merge(level_diagnostics, interface_diagnostics, (; pblh = surface_field(grid)))

    scratch = (du = CenterField(grid), dl = CenterField(grid), d = CenterField(grid))
    increments = (u = CenterField(grid), v = CenterField(grid))

    return SHOCFields(state, host, surface, surface_specification, diagnostics, scratch, increments)
end

# SHOC has no diffusivity that the dynamics uses, and nothing to recompute from the model state.
Oceananigans.TurbulenceClosures.compute_closure_fields!(::SHOCFields, ::FlavorOfSHOC, model; kw...) = nothing

#####
##### Loading a column from the model, and storing it back
#####

# The horizontal mesh size, which caps the mixing length
@inline host_Δx(i, j, grid, Δ) = Δxᶜᶜᶜ(i, j, 1, grid)
@inline host_Δy(i, j, grid, Δ) = Δyᶜᶜᶜ(i, j, 1, grid)
@inline host_Δx(i, j, ::AbstractGrid{<:Any, <:Flat}, Δ) = Δ
@inline host_Δy(i, j, ::AbstractGrid{<:Any, <:Any, <:Flat}, Δ) = Δ

@inline specific_valueᶜᶜᶜ(i, j, k, grid, ρc, ρ) = @inbounds ρc[i, j, k] / ρ[i, j, k]
@inline level_increment_density(i, j, k, grid, ρ, δ) = @inbounds ρ[i, j, k] * δ[i, j, k]

@inline function load_shoc_column!(i, j, grid, closure, fields, ρ, pressure, ρθ, ρqᵗ, ρu, ρv, ρw, ρe)
    Nz = size(grid, 3)
    p = closure.parameters
    state, host, diagnostics = fields.state, fields.host, fields.diagnostics
    specification, surface = fields.surface_specification, fields.surface
    c = Center()
    f = Face()

    z_surface = znode(i, j, 1, grid, c, c, f)

    @inbounds for k in 1:Nz
        ρᵏ = ρ[i, j, k]
        thetal = ρθ[i, j, k] / ρᵏ
        qw = ρqᵗ[i, j, k] / ρᵏ

        state.thetal[i, j, k] = thetal
        state.qw[i, j, k] = qw
        state.tke[i, j, k] = ρe[i, j, k] / ρᵏ
        state.u_wind[i, j, k] = ℑxᶜᵃᵃ(i, j, k, grid, ρu) / ρᵏ
        state.v_wind[i, j, k] = ℑyᵃᶜᵃ(i, j, k, grid, ρv) / ρᵏ

        # The level winds before the step, for the increments
        fields.increments.u[i, j, k] = state.u_wind[i, j, k]
        fields.increments.v[i, j, k] = state.v_wind[i, j, k]

        pres = pressure[i, j, k]
        inv_exner = (p.basepres / pres)^(p.rgas / p.cp)
        host.zt_grid[i, j, k] = znode(i, j, k, grid, c, c, c) - z_surface
        host.zi_grid[i, j, k] = znode(i, j, k, grid, c, c, f) - z_surface
        host.pres[i, j, k] = pres
        host.pdel[i, j, k] = ρᵏ * p.ggr * Δzᶜᶜᶜ(i, j, k, grid)
        host.inv_exner[i, j, k] = inv_exner
        host.w_field[i, j, k] = ℑzᵃᵃᶜ(i, j, k, grid, ρw) / ρᵏ

        # Virtual potential temperature, with the subgrid cloud liquid of the previous step
        ql = diagnostics.shoc_ql[i, j, k]
        theta = thetal + (p.lcond / p.cp) * inv_exner * ql
        host.thv[i, j, k] = theta * (1 + p.eps * (qw - ql) - ql)
    end

    @inbounds begin
        host.zi_grid[i, j, Nz+1] = znode(i, j, Nz+1, grid, c, c, f) - z_surface
        host.host_dx[i, j, 1] = host_Δx(i, j, grid, closure.host_grid_spacing)
        host.host_dy[i, j, 1] = host_Δy(i, j, grid, closure.host_grid_spacing)

        # Surface fluxes: prescribed, or a momentum flux diagnosed from a friction velocity
        u₁ = state.u_wind[i, j, 1]
        v₁ = state.v_wind[i, j, 1]
        u★ = specification.friction_velocity[i, j, 1]
        speed = zero_safe_sqrt(u₁ * u₁ + v₁ * v₁)
        drag = (u★ > 0) & (speed > 0)
        speed = ifelse(drag, speed, one(speed))

        surface.wthl_sfc[i, j, 1] = specification.wthl_sfc[i, j, 1]
        surface.wqw_sfc[i, j, 1] = specification.wqw_sfc[i, j, 1]
        surface.uw_sfc[i, j, 1] = ifelse(drag, -(u★ * u★) * u₁ / speed, specification.uw_sfc[i, j, 1])
        surface.vw_sfc[i, j, 1] = ifelse(drag, -(u★ * u★) * v₁ / speed, specification.vw_sfc[i, j, 1])
        surface.friction_velocity[i, j, 1] = u★
    end

    return nothing
end

@inline function store_shoc_column!(i, j, grid, fields, ρ, ρθ, ρqᵗ, ρe)
    Nz = size(grid, 3)
    state = fields.state

    @inbounds for k in 1:Nz
        ρᵏ = ρ[i, j, k]
        ρθ[i, j, k] = ρᵏ * state.thetal[i, j, k]
        ρqᵗ[i, j, k] = ρᵏ * state.qw[i, j, k]
        ρe[i, j, k] = ρᵏ * state.tke[i, j, k]
        fields.increments.u[i, j, k] = state.u_wind[i, j, k] - fields.increments.u[i, j, k]
        fields.increments.v[i, j, k] = state.v_wind[i, j, k] - fields.increments.v[i, j, k]
    end

    return nothing
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

One SHOC step of every column of the model: load the specific state from the prognostic
densities, step SHOC, and store θˡ, qᵗ and turbulent kinetic energy back as densities. One thread
per column, with every vertical loop inside the thread.
"""
@kernel function _step_shoc!(grid, Δt, closure, fields, ρ, pressure, ρθ, ρqᵗ, ρu, ρv, ρw, ρe)
    i, j = @index(Global, NTuple)
    closure_ij = getclosure(i, j, closure)
    Nz = size(grid, 3)

    load_shoc_column!(i, j, grid, closure_ij, fields, ρ, pressure, ρθ, ρqᵗ, ρu, ρv, ρw, ρe)

    shoc_column!(i, j, Nz, Δt, closure_ij.parameters,
                 fields.state, fields.host, fields.surface, fields.diagnostics, fields.scratch)

    store_shoc_column!(i, j, grid, fields, ρ, ρθ, ρqᵗ, ρe)
end

# The momentum densities live on cell faces, so the increments of the level winds are carried there.
@kernel function _add_shoc_momentum_increments!(ρu, ρv, grid, ρ, increments)
    i, j, k = @index(Global, NTuple)
    c = Center()
    f = Face()

    δρu = ℑxᶠᵃᵃ(i, j, k, grid, level_increment_density, ρ, increments.u)
    δρv = ℑyᵃᶠᵃ(i, j, k, grid, level_increment_density, ρ, increments.v)

    @inbounds begin
        ρu[i, j, k] += ifelse(peripheral_node(i, j, k, grid, f, c, c), zero(δρu), δρu)
        ρv[i, j, k] += ifelse(peripheral_node(i, j, k, grid, c, f, c), zero(δρv), δρv)
    end
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Threads per block of the column kernel on a GPU, for `Nx × Ny` columns: one thread per column
leaves few threads in all, so the blocks are made small enough that the columns spread over all
multiprocessors (`2 × 108` blocks on an A100) before they are made larger, up to `maxthreads`.
The block spans `x` first, so that the threads of a warp read adjacent memory.
"""
function shoc_gpu_workgroup(Nx, Ny; target_blocks = 216, maxthreads = 128)
    threads = clamp(prevpow(2, max(1, (Nx * Ny) ÷ target_blocks)), 32, maxthreads)
    wx = min(threads, Nx)
    wy = clamp(threads ÷ wx, 1, Ny)
    return (wx, wy)
end

function launch_shoc_columns!(arch, grid, args...)
    dev = device(arch)
    if dev isa KernelAbstractions.GPU
        Nx, Ny, _ = size(grid)
        kernel! = _step_shoc!(dev, shoc_gpu_workgroup(Nx, Ny), (Nx, Ny))
        kernel!(grid, args...)
    else
        launch!(arch, grid, :xy, _step_shoc!, grid, args...)
    end
    return nothing
end

function shoc_thermodynamic_density(formulation)
    AtmosphereModels.thermodynamic_density_name(formulation) === :ρθ ||
        throw(ArgumentError("SimplifiedHigherOrderClosure requires the liquid-ice potential temperature " *
                            "formulation, but the model formulation is $(summary(formulation))."))
    return Oceananigans.prognostic_fields(formulation).ρθ
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Take one SHOC step of length `Δt` in every column of `model`. Called once per time step by the
time steppers, after the last Runge–Kutta stage, when the prognostic fields hold densities.
"""
function Oceananigans.TimeSteppers.step_closure_prognostics!(fields::SHOCFields, closure::FlavorOfSHOC, model, Δt)
    grid = model.grid
    arch = grid.architecture
    dynamics = model.dynamics

    ρ = AtmosphereModels.total_density(dynamics)
    pressure = AtmosphereModels.dynamics_pressure(dynamics)
    ρθ = shoc_thermodynamic_density(model.formulation)
    ρqᵗ = model.moisture_density
    ρu, ρv, ρw = model.momentum.ρu, model.momentum.ρv, model.momentum.ρw
    ρe = model.tracers[TKE_NAME]

    launch_shoc_columns!(arch, grid, Δt, closure, fields, ρ, pressure, ρθ, ρqᵗ, ρu, ρv, ρw, ρe)

    fill_halo_regions!((fields.increments.u, fields.increments.v))

    launch!(arch, grid, :xyz, _add_shoc_momentum_increments!, ρu, ρv, grid, ρ, fields.increments)

    return nothing
end

# The state SHOC carries between steps lives in the closure fields.
# TODO: checkpoint `shoc_ql`, `shoc_cldfrac`, `wthv_sec`, `tkh` and `tk`.
Oceananigans.prognostic_state(::SHOCFields) = nothing
Oceananigans.restore_prognostic_state!(fields::SHOCFields, ::Nothing) = fields
