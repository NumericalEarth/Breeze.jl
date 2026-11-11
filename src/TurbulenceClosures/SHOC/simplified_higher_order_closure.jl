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
struct SimplifiedHigherOrderClosure{ML, TS, TK, BS, SM, TM, PDF, BL, CP, C, FT} <:
       AbstractTurbulenceClosure{ExplicitTimeDiscretization, 1}
    "The mixing length"
    mixing_length :: ML
    "The stability dependence: return-to-isotropy time scale, eddy diffusivities and dissipation coefficient"
    turbulence_time_scale :: TS
    "The update of turbulent kinetic energy"
    tke_update :: TK
    "The buoyancy source of turbulent kinetic energy"
    buoyancy_source :: BS
    "The second moments"
    second_moments :: SM
    "The third moment of vertical velocity"
    third_moment :: TM
    "The assumed PDF: subgrid cloud and buoyancy flux"
    assumed_pdf :: PDF
    "The diagnosis of the boundary layer depth"
    boundary_layer_depth :: BL
    "How bounds are imposed"
    clipping :: CP
    "Physical constants, reference values and the bounds on turbulent kinetic energy"
    constants :: C
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

Return a `SimplifiedHigherOrderClosure` (SHOC) for an `AtmosphereModel`, with scalars of type `FT`.

The closure is made of components, one per relation of the scheme, each given by its own keyword
argument and each holding the empirical coefficients (written ℂ) of its relation. With no
keyword arguments the components are those of E3SM `shoc.F90` (pinned at E3SM `ba53243f`) with
the published coefficients, and the column physics reproduces `shoc.F90` to round-off. Every
component is replaced independently of the others, either to change coefficients,
`SimplifiedHigherOrderClosure(turbulence_time_scale = ColumnStabilityDamping(ℂᶜ=0.2))`, or to
swap the relation,
`SimplifiedHigherOrderClosure(mixing_length = GradientLimitedMixingLength(), clipping = SmoothClipping())`;
any combination of the components listed below is a valid closure.
[`differentiable_shoc_components`](@ref) returns a set of components without the hard switches
and clips of `shoc.F90`, for calibration with gradients, to splat into the keyword arguments.

[`closure_parameters`](@ref) lists the coefficients of all components and
[`with_parameters`](@ref) replaces some of them, whichever components hold them. A component
that another closure of Breeze also uses ([`GradientLimitedMixingLength`](@ref),
[`RationalStabilityFunction`](@ref)) is passed under its own name, and the closure holds its own
copy of its coefficients: they are coefficients of this closure like any other, listed by
`closure_parameters`, replaced by `with_parameters` and calibrated within SHOC, and are not tied
to the values that [`TKEBasedTurbulenceClosure`](@ref) holds or was calibrated to.

SHOC is operator split from the dynamics: it contributes nothing during the Runge–Kutta stages,
and once per time step every column takes one SHOC step over the whole time step. The closure
adds the prognostic tracer `ρe` (turbulent kinetic energy) to the model, and keeps the state that
SHOC carries between steps — the subgrid cloud liquid `shoc_ql`, the cloud fraction
`shoc_cldfrac`, the buoyancy flux `wthv_sec`, the diffusivities `tkh` and `tk` — together with
every diagnosed moment in `model.closure_fields.diagnostics`.

SHOC applies the surface fluxes of heat, moisture and momentum inside its implicit solve, and it is
the only place they are applied. They are the sum of two parts:

1. The model's own bottom `Flux` boundary conditions on the thermodynamic density `ρθ` (or the
   energy flux given under the key `ρE`, which the model converts to a `ρθ` flux), the moisture
   density, `ρu` and `ρv`. Once per step SHOC evaluates them at the bottom of each column, exactly
   as the tendency kernels would (`total_boundary_flux`, so constants, arrays, fields,
   `FieldTimeSeries`, continuous and discrete functions and bulk formulae all work), at the time
   and state the step ends with, and divides the density-weighted flux by the density SHOC uses
   at the surface to obtain the kinematic flux. The Runge–Kutta stages skip the bottom boundary
   flux of these fields (see `AtmosphereModels.closure_applies_bottom_flux`), so nothing is
   applied twice and the boundary conditions stay the single source of truth.
2. A closure-owned specification, in kinematic units, held in the two-dimensional fields of
   `model.closure_fields.surface_specification` (`wthl_sfc`, `wqw_sfc`, `uw_sfc`, `vw_sfc`,
   `friction_velocity`). They are initialized uniformly from the keyword arguments below (zero by
   default) and may be `set!` afterwards, column by column.

SHOC has its own surface source of turbulent kinetic energy (`u★³` from the surface stress), so a
bottom flux boundary condition on `ρe` is not applied at all: neither by the stages nor by SHOC.

The model must use the liquid-ice potential temperature formulation; `ρθ / ρ` is SHOC's liquid
water potential temperature and the model's moisture density over `ρ` its total water. By
default the liquid water SHOC diagnoses is not fed back to the thermodynamics of the model. With
a microphysics scheme whose subgrid choice is `Breeze.AtmosphereModels.ClosureCloud` (for example
`PredictedParticlePropertiesMicrophysics(subgrid=ClosureCloud())`) SHOC owns the cloud: its total
water is the vapor plus the cloud liquid of the model, after every step the cloud liquid of the
model is SHOC's `shoc_ql` and the vapor the rest, and the microphysics reads `shoc_cldfrac` as
the cloud fraction and keeps the precipitation.

Keyword arguments
=================

Each of the first eight takes a component; the first one named is the default, that of `shoc.F90`.

- `mixing_length`: the mixing length; [`SHOCMixingLength`](@ref), or
  [`GradientLimitedMixingLength`](@ref).
- `turbulence_time_scale`: the return-to-isotropy time scale, the eddy diffusivities and the
  dissipation coefficient; [`ColumnStabilityDamping`](@ref), or
  [`RichardsonStabilityFunctions`](@ref) of a [`RationalStabilityFunction`](@ref).
- `tke_update`: the update of turbulent kinetic energy; [`ForwardEulerTKE`](@ref), or
  [`ImplicitSinkTKE`](@ref).
- `buoyancy_source`: the buoyancy source of turbulent kinetic energy; [`PDFBuoyancyFlux`](@ref),
  or [`DowngradientBuoyancyFlux`](@ref) for comparison.
- `second_moments`: [`DowngradientSecondMoments`](@ref).
- `third_moment`: the third moment of vertical velocity; [`CanutoThirdMoment`](@ref) with
  [`ResetBound`](@ref), or with [`ClampBound`](@ref) or [`TanhBound`](@ref).
- `assumed_pdf`: the assumed PDF of the subgrid cloud; [`DoubleGaussianPDF`](@ref), or
  [`SmoothDoubleGaussianPDF`](@ref).
- `clipping`: how bounds are imposed; [`HardClipping`](@ref), or [`SmoothClipping`](@ref).
- `boundary_layer_depth`: [`HoltslagBovilleDepth`](@ref).
- `constants`: the [`SHOCConstants`](@ref).
- `surface_temperature_flux`, `surface_moisture_flux`, `surface_x_momentum_flux`,
  `surface_y_momentum_flux`: uniform kinematic surface fluxes.
- `friction_velocity`: where positive, the surface momentum flux is diagnosed every step as
  `−u★² u / |u|` from the wind of the lowest level, in place of the closure-owned
  `surface_x_momentum_flux` and `surface_y_momentum_flux` (the model's boundary-condition fluxes
  are still added).
- `host_grid_spacing`: the horizontal mesh size that caps the mixing length along `Flat`
  directions (elsewhere the grid spacing is used).

```jldoctest
using Breeze

closure = SimplifiedHigherOrderClosure(surface_temperature_flux=0.1)

# output
SimplifiedHigherOrderClosure{Float64}
├── mixing_length: SHOCMixingLength{Float64}
├── turbulence_time_scale: ColumnStabilityDamping{Float64}
├── tke_update: ForwardEulerTKE{Float64}
├── buoyancy_source: PDFBuoyancyFlux
├── second_moments: DowngradientSecondMoments{Float64}
├── third_moment: CanutoThirdMoment{Float64} with ResetBound
├── assumed_pdf: DoubleGaussianPDF{Float64}
├── boundary_layer_depth: HoltslagBovilleDepth{Float64}
├── clipping: HardClipping
├── surface fluxes: wthl_sfc = 0.1, wqw_sfc = 0.0, uw_sfc = 0.0, vw_sfc = 0.0, friction_velocity = 0.0
└── host_grid_spacing: 100000.0
```
"""
function SimplifiedHigherOrderClosure(FT::DataType = Oceananigans.defaults.FloatType;
                                      mixing_length = SHOCMixingLength(Float64),
                                      turbulence_time_scale = ColumnStabilityDamping(Float64),
                                      tke_update = ForwardEulerTKE(Float64),
                                      buoyancy_source = PDFBuoyancyFlux(),
                                      second_moments = DowngradientSecondMoments(Float64),
                                      third_moment = CanutoThirdMoment(Float64),
                                      assumed_pdf = DoubleGaussianPDF(Float64),
                                      clipping = HardClipping(),
                                      boundary_layer_depth = HoltslagBovilleDepth(Float64),
                                      constants = SHOCConstants(Float64),
                                      surface_temperature_flux = 0,
                                      surface_moisture_flux = 0,
                                      surface_x_momentum_flux = 0,
                                      surface_y_momentum_flux = 0,
                                      friction_velocity = 0,
                                      host_grid_spacing = 100000)

    # The published coefficients are built in double precision, whatever the default float type,
    # and converted once here.
    components = (; mixing_length, turbulence_time_scale, tke_update, buoyancy_source, second_moments,
                    third_moment, assumed_pdf, boundary_layer_depth, clipping, constants)

    check_component_slots(components)
    components = map(component -> convert_eltype(FT, component), values(components))
    check_unique_scalar_names(components)

    return SimplifiedHigherOrderClosure(components...,
                                        convert(FT, surface_temperature_flux),
                                        convert(FT, surface_moisture_flux),
                                        convert(FT, surface_x_momentum_flux),
                                        convert(FT, surface_y_momentum_flux),
                                        convert(FT, friction_velocity),
                                        convert(FT, host_grid_spacing))
end

"""
A shorter name for [`SimplifiedHigherOrderClosure`](@ref).
"""
const SHOC = SimplifiedHigherOrderClosure

# Recurse into the components and the surface scalars when the closure is adapted for a kernel
# (on `ReactantState` they are traced scalars during automatic differentiation).
Adapt.@adapt_structure SimplifiedHigherOrderClosure

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

Base.summary(closure::SimplifiedHigherOrderClosure) = "SimplifiedHigherOrderClosure{$(typeof(closure.host_grid_spacing))}"

function Base.show(io::IO, closure::SimplifiedHigherOrderClosure)
    print(io, summary(closure), '\n')
    for slot in COMPONENT_SLOTS
        slot === :constants && continue
        print(io, "├── ", slot, ": ", summary(getfield(closure, slot)), '\n')
    end
    print(io,
          "├── surface fluxes: wthl_sfc = ", prettysummary(closure.surface_temperature_flux),
          ", wqw_sfc = ", prettysummary(closure.surface_moisture_flux),
          ", uw_sfc = ", prettysummary(closure.surface_x_momentum_flux),
          ", vw_sfc = ", prettysummary(closure.surface_y_momentum_flux),
          ", friction_velocity = ", prettysummary(closure.friction_velocity), '\n',
          "└── host_grid_spacing: ", prettysummary(closure.host_grid_spacing))
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

closure_component_fields(grid, closure::SimplifiedHigherOrderClosure) =
    merge(map(component -> component_fields(grid, component), closure_components(closure))...)

# The members of an ensemble array of closures have the same components
closure_component_fields(grid, closures::SHOCClosureArray) = closure_component_fields(grid, @allowscalar(first(closures)))

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

    # The tridiagonal factorization, and the fields of the components that need their own
    scratch = merge((du = CenterField(grid), dl = CenterField(grid), d = CenterField(grid)),
                    closure_component_fields(grid, closure))
    increments = (u = CenterField(grid), v = CenterField(grid))

    return SHOCFields(state, host, surface, surface_specification, diagnostics, scratch, increments)
end

# SHOC has no diffusivity that the dynamics uses, and nothing to recompute from the model state.
Oceananigans.TurbulenceClosures.compute_closure_fields!(::SHOCFields, ::FlavorOfSHOC, model; kw...) = nothing

#####
##### Surface fluxes from the model's bottom boundary conditions
#####

# The density-weighted flux through the bottom of column (i, j) that the boundary condition `bc`
# of the prognostic field `c` prescribes: positive upward, into the domain. Only `Flux` boundary
# conditions carry one (the default no-flux condition gives zero).
@inline bottom_boundary_flux(bc::BoundaryCondition{<:Flux}, i, j, grid, c, args...) =
    total_boundary_flux(bc, Bottom(), i, j, 1, grid, c, args...)

@inline bottom_boundary_flux(bc, i, j, grid, c, args...) = zero(grid)

# SHOC applies the bottom fluxes of the fields it steps, so the Runge–Kutta stages must not.
# SHOC's own surface source of turbulent kinetic energy replaces a wall flux of `ρe`, which is
# therefore claimed here too and applied nowhere.
function AtmosphereModels.closure_applies_bottom_flux(::FlavorOfSHOC, model, name)
    thermodynamic_name = AtmosphereModels.thermodynamic_density_name(model.formulation)
    moisture_name = AtmosphereModels.moisture_prognostic_name(model.microphysics)
    return name ∈ (thermodynamic_name, moisture_name, :ρu, :ρv, TKE_NAME)
end

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

@inline function load_shoc_column!(i, j, grid, closure, fields, ρ, pressure, ρθ, ρqᵗ, ρu, ρv, ρw, ρe,
                                   bottom_bcs, bc_args)
    Nz = size(grid, 3)
    p = closure.constants
    state, host, diagnostics = fields.state, fields.host, fields.diagnostics
    specification, surface = fields.surface_specification, fields.surface
    c = Center()
    f = Face()

    z_surface = znode(i, j, 1, grid, c, c, f)

    @inbounds for k in 1:Nz
        ρᵏ = ρ[i, j, k]
        thetal = ρθ[i, j, k] / ρᵏ
        qw = total_water_density(i, j, k, ρqᵗ) / ρᵏ

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

        # The model's bottom boundary-condition fluxes are density weighted. SHOC multiplies its
        # kinematic surface fluxes by the density it interpolates to the surface, so dividing by
        # that same density puts exactly the boundary-condition flux into the lowest level.
        ρ_surface = rho_zi(i, j, 1, Nz, p, host.zt_grid, host.zi_grid, host.pdel)
        Jᶿ = bottom_boundary_flux(bottom_bcs.ρθ, i, j, grid, ρθ, bc_args...)
        Jᵗ = bottom_boundary_flux(bottom_bcs.ρqᵗ, i, j, grid, surface_moisture_density(ρqᵗ), bc_args...)
        𝒯ˣ = bottom_boundary_flux(bottom_bcs.ρu, i, j, grid, ρu, bc_args...)
        𝒯ʸ = bottom_boundary_flux(bottom_bcs.ρv, i, j, grid, ρv, bc_args...)

        surface.wthl_sfc[i, j, 1] = specification.wthl_sfc[i, j, 1] + Jᶿ / ρ_surface
        surface.wqw_sfc[i, j, 1] = specification.wqw_sfc[i, j, 1] + Jᵗ / ρ_surface
        surface.uw_sfc[i, j, 1] = ifelse(drag, -(u★ * u★) * u₁ / speed, specification.uw_sfc[i, j, 1]) + 𝒯ˣ / ρ_surface
        surface.vw_sfc[i, j, 1] = ifelse(drag, -(u★ * u★) * v₁ / speed, specification.vw_sfc[i, j, 1]) + 𝒯ʸ / ρ_surface
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
        store_total_water!(i, j, k, ρqᵗ, ρᵏ, state.qw[i, j, k], fields.diagnostics.shoc_ql[i, j, k])
        ρe[i, j, k] = ρᵏ * state.tke[i, j, k]
        fields.increments.u[i, j, k] = state.u_wind[i, j, k] - fields.increments.u[i, j, k]
        fields.increments.v[i, j, k] = state.v_wind[i, j, k] - fields.increments.v[i, j, k]
    end

    return nothing
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

The first part of a SHOC step of every column of the model: load the specific state from the
prognostic densities and take `shoc_turbulence!`. One thread per column, with every vertical
loop inside the thread. `_step_shoc_cloud!` completes the step; the step is split in
three kernels because each part needs fewer registers than the whole.
"""
@kernel function _step_shoc_turbulence!(grid, Δt, closure, fields, ρ, pressure, ρθ, ρqᵗ, ρu, ρv, ρw, ρe,
                                        bottom_bcs, bc_args)
    i, j = @index(Global, NTuple)
    closure_ij = getclosure(i, j, closure)
    Nz = size(grid, 3)

    load_shoc_column!(i, j, grid, closure_ij, fields, ρ, pressure, ρθ, ρqᵗ, ρu, ρv, ρw, ρe, bottom_bcs, bc_args)

    shoc_turbulence!(i, j, Nz, Δt, closure_ij, grid,
                     fields.state, fields.host, fields.surface, fields.diagnostics, fields.scratch)
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

The implicit diffusion and the second and third moments of a SHOC step of every column
(`shoc_implicit_moments!`), between `_step_shoc_turbulence!` and `_step_shoc_cloud!`.
"""
@kernel function _step_shoc_implicit!(grid, Δt, closure, fields, ρ, pressure, ρθ, ρqᵗ, ρu, ρv, ρw, ρe)
    i, j = @index(Global, NTuple)
    closure_ij = getclosure(i, j, closure)
    Nz = size(grid, 3)

    shoc_implicit_moments!(i, j, Nz, Δt, closure_ij, grid,
                           fields.state, fields.host, fields.surface, fields.diagnostics, fields.scratch)
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

The last part of a SHOC step of every column: `shoc_cloud!`, then store θˡ, qᵗ and turbulent
kinetic energy back as densities.
"""
@kernel function _step_shoc_cloud!(grid, Δt, closure, fields, ρ, pressure, ρθ, ρqᵗ, ρu, ρv, ρw, ρe)
    i, j = @index(Global, NTuple)
    closure_ij = getclosure(i, j, closure)
    Nz = size(grid, 3)

    shoc_cloud!(i, j, Nz, closure_ij, grid, fields.state, fields.host, fields.surface, fields.diagnostics)

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

# Only the first kernel, which loads the column, evaluates the bottom boundary conditions.
function launch_shoc_columns!(arch, grid, bottom_bcs, bc_args, args...)
    dev = device(arch)
    if dev isa KernelAbstractions.GPU
        Nx, Ny, _ = size(grid)
        workgroup = shoc_gpu_workgroup(Nx, Ny)
        _step_shoc_turbulence!(dev, workgroup, (Nx, Ny))(grid, args..., bottom_bcs, bc_args)
        _step_shoc_implicit!(dev, workgroup, (Nx, Ny))(grid, args...)
        _step_shoc_cloud!(dev, workgroup, (Nx, Ny))(grid, args...)
    else
        launch!(arch, grid, :xy, _step_shoc_turbulence!, grid, args..., bottom_bcs, bc_args)
        launch!(arch, grid, :xy, _step_shoc_implicit!, grid, args...)
        launch!(arch, grid, :xy, _step_shoc_cloud!, grid, args...)
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
    ρqᵗ = shoc_moisture(model)
    ρu, ρv, ρw = model.momentum.ρu, model.momentum.ρv, model.momentum.ρw
    ρe = model.tracers[TKE_NAME]

    # The surface fluxes: the model's bottom boundary conditions, evaluated in the kernel with the
    # arguments the tendency kernels give them.
    bottom_bcs = (ρθ = ρθ.boundary_conditions.bottom,
                  ρqᵗ = surface_moisture_density(ρqᵗ).boundary_conditions.bottom,
                  ρu = ρu.boundary_conditions.bottom,
                  ρv = ρv.boundary_conditions.bottom)
    bc_args = boundary_condition_args(model)

    launch_shoc_columns!(arch, grid, bottom_bcs, bc_args, Δt, closure, fields, ρ, pressure, ρθ, ρqᵗ, ρu, ρv, ρw, ρe)

    fill_halo_regions!((fields.increments.u, fields.increments.v))

    launch!(arch, grid, :xyz, _add_shoc_momentum_increments!, ρu, ρv, grid, ρ, fields.increments)

    return nothing
end

# The state SHOC carries between steps lives in the closure fields.
# TODO: checkpoint `shoc_ql`, `shoc_cldfrac`, `wthv_sec`, `tkh` and `tk`.
Oceananigans.prognostic_state(::SHOCFields) = nothing
Oceananigans.restore_prognostic_state!(fields::SHOCFields, ::Nothing) = fields
