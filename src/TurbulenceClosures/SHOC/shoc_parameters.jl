#####
##### Parameters of SHOC: the calibratable empirical coefficients (written ℂ, as elsewhere in
##### Breeze), the physical constants SHOC is handed by its host, and the reference values and
##### numerical safeguards ("private module parameters") of E3SM shoc.F90.
#####

"""
$(TYPEDEF)

Every number that the SHOC column physics depends on. All fields have one scalar type `FT` and
none is converted inside the column code, so `SHOCParameters` may be rebuilt with any `Real`
scalar type (for example a traced scalar during automatic differentiation).

Calibratable empirical coefficients are written `ℂˣᵢ`, the `i`-th coefficient of relation `x`;
physical constants, reference values and numerical safeguards keep the names of E3SM's
`shoc.F90`. The defaults are the published values of `shoc.F90` at E3SM `ba53243f`, and with them
the column code reproduces `shoc.F90` to round-off. The first fourteen coefficients below
(`ℂᵘ` to `ℂᵝ₄` excepting `ℂˡ₂` to `ℂˡ₅`) are the run-time tunables of `shoc_init`; the others are
literals in `shoc.F90`.

| coefficient | `shoc.F90` | default | meaning |
|:------------|:-----------|:--------|:--------|
| `ℂᵘ`  | `Ckm`            | 0.1    | eddy diffusivity of momentum, `tk = ℂᵘ τ e` with `τ` the return-to-isotropy time scale |
| `ℂᶜ`  | `Ckh`            | 0.1    | eddy diffusivity of heat and scalars, `tkh = ℂᶜ τ e` |
| `ℂᵘₛ` | `Ckm_s`          | 0.1    | momentum diffusivity of the stable-boundary-layer fallback, `tk = ℂᵘₛ ℓ² √(shear term)` |
| `ℂᶜₛ` | `Ckh_s`          | 0.1    | heat diffusivity of the stable-boundary-layer fallback |
| `ℂˡ₁` | `length_fac`     | 0.5    | divides the mixing length |
| `ℂˡ₂` | literal `2.8284` | 2.8284 | prefactor of the mixing length (`2√2`, truncated as in the Fortran) |
| `ℂˡ₃` | literal `0.01`   | 0.01   | weight of the stratification term `N² / e` in the mixing length |
| `ℂˡ₄` | literal `0.1`    | 0.1    | the asymptotic length `l∞` as a fraction of the TKE-weighted mean height |
| `ℂˡ₅` | `tscale`         | 400    | eddy turnover time scale of the mixing length, **in seconds** (the one dimensional coefficient) |
| `ℂᵛ₁` | `thl2tune`       | 1      | scales the variance of liquid water potential temperature |
| `ℂᵛ₂` | `qw2tune`        | 1      | scales the variance of total water |
| `ℂᵛ₃` | `qwthl2tune`     | 1      | scales the covariance of total water and liquid water potential temperature |
| `ℂʷ₁` | `w2tune`         | 1      | scales the vertical velocity variance, `w'w' = ℂʷ₁ (2/3) e` |
| `ℂʷ₂` | `c_diag_3rd_mom` | 7      | the coefficient `c` of the diagnostic third moment of vertical velocity (Canuto et al. 2001) |
| `ℂᵝ₁` | `lambda_low`     | 0.001  | lowest value of the stability damping `λ` of the return-to-isotropy time scale |
| `ℂᵝ₂` | `lambda_high`    | 0.08   | highest value of `λ` |
| `ℂᵝ₃` | `lambda_slope`   | 2.65   | slope of `λ` in the integrated column stability |
| `ℂᵝ₄` | `lambda_thresh`  | 0.02   | integrated column stability at which `λ` starts to rise |
| `ℂᴰ₁` | `Ck` (`adv_sgs_tke`) | 0.1  | dissipation: `Ce = ℂᴰ₁³ / ℂᴰ₂⁴`, `ε = (0.19 + 0.51) / 0.7 Ce e^{3/2} / ℓ` |
| `ℂᴰ₂` | `Cs` (`adv_sgs_tke`) | 0.15 | dissipation, see `ℂᴰ₁` |
| `ℂᴾ`  | `Ck_sh`          | 0.1    | shear production term, `ℂᴾ |∂z u|²` |
| `ℂᶠ`  | `a_const`        | 1.8    | surface variances and covariance, `θˡ'² = 0.4 ℂᶠ (w'θˡ' / u_f)²` at the lowest interface |
| `ℂᵍ₁` | `w2_1`, `w2_2` (literal `0.4`) | 0.4 | within-plume fraction of the vertical velocity variance in the assumed PDF |
| `ℂᵍ₂` | literal `1.2`    | 1.2    | skewness of total water relative to that of vertical velocity |
| `ℂᵍ₃` | literal `0.2`    | 0.2    | plume separation below which total water is unskewed |
| `ℂᵍ₄` | literal `0.4`    | 0.4    | plume separation above which total water takes the full skewness `ℂᵍ₂ Skew_w` |
| `ℂᴴ₁` | `ricr`           | 0.3    | critical bulk Richardson number of the boundary layer top |
| `ℂᴴ₂` | `fac`            | 100    | friction velocity factor in the squared wind of the bulk Richardson number |
| `ℂᴴ₃` | `fak`            | 8.5    | surface temperature excess of convective thermals |
| `ℂᴴ₄` | `betam`          | 15     | wind-gradient constant of the temperature excess (enters as `ℂᴴ₄ × 0.1`, with `sffrac = 0.1`) |

Between `ℂᵍ₃` and `ℂᵍ₄` the skewness of total water ramps linearly; the Fortran writes the width
of the ramp as the literal `0.2`, which is taken here as `ℂᵍ₄ - ℂᵍ₃` so that the ramp stays
continuous when the two are changed.

$(TYPEDFIELDS)
"""
struct SHOCParameters{FT}
    "Eddy diffusivity coefficient for momentum (`Ckm`)"
    ℂᵘ :: FT
    "Eddy diffusivity coefficient for heat and scalars (`Ckh`)"
    ℂᶜ :: FT
    "Stable boundary layer diffusivity coefficient for momentum (`Ckm_s`)"
    ℂᵘₛ :: FT
    "Stable boundary layer diffusivity coefficient for heat (`Ckh_s`)"
    ℂᶜₛ :: FT
    "Mixing length divisor (`length_fac`)"
    ℂˡ₁ :: FT
    "Mixing length prefactor (literal `2.8284`)"
    ℂˡ₂ :: FT
    "Weight of the stratification term of the mixing length (literal `0.01`)"
    ℂˡ₃ :: FT
    "Asymptotic length as a fraction of the TKE-weighted mean height (literal `0.1`)"
    ℂˡ₄ :: FT
    "Eddy turnover time scale of the mixing length, s (`tscale`)"
    ℂˡ₅ :: FT
    "Temperature variance factor (`thl2tune`)"
    ℂᵛ₁ :: FT
    "Moisture variance factor (`qw2tune`)"
    ℂᵛ₂ :: FT
    "Temperature-moisture covariance factor (`qwthl2tune`)"
    ℂᵛ₃ :: FT
    "Vertical velocity variance factor (`w2tune`)"
    ℂʷ₁ :: FT
    "Coefficient of the diagnostic third moment of vertical velocity (`c_diag_3rd_mom`)"
    ℂʷ₂ :: FT
    "Lowest stability damping of the return-to-isotropy time scale (`lambda_low`)"
    ℂᵝ₁ :: FT
    "Highest stability damping (`lambda_high`)"
    ℂᵝ₂ :: FT
    "Slope of the stability damping (`lambda_slope`)"
    ℂᵝ₃ :: FT
    "Column stability at which the stability damping rises (`lambda_thresh`)"
    ℂᵝ₄ :: FT
    "Dissipation coefficient (`Ck` of `adv_sgs_tke`)"
    ℂᴰ₁ :: FT
    "Dissipation coefficient (`Cs` of `adv_sgs_tke`)"
    ℂᴰ₂ :: FT
    "Shear production coefficient (`Ck_sh`)"
    ℂᴾ :: FT
    "Surface variance coefficient (`a_const`)"
    ℂᶠ :: FT
    "Within-plume fraction of the vertical velocity variance in the assumed PDF (literal `0.4`)"
    ℂᵍ₁ :: FT
    "Skewness of total water relative to that of vertical velocity (literal `1.2`)"
    ℂᵍ₂ :: FT
    "Plume separation below which total water is unskewed (literal `0.2`)"
    ℂᵍ₃ :: FT
    "Plume separation above which total water is fully skewed (literal `0.4`)"
    ℂᵍ₄ :: FT
    "Critical bulk Richardson number (`ricr`)"
    ℂᴴ₁ :: FT
    "Friction-velocity factor in the bulk Richardson number (`fac`)"
    ℂᴴ₂ :: FT
    "Surface temperature excess constant (`fak`)"
    ℂᴴ₃ :: FT
    "Wind-gradient constant of the surface temperature excess (`betam`)"
    ℂᴴ₄ :: FT

    "Gravitational acceleration, m s⁻² (default 9.80616)"
    ggr :: FT
    "Dry air gas constant, J kg⁻¹ K⁻¹ (default 287.042)"
    rgas :: FT
    "Water vapor gas constant, J kg⁻¹ K⁻¹ (default 461.505)"
    rv :: FT
    "Specific heat of dry air, J kg⁻¹ K⁻¹ (default 1004.64)"
    cp :: FT
    "Latent heat of vaporization, J kg⁻¹ (default 2.501e6)"
    lcond :: FT
    "`rv / rgas - 1` (E3SM `zvir`)"
    eps :: FT
    "Von Kármán constant (default 0.4)"
    vk :: FT

    "Reference temperature, K (default 300)"
    basetemp :: FT
    "Reference pressure, Pa (default 1e5)"
    basepres :: FT
    "Pressure above which the column stability is integrated, Pa (default 8e4)"
    troppres :: FT
    "Minimum surface friction velocity, m s⁻¹ (default 0.01)"
    ustar_min :: FT
    "Lowest pressure at which the boundary layer top may be diagnosed, Pa (default 4e4)"
    pblmaxp :: FT
    "Clipping factor for the third moment of vertical velocity (default 1.2)"
    w3clip :: FT
    "Maximum mixing length, m (default 20000)"
    maxlen :: FT
    "Minimum mixing length, m (default 20)"
    minlen :: FT
    "Maximum turbulent kinetic energy, m² s⁻² (default 50)"
    maxtke :: FT
    "Minimum turbulent kinetic energy, m² s⁻² (default 4e-4)"
    mintke :: FT
    "Lower bound on the squared wind in the bulk Richardson number, m² s⁻² (default 1e-36)"
    tinyw :: FT
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return `SHOCParameters{FT}` with the defaults of E3SM `shoc.F90`. Every field is a keyword
argument.
"""
function SHOCParameters(FT::DataType = Oceananigans.defaults.FloatType;
                        ℂᵘ = 0.1,
                        ℂᶜ = 0.1,
                        ℂᵘₛ = 0.1,
                        ℂᶜₛ = 0.1,
                        ℂˡ₁ = 0.5,
                        ℂˡ₂ = 2.8284,
                        ℂˡ₃ = 0.01,
                        ℂˡ₄ = 0.1,
                        ℂˡ₅ = 400,
                        ℂᵛ₁ = 1,
                        ℂᵛ₂ = 1,
                        ℂᵛ₃ = 1,
                        ℂʷ₁ = 1,
                        ℂʷ₂ = 7,
                        ℂᵝ₁ = 0.001,
                        ℂᵝ₂ = 0.08,
                        ℂᵝ₃ = 2.65,
                        ℂᵝ₄ = 0.02,
                        ℂᴰ₁ = 0.1,
                        ℂᴰ₂ = 0.15,
                        ℂᴾ = 0.1,
                        ℂᶠ = 1.8,
                        ℂᵍ₁ = 0.4,
                        ℂᵍ₂ = 1.2,
                        ℂᵍ₃ = 0.2,
                        ℂᵍ₄ = 0.4,
                        ℂᴴ₁ = 0.3,
                        ℂᴴ₂ = 100,
                        ℂᴴ₃ = 8.5,
                        ℂᴴ₄ = 15,
                        ggr = 9.80616,
                        rgas = 287.042,
                        rv = 461.505,
                        cp = 1004.64,
                        lcond = 2.501e6,
                        eps = rv / rgas - 1,
                        vk = 0.4,
                        basetemp = 300,
                        basepres = 100000,
                        troppres = 80000,
                        ustar_min = 0.01,
                        pblmaxp = 4e4,
                        w3clip = 1.2,
                        maxlen = 20000,
                        minlen = 20,
                        maxtke = 50,
                        mintke = 0.0004,
                        tinyw = 1e-36)

    return SHOCParameters{FT}(ℂᵘ, ℂᶜ, ℂᵘₛ, ℂᶜₛ, ℂˡ₁, ℂˡ₂, ℂˡ₃, ℂˡ₄, ℂˡ₅, ℂᵛ₁, ℂᵛ₂, ℂᵛ₃, ℂʷ₁, ℂʷ₂,
                              ℂᵝ₁, ℂᵝ₂, ℂᵝ₃, ℂᵝ₄, ℂᴰ₁, ℂᴰ₂, ℂᴾ, ℂᶠ, ℂᵍ₁, ℂᵍ₂, ℂᵍ₃, ℂᵍ₄, ℂᴴ₁, ℂᴴ₂, ℂᴴ₃, ℂᴴ₄,
                              ggr, rgas, rv, cp, lcond, eps, vk,
                              basetemp, basepres, troppres, ustar_min, pblmaxp, w3clip,
                              maxlen, minlen, maxtke, mintke, tinyw)
end

"""
The calibratable empirical coefficients of [`SHOCParameters`](@ref), each with the name it has in
E3SM's `shoc.F90` (`nothing` where `shoc.F90` has only a literal).
"""
const SHOC_COEFFICIENT_NAMES = (ℂᵘ = :Ckm, ℂᶜ = :Ckh, ℂᵘₛ = :Ckm_s, ℂᶜₛ = :Ckh_s,
                                ℂˡ₁ = :length_fac, ℂˡ₂ = nothing, ℂˡ₃ = nothing, ℂˡ₄ = nothing, ℂˡ₅ = :tscale,
                                ℂᵛ₁ = :thl2tune, ℂᵛ₂ = :qw2tune, ℂᵛ₃ = :qwthl2tune,
                                ℂʷ₁ = :w2tune, ℂʷ₂ = :c_diag_3rd_mom,
                                ℂᵝ₁ = :lambda_low, ℂᵝ₂ = :lambda_high, ℂᵝ₃ = :lambda_slope, ℂᵝ₄ = :lambda_thresh,
                                ℂᴰ₁ = :Ck, ℂᴰ₂ = :Cs, ℂᴾ = :Ck_sh, ℂᶠ = :a_const,
                                ℂᵍ₁ = :w2_1, ℂᵍ₂ = nothing, ℂᵍ₃ = nothing, ℂᵍ₄ = nothing,
                                ℂᴴ₁ = :ricr, ℂᴴ₂ = :fac, ℂᴴ₃ = :fak, ℂᴴ₄ = :betam)

"""
The run-time tunables of `shoc_init` in E3SM's `shoc.F90`, by their coefficient name in
[`SHOCParameters`](@ref).
"""
const SHOC_INIT_TUNABLE_NAMES = (ℂᵛ₁ = :thl2tune, ℂᵛ₂ = :qw2tune, ℂᵛ₃ = :qwthl2tune, ℂʷ₁ = :w2tune,
                                 ℂˡ₁ = :length_fac, ℂʷ₂ = :c_diag_3rd_mom,
                                 ℂᵝ₁ = :lambda_low, ℂᵝ₂ = :lambda_high, ℂᵝ₃ = :lambda_slope, ℂᵝ₄ = :lambda_thresh,
                                 ℂᶜ = :Ckh, ℂᵘ = :Ckm, ℂᶜₛ = :Ckh_s, ℂᵘₛ = :Ckm_s)

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return a copy of `parameters` with the fields named by the keyword arguments replaced. The
replacements are promoted with the retained fields, so that passing scalars of another `Real`
type (e.g. traced scalars) returns `SHOCParameters` of the promoted type.
"""
function with_parameters(parameters::SHOCParameters; kw...)
    names = fieldnames(SHOCParameters)
    all(name -> name ∈ names, keys(kw)) ||
        throw(ArgumentError("with_parameters: $(setdiff(keys(kw), names)) are not fields of SHOCParameters."))
    values = map(name -> haskey(kw, name) ? kw[name] : getfield(parameters, name), names)
    return SHOCParameters(promote(values...)...)
end

Base.eltype(::SHOCParameters{FT}) where FT = FT
Base.summary(::SHOCParameters{FT}) where FT = "SHOCParameters{$FT}"

function Base.show(io::IO, p::SHOCParameters)
    print(io, summary(p), ":")
    for name in keys(SHOC_COEFFICIENT_NAMES)
        print(io, '\n', "    ", name, " = ", prettysummary(getfield(p, name)))
    end
end
