#####
##### Parameters of SHOC: the tunables of `shoc_init`, the physical constants SHOC is handed by
##### its host, and the fixed limits ("private module parameters") of E3SM shoc.F90.
#####
##### The field names are those of shoc.F90 (E3SM ba53243f), so that the column code can be read
##### against the Fortran line by line.
#####

"""
$(TYPEDEF)

Every number that the SHOC column physics depends on: the tunable coefficients, the physical
constants, and the fixed limits of E3SM's `shoc.F90`. Field names are those of `shoc.F90`. All
fields have the floating point type `FT`; none is converted inside the column code, so a
`SHOCParameters` may be rebuilt with any `Real` scalar type (for example a traced scalar during
automatic differentiation).

The defaults are the published defaults of `shoc.F90` at E3SM `ba53243f`, with the physical
constants of E3SM's `shr_const_mod`.

Tunable coefficients (non-dimensional unless noted)
===================================================

$(TYPEDFIELDS)
"""
struct SHOCParameters{FT}
    "Temperature variance tuning factor (default 1)"
    thl2tune :: FT
    "Moisture variance tuning factor (default 1)"
    qw2tune :: FT
    "Temperature-moisture covariance tuning factor (default 1)"
    qwthl2tune :: FT
    "Vertical velocity variance tuning factor (default 1)"
    w2tune :: FT
    "Length scale factor (default 0.5)"
    length_fac :: FT
    "Coefficient of the diagnostic third moment of vertical velocity (default 7)"
    c_diag_3rd_mom :: FT
    "Lowest value of the stability correction of the return-to-isotropy time scale (default 0.001)"
    lambda_low :: FT
    "Highest value of the stability correction (default 0.08)"
    lambda_high :: FT
    "Slope of the stability correction (default 2.65)"
    lambda_slope :: FT
    "Column stability at which the stability correction applies (default 0.02)"
    lambda_thresh :: FT
    "Eddy diffusivity coefficient for heat (default 0.1)"
    Ckh :: FT
    "Eddy diffusivity coefficient for momentum (default 0.1)"
    Ckm :: FT
    "Stable boundary layer diffusivity coefficient for heat (default 0.1)"
    Ckh_s :: FT
    "Stable boundary layer diffusivity coefficient for momentum (default 0.1)"
    Ckm_s :: FT

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
    "Friction-velocity factor in the bulk Richardson number (default 100)"
    fac :: FT
    "Critical bulk Richardson number (default 0.3)"
    ricr :: FT
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return `SHOCParameters{FT}` with the defaults of E3SM `shoc.F90`. Every field is a keyword
argument.
"""
function SHOCParameters(FT::DataType = Oceananigans.defaults.FloatType;
                        thl2tune = 1,
                        qw2tune = 1,
                        qwthl2tune = 1,
                        w2tune = 1,
                        length_fac = 0.5,
                        c_diag_3rd_mom = 7,
                        lambda_low = 0.001,
                        lambda_high = 0.08,
                        lambda_slope = 2.65,
                        lambda_thresh = 0.02,
                        Ckh = 0.1,
                        Ckm = 0.1,
                        Ckh_s = 0.1,
                        Ckm_s = 0.1,
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
                        tinyw = 1e-36,
                        fac = 100,
                        ricr = 0.3)

    return SHOCParameters{FT}(thl2tune, qw2tune, qwthl2tune, w2tune, length_fac, c_diag_3rd_mom,
                              lambda_low, lambda_high, lambda_slope, lambda_thresh,
                              Ckh, Ckm, Ckh_s, Ckm_s,
                              ggr, rgas, rv, cp, lcond, eps, vk,
                              basetemp, basepres, troppres, ustar_min, pblmaxp, w3clip,
                              maxlen, minlen, maxtke, mintke, tinyw, fac, ricr)
end

"""The names of the tunable, non-dimensional coefficients of [`SHOCParameters`](@ref)."""
const SHOC_TUNABLE_PARAMETER_NAMES = (:thl2tune, :qw2tune, :qwthl2tune, :w2tune, :length_fac, :c_diag_3rd_mom,
                                      :lambda_low, :lambda_high, :lambda_slope, :lambda_thresh,
                                      :Ckh, :Ckm, :Ckh_s, :Ckm_s)

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return a copy of `parameters` with the fields named by the keyword arguments replaced. The
replacements are promoted with the retained fields, so that passing scalars of another `Real`
type (e.g. traced scalars) returns `SHOCParameters` of the promoted type.
"""
function with_parameters(parameters::SHOCParameters; kw...)
    names = fieldnames(SHOCParameters)
    values = map(name -> haskey(kw, name) ? kw[name] : getfield(parameters, name), names)
    return SHOCParameters(promote(values...)...)
end

Base.eltype(::SHOCParameters{FT}) where FT = FT

function Base.show(io::IO, p::SHOCParameters{FT}) where FT
    print(io, "SHOCParameters{$FT}:")
    for name in SHOC_TUNABLE_PARAMETER_NAMES
        print(io, '\n', "    ", name, " = ", prettysummary(getfield(p, name)))
    end
end
