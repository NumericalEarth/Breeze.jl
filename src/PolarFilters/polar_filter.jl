using Oceananigans.Grids: φnodes

#####
##### Abstract rolloff type
#####

"""
    AbstractRolloff

Supertype for polar filter spectral rolloff strategies.
"""
abstract type AbstractRolloff end

"""
$(TYPEDEF)

Algebraic 1-2-1 (Shapiro) zonal smoother applied `N(φ)` times per filtered
latitude row, where `N(φ)` is chosen so the smoother response at the
latitude's effective cutoff wavenumber is `≤ target_response`. This is the
default polar filter rolloff, matching NCAR CAM-FV's algebraic smoother.

Fields
======

$(TYPEDFIELDS)
"""
struct Shapiro121{FT} <: AbstractRolloff
    "Target response at the cutoff wavenumber (default 0.1 = 90 % damping)"
    target_response :: FT
    "Maximum passes per row (default 32)"
    max_passes :: Int
end

Shapiro121(; target_response=0.1, max_passes=32) = Shapiro121(Float64(target_response), max_passes)

#####
##### PolarFilter
#####

"""
$(TYPEDEF)

Algebraic polar filter for `LatitudeLongitudeGrid`.

Damps unresolvable high-wavenumber zonal modes poleward of a
`threshold_latitude` using a 1-2-1 Shapiro smoother applied a
latitude-dependent number of times. For each filtered row, the pass count
is chosen so the smoother response at
`k_max(φ) = floor(Nλ cos φ / cos φ_c)` is `≤ target_response`.

Construct with `PolarFilter(; threshold_latitude, rolloff)` to produce
a *skeleton* (no grid, no buffers). The skeleton is materialized into
the fully-typed working form by [`materialize_polar_filter`](@ref) when
`AtmosphereModel` is constructed.

Fields
======

$(TYPEDFIELDS)
"""
struct PolarFilter{FT, RO <: AbstractRolloff, IPV, NPV, BUF, G}
    "Latitude (degrees) above which filtering is applied"
    threshold_latitude :: FT
    "Spectral rolloff strategy"
    rolloff :: RO
    "Interior j-indices of rows needing filtering (both hemispheres)"
    filtered_indices :: IPV
    "Number of smoother passes per filtered row"
    passes_per_row :: NPV
    "Ping buffer of shape (Nλ, N_filtered, Nz)"
    buffer_a :: BUF
    "Pong buffer of shape (Nλ, N_filtered, Nz)"
    buffer_b :: BUF
    "The grid on which the filter operates"
    grid :: G
end

"""
$(TYPEDSIGNATURES)

Construct a `PolarFilter` skeleton (no grid, no buffers). Pass this to
`CompressibleDynamics(; polar_filter = PolarFilter(...))` and it will be
materialized when the model is built.

Keyword Arguments
=================

- `threshold_latitude`: latitude in degrees above which filtering is applied (default `60`).
- `rolloff`: spectral rolloff strategy. Default: [`Shapiro121`](@ref)`()`.
"""
function PolarFilter(; threshold_latitude=60, rolloff=Shapiro121())
    FT = Float64
    return PolarFilter(FT(threshold_latitude), rolloff,
                        nothing, nothing, nothing, nothing, nothing)
end

#####
##### Materialization
#####

"""
$(TYPEDSIGNATURES)

Materialize a `PolarFilter` skeleton into a fully-typed working form with
device-side buffers and per-row pass counts. Called during
`materialize_dynamics`.
"""
function materialize_polar_filter(grid, pf::PolarFilter)
    FT = eltype(grid)
    threshold_latitude = FT(pf.threshold_latitude)
    Nλ = grid.Nx
    Nz = grid.Nz

    ## Identify filtered rows (both hemispheres)
    φ_centers = φnodes(grid, Center())
    filtered_cpu = Int[]
    for j in 1:grid.Ny
        abs(φ_centers[j]) > threshold_latitude && push!(filtered_cpu, j)
    end
    N_filtered = length(filtered_cpu)
    N_filtered == 0 && return nothing

    ## Compute per-row pass count
    passes_cpu = _compute_passes_per_row(pf.rolloff, φ_centers, filtered_cpu, Nλ, threshold_latitude)

    ## Move to device
    arch = Oceananigans.architecture(grid)
    filtered_dev = Oceananigans.Architectures.on_architecture(arch, filtered_cpu)
    passes_dev = Oceananigans.Architectures.on_architecture(arch, passes_cpu)

    ## Allocate ping-pong buffers on device — sized at Nz+1 to accommodate ZFaceFields
    Nz_buf = Nz + 1
    buffer_a = Oceananigans.Architectures.on_architecture(arch, zeros(FT, Nλ, N_filtered, Nz_buf))
    buffer_b = Oceananigans.Architectures.on_architecture(arch, zeros(FT, Nλ, N_filtered, Nz_buf))

    return PolarFilter(threshold_latitude, pf.rolloff,
                        filtered_dev, passes_dev,
                        buffer_a, buffer_b, grid)
end

## No-op for no filter or non-LLG grids
materialize_polar_filter(grid, ::Nothing) = nothing

#####
##### Pass count computation
#####

function _compute_passes_per_row(rolloff::Shapiro121, φ_centers, filtered_indices, Nλ, threshold_latitude)
    cos_threshold = cosd(threshold_latitude)
    passes = zeros(Int, length(filtered_indices))

    for (row, j) in enumerate(filtered_indices)
        φ = abs(φ_centers[j])
        k_max = floor(Int, Nλ * cosd(φ) / cos_threshold)
        ## Clamp: if k_max ≥ Nλ/2, no filtering needed
        if k_max >= Nλ ÷ 2
            passes[row] = 0
        else
            ## cos²ᴺ(π k_max / Nλ) ≤ target_response
            ## N ≥ log(target_response) / (2 log(cos(π k_max / Nλ)))
            cosval = cos(π * k_max / Nλ)
            if cosval ≤ 0 || cosval ≥ 1
                passes[row] = 0
            else
                N = ceil(Int, log(rolloff.target_response) / (2 * log(cosval)))
                passes[row] = clamp(N, 0, rolloff.max_passes)
            end
        end
    end
    return passes
end

#####
##### Adapt
#####

Adapt.adapt_structure(to, pf::PolarFilter) =
    PolarFilter(pf.threshold_latitude,
                pf.rolloff,
                adapt(to, pf.filtered_indices),
                adapt(to, pf.passes_per_row),
                adapt(to, pf.buffer_a),
                adapt(to, pf.buffer_b),
                adapt(to, pf.grid))
