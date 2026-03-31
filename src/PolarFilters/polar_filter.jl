using Oceananigans.Grids: φnodes

## FFTW is a transitive dependency loaded by Oceananigans.
## Access it via Base.loaded_modules to avoid a direct [deps] entry
## while satisfying ExplicitImports qualified-access checks.
const _FFTW_PKGID = Base.PkgId(Base.UUID("7a1cc6ca-52ef-59f5-83cd-3a7055c09341"), "FFTW")
_fftw() = Base.loaded_modules[_FFTW_PKGID]
_plan_rfft(args...; kw...) = _fftw().plan_rfft(args...; kw...)
_plan_brfft(args...; kw...) = _fftw().plan_brfft(args...; kw...)

"""
    AbstractFilterMode

Supertype for polar filter spectral truncation strategies.
"""
abstract type AbstractFilterMode end

"""
$(TYPEDEF)

Truncate wavenumbers above `k_max(φ)` to zero (sharp cutoff).
This is the approach used by WRF but may produce Gibbs ringing.
"""
struct SharpTruncation <: AbstractFilterMode end

"""
$(TYPEDEF)

Apply an exponential rolloff above `k_max(φ)`:
```math
w(k) = \\exp\\!\\left(-\\left(\\frac{k - k_{\\max}}{N_k - k_{\\max}}\\right)^p\\right)
```
where `p` is the `order` (higher = steeper). Default `order = 8`.
"""
struct ExponentialRolloff{FT} <: AbstractFilterMode
    order :: FT
end

ExponentialRolloff(order::Integer) = ExponentialRolloff(Float64(order))

"""
$(TYPEDEF)

FFT-based polar filter for `LatitudeLongitudeGrid`.

Removes unresolvable high-wavenumber zonal modes poleward of a
`threshold_latitude`. For each filtered latitude row, wavenumbers above
`k_max(φ) = floor(Nλ cos φ / cos φ_c)` are damped or zeroed,
where `φ_c` is the threshold latitude.

This follows the WRF polar filtering approach
([Skamarock et al., 2008](@cite Skamarock2008Description), Section 2.5).

Fields
======

$(TYPEDFIELDS)
"""
struct PolarFilter{FT, FM <: AbstractFilterMode, FP, IP, SM, BF, BC, FI, G}
    "Latitude (degrees) above which filtering is applied"
    threshold_latitude :: FT
    "Spectral truncation strategy"
    filter_mode :: FM
    "Pre-planned forward rfft (batched along dim 1)"
    forward_plan :: FP
    "Pre-planned inverse brfft (batched along dim 1)"
    inverse_plan :: IP
    "Spectral mask array of shape (Nk, N_filtered_rows)"
    spectral_mask :: SM
    "Real-valued work buffer of shape (Nλ, N_filtered_rows * Nz)"
    buffer_real :: BF
    "Complex-valued work buffer of shape (Nk, N_filtered_rows * Nz)"
    buffer_complex :: BC
    "Interior j-indices of rows needing filtering (both hemispheres)"
    filtered_indices :: FI
    "The grid on which the filter operates"
    grid :: G
end

"""
$(TYPEDSIGNATURES)

Construct a `PolarFilter` for `grid`.

Keyword Arguments
=================

- `threshold_latitude`: latitude in degrees above which filtering is applied (default `60`).
- `filter_mode`: spectral truncation strategy. Either [`SharpTruncation`](@ref) or
  [`ExponentialRolloff`](@ref)`(order)` (default `ExponentialRolloff(8)`).
"""
function PolarFilter(grid;
                     threshold_latitude = 60,
                     filter_mode = ExponentialRolloff(8))

    FT = eltype(grid)
    threshold_latitude = FT(threshold_latitude)

    Nλ = grid.Nx
    Nz = grid.Nz
    Nk = Nλ ÷ 2 + 1

    ## Identify filtered rows (both hemispheres)
    φ_centers = φnodes(grid, Center())
    filtered_indices = Int[]
    for j in 1:grid.Ny
        abs(φ_centers[j]) > threshold_latitude && push!(filtered_indices, j)
    end
    N_filtered = length(filtered_indices)

    ## Build spectral mask: (Nk, N_filtered)
    cos_threshold = cosd(threshold_latitude)
    spectral_mask = ones(FT, Nk, N_filtered)

    for (row, j) in enumerate(filtered_indices)
        φ = abs(φ_centers[j])
        k_max = max(1, floor(Int, Nλ * cosd(φ) / cos_threshold))
        _fill_spectral_mask!(view(spectral_mask, :, row), k_max, Nk, filter_mode)
    end

    ## Allocate work buffers
    N_batch = N_filtered * Nz
    buffer_real = zeros(FT, Nλ, N_batch)
    buffer_complex = zeros(Complex{FT}, Nk, N_batch)

    ## Create FFTW plans (batched 1D transforms along dim 1)
    forward_plan = _plan_rfft(buffer_real, 1)
    inverse_plan = _plan_brfft(buffer_complex, Nλ, 1)

    return PolarFilter(threshold_latitude, filter_mode,
                       forward_plan, inverse_plan,
                       spectral_mask, buffer_real, buffer_complex,
                       filtered_indices, grid)
end

## Spectral mask construction

function _fill_spectral_mask!(mask, k_max, Nk, ::SharpTruncation)
    for k in 1:Nk
        mask[k] = ifelse(k <= k_max, 1, 0)
    end
    return nothing
end

function _fill_spectral_mask!(mask, k_max, Nk, mode::ExponentialRolloff)
    p = mode.order
    ## α chosen so the Nyquist wavenumber (η = 1) is attenuated to machine precision:
    ##   mask(Nk) = exp(-α · 1^p) = exp(-α) ≈ ε  →  α = -log(ε)
    α = -log(eps(eltype(mask)))
    width = max(1, Nk - k_max)
    for k in 1:Nk
        if k <= k_max
            mask[k] = 1
        else
            η = (k - k_max) / width
            mask[k] = exp(-α * η^p)
        end
    end
    return nothing
end
