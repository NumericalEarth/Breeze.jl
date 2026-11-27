#####
##### The scalar second moments θˡⁱ′², qᵗ′² and θˡⁱ′qᵗ′ from their steady budget with vertical transport
#####
##### The local diagnosis of the distribution, X = ℂᵛ τ Kᶜ ∂z a ∂z b, is the steady budget without transport,
##### X / τᵈ = 2 ℂᵛ Kᶜ ∂z a ∂z b with the dissipation time τᵈ = τ / 2 (`DissipationTimeScale`: τ = 2 τᵠ). With
##### the transport, at each interior face,
#####
#####     X / τᵈ - ∂z (Kᵛ ∂z X) + ∂z T = 2 ℂᵛ Kᶜ ∂z a ∂z b,    Kᵛ = ℂᵛᵗ Kᶜ,
#####
##### that is X - τᵈ ∂z (Kᵛ ∂z X) = X_local - τᵈ ∂z T, with no flux through the bottom and top faces. The three
##### moments share the matrix: one Thomas elimination per column, three right-hand sides. T is the
##### third moment w′a′b′, or zero.
#####
##### The matrix has a positive diagonal, non-positive off-diagonals and is strictly diagonally dominant
##### by rows: it is an M-matrix, its inverse G is non-negative, so θˡⁱ′² and qᵗ′² are non-negative where
##### their productions are (T = 0). The local moments have correlation ±1 (θˡⁱ′qᵗ′² = θˡⁱ′² qᵗ′²); with
##### G ≥ 0, Cauchy–Schwarz on the sums Σₖ Gⱼₖ sₖ ∂θₖ ∂qₖ gives (θˡⁱ′qᵗ′)² ≤ θˡⁱ′² qᵗ′² after the solve,
##### with the correlation below 1 where gradients of different ratio mix. With T ≠ 0 neither holds by
##### construction.
#####

"""
$(TYPEDEF)

The vertical transport of the scalar second moments `θˡⁱ′²`, `qᵗ′²` and `θˡⁱ′qᵗ′` of a
[`DoubleGaussian`](@ref) (`variance_transport`). At each interior face the moments solve the
steady budget

    X / τᵈ - ∂z (ℂᵛᵗ Kᶜ ∂z X) + ∂z T = X_local / τᵈ,    τᵈ = τ / 2,

with `X_local` the down-gradient diagnosis, `τ` the moment time scale and no flux of `X` through
the bottom and top faces; one tridiagonal elimination per column serves the three moments.
`T` is zero, or with `component_third_moment = true` the third moment `w′a′b′` of the components
of the double-Gaussian PDF, whose means carry the fluxes,

    w′a′b′ = w′³ w′a′ w′b′ / ((1 - ℂᵍ(Sk)) w′²)²,

evaluated at the centers with the third moment `w′³` of the local moments (one fixed-point pass).
It transports variance against the gradient where the skewness and the fluxes are positive.

$(TYPEDFIELDS)
"""
struct ScalarVarianceTransport{FT}
    ℂᵛᵗ :: FT # ratio of the diffusivity of the scalar second moments to Kᶜ
end

Adapt.@adapt_structure ScalarVarianceTransport

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return a [`ScalarVarianceTransport`](@ref).
"""
function ScalarVarianceTransport(FT::DataType = Oceananigans.defaults.FloatType; ℂᵛᵗ = 1)
    ℂᵛᵗ ≥ 0 || throw(ArgumentError("ℂᵛᵗ must be non-negative, got $ℂᵛᵗ"))
    return ScalarVarianceTransport(convert(FT, ℂᵛᵗ))
end

convert_eltype(::Type{FT}, t::ScalarVarianceTransport) where FT = ScalarVarianceTransport(convert(FT, t.ℂᵛᵗ))

scalar_variance_transport_fields(grid, ::Nothing) = NamedTuple()
scalar_variance_transport_fields(grid, ::ScalarVarianceTransport) = (variance_transport_work = ZFaceField(grid),)

# The third moments w′θˡⁱ′², w′qᵗ′², w′θˡⁱ′qᵗ′ at the center k
@inline transport_scalar_variances!(::Nothing, m, cl, i, j, grid, d, closure, closure_fields, g, θ, e) = nothing

@inline function transport_scalar_variances!(t::ScalarVarianceTransport, m, cl, i, j, grid, d, closure, closure_fields, g, θ, e)
    Nz = size(grid, 3)
    FT = grid_float_type(grid)
    work = m.variance_transport_work
    Kᶜ = closure_fields.Kᶜ
    @inbounds begin
        # Thomas algorithm over the faces 2 … Nz; the rows are diagonally dominant
        c′ = zero(FT)
        dθ = zero(FT)
        dq = zero(FT)
        dθq = zero(FT)
        a⁻ = zero(FT) # Kᵛ / Δz at the center below the face (zero below face 2: no flux through the bottom)
        for k in 2:Nz
            a = zero(FT)
            if k < Nz # the center k, between the faces k and k + 1; none above face Nz: no flux through the top
                a = t.ℂᵛᵗ * (Kᶜ[i, j, k] + Kᶜ[i, j, k+1]) / 2 / Δzᶜᶜᶜ(i, j, k, grid)
            end
            r = m.τ[i, j, k] / 2 / Δzᶜᶜᶠ(i, j, k, grid)
            lower = -r * a⁻
            upper = -r * a
            pivot = 1 + r * (a + a⁻) - lower * c′
            dθ = (m.θˡⁱ′²[i, j, k] - lower * dθ) / pivot
            dq = (m.qᵗ′²[i, j, k] - lower * dq) / pivot
            dθq = (m.θˡⁱ′qᵗ′[i, j, k] - lower * dθq) / pivot
            c′ = upper / pivot
            work[i, j, k] = c′
            m.θˡⁱ′²[i, j, k] = dθ
            m.qᵗ′²[i, j, k] = dq
            m.θˡⁱ′qᵗ′[i, j, k] = dθq
            a⁻ = a
        end
        for k in Nz-1:-1:2
            m.θˡⁱ′²[i, j, k] -= work[i, j, k] * m.θˡⁱ′²[i, j, k+1]
            m.qᵗ′²[i, j, k] -= work[i, j, k] * m.qᵗ′²[i, j, k+1]
            m.θˡⁱ′qᵗ′[i, j, k] -= work[i, j, k] * m.θˡⁱ′qᵗ′[i, j, k+1]
        end
    end
    return nothing
end
