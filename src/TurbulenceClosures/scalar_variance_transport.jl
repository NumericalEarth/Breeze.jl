"""
$(TYPEDEF)

The vertical transport of the scalar second moments `θˡⁱ′²`, `qᵗ′²` and `θˡⁱ′qᵗ′` of a [`DoubleGaussian`](@ref),
diffused with `ℂᵛᵗ Kᶜ` in a steady budget relaxed to the local diagnosis.
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

@inline transport_scalar_variances!(::Nothing, m, cl, i, j, grid, d, closure, closure_fields, g, θ, e) = nothing

@inline function transport_scalar_variances!(t::ScalarVarianceTransport, m, cl, i, j, grid, d, closure, closure_fields, g, θ, e)
    Nz = size(grid, 3)
    FT = grid_float_type(grid)
    work = m.variance_transport_work
    Kᶜ = closure_fields.Kᶜ
    @inbounds begin
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
