@inline function w_sedimentation_velocity(i, j, k, grid, microphysics::Microphysics1M, ::Val{:ρq_rai}, ρ, ρq_rai)
    ρᶜᶜᶠ = ℑzᵃᵃᶠ(i, j, k, grid, ρ)
    ρq_raiᶜᶜᶠ = ℑzᵃᵃᶠ(i, j, k, grid, ρq_rai)
    return CM1.terminal_velocity(microphysics.parameters.pr, microphysics.parameters.tv.rain, ρᶜᶜᶠ, ρq_raiᶜᶜᶠ/ ρᶜᶜᶠ)
end

@inline function w_sedimentation_velocity(i, j, k, grid, microphysics::Microphysics1M, ::Val{:ρq_sno}, ρ, ρq_sno)
    ρᶜᶜᶠ = ℑzᵃᵃᶠ(i, j, k, grid, ρ)
    ρq_snoᶜᶜᶠ = ℑzᵃᵃᶠ(i, j, k, grid, ρq_sno)
    return CM1.terminal_velocity(microphysics.parameters.ps, microphysics.parameters.tv.snow, ρᶜᶜᶠ, ρq_snoᶜᶜᶠ/ ρᶜᶜᶠ)
end
