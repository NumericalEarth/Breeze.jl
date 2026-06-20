#####
##### Fused time-averaged-velocity accumulation (performance optimization).
#####
##### Replaces the three per-substep broadcasts
#####   time_averaged_velocities.{u,v,w} .+= momentum_perturbation.{u,v,w}
##### with a single elementwise kernel over the (linear) parent arrays: 3 kernel launches → 1,
##### with the momentum read in one pass. Bit-exact with the broadcasts (same full-parent range,
##### halos included). The time-averaged velocities are diagnostic (not prognostic).
#####
##### Measured +7.5% on the H100 baroclinic-wave throughput benchmark (NWPShootout), with the
##### prognostic state bit-identical to the broadcast path. Toggled by `FUSE_ACCUM=1` (off by
##### default) in `accumulate_momentum_perturbations!`.

using KernelAbstractions: @kernel, @index, get_backend

@kernel function _accumulate_momentum_fused!(tu, tv, tw, mu, mv, mw)
    I = @index(Global, Linear)
    @inbounds begin
        tu[I] += mu[I]
        tv[I] += mv[I]
        tw[I] += mw[I]
    end
end

@inline function accumulate_momentum_fused!(substepper)
    tu = parent(substepper.time_averaged_velocities.u)
    tv = parent(substepper.time_averaged_velocities.v)
    tw = parent(substepper.time_averaged_velocities.w)
    mu = parent(substepper.momentum_perturbation.u)
    mv = parent(substepper.momentum_perturbation.v)
    mw = parent(substepper.momentum_perturbation.w)
    _accumulate_momentum_fused!(get_backend(tu), 256)(tu, tv, tw, mu, mv, mw; ndrange = length(tu))
    return nothing
end
