# Reduce Breeze (u,v) output to azimuthal-mean TANGENTIAL-wind diagnostics for the Willson et al.
# (2024) Fig. 5/7/8 comparison, REPLICATING TempestExtremes' radial_wind_profile approach so the
# Breeze side is processed like the DCMIP2016 ensemble:
#   * stable sub-grid storm center — parabolic refinement of the surface-pressure minimum (removes
#     the ±1-cell grid-snap jitter of a plain argmin that inflated the per-timestep MWS);
#   * azimuthal average by RING INTERPOLATION — M points per radius, bilinearly interpolated from
#     the lat-lon grid and projected onto the cyclonic-tangential direction, on a 0.25°-great-circle
#     radial grid matching the published profiles (rather than binning native cells by radius).
#
# Writes (into postproc/): breeze_fields_rz_010.csv (run,r_km,z_km,vt_ms; days 4-10 mean) -> Fig 8
#         breeze_profiles_010.csv (run,r_km,wind1km_ms,psfc_hPa; days 4-10 mean) -> Fig 7
#         breeze_windpressure_010.csv (run,day,msp_hPa,mws_ms; per fields frame) -> Fig 5
# These feed plot_willson_comparison.jl. Usage:
#   julia --project=<env> extract_willson_comparison_data.jl <weno5_dir> <weno9_dir>
using Oceananigans
using Oceananigans.Grids: λnodes, φnodes, znodes
using Oceananigans.OutputReaders: FieldTimeSeries, OnDisk
using Printf

const A = 6371220.0
const DEG = π / 180
const DR = 0.25 * DEG * A        # 0.25° great-circle radial step (≈ 27.8 km), matches published grid
const M = 72                     # azimuthal samples per ring

# bilinear interpolation of F (Nx×Ny) at geographic (λq°, φq°); longitude periodic, latitude clamped
@inline function bilin(F, λq, φq, λ1, dλ, φ1, dφ, Nx, Ny)
    fi = mod(λq - λ1, 360.0) / dλ
    i0 = floor(Int, fi); ti = fi - i0
    ia = mod(i0, Nx) + 1; ib = mod(i0 + 1, Nx) + 1
    fj = clamp((φq - φ1) / dφ, 0.0, Float64(Ny - 1))
    j0 = floor(Int, fj); tj = fj - j0
    ja = j0 + 1; jb = min(j0 + 2, Ny)
    @inbounds f00 = F[ia,ja]; @inbounds f10 = F[ib,ja]
    @inbounds f01 = F[ia,jb]; @inbounds f11 = F[ib,jb]
    return (1-ti)*(1-tj)*f00 + ti*(1-tj)*f10 + (1-ti)*tj*f01 + ti*tj*f11
end

# sub-grid storm center via parabolic refinement of the pressure minimum
function subgrid_center(pp, λ, φ, dλ, dφ, Nx, Ny)
    i0, j0 = Tuple(argmin(pp))
    λe = λ[i0]; φe = φ[j0]
    if 1 < i0 < Nx
        a, b, c = pp[i0-1,j0], pp[i0,j0], pp[i0+1,j0]; den = a - 2b + c
        den != 0 && (λe += clamp(0.5*(a-c)/den, -0.5, 0.5) * dλ)
    end
    if 1 < j0 < Ny
        a, b, c = pp[i0,j0-1], pp[i0,j0], pp[i0,j0+1]; den = a - 2b + c
        den != 0 && (φe += clamp(0.5*(a-c)/den, -0.5, 0.5) * dφ)
    end
    return λe, φe
end

# ring points (λq,φq per k,m) + per-azimuth tangential projection coeffs, for K radii about (λe,φe)
function ring_geometry(λe, φe, K)
    λq = zeros(K, M); φq = zeros(K, M); ct = zeros(M); st = zeros(M)
    sφe = sin(φe*DEG); cφe = cos(φe*DEG)
    for m in 1:M
        θ = 2π*(m-1)/M; ct[m] = cos(θ); st[m] = sin(θ)
    end
    for k in 1:K
        δ = ((k-1)*DR)/A; sδ = sin(δ); cδ = cos(δ)   # k=1 → r=0 (the eye center)
        for m in 1:M
            θ = 2π*(m-1)/M
            φqr = asin(sφe*cδ + cφe*sδ*cos(θ))
            λqr = λe*DEG + atan(sin(θ)*sδ*cφe, cδ - sφe*sin(φqr))
            φq[k,m] = φqr/DEG; λq[k,m] = mod(λqr/DEG, 360.0)
        end
    end
    return λq, φq, ct, st
end

# azimuthal-mean cyclonic-tangential wind at each radius for a single (u,v) level slab
function azmean_vt(Uk, Vk, λq, φq, ct, st, λ1, dλ, φ1, dφ, Nx, Ny, K)
    out = zeros(K)
    for k in 1:K
        s = 0.0
        for m in 1:M
            u = bilin(Uk, λq[k,m], φq[k,m], λ1, dλ, φ1, dφ, Nx, Ny)
            v = bilin(Vk, λq[k,m], φq[k,m], λ1, dλ, φ1, dφ, Nx, Ny)
            s += -u*ct[m] + v*st[m]
        end
        out[k] = s / M
    end
    return out
end

# azimuthal-mean of a 2-D scalar (e.g. surface pressure) at each radius
function azmean_scalar(P, λq, φq, λ1, dλ, φ1, dφ, Nx, Ny, K)
    out = zeros(K)
    for k in 1:K
        s = 0.0
        for m in 1:M
            s += bilin(P, λq[k,m], φq[k,m], λ1, dλ, φ1, dφ, Nx, Ny)
        end
        out[k] = s / M
    end
    return out
end

function process(lab, dir, io_rz, io_prof, io_wp; mature_s = 6*86400.0, rmax = 1000e3)
    uf = FieldTimeSeries(joinpath(dir, "dcmip_tc_fields.jld2"), "u"; backend = OnDisk())
    vf = FieldTimeSeries(joinpath(dir, "dcmip_tc_fields.jld2"), "v"; backend = OnDisk())
    pt = FieldTimeSeries(joinpath(dir, "dcmip_tc_psfc.jld2"),  "p")
    grid = uf.grid
    λ = λnodes(grid, Center()); φ = φnodes(grid, Center()); z = znodes(grid, Center())
    Nx, Ny, Nz = size(grid)
    λ1 = λ[1]; dλ = λ[2]-λ[1]; φ1 = φ[1]; dφ = φ[2]-φ[1]
    ftimes = uf.times; ptimes = pt.times
    K = round(Int, rmax / DR) + 1                                # +1 to include r=0 (the eye)
    rmid = [(k-1)*DR/1e3 for k in 1:K]                            # r = 0, DR, 2DR, … (matches published grid)
    klo = searchsortedlast(z, 1000.0); khi = min(klo+1, Nz)
    w1 = khi > klo ? (1000.0 - z[klo]) / (z[khi]-z[klo]) : 0.0

    aVT = zeros(K, Nz); aPS = zeros(K); nrz = 0
    for n in 1:length(ftimes)
        t  = ftimes[n]
        pn = argmin(abs.(ptimes .- t))
        pp = interior(pt[pn], :, :, 1)
        msp = minimum(pp) / 100
        λe, φe = subgrid_center(pp, λ, φ, dλ, dφ, Nx, Ny)
        λq, φq, ct, st = ring_geometry(λe, φe, K)
        U = interior(uf[n], :, :, :); V = interior(vf[n], :, :, :)
        U1 = (1-w1) .* U[:,:,klo] .+ w1 .* U[:,:,khi]
        V1 = (1-w1) .* V[:,:,klo] .+ w1 .* V[:,:,khi]
        vt1 = azmean_vt(U1, V1, λq, φq, ct, st, λ1, dλ, φ1, dφ, Nx, Ny, K)
        println(io_wp, @sprintf("%s,%.4f,%.4f,%.4f", lab, t/86400, msp, maximum(vt1)))
        if t >= ftimes[end] - mature_s
            for k in 1:Nz
                @views aVT[:,k] .+= azmean_vt(U[:,:,k], V[:,:,k], λq, φq, ct, st, λ1, dλ, φ1, dφ, Nx, Ny, K)
            end
            aPS .+= azmean_scalar(pp, λq, φq, λ1, dλ, φ1, dφ, Nx, Ny, K) ./ 100
            nrz += 1
        end
    end
    aVT ./= nrz; aPS ./= nrz
    for k in 1:K
        for kz in 1:Nz
            println(io_rz, @sprintf("%s,%.3f,%.4f,%.5f", lab, rmid[k], z[kz]/1000, aVT[k,kz]))
        end
        w1km = (1-w1)*aVT[k,klo] + w1*aVT[k,khi]
        println(io_prof, @sprintf("%s,%.3f,%.6f,%.6f", lab, rmid[k], w1km, aPS[k]))
    end
    @info "dumped $lab ($nrz days-4-10 frames, $(length(ftimes)) total; K=$K radii, M=$M azimuths)"
end

runs = [
    ("weno5_0.25deg", length(ARGS) >= 1 ? ARGS[1] : "run_weno5_0.25deg_fields"),
    ("weno9_0.25deg", length(ARGS) >= 2 ? ARGS[2] : "run_main_0.25deg_weno9"),
]
mkpath("postproc")
io_rz   = open("postproc/breeze_fields_rz_010.csv", "w");    println(io_rz,   "run,r_km,z_km,vt_ms")
io_prof = open("postproc/breeze_profiles_010.csv", "w");     println(io_prof, "run,r_km,wind1km_ms,psfc_hPa")
io_wp   = open("postproc/breeze_windpressure_010.csv", "w"); println(io_wp,   "run,day,msp_hPa,mws_ms")
for (lab, dir) in runs
    process(lab, dir, io_rz, io_prof, io_wp)
end
close(io_rz); close(io_prof); close(io_wp)
println("wrote postproc/{breeze_fields_rz_010,breeze_profiles_010,breeze_windpressure_010}.csv")
