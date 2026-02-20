#=
MWE: Slice Bounds Error — Round 7: Pure Oceananigans Reproduction
=================================================================
Reactant v0.2.221+, Julia 1.11+

CONFIRMED from Round 6:
  Mx (x-momentum tendency at Face,Center,Center on Bounded x) → FAIL
  My (y-momentum tendency at Center,Face,Center on Bounded y) → PASS

Root cause hypothesis:
  Oceananigans' `work_layout(grid, :xyz)` returns (Nx, Ny, Nz) = (5,5,1).
  The x-momentum kernel iterates i=1:5 and its stencil accesses i+1=6 on
  Face fields (u, ρu) which have N+1=6 interior x-points on Bounded.
  Similarly, the y-halo fill for Face-in-x fields uses worksize (5,1) in :xz.
  The MLIR compiler creates intermediate slices of size 5 in dim 1, then the
  tendency stencil tries to read index 6 → "limit index 6 > dimension size 5".

This MWE tests with ONLY Oceananigans (no Breeze) to confirm the bug is at
the framework level. Uses simple @kernel stencils on Face vs Center fields.

Run:  julia --project=test test/bounded-segfault/mwe_slice_bounds_round7.jl
=#

using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: interior, set!, Face, Center
using Oceananigans.Utils: launch!
using Oceananigans.Operators: δxᶠᵃᵃ, δxᶜᵃᵃ, δyᵃᶜᵃ, ℑxᶜᵃᵃ
using KernelAbstractions: @kernel, @index
using Reactant
using Statistics: mean

Reactant.set_default_backend("cpu")

grid = RectilinearGrid(ReactantState();
    size=(5, 5), extent=(1e3, 1e3), halo=(3, 3),
    topology=(Bounded, Bounded, Flat))

nsteps = 2

# Fields
u_face  = Field{Face, Center, Center}(grid)    # Face in x (dim 1) → 6 interior x-points
v_face  = Field{Center, Face, Center}(grid)    # Face in y (dim 2) → 6 interior y-points
c_center = CenterField(grid)                    # Center everywhere → 5 interior x-points
out_fcc = Field{Face, Center, Center}(grid)    # output at Face,Center,Center
out_cfc = Field{Center, Face, Center}(grid)    # output at Center,Face,Center
out_ccc = CenterField(grid)                    # output at Center,Center,Center

set!(u_face, (x, y) -> sin(2π * x / 1e3))
set!(v_face, (x, y) -> cos(2π * y / 1e3))
set!(c_center, (x, y) -> 300 + 0.01x)

# ─────────────────────────────────────────────────────────────────────────────
# Stencil kernels that mimic momentum tendency access patterns
# ─────────────────────────────────────────────────────────────────────────────

# Kernel at (Face,Center,Center) reading Face field with δxᶜᵃᵃ → accesses u[i+1]
@kernel function face_x_stencil!(out, grid, u)
    i, j, k = @index(Global, NTuple)
    @inbounds out[i, j, k] = δxᶜᵃᵃ(i, j, k, grid, u) + u[i, j, k]
end

# Kernel at (Center,Face,Center) reading Face field with δyᵃᶜᵃ → accesses v[j+1]
@kernel function face_y_stencil!(out, grid, v)
    i, j, k = @index(Global, NTuple)
    @inbounds out[i, j, k] = δyᵃᶜᵃ(i, j, k, grid, v) + v[i, j, k]
end

# Kernel at (Face,Center,Center) using δxᶠᵃᵃ on Center field → accesses c[i-1]
@kernel function pressure_gradient_stencil!(out, grid, c)
    i, j, k = @index(Global, NTuple)
    @inbounds out[i, j, k] = δxᶠᵃᵃ(i, j, k, grid, c)
end

# Kernel at (Face,Center,Center) mimicking div_𝐯u structure:
# δxᶠᵃᵃ of (ℑxᶜᵃᵃ(u) * ℑxᶜᵃᵃ(u)) — the self-advection term
@kernel function advection_like_stencil!(out, grid, u)
    i, j, k = @index(Global, NTuple)
    flux_i   = ℑxᶜᵃᵃ(i,   j, k, grid, u) * ℑxᶜᵃᵃ(i,   j, k, grid, u)
    flux_im1 = ℑxᶜᵃᵃ(i-1, j, k, grid, u) * ℑxᶜᵃᵃ(i-1, j, k, grid, u)
    @inbounds out[i, j, k] = flux_i - flux_im1
end

# ─────────────────────────────────────────────────────────────────────────────
# Test runner
# ─────────────────────────────────────────────────────────────────────────────

function run_test(label, fn)
    print("  $label: ")
    flush(stdout)
    try
        result = fn()
        println("PASS (result=$result)")
        flush(stdout)
        return :pass
    catch e
        e isa InterruptException && rethrow()
        msg = sprint(showerror, e)
        if contains(msg, "limit index") || contains(msg, "dimension size")
            m = match(r"limit index (\d+) is larger than dimension size (\d+) in dimension (\d+)", msg)
            if m !== nothing
                println("SLICE BOUNDS: limit=$(m[1]) > dim_size=$(m[2]) in dim $(m[3])")
            else
                println("SLICE BOUNDS ERROR")
            end
        elseif contains(msg, "signal (11)") || contains(msg, "SinkDUS")
            println("SEGFAULT")
        else
            first_line = first(split(msg, '\n'))
            println("ERROR: $(first_line[1:min(end, 120)])")
        end
        flush(stdout)
        return :fail
    end
end

println_header(s) = println("\n", "─" ^ 72, "\n", s, "\n", "─" ^ 72)

println("=" ^ 72)
println("Slice Bounds Error — Round 7: Pure Oceananigans (No Breeze)")
println("Grid: size=(5,5), halo=(3,3), topology=(Bounded,Bounded,Flat)")
println("u_face: (Face,Center,Center) → 6 interior x-pts, 5 interior y-pts")
println("v_face: (Center,Face,Center) → 5 interior x-pts, 6 interior y-pts")
println("All tests: forward-only, nsteps=$nsteps")
println("=" ^ 72)

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1: Bare stencils — no halo fill, 1x per iteration
# ═════════════════════════════════════════════════════════════════════════════
println_header("SECTION 1 — Bare stencils, no halo fill, 1x per iteration")

run_test("S1a  δxᶜᵃᵃ(u_face) at FCC — accesses u[i+1]", () -> begin
    function f(u, out, n)
        @trace track_numbers=false for _ in 1:n
            launch!(grid.architecture, grid, :xyz, face_x_stencil!, out, grid, u)
            parent(out) .= parent(out) .* 0.99
        end
        return mean(interior(out) .^ 2)
    end
    c = Reactant.@compile raise_first=true raise=true sync=true f(u_face, out_fcc, nsteps)
    c(u_face, out_fcc, nsteps)
end)

run_test("S1b  δyᵃᶜᵃ(v_face) at CFC — accesses v[j+1]", () -> begin
    function f(v, out, n)
        @trace track_numbers=false for _ in 1:n
            launch!(grid.architecture, grid, :xyz, face_y_stencil!, out, grid, v)
            parent(out) .= parent(out) .* 0.99
        end
        return mean(interior(out) .^ 2)
    end
    c = Reactant.@compile raise_first=true raise=true sync=true f(v_face, out_cfc, nsteps)
    c(v_face, out_cfc, nsteps)
end)

run_test("S1c  ∂xᶠᶜᶜ(c_center) at FCC — accesses c[i-1]", () -> begin
    function f(c, out, n)
        @trace track_numbers=false for _ in 1:n
            launch!(grid.architecture, grid, :xyz, pressure_gradient_stencil!, out, grid, c)
            parent(out) .= parent(out) .* 0.99
        end
        return mean(interior(out) .^ 2)
    end
    c_fn = Reactant.@compile raise_first=true raise=true sync=true f(c_center, out_fcc, nsteps)
    c_fn(c_center, out_fcc, nsteps)
end)

run_test("S1d  advection-like stencil on u_face at FCC", () -> begin
    function f(u, out, n)
        @trace track_numbers=false for _ in 1:n
            launch!(grid.architecture, grid, :xyz, advection_like_stencil!, out, grid, u)
            parent(out) .= parent(out) .* 0.99
        end
        return mean(interior(out) .^ 2)
    end
    c = Reactant.@compile raise_first=true raise=true sync=true f(u_face, out_fcc, nsteps)
    c(u_face, out_fcc, nsteps)
end)

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2: Stencils + halo fill, 1x per iteration
# ═════════════════════════════════════════════════════════════════════════════
println_header("SECTION 2 — Stencils + halo fill, 1x per iteration")

run_test("S2a  halo(u) + δxᶜᵃᵃ(u) stencil at FCC", () -> begin
    function f(u, out, n)
        @trace track_numbers=false for _ in 1:n
            fill_halo_regions!(u)
            launch!(grid.architecture, grid, :xyz, face_x_stencil!, out, grid, u)
            parent(out) .= parent(out) .* 0.99
        end
        return mean(interior(out) .^ 2)
    end
    c = Reactant.@compile raise_first=true raise=true sync=true f(u_face, out_fcc, nsteps)
    c(u_face, out_fcc, nsteps)
end)

run_test("S2b  halo(v) + δyᵃᶜᵃ(v) stencil at CFC", () -> begin
    function f(v, out, n)
        @trace track_numbers=false for _ in 1:n
            fill_halo_regions!(v)
            launch!(grid.architecture, grid, :xyz, face_y_stencil!, out, grid, v)
            parent(out) .= parent(out) .* 0.99
        end
        return mean(interior(out) .^ 2)
    end
    c = Reactant.@compile raise_first=true raise=true sync=true f(v_face, out_cfc, nsteps)
    c(v_face, out_cfc, nsteps)
end)

run_test("S2c  halo(u) + advection-like stencil at FCC", () -> begin
    function f(u, out, n)
        @trace track_numbers=false for _ in 1:n
            fill_halo_regions!(u)
            launch!(grid.architecture, grid, :xyz, advection_like_stencil!, out, grid, u)
            parent(out) .= parent(out) .* 0.99
        end
        return mean(interior(out) .^ 2)
    end
    c = Reactant.@compile raise_first=true raise=true sync=true f(u_face, out_fcc, nsteps)
    c(u_face, out_fcc, nsteps)
end)

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3: Stencils + halo fill, 3x per iteration (matches failure condition)
# ═════════════════════════════════════════════════════════════════════════════
println_header("SECTION 3 — Stencils + halo fill, 3x per iteration")

run_test("S3a  3x [halo(u) + δxᶜᵃᵃ(u) stencil at FCC]", () -> begin
    function f(u, out, n)
        @trace track_numbers=false for _ in 1:n
            for _ in 1:3
                fill_halo_regions!(u)
                launch!(grid.architecture, grid, :xyz, face_x_stencil!, out, grid, u)
                parent(out) .= parent(out) .* 0.99
            end
        end
        return mean(interior(out) .^ 2)
    end
    c = Reactant.@compile raise_first=true raise=true sync=true f(u_face, out_fcc, nsteps)
    c(u_face, out_fcc, nsteps)
end)

run_test("S3b  3x [halo(v) + δyᵃᶜᵃ(v) stencil at CFC]", () -> begin
    function f(v, out, n)
        @trace track_numbers=false for _ in 1:n
            for _ in 1:3
                fill_halo_regions!(v)
                launch!(grid.architecture, grid, :xyz, face_y_stencil!, out, grid, v)
                parent(out) .= parent(out) .* 0.99
            end
        end
        return mean(interior(out) .^ 2)
    end
    c = Reactant.@compile raise_first=true raise=true sync=true f(v_face, out_cfc, nsteps)
    c(v_face, out_cfc, nsteps)
end)

run_test("S3c  3x [halo(u) + advection-like stencil at FCC]", () -> begin
    function f(u, out, n)
        @trace track_numbers=false for _ in 1:n
            for _ in 1:3
                fill_halo_regions!(u)
                launch!(grid.architecture, grid, :xyz, advection_like_stencil!, out, grid, u)
                parent(out) .= parent(out) .* 0.99
            end
        end
        return mean(interior(out) .^ 2)
    end
    c = Reactant.@compile raise_first=true raise=true sync=true f(u_face, out_fcc, nsteps)
    c(u_face, out_fcc, nsteps)
end)

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4: Multiple Face fields + halo fills + stencils, 3x (mimics full model)
# ═════════════════════════════════════════════════════════════════════════════
println_header("SECTION 4 — Multiple Face fields + stencils, 3x (model-like)")

run_test("S4a  3x [halo(u)+halo(v)+halo(c) + u-stencil + v-stencil]", () -> begin
    function f(u, v, c, out_u, out_v, n)
        @trace track_numbers=false for _ in 1:n
            for _ in 1:3
                fill_halo_regions!(u)
                fill_halo_regions!(v)
                fill_halo_regions!(c)
                launch!(grid.architecture, grid, :xyz, face_x_stencil!, out_u, grid, u)
                launch!(grid.architecture, grid, :xyz, face_y_stencil!, out_v, grid, v)
                parent(out_u) .= parent(out_u) .* 0.99
            end
        end
        return mean(interior(out_u) .^ 2)
    end
    c = Reactant.@compile raise_first=true raise=true sync=true f(u_face, v_face, c_center, out_fcc, out_cfc, nsteps)
    c(u_face, v_face, c_center, out_fcc, out_cfc, nsteps)
end)

run_test("S4b  3x [halo(u,v,c as tuple) + u-stencil + v-stencil]", () -> begin
    function f(u, v, c, out_u, out_v, n)
        @trace track_numbers=false for _ in 1:n
            for _ in 1:3
                fill_halo_regions!((u, v, c))
                launch!(grid.architecture, grid, :xyz, face_x_stencil!, out_u, grid, u)
                launch!(grid.architecture, grid, :xyz, face_y_stencil!, out_v, grid, v)
                parent(out_u) .= parent(out_u) .* 0.99
            end
        end
        return mean(interior(out_u) .^ 2)
    end
    c = Reactant.@compile raise_first=true raise=true sync=true f(u_face, v_face, c_center, out_fcc, out_cfc, nsteps)
    c(u_face, v_face, c_center, out_fcc, out_cfc, nsteps)
end)

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5: Feedback stencils (output feeds back to input) at 3x
# ═════════════════════════════════════════════════════════════════════════════
println_header("SECTION 5 — Feedback stencils (read+write same Face field), 3x")

@kernel function face_x_feedback_stencil!(u, grid)
    i, j, k = @index(Global, NTuple)
    @inbounds u[i, j, k] = u[i, j, k] + 0.001 * δxᶜᵃᵃ(i, j, k, grid, u)
end

run_test("S5a  3x [halo(u) + feedback u-stencil at FCC]", () -> begin
    function f(u, n)
        @trace track_numbers=false for _ in 1:n
            for _ in 1:3
                fill_halo_regions!(u)
                launch!(grid.architecture, grid, :xyz, face_x_feedback_stencil!, u, grid)
            end
        end
        return mean(interior(u) .^ 2)
    end
    c = Reactant.@compile raise_first=true raise=true sync=true f(u_face, nsteps)
    c(u_face, nsteps)
end)

@kernel function face_y_feedback_stencil!(v, grid)
    i, j, k = @index(Global, NTuple)
    @inbounds v[i, j, k] = v[i, j, k] + 0.001 * δyᵃᶜᵃ(i, j, k, grid, v)
end

run_test("S5b  3x [halo(v) + feedback v-stencil at CFC]", () -> begin
    function f(v, n)
        @trace track_numbers=false for _ in 1:n
            for _ in 1:3
                fill_halo_regions!(v)
                launch!(grid.architecture, grid, :xyz, face_y_feedback_stencil!, v, grid)
            end
        end
        return mean(interior(v) .^ 2)
    end
    c = Reactant.@compile raise_first=true raise=true sync=true f(v_face, nsteps)
    c(v_face, nsteps)
end)

# ═════════════════════════════════════════════════════════════════════════════
# Summary
# ═════════════════════════════════════════════════════════════════════════════
println("\n" * "=" ^ 72)
println("INTERPRETATION:")
println()
println("• If S1a/S2a/S3a fail → bare stencil on Face-in-x is enough.")
println("  The bug is purely about accessing u[i+1] where worksize=5")
println("  but Face field has 6 interior x-points.")
println()
println("• If only S3a+ fail → the 3x loop + halo fill combination is needed.")
println("  The MLIR optimizer needs sufficient trace complexity.")
println()
println("• If S5a fails but S5b passes → definitive proof that the bug is")
println("  specific to Face-in-dimension-1 (x) with Bounded topology.")
println()
println("• If nothing fails → the bug requires Breeze-level complexity")
println("  (many kernels, large argument tuples, etc.) to manifest.")
println("=" ^ 72)
