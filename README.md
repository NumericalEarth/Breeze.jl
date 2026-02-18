<!-- Title -->
<h1 align="center">
  Breeze.jl
</h1>

<!-- description -->
<p align="center">
  <strong>Fast and friendly Julia software for atmospheric fluid dynamics on CPUs and GPUs. https://numericalearth.github.io/BreezeDocumentation/dev</strong>
</p>

<p align="center">
  <a href="https://numericalearth.github.io/BreezeDocumentation/stable/">
    <img alt="Documentation" src="https://img.shields.io/badge/documentation-stable-blue?style=flat-square">
  </a>
  <a href="https://numericalearth.github.io/BreezeDocumentation/dev/">
    <img alt="Documentation" src="https://img.shields.io/badge/documentation-in%20development-orange?style=flat-square">
  </a>
  <a href="https://doi.org/10.5281/zenodo.18050353">
    <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.18050353.svg" alt="DOI">
  </a>
  <a href="https://codecov.io/gh/NumericalEarth/Breeze.jl" >
    <img src="https://codecov.io/gh/NumericalEarth/Breeze.jl/graph/badge.svg?token=09TZGWKUPV"/>
  </a>
  </br>
  <a href="https://github.com/NumericalEarth/Breeze.jl/discussions">
    <img alt="Ask us anything" src="https://img.shields.io/badge/Ask%20us-anything-1abc9c.svg?style=flat-square">
  </a>
  <a href="https://github.com/SciML/ColPrac">
    <img alt="ColPrac: Contributor's Guide on Collaborative Practices for Community Packages" src="https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet?style=flat-square">
  </a>
  <a href="https://github.com/JuliaTesting/Aqua.jl" >
    <img src="https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg"/>
  </a>
</p>

Breeze is a library for simulating atmospheric flows and weather phenomena, such as clouds and hurricanes, on both CPUs and GPUs.
Built on [Oceananigans](https://github.com/CliMA/Oceananigans.jl), Breeze extends its grids, solvers, and advection schemes with atmospheric dynamics, thermodynamics, microphysics, and radiation.

Learn more in [the documentation](https://numericalearth.github.io/BreezeDocumentation/dev/) and [examples](https://github.com/NumericalEarth/Breeze.jl/tree/main/examples), or get in touch on the [NumericalEarth Slack](https://join.slack.com/t/numericalearth/shared_invite/zt-3pwpvky4k-XX7RkgQgHLIUt~wtwGXN~Q) or [GitHub discussions](https://github.com/NumericalEarth/Breeze.jl/discussions).

## Installation

Breeze is a registered Julia package. First [install Julia](https://julialang.org/install/); suggested version 1.12. See [juliaup](https://github.com/JuliaLang/juliaup) README for how to install 1.12 and make that version the default.

Then launch Julia and type

```julia
julia> using Pkg

julia> Pkg.add("Breeze")
```

If you want to live on the cutting edge, you can use
`Pkg.add(; url="https://github.com/NumericalEarth/Breeze.jl.git", rev="main")` to install from `main`.
For more information, see the [Pkg.jl documentation](https://pkgdocs.julialang.org).

## Quick start

A warm bubble rising through a neutral atmosphere in 15 lines:

```julia
using Breeze, Oceananigans.Units, CairoMakie

grid = RectilinearGrid(CPU(); size=(256, 256), x=(-10e3, 10e3), z=(0, 10e3),
                       topology=(Periodic, Flat, Bounded))

reference = ReferenceState(grid; potential_temperature=300)
model = AtmosphereModel(grid; dynamics=AnelasticDynamics(reference), advection=WENO(order=5))
set!(model, θ = (x, z) -> 300 + 2cos(π/2 * min(1, √(x^2 + (z - 2000)^2) / 2000))^2)

simulation = Simulation(model; Δt=2, stop_time=25minutes)
conjure_time_step_wizard!(simulation, cfl=0.7)
run!(simulation)

heatmap(liquid_ice_potential_temperature(model), colormap=:thermal, axis=(; aspect=2))
```

<img width="1192" height="631" alt="image" src="https://github.com/user-attachments/assets/c5681645-045d-483a-9ff1-140a3126acbc" />

Swap `CPU()` for `GPU()` to run on an NVIDIA GPU.

## Features

- **Anelastic dynamics** with a pressure Poisson solver that filters sound waves — [BOMEX shallow convection](https://numericalearth.github.io/BreezeDocumentation/dev/literated/bomex/), [RICO trade-wind cumulus](https://numericalearth.github.io/BreezeDocumentation/dev/literated/rico/)
- **Compressible dynamics** with split-explicit acoustic substepping (horizontally explicit, vertically implicit) using SSP-RK3 or Wicker-Skamarock RK3 — [inertia-gravity waves](https://numericalearth.github.io/BreezeDocumentation/dev/literated/inertia_gravity_wave/), [acoustic waves](https://numericalearth.github.io/BreezeDocumentation/dev/literated/acoustic_wave/)
- **Moist thermodynamics** with liquid-ice potential temperature and static energy formulations — [cloudy thermal bubble](https://numericalearth.github.io/BreezeDocumentation/dev/literated/cloudy_thermal_bubble/)
- **Cloud microphysics**: saturation adjustment, Kessler, one- and two-moment bulk schemes via [CloudMicrophysics.jl](https://github.com/CliMA/CloudMicrophysics.jl) — [splitting supercell](https://numericalearth.github.io/BreezeDocumentation/dev/literated/splitting_supercell/)
- **Radiative transfer**: gray, clear-sky, and all-sky solvers via [RRTMGP.jl](https://github.com/CliMA/RRTMGP.jl) — [single column radiation](https://numericalearth.github.io/BreezeDocumentation/dev/literated/single_column_radiation/)
- **High-order advection** including bounds-preserving WENO schemes
- **LES turbulence closures** for subgrid-scale mixing — [prescribed SST convection](https://numericalearth.github.io/BreezeDocumentation/dev/literated/prescribed_sea_surface_temperature/)
- **Surface physics**: Coriolis forces, bulk drag, heat and moisture fluxes — [tropical cyclones](https://numericalearth.github.io/BreezeDocumentation/dev/literated/tropical_cyclone_world/)
- **Kinematic driver and parcel model** for rapid physics prototyping — [kinematic driver](https://numericalearth.github.io/BreezeDocumentation/dev/literated/kinematic_driver/), [rising parcels](https://numericalearth.github.io/BreezeDocumentation/dev/literated/rising_parcels/)
- **GPU-first**: use `GPU()` to run _very fast_ on NVIDIA GPUs

## Selected examples

Below we've included thumbnails that link to a few of Breeze's examples.
Check out the [documentation](https://numericalearth.github.io/BreezeDocumentation/dev/) for the full list.

<table>
  <tr>
    <td width="33%" align="center" valign="top">
      <a href="https://numericalearth.github.io/BreezeDocumentation/dev/literated/cloudy_thermal_bubble/">
        <img src="https://github.com/user-attachments/assets/1ebc76bd-0ec5-4930-9d12-970caf3c8036" width="100%"><br>
        Cloudy thermal bubble
      </a>
    </td>
    <td width="33%" align="center" valign="top">
      <a href="https://numericalearth.github.io/BreezeDocumentation/dev/literated/bomex/">
        <img src="https://github.com/user-attachments/assets/0264d13f-31a7-4ca1-830d-2aa05f27ec4a" width="100%"><br>
        BOMEX shallow convection
      </a>
    </td>
    <td width="33%" align="center" valign="top">
      <a href="https://numericalearth.github.io/BreezeDocumentation/dev/literated/rico/">
        <img src="https://github.com/user-attachments/assets/6a041b42-a828-41e5-91fd-b4bc89e0f63a" width="100%"><br>
        RICO trade-wind cumulus
      </a>
    </td>
  </tr>
  <tr>
    <td width="33%" align="center" valign="top">
      <a href="https://numericalearth.github.io/BreezeDocumentation/dev/literated/prescribed_sea_surface_temperature/">
        <img src="https://github.com/user-attachments/assets/44a4b21c-23a6-401d-b938-e4ec00f24704" width="100%"><br>
        Prescribed SST convection
      </a>
    </td>
    <td width="33%" align="center" valign="top">
      <a href="https://numericalearth.github.io/BreezeDocumentation/dev/literated/cloudy_kelvin_helmholtz/">
        <img src="https://github.com/user-attachments/assets/1bcc787a-5b29-4bb7-b686-2d4465374b7d" width="100%"><br>
        Cloudy Kelvin-Helmholtz instability
      </a>
    </td>
    <td width="33%" align="center" valign="top">
      <a href="https://numericalearth.github.io/BreezeDocumentation/dev/literated/acoustic_wave/">
        <img src="https://github.com/user-attachments/assets/fa2992d0-a289-4de7-aeb3-f59df7cbef28" width="100%"><br>
        Acoustic wave in shear flow
      </a>
    </td>
  </tr>
</table>

For instance, by increasing the resolution of the cloudy Kelvin-Helmholtz instability
to `Nx=1536` and `Nz=1024` and decreasing the timestep to `Δt = 0.1`, we get

https://github.com/user-attachments/assets/f47ff268-b2e4-401c-a114-a0aaf0c7ead3

Or cranking up the resolution of the thermal bubble example to `size = (1024, 512)`:

https://github.com/user-attachments/assets/c9a0c9c3-c199-48b8-9850-f967bdcc4bed

We ran the [BOMEX example](https://numericalearth.github.io/BreezeDocumentation/dev/literated/bomex/) at 25 m resolution and a 2x bigger grid, and used the data to produce a visualization of the resulting clouds:

https://github.com/user-attachments/assets/3c378cc7-c71b-420d-b301-33d45c7521e2

## Roadmap and a call to action

Our goal is to build a very fast, easy-to-learn, productive tool for atmospheric research, teaching, and forecasting, as well as a platform for the development of algorithms, numerical methods, parameterizations, microphysical schemes, and atmosphere model components.
This goal can't be achieved by the efforts of a single group, project, or even a single community.
Such a lofty aim can only be realized by a wide-ranging and sustained collaboration of passionate people.
Maybe that includes you - consider it!
Model development is hard but rewarding, and builds useful skills for a myriad of pursuits.

The goals of the current group of model developers include developing

- **Advanced microphysics**: Predicted Particle Property (P3) bulk microphysics, spectral bin schemes, and Lagrangian superdroplet methods for high-fidelity cloud and precipitation modeling.
- **Terrain-following coordinates**: Smooth [sigma coordinates](https://en.wikipedia.org/wiki/Sigma_coordinate_system) for flow over complex topography
- **Open boundaries and nesting**: Open boundary conditions are useful for both idealized simulations and realistic one- and two-way nested simulations for high-resolution downscaling.
- **Coupled atmosphere-ocean simulations**: Support for high-resolution coupled atmosphere-ocean simulations via [NumericalEarth.jl](https://github.com/NumericalEarth/NumericalEarth.jl).

If you have ideas, dreams, or criticisms that can make Breeze and its future better, don't hesitate to speak up by [opening issues](https://github.com/NumericalEarth/Breeze.jl/issues/new/choose) and contributing pull requests.

## Relationship to Oceananigans

Breeze is built on [Oceananigans.jl](https://github.com/CliMA/Oceananigans.jl), an ocean modeling package that provides grids, fields, operators, advection schemes, time-steppers, turbulence closures, and output infrastructure.
Breeze extends Oceananigans with atmospheric dynamics, thermodynamics, microphysics, and radiation to create a complete atmosphere simulation capability.
The two packages share a common philosophy: fast, flexible, GPU-native Julia code with a user interface designed for productivity and experimentation.
To learn these foundational components of Breeze, please see the [Oceananigans documentation](https://clima.github.io/OceananigansDocumentation/stable/).

If you're familiar with Oceananigans, you'll feel right at home with Breeze.
If you're new to both, Breeze is a great entry point—and the skills you develop transfer directly to ocean and climate modeling with Oceananigans and [ClimaOcean.jl](https://github.com/CliMA/ClimaOcean.jl).

## Citing

If you use Breeze for research, teaching, or fun, we'd be grateful if you give credit by citing the corresponding Zenodo record, e.g.,

> Wagner, G. L. et al. (2026). NumericalEarth/Breeze.jl. Zenodo. DOI:[10.5281/zenodo.18050353](https://doi.org/10.5281/zenodo.18050353)
