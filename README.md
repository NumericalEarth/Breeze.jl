<!-- Title -->
<h1 align="center">
  Breeze.jl
</h1>

<!-- description -->
<p align="center">
  <strong>🌪 Fast and friendly Julia software for atmospheric fluid dynamics on CPUs and GPUs. https://numericalearth.github.io/BreezeDocumentation/stable</strong>
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

Breeze is a library for simulating atmospheric flows, convection, clouds, weather, and hurricanes on CPUs and GPUs.
Much of Breeze's power flows from [Oceananigans](https://github.com/CliMA/Oceananigans.jl), which provides a user interface, grids, fields, solvers, advection schemes, Lagrangian particles, physics, and more.

Breeze's AtmosphereModel features anelastic, compressible, and prescribed (kinematic) dynamics, closures for large eddy simulation, saturation adjustment microphysics, Kessler microphysics, and one- and two-moment bulk schemes via an extension to the [Climate Modeling Alliance's](clima.caltech.edu) [CloudMicrophysics.jl](https://github.com/CliMA/CloudMicrophysics.jl).
An extension to [RRTMGP.jl](https://github.com/CliMA/RRTMGP.jl) provides solvers for gray, clear-sky, and all-sky radiative transfer.
Breeze's examples include single column radiation, idealized thermal bubbles and inertia-gravity waves and Kelvin-Helmholtz, BOMEX shallow convection, RICO trade-wind cumulus, supercells, mountain waves, and more.

Check out [the documentation](https://numericalearth.github.io/BreezeDocumentation/stable/) and [examples](https://github.com/NumericalEarth/Breeze.jl/tree/main/examples).
Don't hesitate to get in touch on the [NumericalEarth slack](https://join.slack.com/t/numericalearth/shared_invite/zt-3kjcowmpg-B0s3nalWkvZg8IBc~BIJEA) or by opening a new [discussion](https://github.com/NumericalEarth/Breeze.jl/discussions)!

## Roadmap and a call to action

Our goal is to build the world's fastest, easiest to use, most productive tool for weather research and forecasting, atmospheric model development, cloud physics, and more.
This won't be the effort of a single group or even a single community.
It can only be achieved by a wide-ranging and sustained collaboration of passionate people.
Maybe that includes you --- consider it!
Model development is hard, but rewarding and builds useful skills for a myriad of pursuits.

Right now, the goals of the current group of model developers is to implement

- ⛈️ **Advanced microphysics**: Predicted Particle Property (P3) bulk microphysics, spectral bin schemes, and Lagrangian superdroplet methods for high-fidelity cloud and precipitation modeling
- 🌊 **Acoustic substepping**: A fully compressible dynamical core that resolves sound waves, for applications from convective dynamics to acoustic propagation
- 🏔️ **Terrain-following coordinates**: Flow over complex topography with smooth sigma coordinates
- 🔬 **Nesting**: Two-way nesting from turbulence-resolving to mesoscales
- 🌀 **Coupled atmosphere-ocean simulations**: Support for high-resolution coupled atmosphere-ocean simulations via [ClimaOcean.jl](https://github.com/CliMA/ClimaOcean.jl)

If you have ideas, dreams, or criticisms that can make Breeze and it's future better, don't hesitate to speak up.

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
        <img src="https://github.com/user-attachments/assets/89876fed-c2d0-43da-aa54-ab69b97e0283" width="100%"><br>
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
      <a href="https://numericalearth.github.io/BreezeDocumentation/dev/literated/inertia_gravity_wave/">
        <img src="https://github.com/user-attachments/assets/54837848-c1b5-4ffc-943b-ba3f1755b6ee" width="100%"><br>
        Inertia-gravity wave
      </a>
    </td>
    <td width="33%" align="center" valign="top">
      <a href="https://numericalearth.github.io/BreezeDocumentation/dev/literated/acoustic_wave/">
        <img src="https://github.com/user-attachments/assets/fa2992d0-a289-4de7-aeb3-f59df7cbef28" width="100%"><br>
        Acoustic wave
      </a>
    </td>
  </tr>
</table>

## Installation

Breeze is a registered Julia package. First [install Julia](https://julialang.org/install/); suggested version 1.12. See [juliaup](https://github.com/JuliaLang/juliaup) README for how to install 1.12 and make that version the default.

Then launch Julia and type

```julia
julia> using Pkg

julia> Pkg.add("Breeze")
```

which will install the latest stable version of Breeze that's compatible with your current environment.

You can check which version of Breeze you got via

```julia
Pkg.status("Breeze")
```

If you want to live on the cutting edge, you can use, e.g.,
`Pkg.add(; url="https://github.com/NumericalEarth/Breeze.jl.git", rev="main")` to install the latest version of
Breeze from `main` branch. For more information, see the
[Pkg.jl documentation](https://pkgdocs.julialang.org).

## Using Breeze

Now we are ready to run any of the examples!

For instance, by increasing the resolution of the cloudy Kelvin-Helmholtz instability
to `Nx=1536` and `Nz=1024`, decrease the timestep to `Δt = 0.1`, and running

```julia
julia> include("examples/cloudy_kelvin_helmholtz.jl")
```

to get

https://github.com/user-attachments/assets/f47ff268-b2e4-401c-a114-a0aaf0c7ead3

Or cranking up the spatial resolution of the thermal bubble example to to `size = (1024, 512)` and running

```julia
julia> include("examples/dry_thermal_bubble.jl")
```

we get

https://github.com/user-attachments/assets/c9a0c9c3-c199-48b8-9850-f967bdcc4bed

## Citing

If you use Breeze for research, teaching, or fun, we'd be grateful if you give credit by citing the corresponding Zenodo record, e.g.,

> Wagner, G. L. et al. (2025). NumericalEarth/Breeze.jl. Zenodo. DOI:[10.5281/zenodo.18050353](https://doi.org/10.5281/zenodo.18050353)
