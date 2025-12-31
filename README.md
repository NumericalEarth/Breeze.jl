<!-- Title -->
<h1 align="center">
  Breeze.jl
</h1>

<!-- description -->
<p align="center">
  <strong>🌪 Fast and friendly Julia software for atmospheric fluid dynamics on CPUs and GPUs</strong>
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
Right now, `Breeze.AtmosphereModel` is in an early stage of development, and supports simple simulations that use the anelastic formulation of the Euler equations on `RectilinearGrid`.
But we're working feverishly towards a future with bulk, bin and superdroplet microphysics, radiation, and a fully compressible formulation with acoustic substepping (and note, the roadmap and vision for Breeze is still something of a work in progress).
Check out [the documentation](https://numericalearth.github.io/BreezeDocumentation/stable/) to see what we can do now, and watch this space (or get in touch to [discuss](https://github.com/NumericalEarth/Breeze.jl/discussions)!) its crystallization.

## Gallery

<table>
  <tr>
    <th width="25%">
      <a href="https://numericalearth.github.io/BreezeDocumentation/stable/literated/dry_thermal_bubble/">
        <img src="docs/src/assets/dry_thermal_bubble.png" alt="Dry thermal bubble">
        <br><em>Dry thermal bubble</em>
      </a>
    </th>
    <th width="25%">
      <a href="https://numericalearth.github.io/BreezeDocumentation/stable/literated/cloudy_thermal_bubble/">
        <img src="docs/src/assets/cloudy_thermal_bubble.png" alt="Cloudy thermal bubble">
        <br><em>Cloudy thermal bubble</em>
      </a>
    </th>
    <th width="25%">
      <a href="https://numericalearth.github.io/BreezeDocumentation/stable/literated/cloudy_kelvin_helmholtz/">
        <img src="docs/src/assets/cloudy_kelvin_helmholtz.png" alt="Cloudy Kelvin-Helmholtz">
        <br><em>Cloudy Kelvin-Helmholtz</em>
      </a>
    </th>
    <th width="25%">
      <a href="examples/mountain_wave.jl">
        <img src="docs/src/assets/mountain_wave.png" alt="Mountain waves">
        <br><em>Mountain waves</em>
      </a>
    </th>
  </tr>
  <tr>
    <th width="25%">
      <a href="https://numericalearth.github.io/BreezeDocumentation/stable/literated/bomex/">
        <img src="docs/src/assets/bomex.png" alt="BOMEX shallow cumulus">
        <br><em>BOMEX shallow cumulus</em>
      </a>
    </th>
    <th width="25%">
      <a href="examples/rico.jl">
        <img src="docs/src/assets/rico.png" alt="RICO precipitating cumulus">
        <br><em>RICO precipitating cumulus</em>
      </a>
    </th>
    <th width="25%">
      <a href="https://numericalearth.github.io/BreezeDocumentation/stable/literated/single_column_radiation/">
        <img src="docs/src/assets/single_column_radiation.png" alt="Single column radiation">
        <br><em>Single column radiation</em>
      </a>
    </th>
    <th width="25%">
      <a href="examples/supercell.jl">
        <img src="docs/src/assets/supercell.png" alt="Supercell thunderstorm">
        <br><em>Supercell thunderstorm</em>
      </a>
    </th>
  </tr>
</table>

### Simulations in motion

<details>
<summary>🌊 <strong>Cloudy Kelvin-Helmholtz instability</strong> — wave clouds rolling through a moist shear layer</summary>
<br>

Kelvin-Helmholtz billows in a stably-stratified, moist atmosphere. As the shear layer rolls up, a moist layer is advected and deformed, producing billow-like patterns reminiscent of observed "wave clouds".

https://github.com/user-attachments/assets/f47ff268-b2e4-401c-a114-a0aaf0c7ead3

</details>

<details>
<summary>🔥 <strong>Dry thermal bubble</strong> — a warm bubble rising through stable stratification</summary>
<br>

A classic test case: a localized warm perturbation rises buoyantly through a stably-stratified dry atmosphere, demonstrating the model's ability to capture buoyancy-driven convection.

https://github.com/user-attachments/assets/c9a0c9c3-c199-48b8-9850-f967bdcc4bed

</details>

<details>
<summary>☁️ <strong>Cloudy thermal bubble</strong> — moist convection with cloud formation</summary>
<br>

A moist thermal bubble rises and cools adiabatically. As the air reaches saturation, cloud condensate forms, releasing latent heat that further enhances the buoyancy.

<!-- TODO: Upload cloudy_thermal_bubble.mp4 to GitHub and replace with video link -->
![Cloudy thermal bubble](docs/src/literated/cloudy_thermal_bubble.mp4)

</details>

<details>
<summary>⛰️ <strong>Mountain waves</strong> — gravity waves excited by flow over topography</summary>
<br>

Flow over an idealized mountain ridge generates internal gravity waves that propagate upward through the stratified atmosphere.

<!-- TODO: Upload mountain_wave.mp4 to GitHub and replace with video link -->
![Mountain waves](mountain_wave.mp4)

</details>

<details>
<summary>🌀 <strong>Inertia-gravity waves</strong> — rotating stratified dynamics</summary>
<br>

Inertia-gravity waves in a rotating, stratified atmosphere, demonstrating the model's ability to capture geostrophic adjustment.

<!-- TODO: Upload inertia_gravity_wave.mp4 to GitHub and replace with video link -->
![Inertia-gravity waves](docs/src/literated/inertia_gravity_wave.mp4)

</details>

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

Now we are ready to run any of the examples! For instance:

```julia
julia> include("examples/cloudy_kelvin_helmholtz.jl")
```

or dive into the [documentation](https://numericalearth.github.io/BreezeDocumentation/stable/) for tutorials and API reference.

## Citing

If you use Breeze for research, teaching, or fun, we'd be grateful if you give credit by citing the corresponding Zenodo record, e.g.,

> Wagner, G. L. et al. (2025). NumericalEarth/Breeze.jl. Zenodo. DOI:[10.5281/zenodo.18050353](https://doi.org/10.5281/zenodo.18050353)
