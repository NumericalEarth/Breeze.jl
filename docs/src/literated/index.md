# Examples

The examples are the best way to learn Breeze.
Each one is a complete, self-contained script that builds a grid, configures an `AtmosphereModel`, runs a `Simulation`, and visualizes the result, so it shows in one place how the pieces of Breeze fit together.
Reading an example alongside the reference pages is usually faster than reading the reference pages alone.

Together, the examples span much of what Breeze can do: dry and moist thermal bubbles, gravity and acoustic waves, large-eddy simulations of shallow cumulus convection and boundary layers, radiative transfer in a single column, flow over mountains, supercells, tropical cyclones, and baroclinic waves on the sphere.
Some are classic benchmark cases from the literature that are useful for validating the model and comparing numerical methods.
Others are physically motivated setups that show how to combine dynamics, thermodynamics, microphysics, radiation, and surface fluxes into a realistic simulation.
Parcel-model and kinematic-driver examples strip the dynamics away entirely, which makes them a convenient playground for prototyping microphysics and radiation schemes.

Every example is executed when this documentation is built, so the figures, animations, and printed output on each page are the actual result of running the script with the package version documented here.
The pages are generated from the scripts in the [`examples`](https://github.com/NumericalEarth/Breeze.jl/tree/main/examples) directory of the repository with [Literate.jl](https://github.com/fredrikekre/Literate.jl), and the bottom of each page records the Julia version and the packages used to run it.

The examples are also meant to be copied and modified.
Most of them run on a laptop in a few minutes at their default resolution, while the more demanding ones use a GPU.
Any example can be moved between architectures by replacing `CPU()` with `GPU()` in the grid constructor, or vice versa, and the resolution, domain size, and physics can be changed by editing a few lines at the top of the script.
Starting from the example closest to your problem and changing one thing at a time is a good way to build a new simulation.
