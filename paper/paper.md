---
title: 'Breeze.jl: A Julia Package for Simulating Atmospheric Flows'
tags:
  - Julia
  - atmospheric science
  - large-eddy simulation
  - cloud physics
  - computational fluid dynamics
  - GPU computing
authors:
  - name: Gregory L. Wagner
    orcid: 0000-0001-5317-2445
    affiliation: 1
    corresponding: true
  - name: Navid C. Constantinou
    orcid: 0000-0002-8149-4094
    affiliation: 2
  - name: Madelaine Gamble Rosevear
    orcid: 0000-0002-3825-0462
    affiliation: 3
  - name: Tobias Bischoff
    orcid: 0000-0002-5629-5765
    affiliation: 4
  - name: Kaiyuan Cheng
    orcid: 0000-0000-0000-0000
    affiliation: 4
  - name: Mosè Giordano
    orcid: 0000-0002-7218-2873
    affiliation: 5
  - name: Daniel Rosenfeld
    orcid: 0000-0002-0784-7656
    affiliation: 6
affiliations:
  - name: Aeolus Labs, CA, USA and Massachusetts Institute of Technology, Cambridge, MA, USA
    index: 1
  - name: Australian National University, Canberra, Australia
    index: 2
  - name: University of Tasmania, Hobart, Australia
    index: 3
  - name: Aeolus Labs, CA, USA
    index: 4
  - name: University College London, London, UK
    index: 5
  - name: The Hebrew University of Jerusalem, Jerusalem, Israel
    index: 6
date: 31 December 2024
bibliography: paper.bib
---

## TODO / Table of contents

- Summary
- [ ] TODO: *Introduction / summary figure*
- Statement of Need
- Key Features
  - Dynamics equations sets
    - Static energy
    - Liquid-ice potential temperature
    - [ ] Conservative potential temperature (Bryan and Fritsch 2002)
    - [ ] Total energy (Romps 2008, Yatunin 2025)
    - [ ] Entropy for non-phase-equilibrium (?)
  - Thermodynamic Formulations
  - Radiative Transfer
    - Gray radiation
    - Clear sky radiation
    - [ ] All sky radiation
    - [ ] Aerosol scatting radiation (?)
  - Microphysics schemes
    - No microphysics (!)
    - Warm- and mixed-phase SaturationAdjustment
    - CloudMicrophysics 1M scheme
    - [ ] CloudMicrophysics 2M warm-phase scheme
    - [ ] CloudMicrophysics P3+2M scheme
  - [ ] Time stepping methods
- Example Applications
  - Idealized stuff
  - Intercomparisons:
    - BOMEX
    - RICO
    - [ ] GATE
  - [ ] Radiative convective equilibrium
  - [ ] Open boundary case?
  - Prototype mesoscale applications
    - [ ] Dry tropical cyclone setup (Cronin and Chavas 2019)
- Acknowledgements
- References

# Summary

Breeze.jl is an open-source Julia package for simulating atmospheric flows, with a focus on large-eddy simulation (LES) of boundary layer turbulence, cloud dynamics, and moist convection. Built on top of Oceananigans.jl [@Oceananigans], Breeze.jl provides specialized physics for atmospheric applications including anelastic and compressible dynamics, moist thermodynamics with saturation adjustment, cloud microphysics, and radiative transfer. The package is designed for high-performance computing on both CPUs and GPUs, enabling researchers to perform high-resolution atmospheric simulations efficiently.

# Statement of Need

Numerical simulation of atmospheric flows spans scales from meters to thousands of kilometers, requiring diverse computational tools with distinct capabilities. At the smallest scales, large-eddy simulation (LES) models resolve turbulent eddies in the atmospheric boundary layer and have become essential for understanding cloud processes and convective dynamics. Established LES codes include DALES [@DALES], microHH [@MicroHH], PyCLES [@PyCLES], PTerodaC³TILES [@PTerodaC3TILES], and the System for Atmospheric Modeling (SAM) [@SAM]. At larger scales, mesoscale models simulate weather systems from a few kilometers to regional scales, with prominent examples including Cloud Model 1 (CM1) [@CM1], the Weather Research and Forecasting model (WRF) [@WRF], Meso-NH [@MesoNH], and the ICON nonhydrostatic model [@ICON]. Some models, notably SAM and CM1, bridge these scales by operating in both LES and cloud-resolving configurations.

At the global convection-permitting end of this spectrum, the Simple Convection-Permitting E3SM Atmosphere Model (SCREAM) has also been adapted into a doubly-periodic configuration (DP-SCREAM), enabling efficient studies of horizontal-resolution sensitivity and scale awareness in a controlled setting [@DPSCREAM]. At the global-model end more broadly, modern efforts include the CliMA atmospheric dycore, a spectral-element dynamical core designed for climate and global simulations [@ClimaCore; @ClimaAtmos], simplified Julia-first global models such as SpeedyWeather.jl [@SpeedyWeather], and differentiable / machine-learning-oriented global modeling frameworks such as JAX GCM [@JAXGCM] and NeuralGCM [@NeuralGCM].

The Weather Research and Forecasting model (WRF) exemplifies a "dual mandate" approach: its name emphasizes both *research* and *forecasting* as co-equal goals. WRF serves operational weather prediction centers worldwide while simultaneously enabling cutting-edge atmospheric research. This dual mandate—world-class performance for applications alongside flexibility and accessibility for research and education—is central to the design philosophy of Breeze.jl.

Existing atmospheric models face several challenges. Many legacy codes are written in Fortran and can be difficult to extend, modify, or learn for new users. While these codes achieve excellent performance, their complexity often creates barriers to entry for students and researchers from adjacent fields. Modern codes may offer improved usability but sometimes lack either the physical fidelity or computational performance required for production simulations.

Breeze.jl addresses these challenges by combining high performance with accessibility. Key design principles include:

1. **GPU-first architecture**: Breeze.jl is designed from the ground up for GPU computing. Leveraging KernelAbstractions.jl, the same code runs efficiently on both CPUs and GPUs, enabling researchers to utilize modern accelerated hardware without code modifications. This approach follows the successful model demonstrated by Oceananigans.jl [@OceananigansArxiv], which showed that high-level Julia code can achieve excellent performance across heterogeneous architectures.

2. **Julia scripting paradigm**: Written entirely in Julia [@Bezanson2017], Breeze.jl offers a scripting-based workflow where simulations are configured through human-readable Julia scripts rather than configuration files or compiled executables. This design accelerates the research cycle by enabling rapid prototyping, inline visualization, and interactive exploration of results. The same scripts serve as self-documenting examples for education.

3. **Extreme modularity**: Julia's multiple dispatch enables flexible composition of physical parameterizations. Users can swap thermodynamic formulations, microphysics schemes, turbulence closures, and other components independently. This modularity facilitates both research into individual processes and education about atmospheric physics.

4. **Oceananigans.jl foundation**: By building on Oceananigans.jl [@Oceananigans; @OceananigansArxiv], Breeze.jl inherits a battle-tested infrastructure for structured grids, advection schemes, time-stepping, and diagnostics, while adding atmosphere-specific physics including anelastic dynamics, moist thermodynamics, cloud microphysics, and radiative transfer.

5. **Coupled Earth system modeling**: Breeze.jl interfaces with ClimaOcean for coupled atmosphere-ocean simulations, enabling studies of air-sea interaction and supporting the development of next-generation Earth system models.

# Key Features

## Dynamical Cores

Breeze.jl supports multiple dynamical formulations:

- **Anelastic dynamics**: Filters acoustic waves while preserving gravity waves and convective motions, ideal for simulating atmospheric boundary layers and shallow convection.
- **Compressible dynamics**: Fully compressible equations for applications requiring acoustic wave propagation.

## Thermodynamic Formulations

The package offers flexible thermodynamic formulations:

- **Liquid-ice potential temperature thermodynamics**: Prognostic potential temperature density for anelastic simulations.
- **Static energy thermodynamics**: Prognostic static energy density, useful for energy-conserving formulations.

## Cloud Microphysics

Through an extension to CloudMicrophysics.jl, Breeze.jl supports:

- Zero-moment (saturation adjustment) schemes
- One-moment bulk microphysics schemes
- A flexible interface for implementing additional schemes

## Radiative Transfer

Integration with RRTMGP.jl provides:

- Clear-sky radiative transfer calculations
- Gray atmosphere approximations for idealized studies

# Example Applications

Breeze.jl includes documented examples demonstrating its capabilities:

- Dry and moist thermal bubbles
- BOMEX shallow cumulus intercomparison case [@Siebesma2003]
- RICO precipitating shallow convection [@vanZanten2011]
- Kelvin-Helmholtz instability with cloud formation
- Mountain wave dynamics
- Single-column radiation

# Acknowledgements

We acknowledge contributions from the Climate Modeling Alliance (CliMA) and the broader Julia community. This work was supported by [funding sources].

# References

