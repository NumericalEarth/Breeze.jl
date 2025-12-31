---
title: 'Breeze.jl: A Julia Package for Simulating Atmospheric Flows'
tags:
  - Julia
  - atmospheric science
  - large-eddy simulation
  - cloud dynamics
  - computational fluid dynamics
  - GPU computing
authors:
  - name: Gregory L. Wagner
    orcid: 0000-0001-5317-2445
    affiliation: 1
    corresponding: true
  - name: Author Two
    orcid: 0000-0000-0000-0000
    affiliation: 2
affiliations:
  - name: Massachusetts Institute of Technology, Cambridge, MA, USA
    index: 1
  - name: Institution Two, Country
    index: 2
date: 31 December 2024
bibliography: paper.bib
---

# Summary

Breeze.jl is an open-source Julia package for simulating atmospheric flows, with a focus on large-eddy simulation (LES) of boundary layer turbulence, cloud dynamics, and moist convection. Built on top of Oceananigans.jl [@Oceananigans], Breeze.jl provides specialized physics for atmospheric applications including anelastic and compressible dynamics, moist thermodynamics with saturation adjustment, cloud microphysics, and radiative transfer. The package is designed for high-performance computing on both CPUs and GPUs, enabling researchers to perform high-resolution atmospheric simulations efficiently.

# Statement of Need

Large-eddy simulation has become an essential tool for understanding atmospheric turbulence, cloud processes, and boundary layer dynamics. However, existing LES codes often face challenges: legacy Fortran codes can be difficult to extend and maintain, while modern codes may lack the physical fidelity needed for atmospheric applications or the performance required for high-resolution simulations.

Breeze.jl addresses these challenges by providing:

1. **Modern software design**: Written in Julia, Breeze.jl offers a clean, extensible codebase that is easy to understand, modify, and contribute to. The use of multiple dispatch enables flexible composition of physical parameterizations.

2. **GPU acceleration**: Leveraging KernelAbstractions.jl, Breeze.jl runs efficiently on both CPUs and NVIDIA GPUs, enabling researchers to utilize modern computing hardware without code modifications.

3. **Comprehensive atmospheric physics**: The package includes anelastic dynamics for filtering acoustic waves, moist thermodynamics with multiple formulations, cloud microphysics through integration with CloudMicrophysics.jl, and radiative transfer through RRTMGP.jl.

4. **Oceananigans.jl integration**: By building on Oceananigans.jl, Breeze.jl inherits a robust infrastructure for structured grids, advection schemes, and time-stepping, while adding atmosphere-specific physics.

5. **Coupled simulations**: Breeze.jl interfaces with ClimaOcean for coupled atmosphere-ocean simulations, enabling studies of air-sea interaction.

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

