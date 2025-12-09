# [Microphysics](@id section:microphysics-overview)

In atmospheric modeling, "microphysics" largely overlaps with ["cloud physics"](https://en.wikipedia.org/wiki/Cloud_physics) and concerns the formation, development, and precipitation of clouds.
More generally, "microphysics" encompasses panoply of physical processes associated with _(i)_ the conversion of water between vapor, liquid, and ice phases, and _(ii)_ the interaction and transformation of "cloud particles", which include droplets, ice particles, and aerosols.
For example: microphysical processes include droplet nucleation on aerosol particles; liquid freezing and vapor solidification; chemical and humidity-based transformations of aerosol particles; the agglomeration of ice particles into snowflakes; collisions and breakup of falling droplets (rain); collision between ice particles and droplets or wholesale freezing of droplets to form hail (e.g. [Straka2009](@citet)).

Breeze microphysics is nascent and under active development.
Breeze aims to eventually provide a wide range of microphysical models, ranging from simple warm-phase saturation adjustment, to the bulk schemes provided by the Climate Modeling Alliance's [CloudMicrophysics.jl](https://github.com/CliMA/CloudMicrophysics.jl), to superdroplet schemes, to spectral bin schemes that include a spectrum of droplet sizes, ice particle shapes and aerosol types.
