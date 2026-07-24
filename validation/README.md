# Breeze.jl validation studies

In-depth studies that validate Breeze.jl against published benchmarks and quantify the
sensitivity of the results to numerics and physics.

## How this differs from `examples/`

The `examples/` are short, lightweight demonstrations integrated into the documentation:
`docs/make.jl` runs them through Literate, *executing* each to capture its output and figures.
The lighter ones build on every PR; heavier ones are gated with `build_always = false` so they
build only on `main` or on demand (`BREEZE_BUILD_ALL_EXAMPLES = true`) — but all are cheap
enough to run as part of the docs build.

Validation studies are the opposite. They are **in-depth** and **computationally expensive**
— typically multi-day global or high-resolution runs that take minutes to hours on a GPU, and
sometimes a sweep of several such runs. They are therefore **run manually** on appropriate
hardware and are deliberately kept **out of CI and the automatic docs build**. This directory
holds them, together with a manual *literate-and-publish* step that renders each study's
committed narrative and figures to a GitHub-viewable markdown document **without re-running
the simulations**.

## Layout

```
validation/
├── Project.toml          # rendering tooling (Literate + Documenter)
├── make.jl               # manual literate-and-publish driver
└── <Study>/              # one self-contained study per directory
    ├── Project.toml      #   the study's run environment (heavy deps: Breeze, CUDA, …)
    ├── README.md         #   how to run this study, expected results
    ├── *.jl              #   Literate-formatted scripts (the simulation + the study)
    ├── *.png             #   committed figures produced by running the study
    └── *.md              #   the rendered, published document (from make.jl)
```

Each study is self-contained: it carries its own `Project.toml` run environment so it can be
reproduced independently, and it commits the figures it produces so the rendered document is
viewable without re-running anything.

## Reproducing a study

Run inside the study's own environment, on suitable hardware (most are GPU runs):

```bash
cd validation/<Study>
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. <study_script>.jl
```

See each study's `README.md` for the specific scripts, configurations, and expected results.

## Literate-and-publish

Rendering is decoupled from running and is cheap (CPU only — needs only Literate, plus
Documenter for the HTML site). From the repository root:

```bash
julia --project=validation -e 'using Pkg; Pkg.instantiate()'
julia --project=validation validation/make.jl                      # markdown, all studies
julia --project=validation validation/make.jl DCMIP2016_TC         # markdown, selected study/studies
julia --project=validation validation/make.jl --html              # HTML site, all studies
julia --project=validation validation/make.jl --html DCMIP2016_TC # HTML site, selected studies
```

Both modes render with `execute = false`: they do **not** re-run the simulations, but embed
the already-committed figures the documents reference.

- **Markdown (default).** Writes each `<study>.md` in place next to its figures; renders
  directly on GitHub. Re-run after updating a study's narrative or figures, and commit the
  regenerated markdown.
- **HTML (`--html`).** Builds a browsable Documenter site in `validation/build/` (open
  `build/index.html`): the top-level `README.md` is the home page and each study is an entry in
  the sidebar table of contents. The site is generated, not committed (`build/` is gitignored).

Which scripts are rendered is set by the `STUDIES` registry at the top of `make.jl`. Each
entry is a tuple `(subdirectory, document_script, title)` naming the single Literate document
to render in that study directory (e.g. the intercomparison, not the simulation generator it
`include`s). The optional `<Study>` argument selects entries by their **subdirectory** — the
first tuple element — so `validation/make.jl DCMIP2016_TC` renders only that study.

## Studies

| Study | Description |
|-------|-------------|
| [DCMIP2016 tropical cyclone](DCMIP2016_TC/dcmip2016_tc_intercomparison.md) | Reed–Jablonowski tropical cyclone: intensification benchmark and a resolution × advection × vertical-grid sensitivity intercomparison. |

## Adding a study

1. Create `validation/<Study>/` with its own `Project.toml`, Literate-formatted script(s),
   and a `README.md`.
2. Run it on appropriate hardware and commit the resulting figures.
3. Register it in `validation/make.jl`'s `STUDIES` as `(subdirectory, document_script, title)`,
   and add a row to the table above.
4. Render it with `make.jl` (preview the site with `make.jl --html`); commit the generated markdown.
