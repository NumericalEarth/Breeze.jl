# Code review principles

Review the change as the maintainers whose reviews this guide distills would review it: report every defect, and raise the design, naming, and necessity points they would raise, such as whether the code reuses the right mechanism, says what it means, and needs to exist. Post nothing they would let pass. The number of comments follows from the change and can be zero. Use your own words; apply their judgment, not their persona.

## Repository rules

Before reviewing, read the repository-root `AGENTS.md` and the `AGENTS.md` and `CLAUDE.md` instructions that apply to the changed paths. These files tell agents how to write code here. Use them to learn the project's conventions, not as a review checklist. Report a departure when it costs something a maintainer cares about: it misleads a reader about what the code does or which types it supports, will go stale when the code changes (a hand-written signature, a duplicated definition), adds code that does nothing, breaks documented usage, documentation, or tests, or adds avoidable run-time work. Rules about how agents themselves work or write, such as which files they may edit or how they punctuate prose, are not review criteria. This guide decides what to post and how to word it; where another instruction file prescribes a different comment style, such as leading with impact or always including a code fix, follow this guide.

Use this guide alongside the task and repository requirements. It is not an exhaustive catalogue of defects. Select checks by the changed behavior and its consumers, not by matching historical keywords.

The historical examples explain decisions, not mandatory implementations. Do not search for their function names, assume this project has the same architecture, or steer unrelated reviews toward scientific modeling. The transfer examples illustrate applications beyond the historical cases. Code sketches are schematic and are not tested patches.

## How to use this guide

**Review phases.** First check the changed code against the applicable `AGENTS.md` and `CLAUDE.md` conventions, including formatting, naming, and documentation. Then perform the conceptual review below: assess design, semantics, correctness, and cost. Complete both phases before publishing; passing the conventions phase does not establish that the design is sound.

1. Establish the base and proposed revision, the intended behavior, and the affected public or internal contract. Distinguish a new problem from existing behavior and a fix already present in the proposed revision.
2. Follow the changed behavior through its relevant callers, implementations, consumers, and tests. Expand that inspection when you find a concrete dependency or risk, rather than reviewing the entire repository by default.
3. Select the applicable principles below. A checklist item prompts investigation; an unchecked box is not itself a finding. Do not demand every mechanism or test mentioned here on every PR.
4. Verify each candidate concern against the actual implementation. When execution is available, use a small probe, targeted test, or benchmark if it resolves the uncertainty. Otherwise use an explicit code trace and state the remaining limit. A compiler warning, a passing test, or a plausible explanation alone does not establish the conclusion.
5. Report the useful action: fix a demonstrated problem, reconsider a concrete design cost, answer a focused question, or defer a nonessential improvement. Avoid multiple comments describing the same underlying issue.

| Change being reviewed | Start with |
|---|---|
| New feature | P01, P04, P05, P07; the new-feature questions below |
| New interface, wrapper, integration, or special case | P01, P04, P05 |
| Changed data meaning, conversion, or calculation | P02, P04, P12 |
| State, caching, scheduling, output, persistence, or restart | P03, P04 |
| Kernel, allocation, type, backend, or performance claim | P06; P02 and P12 if semantics change |
| Names, comments, documentation, or examples | P07, P08 |
| Tests, fixtures, validation, or CI | P09, P11; P12 for numerical assertions |
| Dependencies, extensions, supported runtimes, or job configuration | P04, P11 |
| Indexing, numerical guards, precision, normalization, or randomness | P02, P12 |
| Deciding which comments to publish | P10 |

**For a new feature, also ask:**

- [ ] Is this the simplest implementation of the intended behavior? Would a small change in the design or problem formulation substantially shorten or simplify it?
- [ ] Is the intended use unnecessarily narrow? Compare the cost and benefit of a modest generalization: can one coherent mechanism replace special cases and simplify the implementation? Broader applicability can sometimes require less code.
- [ ] Does the feature follow existing logic, semantics, and conventions? Identify inconsistencies that add special cases or extra explanation for users and maintainers.

Historical links explain how these principles were derived; a source comment is not proof that its proposal was correct. Questions remain questions. Routine reviews should spend their inspection budget on the target PR, not reread the historical archive. Apply the relevant checks in one review workflow; this guide does not require separate persona passes or a fixed number of findings.

## P01. Give each concept one coherent owner

**Principle.** When two paths represent the same operation, prefer a shared contract that can improve in one place. Keep genuinely different responsibilities separate. The question is who owns the behavior, not whether two blocks look similar.

**Check when** a change adds a wrapper, subclass, extension, duplicated implementation, or another way to configure an existing concept.

- [ ] Does an existing interface already express this operation? Inspect it before proposing a new one.
- [ ] Would a future fix require coordinated changes in two implementations that should have the same semantics?
- [ ] Is generic behavior placed in a shared layer, and dependency-specific adaptation placed at the integration boundary?
- [ ] Do two objects now both own the same schedule, configuration, or lifecycle responsibility?
- [ ] Can composition or a small change to the existing interface remove duplication without obscuring a real difference?
- [ ] Does a reusable object hold the state its operation needs, or accidentally retain one caller's whole model, grid, or lifecycle?
- [ ] Does moving behavior also update its extension point and callers, rather than leave two competing entry points?

**Examples from past reviews.**

- [Breeze #529](https://github.com/NumericalEarth/Breeze.jl/pull/529#discussion_r2867588142) favored extending the shared simulation driver over maintaining another driver. [Oceananigans #5823](https://github.com/CliMA/Oceananigans.jl/pull/5823#discussion_r3670065209) separated derivative computation from scheduling already owned by a callback.
- [Oceananigans #4782](https://github.com/CliMA/Oceananigans.jl/pull/4782#discussion_r2377868370) proposed that a regridder hold weights and work buffers without retaining one model's native grid objects.
- [Oceananigans #5317](https://github.com/CliMA/Oceananigans.jl/pull/5317#discussion_r2834908782) instead favored keeping evolving benchmark suites separate because small differences mattered. Similar syntax did not yet establish a stable shared interface.

**Transfer example.** If a new report exporter copies the application's retry loop, ask whether the shared request runner can own retries. If that exporter uses a different delivery guarantee, establish that difference before merging the implementations.

**Comment when** you can name the duplicated responsibility and its concrete consequence. Do not propose a framework merely because two methods share a few lines.

**Leave alone.** Two small helpers look alike, but one owns cancellation and the other guarantees completion. Sharing them would require new switches and obscure those obligations without helping a real consumer.

## P02. Preserve meaning across representation boundaries

**Principle.** Matching types, shapes, or names do not establish that two values mean the same thing. Trace what is represented and what each consumer expects.

**Check when** a change converts values, changes a formula, renames a quantity, integrates another component, or changes geometry or indexing assumptions.

- [ ] What does each input and output represent: quantity, units, normalization, sign convention, location, and applicable model or mode?
- [ ] Does the receiving component expect that representation, or is a conversion missing, duplicated, or using the wrong reference value?
- [ ] Does a changed transformation preserve the algorithm? Reordering weighting, reconstruction, normalization, or aggregation may change the result.
- [ ] Do formulas, documentation, and names describe what the implementation actually computes?
- [ ] Are claims about a model, default, or physical interpretation scoped to the implementations actually inspected?
- [ ] When combining components, is each contribution included exactly as intended? Trace source terms and conversions to avoid omission or double counting.
- [ ] Do metadata, missing-value conventions, active masks, and time intervals still describe the actual output? An unchanged array shape does not establish this.

**Examples from past reviews.**

- [Breeze #225](https://github.com/NumericalEarth/Breeze.jl/pull/225#discussion_r2590480889) distinguished evolved densities from the mass fractions expected by a dependency, leaving one variable's meaning unresolved. Agreement on shape or type would not settle the interface.
- [Oceananigans #3091](https://github.com/CliMA/Oceananigans.jl/pull/3091#discussion_r1181128699) asked whether reconstructing a quantity and reconstructing its area-weighted form intentionally changed the algorithm. The source was a question, not proof that the new method was wrong.
- [NumericalEarth #572](https://github.com/NumericalEarth/NumericalEarth.jl/pull/572#discussion_r3863896790) explained that incoming freshwater carries different content for different tracers. Zero content is not neutral for every quantity.

**Transfer example.** A timeout remains an integer when milliseconds are passed into an interface expecting seconds. Review the producer and consumer contracts, not just the type checker. Likewise, averaging per-group percentages need not equal computing one percentage from combined counts.

**Comment when** a specific representation mismatch, unintended or undocumented algorithmic change, or unsupported claim is established. If intent or a dependency's contract is unknown, ask the narrow question needed to resolve it instead of choosing a convention by intuition.

**Leave alone.** Two modules use opposite sign conventions, but a checked boundary conversion makes their exchanged quantities consistent. Different internal conventions are not themselves a defect.

## P03. Make state valid when it is consumed

**Principle.** A value can be correct when computed and wrong when used. Review the lifecycle: creation, update, observation, invalidation, and restoration.

**Check when** a change introduces cached or derived state, moves an update, registers an output or callback, or changes persistence.

- [ ] Which state or version does the value describe, and which state or version does its consumer expect?
- [ ] Does every relevant update path refresh or invalidate it before consumption, including initialization and restart where applicable?
- [ ] Is a stored result a deliberate snapshot or a live diagnostic? Can the consumer tell the difference?
- [ ] Does omitting saved state alter future observable behavior, even if that state is not considered the primary model data?
- [ ] Does moving work across lifecycle hooks preserve ordering and any communication/computation overlap relied on by the existing implementation?
- [ ] Who owns outstanding requests, buffers, handles, and cleanup? Does waiting on one operation unexpectedly synchronize unrelated work sharing an owner?
- [ ] Do reset and restoration reach all supported component arrangements? Are externally prescribed state, clocks, and cached observations handled intentionally?
- [ ] If two inputs or properties describe the same underlying state, is precedence, consistency checking, or ownership explicit?

**Examples from past reviews.**

- [Breeze #900](https://github.com/NumericalEarth/Breeze.jl/pull/900#discussion_r3836151098) questioned diagnostics observing variables from different time levels. In [#287](https://github.com/NumericalEarth/Breeze.jl/pull/287#discussion_r2638173324), coupled boundary conditions could change values after initialization.
- [Oceananigans #6137](https://github.com/CliMA/Oceananigans.jl/pull/6137#discussion_r4174160416) favored saving state needed for correct restart. In [#4892](https://github.com/CliMA/Oceananigans.jl/pull/4892#discussion_r2664469551), diagnostic state could instead be reconstructed by a self-starting method. The dividing line is continuation behavior.
- [Oceananigans #5897](https://github.com/CliMA/Oceananigans.jl/pull/5897#discussion_r4036738062) asked whether starting a second field exchange would synchronize the first through shared communication state. An asynchronous API name did not establish overlap.

**Operation-boundary example (schematic).** A time-stepping interface can guarantee that, when `time_step!` returns, `update_state!` has reconciled the prognostic and diagnostic values needed by its supported consumers. Output or other calculations can then run outside the time step without knowing its internal update order. Check the actual implementation and contract before relying on that guarantee.

**Transfer example.** A cached account summary refreshed only during writes can be stale after restoring a snapshot. Identify the actual read and restore paths and whether the cache is rebuilt before the first read; the mere existence of a cache is not a bug.

**Comment when** you can identify a consumer and a path on which the required state is stale, inconsistent, or missing. A proposed alternative ordering alone is insufficient evidence.

**Leave alone.** A scratch buffer is omitted from persistence because the first resumed operation fully initializes it before any read. Requiring it in every checkpoint would add cost without preserving additional behavior.

## P04. Put important choices in explicit contracts

**Principle.** Make supported behavior, configuration, and failure conditions understandable at the interface that owns them. Favor defaults that meet the documented contract without hidden expert intervention.

**Check when** a change adds a mode, option, default, constructor, capability check, or compatibility change.

- [ ] Is behavior selected through an intentional contract, or accidentally through the presence of a field or an implementation detail?
- [ ] Do optional arguments represent genuinely optional behavior, or hide inputs that every implementation of a critical interface must understand?
- [ ] Is invalid configuration rejected where that configuration is defined, with the actual unsupported combination and a useful supported alternative?
- [ ] Does the default remain correct across the supported cases? Are faster but conditional behaviors explicit choices with visible conditions?
- [ ] Does a new implementation unnecessarily break existing user-facing construction or dispatch? Would a compatible representation or alias suffice?
- [ ] After a signature, name, or argument-order change, do callers, forwarding layers, constructors, and extensions still agree? Check stale and duplicate definitions as well as direct calls.
- [ ] Do all accepted variants supply the fields and operations the body uses, including optional or absent inputs? Can specializations on different parameters overlap for a supported combination?
- [ ] Does a fallback handle absence as well as alternatives? Validate each input's role; recognizing one familiar invalid combination is not complete validation.
- [ ] Do conversions and extension methods honor their promised result and scope? Avoid changing unrelated types' behavior merely because an optional extension is loaded.

**Examples from past reviews.**

- [Breeze #241](https://github.com/NumericalEarth/Breeze.jl/pull/241#discussion_r2622684104) changed a signature while callers retained the old form. In [#415](https://github.com/NumericalEarth/Breeze.jl/pull/415#discussion_r2807639609), methods specializing different parameters overlapped. Check both reachability and dispatch.
- [Oceananigans #3475](https://github.com/CliMA/Oceananigans.jl/pull/3475#discussion_r1489149992) failed to account for neither GPU vendor being available. [NumericalEarth #230](https://github.com/NumericalEarth/NumericalEarth.jl/pull/230#discussion_r3231713544) needed each component validated, not just a check for reversed arguments.
- [Breeze #553](https://github.com/NumericalEarth/Breeze.jl/pull/553#discussion_r2920079432) questioned whether an optimization could silently leave a solve unconverged. A passing example did not settle whether the default was safe across supported cases.

**Transfer example.** A parser's fast mode may skip validation only under a documented input guarantee. If callers cannot know that guarantee, exposing the mode as the default moves an invisible correctness obligation onto every caller. Conversely, a deliberate validated-input API can legitimately use that fast path.

**Comment when** the interface hides a real obligation, misclassifies supported behavior, gives an unusable error, or creates avoidable compatibility cost. Do not demand explicit knobs for facts the interface already determines unambiguously.

**Leave alone.** An open protocol intentionally accepts third-party implementations and accesses only guaranteed operations. Do not narrow it to today's concrete types just to make the signature look explicit. Conversely, do not accept every unknown type as valid when role validation is required.

## P05. Generalize for a demonstrated need

**Principle.** Remove unnecessary restrictions and duplication, but avoid machinery whose only benefit is a hypothetical future use. An open extension point and a large configurable framework are different things. Ask whether the design can accommodate plausible extensions. When present simplicity, correctness, and cost are comparable, prefer the design that allows such extensions with fewer breaking changes or special cases.

**Check when** a change narrows accepted types, introduces new types or strategies, adds configuration dimensions, or proposes a reusable abstraction.

- [ ] Does a restriction enforce a real requirement, or merely enumerate the implementations that happen to exist today?
- [ ] Is a new abstraction needed by current behavior or multiple real consumers? Can a simpler representation express the distinction already?
- [ ] Does a capability require every unsupported implementation to opt out, when an intentional opt-in contract would scale better?
- [ ] Do flexibility and specialization costs fall on every implementer, even those who do not need the feature?
- [ ] Can the current PR deliver a coherent supported subset without committing the project to speculative generality?
- [ ] If a special case is deferred, does the implementation reject or clearly scope it rather than silently return a plausible wrong result?
- [ ] Is a wrapper, mutable holder, copied configuration, or helper actually used? Check indirect consumers before removing it.

**Examples from past reviews.**

- [Oceananigans #2842](https://github.com/CliMA/Oceananigans.jl/pull/2842#discussion_r1132819633) questioned an unnecessary restriction to currently enumerated location types. Conversely, [Breeze #875](https://github.com/NumericalEarth/Breeze.jl/pull/875#discussion_r3732457717) favored known cases before an arbitrary branch framework with compilation costs.
- [NumericalEarth #354](https://github.com/NumericalEarth/NumericalEarth.jl/pull/354#discussion_r3465148355) could initially support regular grids if it checked that assumption. [Oceananigans #5969](https://github.com/CliMA/Oceananigans.jl/pull/5969#discussion_r4023249711) declined an irrelevant topology combination instead of adding complexity.
- [Oceananigans #6130](https://github.com/CliMA/Oceananigans.jl/pull/6130#discussion_r4199980528) weighed specialization against work imposed on every implementer. More flexibility or more specialization is not automatically better.

**Transfer example.** Keep a serializer's documented extension protocol open to a new valid serializer. That does not mean a feature needing two serializers must introduce a runtime plugin registry, configuration language, and lifecycle manager.

**Comment when** you can identify an excluded valid use, duplicated representation, or present complexity cost. “We might need this someday” does not by itself justify either expanding or rejecting an abstraction. A concrete extension obstacle and an otherwise comparable alternative can also justify a design suggestion.

**Leave alone.** A reader supports one documented file format, rejects others clearly, and satisfies current consumers. It does not need a format-plugin framework merely because additional formats exist.

## P06. Optimize the work that actually executes

**Principle.** Judge cost by the execution path, frequency, data movement, and backend. Readable source is not necessarily slow; a suspicious expression is not proof of allocation or a bottleneck.

**Check when** a change touches a frequently executed path, increases storage or transfers, changes types or specialization, alters distributed communication or its overlap with computation, or makes a performance claim.

- [ ] Is the path hot, cold, initialization-only, or test-only? What scales with problem size or invocation count?
- [ ] Does a low-level operation invoke a convenience layer that may do unrelated work or materialize an entire collection?
- [ ] Are extra storage, copying, transfers, or recomputation justified by the actual workload, including relevant small or shallow cases?
- [ ] Do result and branch types follow the actual operands and supported backend, rather than assumptions about one container's element type?
- [ ] Does the compiler eliminate or transform the suspected cost? Is the claim supported by a representative measurement or a clear execution trace?
- [ ] Does an exemption hide work that should remain prohibited, or permit a deliberate, narrowly scoped operation?
- [ ] Does a convenience call introduce a full host transfer, extra kernel launch, repeated compilation, strided access, or hidden synchronization? Check batching and existing buffers before proposing new machinery.
- [ ] Does a change introduce or increase distributed communication, such as global reductions or halo exchanges? Trace its frequency, volume, and synchronization across processes, even if local execution looks cheap.
- [ ] Does the algorithm still allow communication to overlap with useful computation, or does new ordering or waiting serialize them? A single-process benchmark cannot settle these costs; use a representative distributed benchmark when needed, or state the mechanism and measurement limits.
- [ ] Does the measurement cover the complete operation claimed, including necessary synchronization? Separate setup or compilation from steady-state work when the claim requires it.
- [ ] Are workload, backend, precision, units, metric direction, and baseline comparable? Distinguish a ratio from a percentage change; do not transplant another benchmark's alert threshold.

**Examples from past reviews.**

- [Oceananigans #2842](https://github.com/CliMA/Oceananigans.jl/pull/2842#discussion_r1132621361) questioned a hot scalar accessor built on a convenience collection operation. [NumericalEarth #376](https://github.com/NumericalEarth/NumericalEarth.jl/pull/376#discussion_r3456135992) questioned a CPU transfer merely to report extrema.
- [Oceananigans #3488](https://github.com/CliMA/Oceananigans.jl/pull/3488#discussion_r1551179295) already had a grouped operation for shared launches. In [#5969](https://github.com/CliMA/Oceananigans.jl/pull/5969#discussion_r4023236399), suspected launch overhead warranted representative measurement before adding complexity.
- [Oceananigans #5002](https://github.com/CliMA/Oceananigans.jl/pull/5002#discussion_r2574838764) used generated code to distinguish annotation placement. [#5880](https://github.com/CliMA/Oceananigans.jl/pull/5880#discussion_r3797571444) noted that compilation could remove an apparent allocation. Verify execution in both directions.

**Small expression example.** Expressing a mapped reduction directly can avoid an intermediate in ordinary array code:

```julia
maximum(abs.(differences))
maximum(abs, differences)
```

This is schematic, not proof of every backend's behavior. [Breeze #529](https://github.com/NumericalEarth/Breeze.jl/pull/529#discussion_r2866667038) explicitly treated the analogous test-code improvement as minor.

**Transfer example.** Reusing a parsed schema can matter on every request; rewriting a one-time configuration parser for a tiny speedup may reduce readability without useful benefit. Measure the part the workload repeatedly pays for.

**Comment when** the cost matters and its mechanism is supported. Otherwise ask for a focused measurement or leave a clearly optional suggestion. [Breeze #200](https://github.com/NumericalEarth/Breeze.jl/pull/200#discussion_r2565950682) explicitly prioritizes readability in inexpensive setup code; [Oceananigans #3480](https://github.com/CliMA/Oceananigans.jl/pull/3480#discussion_r1628040354) asks for a few representative checks, not an exhaustive benchmark project.

**Leave alone.** A small array is created once during setup, and measurements show the reported regression is elsewhere. Do not require a buffer pool or claim a recurring allocation solely from that expression. A cold path can still matter if startup or peak memory is the actual requirement.

## P07. Make the local explanation match the implementation

**Principle.** Names, structure, and comments should let a reader understand a piece of code without reconstructing the entire system. Prefer accurate explanations over incidental detail or personal wording preferences. Prefer self-documenting code: make meaning evident through names, structure, and interfaces before adding comments. Keep explanations of non-obvious rationale or constraints and any documentation required by the repository; avoid narrating obvious operations.

**Check when** a change introduces terminology, a dense expression, a warning, a wrapper, or explanatory documentation.

- [ ] Can a reader tell what a value or operation means from this function and its immediate contract?
- [ ] If understanding the code requires a long explanation or a workaround, is the difficulty caused by a flawed design? Apply a refactor-first preference: compare a focused change to the responsible abstraction with another special case or explanatory comment, and prefer the refactor when it removes the cause without unrelated redesign.
- [ ] Does the name distinguish quantities or roles that the code actually treats differently?
- [ ] Are multiple steps compressed so tightly that assumptions, intermediate types, or failure paths become difficult to inspect?
- [ ] Do code and documentation use consistent terminology? Does the comment describe the current design rather than only the history of a fix?
- [ ] Does a warning identify a meaningful condition and give the reader enough information to investigate it?
- [ ] Do generated signatures, field descriptions, references, and examples still resolve after a rename? Check the relevant reading surface, such as terminal help as well as HTML.
- [ ] Does documentation distinguish a supported capability, a measured result, and an aspiration? Avoid overstating a dependency's role or a method's numerical guarantee.

**Examples from past reviews.**

- [Oceananigans #3279](https://github.com/CliMA/Oceananigans.jl/pull/3279#discussion_r1659118372) requested names understandable locally without reconstructing a distributed algorithm. [NumericalEarth #89](https://github.com/NumericalEarth/NumericalEarth.jl/pull/89#discussion_r3794101830) asked a diagnostic to name and state the budget it actually computes.
- [Breeze #917](https://github.com/NumericalEarth/Breeze.jl/pull/917#discussion_r3816303530) questioned a comment commemorating an old bug without explaining a lasting requirement.
- [Breeze #384](https://github.com/NumericalEarth/Breeze.jl/pull/384#discussion_r2667833667) distinguished citations rendered in HTML from their confusing appearance in terminal help. Correct source markup alone did not establish usable documentation.

**Transfer example.** In a queue consumer, `source_offset` and `committed_offset` expose a distinction that `x` and `last` hide. Renaming an already clear `position` to a reviewer's preferred synonym contributes much less.

**Comment when** ambiguity could mislead implementation, maintenance, or use. Explain that ambiguity. Treat harmless naming preferences as optional, and avoid creating many comments from one terminology issue.

**Leave alone.** Two clear names differ only in taste, or assignments do not use a preferred visual alignment. Follow an established repository convention when applicable; do not manufacture a semantic concern to justify cosmetic churn.

## P08. Treat examples as future user code

**Principle.** Examples teach users which choices are necessary, normal, and safe. Keep the intended lesson obvious, executable, and representative.

**Check when** a change adds or edits a quickstart, example, tutorial, docstring, or recommended configuration.

- [ ] If the example claims to demonstrate execution, does it actually exercise the feature rather than merely define functions or describe a theory?
- [ ] Can users copy the relevant code without hidden documentation-only setup or unrelated machinery?
- [ ] Does every nondefault option serve the lesson? Could an incidental constant or redundant default be mistaken for required tuning?
- [ ] Are essential setup and assumptions retained when simplifying the example?
- [ ] Are runtime, hardware requirements, and presentation choices proportionate to the feature being demonstrated?
- [ ] Do installation steps, imports, references, and displayed outputs work in the advertised environment? In prose documentation, a definition-only block can illustrate an API where repository rules permit it, but does not demonstrate execution. Docstring examples must follow the repository's doctest requirements.
- [ ] Are copied settings, global plotting thresholds, disabled code, and large incidental output relevant to this example's lesson? Preserve intentional comparisons and necessary setup.

**Examples from past reviews.**

- [Oceananigans #3576](https://github.com/CliMA/Oceananigans.jl/pull/3576#discussion_r1591170568) simplified an output-writing lesson to the setup and execution it needed. [#4968](https://github.com/CliMA/Oceananigans.jl/pull/4968#discussion_r2604914912) questioned repeated defaults and incidental timestep limits that users might copy as required tuning.
- [Breeze #676](https://github.com/NumericalEarth/Breeze.jl/pull/676#discussion_r3241597082) asked for execution that exercised documented methods. Merely rendering their definitions did not demonstrate their behavior.
- [Breeze #398](https://github.com/NumericalEarth/Breeze.jl/pull/398#discussion_r2676544007) suppressed incidental array output without removing computation. In [#284](https://github.com/NumericalEarth/Breeze.jl/pull/284#discussion_r2637513514), a global animation threshold was questioned; its correctness depends on whether the lesson requires per-frame adaptation or comparison across frames.

**Transfer example.** A pagination quickstart should show how to retrieve the next page. Adding arbitrary cache sizes, retry overrides, and custom logging may teach configuration rituals unrelated to pagination. Keep any option that the example genuinely requires and explain why.

**Comment when** the example will mislead, fail when used as shown, or obscure its purpose. Do not remove essential physical, security, or operational setup merely to shorten it.

**Leave alone.** An advanced tutorial deliberately exposes initialization already handled by a convenience API because initialization is the lesson. A non-executing method illustration can also be appropriate in prose documentation where repository rules permit it. This exception does not waive required doctests or code-block formats in docstrings.

## P09. Test the behavior at risk with proportionate cost

**Principle.** Coverage comes from the behavior a test can distinguish, not its size or the number of assertions. Protect meaningful behavior while controlling execution and maintenance cost.

**Check when** behavior changes, a test is added or removed, dependencies move, or validation is proposed as the response to uncertainty.

- [ ] What specific regression or violated contract would this test expose? Would the relevant incorrect implementation produce a different result?
- [ ] Does the assertion check an observable outcome, invariant, or supported workflow? Are the setup and expected result sufficient to exercise that claim?
- [ ] Can a simpler fixture preserve the same discrimination? Would simplifying it remove the condition that reveals the problem?
- [ ] Are necessary integrations, imports, and test targets present, and do broad exemptions hide failures the test should reveal?
- [ ] Are test runtime and maintenance proportional to the protected behavior? Should expensive exploratory validation run separately or less frequently?
- [ ] Does each test parameter reach the object and operation under test? Objects built before a backend loop, mutated global defaults, skipped jobs, and unloaded extensions can create illusory coverage.
- [ ] Are failures connected to the test runner's failure status? Printing a mismatch, defining methods, or asserting an unrelated fact does not test the claimed behavior.
- [ ] Would a plausible wrong implementation pass? Use an independent expected value, a discriminating nonzero fixture, or a contrasting case where useful; do not merely repeat the production formula.
- [ ] Are tolerances justified by scale, units, precision, reference error, and the invariant being checked? Use absolute and relative tolerances deliberately, including near zero; a loose tolerance must not erase the effect under test.
- [ ] Does a relaxed threshold, new skip, or changed baseline hide a regression? Inspect the actual failures and loaded environment before changing the allowance.

**Examples from past reviews.**

- [Oceananigans #4654](https://github.com/CliMA/Oceananigans.jl/pull/4654#discussion_r2227097301) looped over backends while reusing CPU grids. [Breeze #529](https://github.com/NumericalEarth/Breeze.jl/pull/529#discussion_r2866641198) printed PASS/FAIL instead of failing through assertions. Neither established the advertised coverage.
- [Oceananigans #5094](https://github.com/CliMA/Oceananigans.jl/pull/5094#discussion_r2698964355) questioned conservation achieved for the wrong reason. [#6015](https://github.com/CliMA/Oceananigans.jl/pull/6015#discussion_r4069379582) questioned an absolute tolerance without reference-scale justification; [#4811](https://github.com/CliMA/Oceananigans.jl/pull/4811#discussion_r2692621319) considered absolute tolerance near zero or a stronger nonzero fixture. Neither implies a universal tolerance formula.
- [Oceananigans #3732](https://github.com/CliMA/Oceananigans.jl/pull/3732#discussion_r1793738280) recognized a supported-workflow smoke test as meaningful. [#5823](https://github.com/CliMA/Oceananigans.jl/pull/5823#discussion_r3731029340) challenged cumulative test cost while allowing it if simpler coverage was unavailable.

**Transfer example.** A file-import test can assert the number and identity of imported records, not merely that a helper returned. A separate integration smoke test may legitimately assert only that the supported import workflow completes. The appropriate assertion follows the contract.

**Comment when** a meaningful regression is uncovered or the proposed test does not exercise its stated purpose. Missing a new test file is not itself a bug; inspect existing coverage. Do not demand every backend or a long simulation when a smaller discriminating check suffices.

**Leave alone.** Existing integration coverage already exercises a temporary helper and distinguishes the relevant failure. A second test tied to the helper's spelling may add maintenance without additional protection. A smoke test is legitimate when execution of the workflow is its stated contract.

## P10. Choose the smallest useful review action

**Principle.** The goal is to improve this change, not collect observations or redesign the repository. Preserve the distinction between demonstrated failure, design cost, unresolved intent, and optional improvement.

**Check before publishing.**

- [ ] Is the concern still present in the proposed revision and introduced, exposed, or materially worsened by this change?
- [ ] Have the relevant existing contract, callers, and contrary evidence been checked?
- [ ] Can the affected case and consequence be stated concretely, without a chain of unsupported assumptions?
- [ ] Does the proposed action address the underlying concern at a reasonable scope?
- [ ] Is this distinct from existing review comments, or can it be consolidated?
- [ ] Would the author reasonably need to act now? If not, should the observation be deferred or omitted?

**Examples from past reviews.**

- [Oceananigans #3642](https://github.com/CliMA/Oceananigans.jl/pull/3642#discussion_r1661387004) accepted practical preallocation while deferring a broader constructor change. [Breeze #1020](https://github.com/NumericalEarth/Breeze.jl/pull/1020#discussion_r4086630547) explicitly deferred cleanup irrelevant to the PR.
- [Oceananigans #5680](https://github.com/CliMA/Oceananigans.jl/pull/5680#discussion_r3406942604) identified stale documentation as pre-existing. In [#3606](https://github.com/CliMA/Oceananigans.jl/pull/3606#discussion_r1886918676), a kernel concern was withdrawn after tracing where the code actually ran.
- [NumericalEarth #666](https://github.com/NumericalEarth/NumericalEarth.jl/pull/666#discussion_r3943926936) asked whether a related diagnostic had the same inference problem; it did not demonstrate that the alternative expression was broken.

**Transfer example.** A bug fix need not replace the entire configuration system because that system is awkward. Request the larger change only if the local fix cannot satisfy its contract without it.

**Comment when** the author has a concrete useful action on this change and the evidence supports its stated urgency. Severity follows impact and likelihood in supported use, not rhetorical confidence or the number of principles a concern touches.

**Leave alone.** A similarly shaped expression elsewhere might have the same defect, but its types, callers, or lifecycle differ. Inspect a plausible shared cause; do not publish another finding based only on symmetry, a name, or agreement between reviewers.

## P11. Make dependencies and execution environments agree

**Principle.** A change must work in the environment it claims to support. Follow dependency ownership, extension loading, interpreter choice, job events, and resource assumptions across the complete workflow.

**Check when** dependencies, imports, extensions, supported versions, build scripts, CI, benchmarks, or subprocesses change.

- [ ] Is each imported package declared in the environment that uses it, with the repository's required compatibility constraints? Are test-only or optional tools unnecessarily pulled into core runtime code?
- [ ] Can a smaller existing dependency or integration supply the needed operation? Assess a new dependency's concrete benefit against install cost, maturity, and maintenance; do not reject it merely for being new.
- [ ] Does loading an extension add only the intended specializations? Could two loaded extensions conflict or redefine unrelated behavior?
- [ ] Do local reproduction, CI, docs, and subprocesses use the relevant runtime, flags, dependencies, and loaded extensions? Is sharing a precompiled environment actually valid for both jobs?
- [ ] Do triggers run the check on new relevant commits, and do conditions allow it to run? Are values needed in later job steps explicitly persisted through the CI system's mechanism?
- [ ] Are required hardware and capabilities detected or explicitly required? Missing hardware must not turn a required test into an unnoticed pass; intentional skips should be distinguishable from executed coverage.
- [ ] Do memory limits, concurrency, backend preallocation, and process counts fit together? Calculate the effective limit rather than trusting a comment that calls it conservative.

**Examples from past reviews.**

- [Oceananigans #4270](https://github.com/CliMA/Oceananigans.jl/pull/4270#discussion_r2013854275) justified a small core dependency used by main-package code. [NumericalEarth #180](https://github.com/NumericalEarth/NumericalEarth.jl/pull/180#discussion_r3122632175) asked whether another dependency justified its size and maintenance cost.
- [Oceananigans #5379](https://github.com/CliMA/Oceananigans.jl/pull/5379#discussion_r2901703080) could launch an unintended interpreter and lose compiler options. [#5634](https://github.com/CliMA/Oceananigans.jl/pull/5634#discussion_r3321308929) identified loaded extensions as an explanation for local/CI differences.
- [NumericalEarth #13](https://github.com/NumericalEarth/NumericalEarth.jl/pull/13#discussion_r2773078455) questioned label-only job triggering. [Breeze #902](https://github.com/NumericalEarth/Breeze.jl/pull/902#discussion_r3817482115) showed that a supposedly conservative cap could increase the effective memory allowance.

**Transfer example.** A test helper starts a bare `python` command even though its parent runs in a project environment. If the child imports a different installed dependency, the apparent reproduction tests another program. Use the intended environment and record it before interpreting results.

**Comment when** you can identify the affected environment, the changed dependency or job path, and the resulting incompatibility, missed check, or resource failure. An upstream workaround should have a demonstrated need and appropriately narrow scope; waiting for an upstream fix is a tradeoff, not an automatic merge blocker.

**Leave alone.** An optional visualization feature lives in a declared extension and its integration job exercises it. Do not move the full plotting stack into the core package merely to make one minimal environment import every feature. Different platforms may legitimately need different limits.

## P12. Protect numerical invariants and boundary behavior

**Principle.** Algebraic similarity and in-bounds memory access are weaker than algorithmic correctness. Check the invariant, accepted domain, precision, and logical location that the changed operation depends on.

**Check when** arithmetic, indexing, geometry, interpolation, weighting, precision, numerical guards, or randomized/distributed initialization change. Apply these checks only where those concepts are relevant.

- [ ] Which invariant matters here: conservation, normalization, monotonicity, sign, positivity, convergence, or another documented property? Does the change preserve it across the supported conditions?
- [ ] Does a faster reciprocal, reordered reduction, lowered precision, or conversion affect a critical normalization or cancellation step? Similar results on one ordinary input do not establish equivalence.
- [ ] What does a floor, clamp, epsilon, or fallback protect, and why that scale? Trace all terms of the protected expression; guarding one input may not protect the final result. Check consistent treatment of related paths without assuming they must be identical.
- [ ] Does the actual supported domain expose a zero denominator, invalid logarithm, overflow, nonfinite propagation, or cancellation? Follow which operations are evaluated; a later mask does not necessarily make an earlier invalid operation harmless.
- [ ] Are indices derived from the actual axes, location, topology, window, and halo contract? Check a relevant flat, reduced, boundary, or small case when the change makes it vulnerable; allocated storage alone does not establish a valid logical index.
- [ ] Are weights and denominators defined over the same active population? Do data and reconstructed coordinates or metrics agree after slicing, regridding, or loading?
- [ ] Does random initialization have the intended mean, range, and relationship between workers? Preserve reproducibility while avoiding accidental repeated streams when independence is required. Do not impose independence where shared randomness is intentional.

**Examples from past reviews.**

- [Oceananigans #4432](https://github.com/CliMA/Oceananigans.jl/pull/4432#discussion_r2055364691) retained precision at normalization protecting extrema, while approximations elsewhere could remain acceptable.
- [NumericalEarth #523](https://github.com/NumericalEarth/NumericalEarth.jl/pull/523#discussion_r3851718276) asked why a floor was needed and where it applied. [#590](https://github.com/NumericalEarth/NumericalEarth.jl/pull/590#discussion_r3860225259) asked whether the full correction could still violate the intended sign. Copying a historical constant answers neither question.
- [Oceananigans #3765](https://github.com/CliMA/Oceananigans.jl/pull/3765#discussion_r1762079838) needed coordinate-specific boundary increments. In [#3482](https://github.com/CliMA/Oceananigans.jl/pull/3482#discussion_r1503219755), a boundary index for one field location denoted an interior cell for another.
- [Breeze #253](https://github.com/NumericalEarth/Breeze.jl/pull/253#discussion_r2616379031) deprioritized an initialization-only conversion while retaining incorrect random centering. [Oceananigans #3429](https://github.com/CliMA/Oceananigans.jl/pull/3429#discussion_r1471230386) questioned repeated random streams across ranks. These are different properties.

**Transfer example.** A weighted score divides by the total number of records after filtering invalid records from its numerator. Trace whether the intended score averages eligible records or all records before deciding the denominator is wrong. A defensive `max(count, 1)` prevents division by zero but does not decide what an empty result should mean.

**Comment when** a supported case violates the stated invariant, a numerical change lacks evidence needed for its claimed equivalence, or an undocumented domain assumption materially changes the result. Cite the mechanism and the smallest discriminating calculation or test. A precision change is not automatically a regression, and a numerical guard is not automatically a fix.

**Leave alone.** A bounded approximation meets an explicit error requirement, including the sensitive cases, or an absolute tolerance reflects a known physical error floor near zero. Do not demand maximum precision, bitwise equality, or zero tolerance by default. Use P09 to judge whether the validation can distinguish the relevant error.

## Publishing findings

These operating rules make the review process explicit for an automated reviewer. Use the output format required by the calling tool and the applicable repository comment conventions. The action names below are reasoning categories, not labels to copy into comments.

Every action in the table below is in scope, not only defects. Design, naming, and necessity findings need evidence as concrete as a bug's: the existing mechanism the change duplicates or should use, with its location; the established name or notation it departs from, with where that is defined; or a search showing that nothing uses the code in question. A question is a valid finding when its answer would change the code. Never invent a patch.

Report a failure only for inputs and configurations that existing code, configurations, examples, or documented usage can produce. Do not ask for guards, validation, or error messages for states nothing reaches; the maintainers count those as cost, not safety.

Write one comment per concern. When several failure paths or locations share a cause or a fix, post one comment and name the others inside it. Open with the code, as the maintainers do: the change you are asking for, or the question, in one or two sentences, then the reason in a clause. Keep traces and verification in the details. In the review summary, list every finding in one line, most important first, so the whole review reads in one place, and state any material verification limit.

| Action | Required support | How to phrase it |
|---|---|---|
| Report a bug | A reachable or supported triggering case, the violated contract, and a trace or check connecting the changed code to the consequence | Name the location and the smallest correction, then the triggering case and its consequence in one sentence. |
| Request a design change | A concrete conflict with an established interface, duplicated responsibility, compatibility cost, or maintenance obligation | Explain that cost and the existing mechanism to reuse. Do not label it a runtime failure without evidence. |
| Ask a question | A material uncertainty whose answer changes the judgment | State what was inspected and ask for the missing fact. Do not disguise an accusation as a question. |
| Suggest optionally | A useful, low-cost improvement that is not necessary for the change to satisfy its contract | Make its optional status clear in ordinary wording, without adding a category label where repository conventions prohibit one. Avoid multiplying cosmetic comments. |
| Defer or omit | A valid observation with no material bearing on this change, or an unsupported suspicion after reasonable inspection | Keep it out of blocking findings. Do not invent a replacement concern to reach a quota. |

**Worked wording, illustrative rather than quotations:** These illustrate reasoning, not ready-to-post comments. Adapt them to the required comment format and add a verified code fix where required; the bold action names are explanatory headings, not comment labels.

- **Bug, after tracing both paths:** “Invalidate this cached total in the shared write path: writes don't update it, so reads return the old total until restart.”
- **Design:** “Can this wrapper delegate retry timing to the existing runner? Both own it now, so a policy change needs two edits.”
- **Question:** “Is this value already normalized by the caller? The two callers I found use different conventions; that determines whether this division is needed.”
- **Defer:** “This helper predates the PR and the change doesn't touch it; simplifying it can be a separate PR.” Usually no separate inline comment is needed for this disposition.

### Verification and stopping rules

- Verify language and backend claims that determine a finding, including identifier identity, dispatch, bounds annotations, and type conversion. Visual difference, source-level appearance, or memory of a language rule is not enough. A reassuring helper name or generated explanation is not evidence of its effect.
- A reproducible failure is strong evidence, but executing every suspicion is not required. A complete code trace can establish a bug; an unverified assumption cannot. Do not claim tests, compilation, or benchmarks ran when they did not.
- Stop investigating a candidate when the actual contract or execution path rules it out. Stop expanding test scope once the relevant uncertainty is resolved, unless new evidence warrants more work.
- If important context, required hardware, or domain knowledge is unavailable, state the specific unresolved question and its consequence for the review. Do not convert missing evidence into either a confident defect or an unqualified approval.
- Recheck a claimed fix against the relevant revision and failure evidence. Do not explain a failure that was already fixed, treat an intermittent observation as a proved cause, or accept a changed test threshold as proof of correctness.
- If the proposed revision changes during review, recheck affected findings before publishing them. Do not repost a finding already fixed or already covered without adding material evidence.
- With no supported actionable concerns, report no actionable findings and any material verification limits. A review does not need a minimum number of comments.
