# Project Review: poly_csp

Date: 2026-03-01  
Reviewer perspective: software engineering + molecular simulation

## Scope reviewed

- `README.md`
- `docs/csp_construction_report.md`
- `docs/csp_construction_workflow_staged_implementation_plan.md`
- `docs/csp_construction_ticket_stages_1_3.md`
- `conf/*`
- `src/poly_csp/*`
- `legacy/*`

## Executive assessment

The project has a strong scientific direction and a sensible staged architecture, but the current codebase is still an early scaffold rather than a usable construction engine. Core functions for geometry, monomer/template handling, polymerization, and functionalization are mostly placeholders (`...`), so the advertised pipeline is not executable yet. I found 17 placeholder stubs across 8 source files in the active package.

Scientifically, your target model (deterministic screw-symmetric CSP construction with selector pre-organization) is well chosen and directly addresses known failure modes of unconstrained MD for these systems. The main gap is implementation depth and hard validation gates.

## Current state by stage

| Stage | Intended outcome | Current status | Notes |
|---|---|---|---|
| Stage 0 | Structure, config, interfaces | Partial | Package skeleton and Hydra config exist; no tests in tree; docs and file layout diverge in places. |
| Stage 1 | Screw transform + helical backbone coords | Not implemented | `geometry/transform.py` and `chemistry/backbone_build.py` are stubs. |
| Stage 2 | Polymer graph + conformer assignment | Not implemented | `chemistry/polymerize.py` is stubbed. |
| Stage 3 | Deterministic selector attachment | Not implemented | `chemistry/selectors.py`, `chemistry/functionalization.py`, `geometry/local_frames.py`, `geometry/dihedrals.py` are stubs. |
| Stage 4 | Rotamer/H-bond pre-organization | Not implemented | No `rotamers.py`, `hbonds.py`, `symmetry_opt.py` in active package. |
| Stage 5 | OpenMM restrained relaxation | Not implemented | `src/poly_csp/mm` directory exists but is empty. |
| Stage 6 | QC metrics and quality gates | Minimal | `ordering/scoring.py` has basic clash and symmetry RMSD, but depends on stubbed transform code and is not enough for CSP ordering quality. |
| Stage 7-9 | Selector generalization, CLI maturity, performance | Not implemented | Planning docs are detailed; code is not yet there. |

## Key findings and risks

### Critical engineering blockers

1. Core execution path is incomplete.
- The pipeline imports and calls stubbed functions from:
  - `src/poly_csp/geometry/transform.py`
  - `src/poly_csp/chemistry/monomers.py`
  - `src/poly_csp/chemistry/backbone_build.py`
  - `src/poly_csp/chemistry/polymerize.py`
  - `src/poly_csp/chemistry/selectors.py`
  - `src/poly_csp/chemistry/functionalization.py`
  - `src/poly_csp/geometry/local_frames.py`
  - `src/poly_csp/geometry/dihedrals.py`

2. Environment is not bootstrapped for runtime.
- In this shell, `python` is unavailable and `python3` lacks `hydra-core`, so `python3 -m poly_csp.pipelines.build_csp` fails before entering your logic.

3. Packaging configuration is risky for current namespace layout.
- `pyproject.toml` uses `[tool.setuptools.packages.find]`, but `src/poly_csp` has no `__init__.py` files.
- With modern setuptools this often requires `find_namespace` configuration, otherwise install/discovery behavior can be inconsistent across environments.

4. Test harness is missing.
- `pyproject.toml` points pytest at `src/poly_csp/tests`, but no tests are present there.
- This removes safety for geometry invariants and chemistry correctness.

### Scientific/modeling risks

1. Helix presets are not yet harmonized with CSP targets.
- The project narrative targets carbamate CSP helices (especially amylose 4/3 and cellulose 3/2 motifs).
- Active cellulose config is `conf/helix/cellulose_i.yaml` with a 2/1-like parameterization (`theta=pi`, right-handed metadata), which may be valid for a different structural context but is not aligned with the CSP-specific framing in your docs.

2. No enforced atom-label contract for carbohydrate chemistry yet.
- `make_glucose_template` is not implemented, so there is no validated, deterministic mapping for C1/O4/C2/C3/C6/O2/O3/O6.
- Without that contract, all downstream steps (polymerization, attachment, torsion control, H-bond logic) are fragile.

3. Selector chemistry and coordinates are not implemented.
- The intended O2/O3/O6 carbamate linkage logic is not encoded yet.
- Pipeline comments explicitly note selector coordinate merge is not done.

4. Quality metrics are not yet physically sufficient for ordered CSP screening.
- Current clash metric is pairwise all-heavy-atom minimum distance, which includes bonded neighbors and cannot reliably distinguish realistic packing from true overlaps.
- No H-bond occupancy metrics, torsion variance metrics, or repeat-level order parameters are implemented.

### Documentation and consistency risks

1. Planning docs are strong, but some content is not production-ready.
- `docs/csp_construction_report.md` contains unresolved citation artifacts (for example `turnXX` markers), which reduces trust and maintainability.

2. Naming is inconsistent in several places.
- `admcp` and `admpc` both appear in filenames/functions/config names.
- This will create avoidable config and import mistakes as the code grows.

3. README describes modules not present in the codebase.
- Example: `geometry/helix.py`, `ordering/hbonds.py`, `ordering/symmetry_opt.py`, and `mm/*` are listed but not implemented in active source.

## What is working well

1. The architectural direction is correct for the scientific problem.
- Deterministic screw construction, repeat-unit optimization mindset, and staged relaxation are exactly the right strategy for avoiding disordered selector brushes.

2. Config-first workflow is a good foundation.
- Hydra + schema-based settings are appropriate for reproducible parameter sweeps.

3. The staged docs are unusually clear.
- `docs/csp_construction_workflow_staged_implementation_plan.md` and `docs/csp_construction_ticket_stages_1_3.md` are actionable and map naturally to sprint execution.

## Recommended path forward

### Phase 0 (immediate stabilization, 1-2 days)

1. Fix package/runtime baseline.
- Choose one:
  - add `__init__.py` files under `src/poly_csp/*`, or
  - switch setuptools to namespace package discovery.
- Create a reproducible bootstrap command in README for env creation and local run.

2. Add a minimal test scaffold.
- Add `src/poly_csp/tests` with:
  - import smoke test
  - screw group property test
  - config load test

3. Resolve naming and preset policy.
- Standardize ADMPC naming (`admpc` vs `admcp`).
- Define one canonical helix preset set for CSP work and document when cellulose 2/1 is intentionally used.

### Phase 1 (vertical slice A: stages 1-2, 3-5 days)

1. Implement deterministic geometry core.
- `rotation_matrix_z`, `ScrewTransform.matrix/apply`, optional Kabsch utility.

2. Implement monomer template contract.
- Deterministic labeled glucose template with explicit site indices.

3. Implement backbone coordinates and polymer graph.
- Build DP chain coordinates by screw replication.
- Build RDKit topology with deterministic glycosidic bonds.
- Assign conformer and export PDB.

4. Acceptance criteria.
- Pipeline runs for backbone-only case.
- Symmetry RMSD near numerical zero for repeat mapping.
- RDKit sanitization passes.

### Phase 2 (vertical slice B: stage 3 + QC hardening, 4-6 days)

1. Implement one selector path only first.
- 3,5-DMPC at C6 for all residues.
- Deterministic local frame placement and bonded attachment.

2. Implement meaningful QC.
- Clash metric excluding bonded and near-bonded pairs.
- Basic selector torsion summary and repeat variance.

3. Acceptance criteria.
- DP=12 amylose/cellulose builds with C6 selector complete and deterministic.
- Build report includes reproducible torsion and clash metrics.

### Phase 3 (scientific ordering core: stages 4-5, 1-2 weeks)

1. Add symmetry-aware rotamer optimization on repeat unit.
- Start with small discrete rotamer grid and objective combining clashes + H-bond geometry.

2. Add restrained OpenMM relaxation (minimal first cut).
- Backbone positional restraints
- Selector dihedral restraints
- Optional soft H-bond restraints
- Staged release schedule

3. Acceptance criteria.
- Pre-MM and post-MM H-bond motif occupancy is measurable and stable.
- Symmetry/order degradation is bounded by thresholds.

## Priority decisions you should lock now

1. Canonical helix presets for target CSP families.
- Decide final default for amylose ADMPC and cellulose carbamate workflows.
- Keep alternative presets, but tag them explicitly as non-default or exploratory.

2. Force-field backend strategy for first production milestone.
- Choose one initial lane and postpone backend abstraction until first end-to-end build is validated.

3. Exact first H-bond motif definition.
- Encode one motif and measure it rigorously before expanding motif complexity.

## Final verdict

The project is scientifically well-conceived and has a good planning backbone, but it is currently pre-implementation for most core functionality. The right move is to execute the existing stage/ticket plan as a strict vertical slice with hard geometry and chemistry acceptance tests before adding broader generalization. If you follow that sequence, you can reach a credible deterministic CSP builder quickly without accumulating untestable complexity.
