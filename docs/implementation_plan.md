# poly_csp Implementation Plan (Full Replacement)
Target document path: `docs/implementation_plan.md`

## Summary
This plan replaces earlier planning docs with a state-aware roadmap that includes completed work and pending work.  
Current baseline: Stages 0-3 and Ticket B4 core APIs are implemented and tested (`23 passed`).  
Primary next milestone: Stage 4 ordering (rotamers + H-bond pre-organization), then Stage 5 restrained OpenMM relaxation, then Stage 6 QC hardening and Stage 8 AMBER handoff.

## Explicit Defaults and Assumptions
1. `polymer.kind` default remains `amylose`; `cellulose` stays secondary until amylose path is stable end-to-end.
2. `polymer.monomer_representation` default is `anhydro`.
3. `polymer.end_mode` default is `open` for now; `periodic` and `capped` are planned extensions.
4. Selector-first scope remains `35dmpc`, with `C6` as first-site baseline and `C2/C3/C6` as expanded mode.
5. The pipeline remains deterministic by default (fixed seeds, no random conformer search in final build path).

## Monomer Representation Decision (Requested Tradeoff)
### Option A: `anhydro` repeating unit (recommended default now)
- Pros: deterministic polymerization by direct inter-residue bonding, stable atom counts/indexing, simpler symmetry mapping, easier per-residue metadata.
- Cons: residue is not a full natural monosaccharide state; open-chain termini are chemically implicit and need explicit terminal policy later.
- Failure mode to watch: any sanitize/valence issue caused by reduced atom model.

### Option B: `natural_oh` monomer (fallback if A fails, or if terminus chemistry is priority)
- Pros: chemically intuitive monomer and termini, straightforward explicit terminal chemistry.
- Cons: polymerization is more complex (must perform deterministic dehydration-style linkage edits), higher indexing churn, more bookkeeping for selector and residue mapping.
- Risk: larger implementation surface before Stage 4/5 progress.

### Decision Rule (locked)
1. Continue with `anhydro`.
2. Keep automated validation gate on every build profile: sanitize success, no valence errors, deterministic indexing, selector attachment success.
3. Switch to `natural_oh` only if `anhydro` creates unresolved valence/sanitization problems or blocks required terminal chemistry for downstream MD/AMBER.
4. If switch occurs, implement `natural_oh` as a second representation backend without removing `anhydro`.

## Public Interfaces and Config Additions
### Existing interfaces to keep
1. `make_glucose_template`, `build_backbone_coords`, `polymerize`, `attach_selector`, `apply_selector_pose_dihedrals`.
2. Selector config path with `selector.pose.dihedral_targets_deg`.

### New interfaces to add
1. Config schema extensions:
   - `polymer.monomer_representation: anhydro | natural_oh`
   - `polymer.end_mode: open | capped | periodic`
   - `polymer.end_caps` (only used when `end_mode=capped`)
2. Export config:
   - `output.export_formats: [pdb, amber]`
   - `amber` block for file naming, charge mode, and parameter backend.
3. Terminal builder API:
   - `apply_terminal_mode(mol, mode, caps, representation) -> mol`

### Hydra note
For dynamic dihedral keys, use `+selector.pose.dihedral_targets_deg.<name>=<deg>`.

## Stage Plan (Complete + Pending)

## Stage 0 — Foundation and invariants (Status: complete, keep hardening)
1. Keep package/layout, deterministic conventions, and root `tests/`.
2. Add metadata invariants doc (`_poly_csp_*` properties and atom props).
3. Acceptance: import tests pass; metadata contract documented and validated in unit tests.

## Stage 1 — Helical symmetry core (Status: complete, refine chemistry realism)
1. Keep `ScrewTransform`, Kabsch, deterministic residue placement.
2. Add optional internal-coordinate refinement hook for glycosidic targets.
3. Acceptance: group property, orthonormality, repeat RMSD, and ring-radius invariance remain green.

## Stage 2 — Polymerization and representation backend (Status: complete for `anhydro`; extend)
1. Add explicit representation switch (`anhydro` now, `natural_oh` backend planned).
2. Add representation-specific polymerization adapters while preserving common residue/site metadata.
3. Acceptance: sanitize passes for DP test set `{1,2,12,24}`, atom ordering/indexing deterministic, metadata preserved.

## Stage 2.5 — Terminal handling (Status: pending, new)
1. Implement `end_mode=open|capped|periodic`.
2. `open`: no added terminal groups.
3. `capped`: deterministic end-group chemistry for both termini.
4. `periodic`: topology/coordinate constraints for seamless repeat boundary handling.
5. Acceptance: each mode sanitizes; termini behavior matches config; no index instability.

## Stage 3 — Selector attachment and pose control (Status: complete baseline, extend)
1. Keep current `35dmpc` template and registry.
2. Expand robust local-frame placement checks and per-site geometry tolerances.
3. Ensure `apply_selector_pose_dihedrals` supports multi-site attachments reliably.
4. Acceptance: all selected sites attach, bond/valence valid, pose target application reproducible.

## Stage 4 — Rotamer and H-bond pre-organization (Status: pending, next priority)
1. Implement `ordering/rotamers.py` with selector-specific rotamer definitions.
2. Implement `ordering/hbonds.py` motif definitions and geometry scoring.
3. Implement `ordering/symmetry_opt.py` for repeat-unit symmetry-aware optimization.
4. Integrate into pipeline with deterministic objective and report output.
5. Acceptance: measurable improvement in target H-bond motif satisfaction vs unoptimized baseline; reproducible rotamer outputs.

## Stage 5 — Symmetry-preserving OpenMM relaxation (Status: pending)
1. Implement `mm/openmm_system.py`, `mm/restraints.py`, `mm/minimize.py`, `mm/anneal.py`.
2. Apply staged restraints: backbone positional, selector dihedral, optional H-bond.
3. Keep symmetry constraints active through early relaxation.
4. Acceptance: no blow-ups; dihedral drift bounded; H-bond satisfaction does not collapse.

## Stage 6 — QC gates and failure criteria (Status: minimal baseline exists; expand)
1. Upgrade clash scoring to exclude bonded and near-bonded pairs (`1-2/1-3`, optional `1-4`).
2. Add per-class clashes: backbone-backbone, backbone-selector, selector-selector.
3. Add selector torsion distributions and H-bond motif occupancy.
4. Add pass/fail thresholds in report and optional CI gating.
5. Acceptance: perturbed/disordered controls fail QC; known-good builds pass.

## Stage 7 — Selector plugin generalization (Status: partial via registry; extend)
1. Add plugin schema for attachment mapping, dihedrals, donors/acceptors, optional aromatic features.
2. Optional donor/acceptor auto-detection path with override support.
3. Acceptance: at least one additional selector template added without core code changes.

## Stage 8 — End-to-end CLI and AMBER handoff (Status: pending, explicit addition)
1. Keep deterministic CLI orchestration and parameter sweep support.
2. Add AMBER export pathway (`prmtop/inpcrd` and intermediate artifacts as needed).
3. Define parameterization backend interface and file contract.
4. Acceptance: one canonical amylose+35dmpc build exports both PDB and AMBER artifacts reproducibly.

## Stage 9 — Scaling and performance (Status: pending)
1. Repeat-unit optimization only; replicate by symmetry.
2. Cache templates/transforms/index maps.
3. Add profiling for DP scaling and selector count scaling.
4. Acceptance: documented runtime/memory profile for DP `{12,24,48}` with trend targets.

## Test Plan and Scenarios
1. Unit geometry tests: rotation, screw composition, dihedral set/measure correctness.
2. Chemistry tests: monomer label stability, polymer sanitize, site attachment correctness, metadata persistence.
3. Integration tests: backbone-only pipeline, selector-enabled pipeline, pose-dihedral override via Hydra.
4. New representation tests: `anhydro` and `natural_oh` backend parity on deterministic indexing contract.
5. Terminal mode tests: `open`, `capped`, `periodic` topology sanity.
6. QC tests: known-good build passes thresholds; intentionally perturbed build fails.
7. Export tests: AMBER export artifacts exist and are internally consistent.

## Milestone Order (Execution)
1. M1: Stage 2 representation switch + Stage 2.5 terminal handling.
2. M2: Stage 4 rotamer/H-bond ordering (primary scientific milestone).
3. M3: Stage 5 restrained OpenMM relaxation.
4. M4: Stage 6 QC hardening and threshold gates.
5. M5: Stage 8 AMBER export handoff.
6. M6: Stage 7 selector generalization and Stage 9 performance tuning.

## Success Criteria (Project-level)
1. Deterministic end-to-end build for amylose + 35dmpc at C6 and C2/C3/C6.
2. Reproducible ordered pre-MM selector configuration with measurable H-bond improvements after Stage 4.
3. Symmetry-preserving MM relaxation that maintains ordering targets.
4. QC-gated outputs with explicit pass/fail.
5. Production handoff artifacts include both structure outputs and AMBER-ready outputs.
