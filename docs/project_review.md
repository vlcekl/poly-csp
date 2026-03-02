# poly_csp Project Review

Date: 2026-03-02  
Scope: repository-level engineering and scientific review of `poly_csp` as a deterministic CSP polymer construction workflow.

## Findings (Ordered by Severity)

### High Severity

1. **Screw symmetry QC is computed on pre-edit backbone coordinates, not the final molecule**
   - Evidence:
     - `src/poly_csp/pipelines/build_csp.py:423-432` computes `qc_screw_symmetry_rmsd_A` from `coords` (stage-1 backbone coordinates), not from final `mol_poly` coordinates after terminal edits, selector placement, ordering, or relaxation.
   - Observed behavior:
     - In representative builds (`scratch/review_*`), `qc_screw_symmetry_rmsd_A` remained ~0.0 even when topology/coordinates changed in later stages.
   - Impact:
     - A key QC number can report "perfect symmetry" while the exported model may no longer satisfy symmetry after downstream operations.
     - This undermines confidence in the main scientific claim (symmetry-preserving construction and relaxation).
   - Recommendation:
     - Recompute symmetry from final coordinates using residue label maps (`_poly_csp_residue_label_map_json`), with explicit masking for selector/cap atoms.
     - Add regression tests where post-build perturbations must change `qc_screw_symmetry_rmsd_A`.

2. **Ordering objective clash term is inconsistent with QC and can be physically misleading**
   - Evidence:
     - `src/poly_csp/ordering/symmetry_opt.py:51-54` uses `min_interatomic_distance(xyz, _heavy_mask(mol))` without bonded exclusions.
     - QC uses bonded exclusions via `bonded_exclusion_pairs` in `build_csp.py:413-420`.
   - Observed behavior:
     - In `scratch/review_ordering/build_report.json`, ordering summary reports `baseline_min_heavy_distance_A = 0.388 A`, while final QC reports `qc_min_heavy_distance_A = 1.444 A`.
   - Impact:
     - Optimizer can score candidates with a clash metric that is not chemically meaningful and not aligned with acceptance criteria.
     - This can bias pose selection unpredictably.
   - Recommendation:
     - Reuse the same exclusion policy and distance routine in both ordering objective and QC.
     - Consider class-aware clash penalties (`backbone-selector`, `selector-selector`) instead of one global minimum.

3. **"Relaxation" is currently a soft-overlap resolution model, not a physically valid force-field relaxation**
   - Evidence:
     - `src/poly_csp/mm/openmm_system.py:99-115` builds only a custom soft repulsive nonbonded force.
     - No bonded, electrostatic, or torsional force field terms are created from actual parameters.
   - Impact:
     - Resulting coordinates may be smoother but are not energetically meaningful for production-level CSP conformational interpretation.
     - Risk of scientific overinterpretation if treated as real MD-ready minimization.
   - Recommendation:
     - Keep this path explicitly labeled as geometric pre-relaxation only.
     - Add parameterized OpenMM path from AmberTools outputs (`ambertools` backend) for scientifically valid minimization.

4. **QC hard-fail is off by default, allowing poor structures to silently pass**
   - Evidence:
     - `conf/qc/basic.yaml:21` sets `fail_on_thresholds: false`.
   - Impact:
     - In automated workflows/CI, low-quality outputs may propagate unless consumers remember to override config.
   - Recommendation:
     - For CI presets, set hard-fail to true.
     - Keep interactive default flexible if needed, but separate "dev" vs "production" QC profiles.

### Medium Severity

5. **Hydrogen-bond metric is distance-only and heavy-atom-only**
   - Evidence:
     - `src/poly_csp/ordering/hbonds.py:59-69` uses donor-acceptor distance threshold only; no angle criteria or explicit proton geometry.
   - Impact:
     - `hbond_fraction` is a useful heuristic for pre-organization, but not a physically robust H-bond occupancy metric.
   - Recommendation:
     - Introduce geometric constraints (D-H...A or surrogate angle-based D...A-X) and residue-pair uniqueness rules.
     - Rename existing metric as "hbond_like_fraction" to prevent over-interpretation.

6. **Test portability is limited by absolute paths and hard dependency on a named conda env**
   - Evidence:
     - Integration tests invoke commands with hardcoded `cd /home/lukas/work/projects/poly_csp` and `conda run -n polycsp` in:
       - `tests/test_pipeline_ordering.py:18-23`
       - `tests/test_pipeline_relax_and_amber.py:17-20`
       - `tests/test_pipeline_end_modes.py:17-20`
   - Impact:
     - CI portability and contributor onboarding are harder than necessary.
   - Recommendation:
     - Use repository-relative execution and current interpreter (`sys.executable`) where possible.
     - Mark environment-heavy tests and gate them by marker.

7. **Documentation and codebase have drift**
   - Evidence:
     - README tree lists files that do not exist (for example `geometry/helix.py`) and misses current modules (`chemistry/terminals.py`, `io/amber.py`) (`README.md:95-124`).
     - `src/poly_csp/config/presets.py` includes placeholder-style citation markers and appears disconnected from Hydra defaults.
   - Impact:
     - New contributors can misread current architecture and active configuration sources.
   - Recommendation:
     - Refresh README structure and "active code path" notes.
     - Either wire presets into runtime config flow or mark them as reference-only.

### Low Severity

8. **Annealing step count truncation**
   - Evidence:
     - `src/poly_csp/mm/anneal.py:19-25` uses integer division for `steps_per_segment`, potentially dropping remainder steps.
   - Impact:
     - Minor schedule drift vs requested total steps.
   - Recommendation:
     - Distribute remainder across segments so executed steps exactly match `n_steps`.

9. **Optional import guards are broad**
   - Evidence:
     - `except Exception` in `build_csp.py:51-68` swallows all import failures for selector and MM modules.
   - Impact:
     - Real coding errors can be hidden as "module unavailable."
   - Recommendation:
     - Catch `ImportError` explicitly; log unexpected exceptions.

---

## Current State Summary

`poly_csp` is a strong early-stage deterministic geometry and topology builder for CSP polymer models. The project already has meaningful modular boundaries (chemistry, geometry, ordering, MM, IO), metadata tracking, and a solid test baseline.

### What is working well

- Deterministic backbone generation via explicit screw transforms.
- Deterministic selector attachment with residue/site metadata propagation.
- End-mode handling (`open`, `capped`, `periodic`) with metadata invariants.
- AMBER export abstraction with a clear placeholder vs AmberTools backend split.
- A broad unit/integration test suite for current scope.

### Empirical validation performed during this review

- Test suite: `conda run -n polycsp pytest -q` -> **41 passed in 12.49s**.
- Representative pipeline builds completed successfully:
  - Baseline backbone build (`scratch/review_baseline`).
  - Selector-enabled build (`scratch/review_selector`).
  - Ordering-enabled build (`scratch/review_ordering`).
  - Relaxation-enabled build (`scratch/review_relax`).
  - Periodic natural-OH build (`scratch/review_periodic`).
  - AMBER placeholder export (`scratch/review_amber`).

### Scientific maturity assessment

- **Mature enough for deterministic model construction and software experimentation.**
- **Not yet mature for physically rigorous simulation claims** without parameterized force fields, stronger QC coupling to final structures, and richer structural validation metrics.

---

## Proposed Path Forward

### Phase 1 (Immediate: 1-2 weeks) - Correctness and consistency

1. Fix QC symmetry metric to operate on final coordinates and backbone-only atom sets.
2. Align ordering clash objective with QC exclusion rules.
3. Add regression tests:
   - symmetry metric changes after deliberate selector/terminal perturbation,
   - ordering summary distance metric agrees with QC conventions.

### Phase 2 (Near-term: 2-4 weeks) - Scientific validity of relaxation

1. Keep current OpenMM path as `"geometry_pre_relax"` mode.
2. Implement/activate parameterized relaxation mode using AmberTools-generated topology/coordinates.
3. Compare pre-relax vs parameterized-relax outputs on the same systems and report divergences.

### Phase 3 (Near-term: 3-6 weeks) - Stronger CSP-specific quality metrics

1. Upgrade H-bond scoring with angle criteria and occupancy logic.
2. Add helix-aware order parameters:
   - per-residue axis projection variance,
   - selector orientation order parameter,
   - repeat-resolved torsion coherence.
3. Introduce stricter QC profile (`qc=production`) with hard failure enabled.

### Phase 4 (Mid-term: 1-2 months) - Reproducibility and usability

1. Remove absolute paths from tests and standardize environment handling.
2. Update README to reflect actual module layout and runtime pathways.
3. Define benchmark set (amylose/cellulose, DP values, site patterns) and track metrics over commits.

---

## Practical Recommendation

Treat the project as a high-quality deterministic **construction engine** today, and as a **simulation-ready CSP platform** only after Phase 1-2 are complete. The most important immediate investment is tightening metric validity (QC and objective coherence) before adding new scientific features.
