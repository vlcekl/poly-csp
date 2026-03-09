**Bottom Line**

Implicit solvent could be useful here, but mostly as a pragmatic late-stage refinement tool, not as a full physical replacement for the current periodic search. For Chiralpak-like carbamate-rich CSPs in alcohol-containing mobile phases, it can reduce some vacuum artifacts, but it will not reproduce the real solvent competition that methanol or isopropanol exert on the selector H-bond network.

**Molecular Science**

The main scientific reason to consider implicit solvent is that your current relaxation is effectively gas-phase-like. The full runtime system is just a standard `NonbondedForce` with charges and LJ, using `NoCutoff` for open systems and `CutoffPeriodic` for periodic ones in [system_builder.py](/home/lukas/work/projects/chiral_csp_poly/src/poly_csp/forcefield/system_builder.py#L859). That tends to favor strong intramolecular electrostatics and internal H-bonds.

For this CSP, that means implicit solvent could plausibly do three useful things:

- weaken over-stabilized internal NH···O=C contacts relative to vacuum
- reduce over-compact selector arrangements driven by unscreened electrostatics
- shift the receptor ensemble toward conformers that are more plausible in a solvated chromatography environment

That said, for methanol or isopropanol, continuum solvent is only a first-order correction. The biggest limitation is that alcohol mobile phases do not just provide dielectric screening. They also compete specifically for carbamate carbonyls, NH donors, hydroxyls, and aromatic exposure. A continuum model cannot represent:

- solvent molecules directly H-bonding to selector carbonyls or NHs
- methanol vs isopropanol steric differences around binding clefts
- solvent-bridged motifs
- preferential solvation of one selector orientation over another

So scientifically, the likely effect is:

- better than vacuum for suppressing some artificial intramolecular locking
- not good enough to claim “this is the true methanol/IPA conformer ensemble”

That is still useful for docking. Docking rarely uses a fully realistic solvent model either, so a solvent-relaxed receptor ensemble can still be more realistic than a vacuum-relaxed one. I would treat it as an ensemble-generation bias, not ground truth.

For your specific goal, I would expect the biggest value in the later relaxation stages, exactly as you suggested. The periodic search is mainly identifying repeat-symmetric selector packings. The place where solvent is most justifiable is the nonperiodic cleanup or final receptor refinement, where you want to remove obviously vacuum-biased local geometry before docking.

**Implementation Reality**

User-facing, this can look like one parameter. Internally, it is more than one switch.

Right now the relaxation protocol has a clean two-stage split in [relaxation.py](/home/lukas/work/projects/chiral_csp_poly/src/poly_csp/forcefield/relaxation.py#L129):

- stage 1: soft repulsive clash resolution
- stage 2: full runtime system minimization and optional annealing

That separation is good news. The efficient implementation would be to leave stage 1 alone and add implicit solvent only to stage 2. That is the lowest-churn design and also the most defensible scientifically.

The other important boundary is export. The current export layer assumes a canonical full runtime system with exactly one `NonbondedForce` and extracts only solute charges/LJ from it in [export_bundle.py](/home/lukas/work/projects/chiral_csp_poly/src/poly_csp/forcefield/export_bundle.py#L106). So if you add implicit solvent to relaxation, the exported `pdbqt`/AMBER receptor can still use the solvent-relaxed coordinates, but the solvent model itself is not something the current export artifacts will carry through. For docking, that is usually fine. It just needs to be explicit in the report.

The harder issue is periodic mode. Your periodic optimization path uses periodic box vectors and `CutoffPeriodic` in [system_builder.py](/home/lukas/work/projects/chiral_csp_poly/src/poly_csp/forcefield/system_builder.py#L861). That makes implicit solvent a poor fit for the periodic cell step. Even if technically possible with a custom model, it is not the clean path here. The natural fit is:

- keep periodic unit-cell optimization as it is
- expand to open handoff receptor
- optionally run the open cleanup relaxation in implicit solvent

That matches both the chemistry and the software architecture.

**Is One Parameter Enough?**

For the user, yes. For the code, no.

A good user-facing interface could be as simple as:

- `forcefield.solvent=none`
- `forcefield.solvent=methanol`
- `forcefield.solvent=isopropanol`

But behind that, the code still needs to decide at least:

- which implicit model to use
- what solvent dielectric to use
- what solute dielectric to use
- which atoms get which GB radii/scale factors
- which stages it applies to
- whether periodic systems are allowed or rejected

So the right design is: one user-facing selector, backed by preset objects.

The simplest viable engine in OpenMM would be a GBSA-style force added to the full nonperiodic system only. That is efficient and local to the existing architecture. The catch is that standard GB/OBC models are mostly calibrated for aqueous use. Using them with methanol or isopropanol by just changing dielectric is workable as an approximation, but it is not truly solvent-specific. That is the central compromise.

**What I Would Recommend**

If you want this feature, I would implement it in this order:

1. Add implicit solvent only to nonperiodic full-stage relaxation and handoff cleanup.
2. Keep stage 1 soft clash resolution unchanged.
3. Keep periodic unit-cell optimization in the current model.
4. Expose named solvent presets, but document them as continuum approximations, not faithful methanol/IPA physics.
5. Compare receptor ensembles from `none`, `methanol`, and `isopropanol` on the same ranked periodic conformers.

Scientifically, that would answer the useful question: does solvent-aware cleanup materially change selector orientation, donor occupancy, or docking pocket shape? If the answer is “barely,” then it is not worth making default. If the answer is “yes, in a chemically sensible way,” then it becomes a strong late-stage option.

If you want the highest realism later, the real upgrade path is not richer implicit solvent. It is short explicit-solvent refinement on a small number of top-ranked receptors. Implicit solvent is the efficient middle ground, not the final physical model.

If you want, the next step can be to turn this into a concrete implementation design for this repo: exact config schema, stage scope, and the least invasive OpenMM integration path.
