# CSP Construction Workflow — Practical, Staged Implementation Plan

This document is a build plan you can hand to coding AI agents. It is designed for a Python pipeline that produces **highly ordered Daicel-style coated CSP polymers** (amylose/cellulose backbones functionalized at C2/C3/C6 with bulky selectors such as 3,5-dimethylphenylcarbamate), with **helical symmetry imposed from the start** and **pre-organized intramolecular H-bond networks**.

The guiding principle is: **construct geometry deterministically using a helical template + internal coordinates**, then **relax locally under symmetry-preserving constraints**, and only then run unconstrained equilibration.

---

## Stage 0 — Repo layout, interfaces, and “single source of truth” for geometry

### Scientific rationale
Ordered CSPs are not obtained reliably from generic MD annealing because selector torsions and backbone glycosidic angles create a rugged landscape. You need a **template-driven construction** that encodes the intended helix and H-bonding motif.

### Software engineering goal
Define clean modules so you can swap (a) backbone helix parameters, (b) selector chemistry, (c) rotamer/H-bond rules, (d) relaxation protocol.

### Deliverables
**Python package skeleton** (suggested):

```
poly_csp/
  __init__.py
  config/
    schema.py
    presets.py
  geometry/
    helix.py
    internal_coords.py
    transform.py
    zmat.py
  chemistry/
    monomers.py
    selectors.py
    functionalization.py
    polymerize.py
  ordering/
    rotamers.py
    hbonds.py
    scoring.py
    symmetry_opt.py
  mm/
    openmm_system.py
    restraints.py
    minimize.py
    anneal.py
  io/
    rdkit_io.py
    pdb.py
    cif.py
  pipelines/
    build_csp.py
  tests/
  examples/
```

**Configuration objects** (Pydantic v2 recommended):
- `HelixSpec`: helix axis, rise per residue, rotation per residue, residues per turn, handedness, target glycosidic torsions (ϕ/ψ/ω), optional unit-cell info.
- `BackboneSpec`: polymer type (`amylose`/`cellulose`), DP (degree of polymerization), ring puckering constraints.
- `SelectorSpec`: RDKit Mol template + attachment atoms, mapping to C2/C3/C6, internal dihedral definitions, rotamer library source.
- `OrderSpec`: H-bond pattern definitions, scoring weights, symmetry enforcement, steric thresholds.
- `RelaxSpec`: minimization schedule, restraint strengths, annealing schedule.

**One invariant**: every module works in a **common coordinate frame** where helix axis is `+z` and residue i has a **local triad**.

---

## Stage 1 — Helical symmetry core: parametric backbone placement

### Scientific rationale
To avoid “oligomer conformations that won’t polymerize,” you must **never sample arbitrary oligomer conformations**. Instead, create polymer coordinates by repeatedly applying a **screw operation** (rotation about axis + translation along axis) combined with **internal-coordinate enforcement** for glycosidic geometry.

### Implementation goal
Implement a **HelixBuilder** that:
1) defines a residue-local frame,
2) places a reference monomer,
3) generates the chain via a screw transform,
4) optionally refines bond lengths/angles via internal coordinates (Z-matrix) while preserving the screw.

### Tasks
1. **Define the screw operator**
   - For residue index `i`, transform is:
     - rotation `Rz(i * θ)` about z
     - translation `Tz(i * h)` along z
   where `θ` = rotation per residue, `h` = rise per residue.

2. **Create monomer reference geometry**
   - Start from RDKit-built glucopyranose ring (or use pre-validated coordinates).
   - Annotate atoms: ring atoms, O4/O1 linkage atoms, C2/C3/C6.

3. **Internal coordinate scaffold**
   - Define target glycosidic torsions (ϕ/ψ/ω) for α-1,4 and β-1,4.
   - Implement an internal-coordinate generator (`ZMatrixChain`) that can enforce:
     - bond lengths/angles fixed
     - key torsions set (or narrow)
     - ring puckering constrained

4. **Hybrid approach recommended**
   - Use the screw to place residue frames and approximate atom coordinates.
   - Then apply a **symmetry-constrained internal coordinate solve**:
     - solve residue i coordinates in local frame
     - map to global via screw
     - ensure linkage bonds align at i↔i+1 boundary.

### Deliverables
- `geometry/helix.py`: `ScrewTransform(theta, rise)` and `apply_screw(i)`
- `geometry/internal_coords.py`: torsion setting + local-to-global mapping
- `chemistry/monomers.py`: `GlucoseMonomer` with atom labels and attachment sites

### Acceptance tests
- Chain bonds lengths are correct; no broken links.
- For DP=20, glycosidic torsions are within tolerance of target.
- Visual inspection: backbone is helical and consistent.

---

## Stage 2 — Symmetry-preserving polymerization (connectivity and topology)

### Scientific rationale
Even if coordinates look right, the **topology** must reflect polymer bonds so that later MM minimization doesn’t “snap” the structure.

### Implementation goal
Create a deterministic polymerizer that builds an RDKit Mol (or OpenMM Topology) for the full chain **without** conformer sampling.

### Tasks
1. Build polymer graph:
   - Duplicate monomer graph DP times.
   - Add glycosidic bonds between residue i O4 and residue i+1 C1 (or appropriate atoms depending on monomer representation).

2. Assign conformer coordinates:
   - Attach the coordinates from Stage 1 to the RDKit conformer.

3. Sanity checks:
   - Ensure valences OK.
   - Ensure stereochemistry preserved.

### Deliverables
- `chemistry/polymerize.py`: `polymerize(monomer, dp, linkage)`
- `io/rdkit_io.py`: conformer export

### Acceptance tests
- RDKit sanitization passes.
- Exported PDB has continuous bonds and correct atom ordering.

---

## Stage 3 — Selector attachment with local frames (deterministic placement)

### Scientific rationale
Selectors must be oriented consistently relative to the backbone so that the **intramolecular carbamate H-bond network** can form. Random dihedral choices trap the system.

### Implementation goal
Attach selectors at C2/C3/C6 using a **local residue frame** and a **dihedral parameterization** of the attachment.

### Tasks
1. Define a **residue-local coordinate triad**:
   - Origin at ring center or C1.
   - Axes: one along C1–O4 (or ring normal), one along C2–C3, etc.

2. Selector templates:
   - RDKit Mol for selector (e.g., phenylcarbamate).
   - Define attachment atom on selector and attachment atom on sugar (C2/C3/C6 substituent oxygen/carbon depending on chemistry).

3. Attachment chemistry:
   - Build link atom/bond.
   - Correctly handle leaving groups (e.g., replace –OH H with carbamate).

4. Initial placement:
   - Use rigid-body transform to place selector in residue-local frame.
   - Then set attachment dihedrals to initial values from a library.

### Deliverables
- `chemistry/selectors.py`: `SelectorTemplate` and atom mapping
- `chemistry/functionalization.py`: `attach_selector(residue, site, selector, pose)`
- `geometry/transform.py`: Kabsch alignment / rigid placement

### Acceptance tests
- All C2/C3/C6 substituted.
- No catastrophic steric overlaps (distance-based check).

---

## Stage 4 — Rotamer library + H-bond network pre-organization

### Scientific rationale
To get 90%+ of the “right” H-bonds from the start, you need **rotamer selection** guided by an H-bond scoring function while keeping helical symmetry.

### Implementation goal
Implement a **symmetry-aware rotamer optimizer**:
- chooses dihedrals for each selector (and optionally glycosidic torsions within narrow bounds)
- maximizes H-bond satisfaction + packing
- enforces screw symmetry (or periodic patterns)

### Tasks
1. Rotamer representation
   - Define dihedrals for selector: e.g., sugar–O–C(=O)–N, C(=O)–N–Ar, etc.
   - Provide discrete bins or continuous ranges.

2. H-bond motif definition
   - Example: carbamate N–H donor → adjacent carbamate C=O acceptor (same residue or i±1 depending on known motif).
   - Represent as constraints: (donor atom, H atom, acceptor atom, target distance/angle).

3. Scoring function
   - Components:
     - H-bond geometry score (distance + angle)
     - steric clash penalty (van der Waals overlaps)
     - aromatic stacking preference (optional)
     - symmetry penalty (difference between residues mapped by screw)

4. Optimization strategy (practical)
   - Start with **unit cell / repeat unit** length `k` (e.g., 1–4 residues) under screw symmetry.
   - Optimize only those k residues’ rotamers.
   - Replicate across DP by symmetry.

   Recommended search:
   - discrete rotamer enumeration for k-residue repeat (manageable)
   - local continuous refinement using SciPy (`minimize`) on dihedral angles

5. Output
   - A deterministic set of dihedrals per site (C2/C3/C6) and per residue in repeat.

### Deliverables
- `ordering/rotamers.py`: rotamer definitions, enumeration
- `ordering/hbonds.py`: motif constraints + scoring
- `ordering/symmetry_opt.py`: optimize repeat unit then replicate

### Acceptance tests
- Report: fraction of motifs satisfied (target >> 5%, e.g., 60–90% pre-MM).
- Visual: ordered “coating” around the helix.

---

## Stage 5 — Symmetry-preserving relaxation: restrained minimization in OpenMM

### Scientific rationale
Even with good initial placement, you’ll have residual strain. You must relax without destroying the order. Use **restraints that preserve screw symmetry and H-bond geometry**, then gradually release.

### Implementation goal
Create an OpenMM relaxation protocol with staged restraints:
1) strong backbone + symmetry restraints
2) moderate dihedral restraints on selectors
3) explicit H-bond restraints (distance/angle)
4) gradual release

### Tasks
1. Build OpenMM System
   - Force field choice (you decide; architecture should allow swapping).
   - Solvent optional; initially vacuum or implicit is OK for ordering.

2. Add restraints
   - Positional restraints on backbone heavy atoms to helix template.
   - Dihedral restraints on key selector torsions to rotamer targets.
   - H-bond restraints: harmonic distance + angular restraint.

3. Minimization schedule
   - Minimize with strong restraints.
   - Reduce restraint force constants stepwise.

4. Optional annealing
   - short simulated anneal with restraints to cross small barriers while keeping order.

### Deliverables
- `mm/openmm_system.py`: System builder
- `mm/restraints.py`: positional/dihedral/hbond restraint forces
- `mm/minimize.py`: staged minimization runner
- `mm/anneal.py`: optional restrained anneal

### Acceptance tests
- No explosion.
- H-bond satisfaction remains high after minimization.
- Selector dihedrals stay near targets (within tolerance).

---

## Stage 6 — Validation & metrics (automated quality gates)

### Scientific rationale
You need objective metrics to detect “disorganized coatings” before spending compute on MD.

### Implementation goal
Add analysis tools that produce a **QC report** and fail builds that don’t meet thresholds.

### Metrics
- H-bond network:
  - fraction satisfied, per motif type
  - distance/angle distributions
- Helical symmetry:
  - RMSD between residue i and screw-mapped residue i+k
  - torsion distributions (glycosidic and selector)
- Packing:
  - clash score
  - radial distribution of selector centroids around helix axis

### Deliverables
- `ordering/scoring.py`: compute metrics
- `pipelines/build_csp.py`: emits JSON report + plots (optional)

### Acceptance tests
- Example systems pass QC.
- Perturbed system fails QC (unit tests).

---

## Stage 7 — Generalization to arbitrary selectors

### Scientific rationale
Different selectors change the preferred rotamers and H-bond patterns. Generalization requires a *selector-agnostic* representation of attachment and scoring.

### Implementation goal
Make selectors plug-ins:
- each selector provides: attachment mapping, dihedral list, donor/acceptor sets, optional aromatic features

### Tasks
1. Selector plugin interface
   - `SelectorTemplate` fields:
     - RDKit mol
     - attachment atom idx
     - dihedral definitions (4-atom tuples)
     - hbond donors/acceptors
     - optional pseudo-atoms for rings (centroids/normals)

2. Automatic donor/acceptor detection (optional)
   - RDKit chemical features factory to label donors/acceptors.

3. Rotamer generation strategies
   - discrete library (user-provided)
   - or automatic sampling around known torsions (e.g., 60° increments) filtered by sterics

### Deliverables
- `chemistry/selectors.py` extended to registry pattern
- `ordering/rotamers.py` supports per-selector definitions

---

## Stage 8 — End-to-end pipeline and CLI

### Scientific rationale
You want reproducible builds, parameter sweeps (DP, selector variants), and easy handoff to MD production.

### Implementation goal
A deterministic CLI that produces:
- PDB/PRMTOP or OpenMM XML
- QC report
- optionally a minimized structure and checkpoint

### CLI outline
`python -m poly_csp.pipelines.build_csp --preset daicel_cellulose_35dmpc --dp 30 --repeat 2 --output out/`

### Deliverables
- `pipelines/build_csp.py` orchestrator
- `config/presets.py` includes common Daicel-like presets

### Acceptance tests
- Running the example preset produces the expected outputs and QC thresholds.

---

## Stage 9 — Performance and scaling considerations

### Scientific rationale
Full DP=100+ with 3× substitution is large. Optimize by working on repeat units and replicating.

### Engineering tactics
- Optimize only repeat unit rotamers (k residues).
- Cache RDKit templates and transforms.
- Use NumPy arrays for coordinates; avoid per-atom Python loops.
- Provide an option to build a **coated segment** and tile it.

---

## Suggested first “vertical slice” implementation (1–2 weeks of agent work)

1) Implement Stage 1–2 to produce a helical amylose/cellulose backbone with correct topology.
2) Implement Stage 3 for one selector (3,5-DMPC) at one position (C6) first.
3) Implement Stage 4 for a 1-residue repeat (k=1) with a small rotamer grid.
4) Implement Stage 5 restrained minimization in OpenMM.
5) Implement Stage 6 QC: H-bond fraction + symmetry RMSD.

Once that works, expand to C2/C3/C6 and k>1 repeats, then generalize selectors.

---

## Notes for AI agents implementing this plan

- Keep every transformation explicit and testable (matrix + translation).
- Prefer deterministic building over RDKit conformer search.
- Encode helix and rotamer targets as data (YAML/JSON), not hardcoded.
- Write unit tests for geometry invariants before moving to MM.
- Add logging at each stage with:
  - seed (even if deterministic)
  - torsion values
  - clash score
  - H-bond fraction

---

## Open questions to resolve during implementation

- Which helix parameters (θ, h) and glycosidic torsions best represent your target Daicel coating? (You can maintain multiple presets.)
- What is the exact intended H-bond motif (intra-residue vs inter-residue, along chain vs circumferential)? Encode as `HbondPattern` objects.
- Which force field stack will you use in OpenMM (GAFF/SMIRNOFF/CHARMM/CGenFF)? Make this a swappable backend.

