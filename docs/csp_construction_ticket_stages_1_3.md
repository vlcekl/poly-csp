## Epic A — Helical symmetry backbone builder (Stage 1)

### Ticket A0 — Project scaffolding + conventions

**Goal:** Establish package layout, core data types, and coordinate conventions so later modules don’t fight each other.

**Files**

* `poly_csp/__init__.py`
* `poly_csp/config/schema.py`
* `poly_csp/geometry/__init__.py`
* `poly_csp/chemistry/__init__.py`
* `poly_csp/io/__init__.py`
* `poly_csp/tests/test_imports.py`
* `pyproject.toml` (if not existing)

**Key decisions to encode**

* Global helix axis = **+z**
* Coordinates stored as `np.ndarray` of shape `(N, 3)` in **angstrom**
* Atom indexing always refers to RDKit conformer indices where applicable

**APIs**

* `poly_csp.config.schema.HelixSpec`
* `poly_csp.config.schema.BackboneSpec`

**Acceptance**

* `pytest` passes; imports work.

---

### Ticket A1 — Screw transform and helix frame utilities

**Goal:** Implement exact screw-axis transform utilities; no chemistry yet.

**Files**

* `poly_csp/geometry/helix.py`
* `poly_csp/geometry/transform.py`
* `poly_csp/tests/test_screw_transform.py`

**APIs**

* `rotation_matrix_z(theta_rad: float) -> np.ndarray  # (3,3)`
* `ScrewTransform(theta_rad: float, rise_A: float)`

  * `.matrix(i: int) -> tuple[R: (3,3), t: (3,)]`
  * `.apply(points: (N,3), i: int) -> (N,3)`
* `kabsch_align(P: (N,3), Q: (N,3)) -> (R,t)` (for later)

**Acceptance tests**

* For random point set `X`, `apply(X, i+j)` equals `apply(apply(X, i), j)` within 1e-10 (group property).
* Rotation matrix orthonormality.
* Translation along z equals `i * rise_A`.

---

### Ticket A2 — Monomer template: labeled glucose scaffold

**Goal:** Create an RDKit monomer template with stable atom labels and attachment-site indices.

**Files**

* `poly_csp/chemistry/monomers.py`
* `poly_csp/io/rdkit_io.py` (helpers)
* `poly_csp/tests/test_monomer_template.py`

**APIs**

* `class GlucoseMonomerTemplate(BaseModel or dataclass):`

  * `mol: Chem.Mol`
  * `atom_idx: dict[str,int]`  (e.g., `"C1"`, `"O4"`, `"C2"`, `"C3"`, `"C6"`, `"O2"`, `"O3"`, `"O6"`, ring atoms)
  * `site_idx: dict[str,int]` mapping `{"C2": idx, "C3": idx, "C6": idx}` and optionally `{"O2","O3","O6"}` depending on functionalization chemistry
* `make_glucose_template(polymer: Literal["amylose","cellulose"]) -> GlucoseMonomerTemplate`

**Implementation notes**

* Don’t rely on RDKit’s conformer generation for “final geometry”; just ensure the template graph + atom naming is consistent.
* Include stereochemistry and ring closure; atom order must be deterministic.

**Acceptance**

* `mol.GetNumAtoms()` stable.
* Required labels exist.
* Stereochemistry not unspecified.

---

### Ticket A3 — Helix backbone placement (coordinates only, no polymer graph)

**Goal:** Given a monomer template + `HelixSpec`, generate DP residue conformers using screw transforms and residue-local frames.

**Files**

* `poly_csp/geometry/internal_coords.py` (stub now; full Z-matrix later)
* `poly_csp/chemistry/backbone_build.py`
* `poly_csp/tests/test_backbone_helical_coords.py`

**APIs**

* `build_backbone_coords(template: GlucoseMonomerTemplate, helix: HelixSpec, dp: int) -> np.ndarray`

  * Returns coordinates for *all atoms in all residues* in concatenated order: residue0 atoms, residue1 atoms, …
* `residue_frame_from_atoms(coords_res: (n,3), labels: dict[str,int]) -> (R,t)` (optional helper)

**Behavior**

* Place residue 0 with a defined orientation (e.g., ring plane roughly perpendicular to helix axis OR whichever convention you choose, but consistent).
* Generate residue i by applying `ScrewTransform` to residue 0 coordinates (initial pass).
* (Optional in this ticket) Apply simple local “link alignment” correction so that O4 of i points roughly toward C1 of i+1 along the helix.

**Acceptance**

* For any residue i, RMSD between residue i and screw-transformed residue 0 is ~0 (numerical tolerance).
* All residues lie on helix axis consistently (e.g., ring centroid radius roughly constant).

---

### Ticket A4 — Polymer graph construction + conformer assignment (Stage 2-lite)

**Goal:** Build the RDKit polymer molecule by repeating monomers and adding glycosidic bonds; assign coordinates from A3.

**Files**

* `poly_csp/chemistry/polymerize.py`
* `poly_csp/tests/test_polymerize_topology.py`

**APIs**

* `polymerize(template: GlucoseMonomerTemplate, dp: int, linkage: Literal["1-4"], anomer: Literal["alpha","beta"]) -> Chem.Mol`
* `assign_conformer(mol: Chem.Mol, coords: np.ndarray) -> Chem.Mol`

**Acceptance**

* RDKit sanitization passes (or passes with minimal exceptions you document).
* Total atoms = `dp * monomer_atoms`.
* Bonds include `dp-1` glycosidic link bonds between the correct labeled atoms (O4(i)–C1(i+1) or your chosen representation).

---

## Epic B — Deterministic selector attachment (Stage 3)

### Ticket B0 — Selector template interface + registry

**Goal:** Define a plugin interface so arbitrary selectors can be added without changing core code.

**Files**

* `poly_csp/chemistry/selectors.py`
* `poly_csp/tests/test_selector_registry.py`

**APIs**

* `class SelectorTemplate:`

  * `name: str`
  * `mol: Chem.Mol` (RDKit)
  * `attach_atom_label: str` or `attach_atom_idx: int`
  * `dihedrals: dict[str, tuple[int,int,int,int]]` (in selector-local atom indices)
  * `features: dict` (donors/acceptors, ring centroids optional)
* `SelectorRegistry.register(template: SelectorTemplate)`
* `SelectorRegistry.get(name: str) -> SelectorTemplate`

**Acceptance**

* Can register and retrieve templates; atom mapping exists.

---

### Ticket B1 — Implement 3,5-dimethylphenylcarbamate selector template

**Goal:** Provide a working selector example with dihedral definitions.

**Files**

* `poly_csp/chemistry/selector_library/dmpc_35.py`
* `poly_csp/tests/test_dmpc_35_template.py`

**APIs**

* `make_35_dmpc_template() -> SelectorTemplate`

**Notes**

* Include stable atom labels or a deterministic substructure match to define dihedrals.
* Provide at least these dihedrals (names are examples):

  * `tau_link = (sugar_attach, O_link, C_carbonyl, N)` will be set later after attachment; in the template define the last 3 and placeholder
  * `tau_ar = (C_carbonyl, N, C_ipso, C_ortho)` etc.

**Acceptance**

* Template builds; donor/acceptor atoms detected or specified.

---

### Ticket B2 — Residue-local frames + rigid placement of selector before bonding

**Goal:** Place a selector near the target site using a residue-local coordinate system so initial orientation is reproducible.

**Files**

* `poly_csp/geometry/local_frames.py`
* `poly_csp/chemistry/functionalization.py`
* `poly_csp/tests/test_selector_rigid_placement.py`

**APIs**

* `compute_residue_local_frame(coords_res: (n,3), labels: dict[str,int]) -> (R: (3,3), t: (3,))`
* `pose_selector_in_frame(selector_coords: (m,3), pose: SelectorPoseSpec, R,t) -> (m,3)`
* `class SelectorPoseSpec:` (in `config/schema.py` or `chemistry/selectors.py`)

  * desired direction vectors (e.g., “carbonyl points toward -radial”, “phenyl normal tangential”), plus initial dihedral guesses.

**Acceptance**

* Given same residue coords + pose spec, selector placement is bitwise deterministic (within floating tolerance).
* No huge overlaps (distance-based check) for a single residue.

---

### Ticket B3 — Chemical attachment at C2/C3/C6 with explicit atom mapping

**Goal:** Actually modify the polymer RDKit Mol: replace –OH with carbamate and connect selector.

**Files**

* `poly_csp/chemistry/functionalization.py`
* `poly_csp/tests/test_attach_selector_sites.py`

**APIs**

* `attach_selector(mol_polymer: Chem.Mol, residue_index: int, site: Literal["C2","C3","C6"], selector: SelectorTemplate, mode: Literal["replace_OH","connect_O"]) -> Chem.Mol`
* `get_residue_atom_index(mol_polymer, residue_index, monomer_template.atom_idx["O2"]) -> int` (helper)

**Implementation requirements**

* Deterministic atom indexing strategy:

  * If residue i atoms are contiguous blocks, compute global index as `i * n_monomer + local_idx`.
* Decide attachment chemistry:

  * Most realistic: use O2/O3/O6 oxygen as attachment point (carbamate via O–C(=O)–N–Ar). That means you are substituting the hydroxyl hydrogen and bonding from sugar oxygen to selector carbonyl carbon.
* Preserve hydrogens/valence.

**Acceptance**

* After attachment at all residues and sites, sanitization passes.
* Atom count increases by `dp * (atoms_in_selector - atoms_removed)` as expected.
* Bond exists between correct sugar oxygen and selector carbonyl carbon.

---

### Ticket B4 — Post-attachment coordinate merge + dihedral setting API

**Goal:** Merge selector coordinates into the polymer conformer and set initial dihedrals deterministically.

**Files**

* `poly_csp/chemistry/functionalization.py`
* `poly_csp/geometry/dihedrals.py`
* `poly_csp/tests/test_dihedral_setting.py`

**APIs**

* `merge_conformers(poly_coords: (N,3), selector_coords: (M,3), mapping) -> (N+M, 3)`
* `set_dihedral(coords, a,b,c,d, angle_rad) -> coords_new` (pure geometry function)
* `apply_selector_pose_dihedrals(mol, residue_index, site, pose_spec) -> mol`

**Acceptance**

* Setting a dihedral changes only downstream atoms (based on a defined rotation mask); include a simple chain test.
* After applying dihedrals, no bond lengths are broken.

---

## Epic C — Minimal pipeline + outputs (so you can see results)

### Ticket C0 — Build script: helix backbone + single-site selector for DP small

**Goal:** Provide a runnable entrypoint producing PDB and a JSON summary.

**Files**

* `poly_csp/pipelines/build_csp.py`
* `poly_csp/io/pdb.py`
* `poly_csp/tests/test_pipeline_smoke.py`
* `examples/build_cellulose_dp12_c6_dmpc.py`

**CLI**

* `python -m poly_csp.pipelines.build_csp --polymer cellulose --dp 12 --selector 35dmpc --sites C6 --out out/`

**Outputs**

* `out/model.pdb`
* `out/build_report.json` containing:

  * dp, helix params, selector name, sites
  * crude clash score (min heavy-atom distance)
  * torsion stats for selector dihedrals

**Acceptance**

* Smoke test runs <30s locally.
* Output files exist and are readable.

---

## Epic D — Quality gates (minimum viable QC)

### Ticket D0 — Steric clash checker + basic symmetry RMSD

**Goal:** Detect obviously disordered builds early (even before MM).

**Files**

* `poly_csp/ordering/scoring.py`
* `poly_csp/tests/test_qc_metrics.py`

**APIs**

* `min_interatomic_distance(coords, heavy_atom_mask) -> float`
* `screw_symmetry_rmsd(coords, residue_atom_count, screw: ScrewTransform, k: int) -> float`

**Acceptance**

* Symmetry RMSD near 0 for backbone-only.
* Clash score flags intentionally overlapped selectors.

---

# Assignment suggestion (parallelization)

To run agents in parallel without conflicts:

* **Agent 1:** A0, A1, A3 (geometry core)
* **Agent 2:** A2, A4 (chemistry template + polymer graph)
* **Agent 3:** B0, B1 (selector interface + 35-DMPC template)
* **Agent 4:** B2, B3, B4 (attachment + placement + dihedrals)
* **Agent 5:** C0, D0 (pipeline + QC)

---

# “Definition of Done” for the ticket set (vertical slice)

You’re done when:

1. `build_csp.py` produces a **helical cellulose/amylose backbone** for DP=12 with **screw symmetry**.
2. You can attach **3,5-DMPC** at **C6** across all residues deterministically.
3. The resulting PDB has **consistent, ordered initial orientations** (not random).
4. QC JSON includes **symmetry RMSD** and **clash score**.