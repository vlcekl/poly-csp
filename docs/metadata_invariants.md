# poly_csp Metadata Invariants

This document defines molecule-level metadata used across polymerization, attachment,
dihedral application, and future terminal/export stages.

## Molecule Properties

- `_poly_csp_dp`:
  Degree of polymerization (residue count in the base backbone before selector addition).

- `_poly_csp_representation`:
  Monomer representation mode (`anhydro` or `natural_oh`).

- `_poly_csp_template_atom_count`:
  Atom count of the monomer template used to generate initial concatenated coordinates.

- `_poly_csp_removed_old_indices_json`:
  JSON list of integer atom indices removed from the initial concatenated template stack.
  Used to prune backbone coordinates before conformer assignment.

- `_poly_csp_residue_label_map_json`:
  JSON list (`len == dp`) of per-residue dictionaries mapping atom labels (`C1`, `O4`,
  `O6`, etc.) to molecule-global atom indices after representation-specific edits.

- `_poly_csp_siteidx_<label>`:
  Local template site index hints (backward compatibility), e.g. `_poly_csp_siteidx_O6`.

- `_poly_csp_end_mode`:
  Terminal policy mode (`open`, `capped`, `periodic`).

- `_poly_csp_end_caps_json`:
  JSON dictionary of configured end caps (currently metadata only).

- `_poly_csp_terminal_topology_pending`:
  Boolean indicating terminal chemistry edits are still pending implementation.

- `_poly_csp_selector_count`:
  Number of selectors attached so far (monotonic with each attach operation).

## Atom Properties (selector atoms)

- `_poly_csp_selector_instance`:
  Integer selector attachment instance id.

- `_poly_csp_residue_index`:
  Backbone residue index this selector is attached to.

- `_poly_csp_site`:
  Attachment site label (`C2`, `C3`, or `C6`).

- `_poly_csp_selector_local_idx`:
  Selector template-local atom index before any merge/removal operations.

## Invariants

1. `_poly_csp_residue_label_map_json` must always refer to valid global atom indices in the current molecule.
2. For `natural_oh` representation, `O1` may be absent on internal residues by construction.
3. Selector attachment and dihedral APIs must resolve sugar-site oxygen indices via residue label maps, not fixed block offsets.
4. Metadata must be copied forward when creating a new `Chem.Mol` from edited `RWMol` objects.
