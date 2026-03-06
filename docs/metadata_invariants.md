# poly_csp Metadata Invariants

This document defines molecule-level metadata used across polymerization, attachment,
dihedral application, atom mapping, and terminal/export stages.

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
  JSON dictionary of configured end caps.

- `_poly_csp_terminal_topology_pending`:
  Boolean indicating terminal chemistry edits are pending (`False` for implemented modes).

- `_poly_csp_terminal_meta_json`:
  JSON dictionary with terminal-mode-specific metadata (e.g., periodic closure bond indices).

- `_poly_csp_terminal_cap_indices_json`:
  JSON dictionary containing atom indices added by left/right cap operations.

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

- `_poly_csp_component`:
  Component tag used by the domain migration (`backbone`, `selector`, `connector`).
  Added during assembly and consumed by `topology.atom_mapping` + forcefield mixing logic.

- `_poly_csp_connector_atom`:
  Optional boolean-like marker for connector boundary atoms (reserved for capped-monomer
  parameter extraction flow).

- `_poly_csp_connector_role`:
  Semantic connector role for attached linker atoms (`carbonyl_c`, `carbonyl_o`,
  `amide_n`, etc.). Connector atoms keep selector instance metadata but must carry
  component tag `connector`.

## Invariants

1. `_poly_csp_residue_label_map_json` must always refer to valid global atom indices in the current molecule.
2. For `natural_oh` representation, `O1` may be absent on internal residues by construction.
3. Selector attachment and dihedral APIs must resolve sugar-site oxygen indices via residue label maps, not fixed block offsets.
4. Metadata must be copied forward when creating a new `Chem.Mol` from edited `RWMol` objects.
5. Component tagging must remain single-valued per atom (`backbone` xor `selector` xor `connector`).
6. Connector atoms must retain `_poly_csp_selector_instance` and `_poly_csp_selector_local_idx` so they can be remapped from capped-fragment parameters back onto the full polymer.
