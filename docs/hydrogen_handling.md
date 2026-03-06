# Hydrogen Handling in the Current Pipeline

This document describes how the current `poly_csp` pipeline handles hydrogens, step by step.

The key design choice is:

- The canonical working molecule for assembly, ordering, QC, and current OpenMM relaxation is still hydrogen-suppressed.
- "Hydrogen-suppressed" does not mean chemically incomplete. Exchangeable sites are represented by correct valence and correct `GetTotalNumHs()` counts on the heavy atoms.
- Explicit hydrogen atoms are introduced only in derived molecules:
  - temporarily during template embedding,
  - for GAFF/AmberTools fragment parameterization,
  - for the final all-atom structure outputs.

For a molecular modeler, the important consequence is that the runtime topology is currently a chemically correct heavy-atom graph, while all-atom structures are generated as derived views when needed.

## Terminology

### Implicit hydrogen

An RDKit heavy atom can carry hydrogen count information without an explicit hydrogen atom being present in the graph. In this codebase, most working molecules use this representation.

Example:

- a glucose hydroxyl oxygen can exist as an oxygen atom with `GetTotalNumHs() == 1`
- there is no separate hydrogen atom in the molecule graph at that stage

### Explicit hydrogen

A hydrogen atom is present as its own atom in the molecular graph and has coordinates, connectivity, and atom metadata.

This is the representation used for:

- temporary 3D embedding support,
- AmberTools fragment parameterization,
- final `model.pdb` and `model.sdf` outputs by default.

### Heavy-atom master

This is the canonical molecule passed through:

- monomer assembly,
- polymerization,
- terminal editing,
- selector attachment,
- ordering,
- current restrained relaxation.

It is hydrogen-suppressed but chemically valid.

## Stage 1: Monomer template definition

The glucose monomer templates are authored with mapped SMILES that explicitly mark hydroxyl-bearing atoms:

- `O2`, `O3`, `O4`, `O6` are written as `[OH:...]`
- `O1` is also `[OH:...]` in `natural_oh`
- ring oxygen `O5` remains non-protonated

Why this is done:

- it preserves the intended chemistry at template definition time
- it prevents the old failure mode where bare bracketed heteroatoms lost their exchangeable hydrogen semantics

However, RDKit interprets bracketed `[OH]` atoms as fixed-hydrogen atoms. That is not suitable for later bond formation, because a fixed `[OH]` oxygen cannot simply "lose" its hydrogen when a glycosidic or selector bond is formed.

So the monomer loader immediately normalizes those hydroxyl oxygens back to implicit-hydrogen behavior:

- `SetNoImplicit(False)`
- `SetNumExplicitHs(0)`
- `SanitizeMol()`

Result:

- the monomer template still reports one total hydrogen on free hydroxyl oxygens
- but the hydrogen is implicit, not a separate atom
- later bond formation can consume that hydrogen automatically through sanitization

## Stage 2: Temporary explicit hydrogens for template embedding

The monomer template is embedded as follows:

1. `Chem.AddHs(mol)` creates an all-atom temporary copy
2. RDKit embeds and optimizes that all-atom copy
3. `Chem.RemoveHs(..., sanitize=True)` strips the explicit hydrogens again

This happens for glucose templates and also for built-in selector templates such as 3,5-DMPC and TMB.

Why this is done:

- RDKit generally embeds and optimizes better when explicit hydrogens are present
- the project still wants a hydrogen-suppressed master after embedding

Result:

- template coordinates come from an all-atom embedding
- the stored template molecule returns to a heavy-atom representation
- chemically important hydrogen counts remain encoded implicitly

## Stage 3: Backbone polymerization

Polymerization duplicates the monomer template and creates `O4(i)-C1(i+1)` glycosidic bonds.

Important point:

- there are still no explicit hydrogen atoms in the working polymer graph
- hydrogen consumption is represented by a change in the heavy atom valence model

What happens chemically:

- when `O4` forms the glycosidic bond, RDKit sanitization reduces its hydrogen count from 1 to 0
- no explicit hydrogen atom is deleted, because none exists in the graph

For `natural_oh` representations, the code also removes `O1` from every residue that receives an incoming glycosidic bond.

This is a heavy-atom deletion, not a hydrogen deletion.

Why `O1` is removed:

- the `natural_oh` monomer includes a reducing-end `O1`
- internal residues in a 1->4 polymer should not retain that extra heavy atom
- removing `O1` keeps `C1` chemically valid after polymerization

Result:

- internal residues have the expected heavy-atom connectivity
- `O4` no longer reports a hydroxyl hydrogen after linkage
- the polymer master remains hydrogen-suppressed

## Stage 4: Terminal editing

After polymerization, terminal policy is applied.

### `open`

No topology change. Hydrogen state stays implicit on the heavy-atom master.

### `periodic`

The code:

- removes `res0:O1` for `natural_oh` chains
- adds the head-to-tail `O4(last)-C1(first)` bond

Again:

- the removed atom is a heavy atom (`O1`), not an explicit hydrogen
- the loss of hydroxyl character on the participating oxygen is handled implicitly by sanitization

### `capped`

The code adds heavy-atom cap fragments such as:

- methyl/methoxy,
- hydroxyl,
- acetyl

These caps are added as heavy atoms only. Their hydrogens are not made explicit at this stage.

Why:

- the heavy-atom master remains the canonical deterministic graph
- correct hydrogen counts are still implied by valence and sanitization

## Stage 5: Selector attachment

Selector attachment also operates on the hydrogen-suppressed master.

The sequence is:

1. combine the polymer and selector heavy-atom graphs
2. add the bond from the sugar OH oxygen to the selector attachment atom
3. remove the selector dummy atom `[*]`
4. sanitize the molecule

Hydrogen effects:

- the sugar attachment oxygen loses its implicit hydroxyl hydrogen during sanitization
- the DMPC carbamate nitrogen keeps one total hydrogen
- no explicit hydrogens are added or removed in the graph itself at this stage

This is now validated explicitly:

- attachment oxygen must have `GetTotalNumHs() == 0`
- DMPC connector `amide_n` must have `GetTotalNumHs() == 1`

Why the pipeline does not attach on an all-atom graph:

- explicit-hydrogen assembly would enlarge the atom-mapping and ordering surface significantly
- the current heavy-atom master already carries enough chemistry to perform the bond-forming step correctly

## Stage 6: Donor and acceptor logic

Selector donor detection now uses total hydrogen count, not only explicit hydrogen neighbors.

That matters because:

- the runtime selector templates are usually hydrogen-suppressed
- an amide `N-H` donor must still be recognized even when the H atom is not explicit

So donor inference uses:

- `GetTotalNumHs() > 0`

This lets the heavy-atom master remain chemically useful for:

- selector registry auto-detection,
- ordering heuristics,
- hydrogen-bond-like QC metrics.

## Stage 7: Current runtime forcefield and relaxation state

The current OpenMM assembly path still works on the hydrogen-suppressed polymer master.

That means:

- ordering runs on the heavy-atom graph
- QC is evaluated from the heavy-atom graph
- current restrained relaxation uses the heavy-atom graph
- the modular force builder still transfers only heavy-atom bonded terms into the runtime system

This is deliberate for now. The project has not yet moved the full polymer simulation path to an all-atom system.

## Stage 8: Explicit hydrogens for AmberTools fragment parameterization

Hydrogens become fully explicit when the code prepares selector and connector fragments for AmberTools.

This is the most important transition in the current hydrogen-aware design.

### 8.1 Isolated selector parameterization

For an isolated selector fragment:

1. dummy atoms are replaced with real hydrogen atoms, because Antechamber cannot type atomic number 0
2. `complete_with_hydrogens()` is called on the fragment
3. hydrogen coordinates are generated
4. a hydrogen-only local optimization is run, with heavy atoms fixed
5. the all-atom fragment is written to PDB and passed into Antechamber/Parmchk2/tleap

### 8.2 Capped-monomer connector parameterization

For connector extraction:

1. the code builds a heavy-atom capped monomer with one attached selector
2. that heavy-atom fragment is sent through the same GAFF fragment preparation path
3. explicit hydrogens are added before any AmberTools file generation
4. AmberTools produces an all-atom fragment prmtop
5. the code maps the all-atom prmtop back to heavy-atom semantic roles and extracts only the connector bonded terms needed by the current runtime builder

Why explicit hydrogens are required here:

- charge derivation and atom typing need chemically complete fragments
- carbamate and hydroxyl protonation states must be represented explicitly
- the source parameterization chemistry should be physically meaningful even if the current runtime system is still heavy-atom

## Stage 9: Optional AMBER export of the whole build

The current `export_amber_artifacts()` path is separate from final structure completion.

Important detail:

- the main build pipeline calls optional AMBER export before final all-atom completion
- the molecule passed into `export_amber_artifacts()` is still the heavy-atom master

For the current residue-aware GLYCAM path, this is acceptable because:

- tleap assembles the polysaccharide backbone from residue templates, not from a fully hydrogenated polymer PDB
- selector fragment parameterization is already handled through a separate explicit-hydrogen GAFF path

So the AMBER export layer is not currently the place where the whole polymer gets hydrogen-completed.

## Stage 10: Final all-atom structure completion

After assembly, optional ordering, optional relaxation, and optional AMBER export, the pipeline generates the final output structure.

By default, it now does:

1. `complete_with_hydrogens(mol_poly, add_coords=True, optimize="h_only")`
2. keep heavy-atom coordinates fixed
3. add explicit hydrogens to all chemically allowed sites
4. optimize only hydrogen positions
5. propagate metadata from each heavy atom to its added hydrogens

This metadata includes:

- parent heavy atom index,
- component classification,
- selector instance metadata,
- residue/site metadata when available

Why the hydrogen-only optimization is used:

- it gives reasonable hydrogen orientations
- it does not perturb the heavy-atom geometry produced by building, ordering, or relaxation

Result:

- `model.pdb` and `model.sdf` are all-atom by default
- optional `model_heavy.pdb` and `model_heavy.sdf` can still be emitted for debugging

## Stage 11: PDB naming and residue assignment for derived hydrogens

When explicit hydrogens are written to PDB:

- each hydrogen carries `_poly_csp_parent_heavy_idx`
- PDB naming and residue assignment are derived from the parent heavy atom

Examples:

- a backbone hydrogen attached to `O6` is assigned using the residue and label for `O6`
- a selector hydrogen inherits the selector instance and local-index naming context from its parent heavy atom

This keeps the all-atom output traceable back to the heavy-atom master.

## What is added, removed, or only reinterpreted

This is the most concise summary of the hydrogen lifecycle.

### Added temporarily

- explicit hydrogens during monomer embedding
- explicit hydrogens during selector template embedding

These are removed before the canonical template is stored.

### Added permanently to derived molecules

- explicit hydrogens in GAFF/AmberTools selector fragments
- explicit hydrogens in GAFF/AmberTools capped-monomer connector fragments
- explicit hydrogens in final `model.pdb` and `model.sdf`

### Never present as explicit atoms in the canonical runtime master

- hydroxyl hydrogens on glucose
- carbamate `N-H` on DMPC
- cap hydrogens on terminal groups

These are represented implicitly until final all-atom completion.

### Removed as heavy atoms

- `O1` on incoming-linkage residues in `natural_oh` polymerization
- `O1` on residue 0 in `periodic` mode for `natural_oh`
- selector dummy atom `[*]` during attachment

### Not explicitly removed because they were never explicit atoms

- the sugar hydroxyl hydrogen consumed during glycosidic bond formation
- the sugar hydroxyl hydrogen consumed during selector attachment

These are represented by a drop in `GetTotalNumHs()` after sanitization.

## Why the pipeline is designed this way

The current approach balances chemical correctness with implementation scope.

Benefits:

- the master graph stays small and deterministic
- atom mapping for selectors and residues remains simpler
- bond-forming chemistry is still represented correctly through implicit hydrogen counts
- fragment parameterization uses chemically complete all-atom inputs
- final outputs are chemically complete all-atom structures

This is a transitional architecture, but it is not a chemically naive one.

## Current limitations

The reader should be aware of what is not yet true.

- The full polymer OpenMM simulation path is not yet all-atom.
- The current modular system builder still consumes heavy-atom terms only.
- The whole-polymer AMBER export path is still separate from final all-atom completion.
- Hydrogen-handling config keys exist, but current runtime behavior is effectively fixed to the intended default:
  - hydrogen-complete fragment parameterization,
  - strict exchangeable-site validation,
  - all-atom final output by default.

## Bottom line

At present, the pipeline uses three hydrogen representations for three different jobs:

1. implicit hydrogens on the canonical heavy-atom master for chemistry-aware assembly,
2. explicit hydrogens on derived fragments for physically meaningful forcefield parameterization,
3. explicit hydrogens on the final exported structure for chemically complete deliverables.

That separation is intentional and is the central hydrogen-handling principle of the current codebase.
