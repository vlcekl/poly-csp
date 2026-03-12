# Backbone Geometry Notes

This note documents the current local geometry rules enforced by the canonical backbone builder in [src/poly_csp/structure/backbone_builder.py](/home/lukas/work/projects/chiral_csp_poly/src/poly_csp/structure/backbone_builder.py).

## Geometry Derivation Rule

Backbone geometry is always derived from a complete chemically valid glucose structure first.

That means:

1. `natural_oh` glucose is embedded as the full monomer.
2. `anhydro` is derived by removing `O1` after embedding.
3. Explicit-H residue templates are built from the complete explicit-H monomer.
4. Residue-state variants are derived by pruning atoms or hydrogens from that complete template.

This avoids the class of bugs where embedding an already-pruned graph distorts the retained local stereochemistry.

## Current Local Linkage Targets

For each `O4(i)-C1(i+1)` linkage, the backbone pose solver targets:

1. `O4-C1` bond length,
2. `C4-O4-C1` donor angle,
3. `O4-C1-O5` acceptor angle,
4. `O4-C1-C2` acceptor-side stereochemistry angle.

These targets are derived from the explicit-H backbone templates rather than handwritten constants.

## Current Diagnostic Metric

The builder also measures `O4···H1` separation on the acceptor residue.

This is not a glycosidic torsion target. It is a regression guard against the previous under-constrained anomeric placement bug, where the incoming `O4` could rotate toward the `H1` direction.

## What Is Not Yet Constrained

The current builder does not yet impose explicit glycosidic:

- `phi`,
- `psi`,
- `omega`

targets.

Those are intentionally deferred to later work. Phase 1 only requires chemically plausible local covalent geometry and clean all-atom construction from explicit templates.
