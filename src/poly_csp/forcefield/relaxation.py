from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
from rdkit import Chem

import openmm as mm
from openmm import unit

from poly_csp.forcefield.anneal import run_heat_cool_cycle, run_temperature_ramp
from poly_csp.forcefield.minimization import (
    RuntimeRestraintSpec,
    TwoStageMinimizationProtocol,
    backbone_all_indices,
    backbone_heavy_indices,
    connector_all_indices,
    new_context,
    potential_energy_kj_mol,
    positions_nm_from_mol,
    prepare_system_for_minimization,
    run_two_stage_minimization,
    set_optional_parameter,
    selector_all_indices,
    update_rdkit_coords,
)
from poly_csp.forcefield.runtime_params import RuntimeParams, load_runtime_params
from poly_csp.forcefield.system_builder import create_system
from poly_csp.topology.selectors import SelectorTemplate


@dataclass(frozen=True)
class RelaxSpec:
    enabled: bool
    positional_k: float
    dihedral_k: float
    hbond_k: float
    n_stages: int = 3
    max_iterations: int = 200
    freeze_backbone: bool = True
    anneal_enabled: bool = False
    t_start_K: float = 50.0
    t_end_K: float = 350.0
    anneal_steps: int = 2000
    anneal_cool_down: bool = True


def _require_forcefield_molecule(mol: Chem.Mol) -> None:
    if not mol.HasProp("_poly_csp_manifest_schema_version"):
        raise ValueError(
            "Relaxation requires a forcefield-domain molecule from build_forcefield_molecule()."
        )


def _apply_stage2_anneal(
    context: mm.Context,
    integrator: mm.LangevinIntegrator,
    spec: RelaxSpec,
) -> None:
    if not spec.anneal_enabled or int(spec.anneal_steps) <= 0:
        return
    if spec.anneal_cool_down:
        run_heat_cool_cycle(
            context=context,
            integrator=integrator,
            t_start_K=float(spec.t_start_K),
            t_peak_K=float(spec.t_end_K),
            n_steps=int(spec.anneal_steps),
            n_segments=10,
        )
        return
    run_temperature_ramp(
        context=context,
        integrator=integrator,
        t_start_K=float(spec.t_start_K),
        t_end_K=float(spec.t_end_K),
        n_steps=int(spec.anneal_steps),
        n_segments=10,
    )


def run_staged_relaxation(
    mol: Chem.Mol,
    spec: RelaxSpec,
    selector: SelectorTemplate | None = None,
    *,
    runtime_params: RuntimeParams | None = None,
    work_dir: str | Path | None = None,
    soft_repulsion_k_kj_per_mol_nm2: float = 800.0,
    soft_repulsion_cutoff_nm: float = 0.6,
    mixing_rules_cfg: Mapping[str, object] | None = None,
) -> tuple[Chem.Mol, Dict[str, object]]:
    """Run the canonical two-stage runtime relaxation on a forcefield-domain molecule."""
    if not spec.enabled:
        return Chem.Mol(mol), {"enabled": False}

    _require_forcefield_molecule(mol)

    runtime = runtime_params
    if runtime is None:
        runtime = load_runtime_params(
            mol,
            selector_template=selector,
            work_dir=None if work_dir is None else Path(work_dir),
        )

    reference_positions_nm = positions_nm_from_mol(mol)

    soft_result = create_system(
        mol,
        glycam_params=runtime.glycam,
        selector_params_by_name=runtime.selector_params_by_name,
        connector_params_by_key=runtime.connector_params_by_key,
        parameter_provenance=runtime.source_manifest,
        nonbonded_mode="soft",
        repulsion_k_kj_per_mol_nm2=float(soft_repulsion_k_kj_per_mol_nm2),
        repulsion_cutoff_nm=float(soft_repulsion_cutoff_nm),
        mixing_rules_cfg=mixing_rules_cfg,
    )
    prepare_system_for_minimization(
        system=soft_result.system,
        mol=mol,
        restraint_spec=RuntimeRestraintSpec(
            positional_k=float(spec.positional_k),
            dihedral_k=float(spec.dihedral_k),
            hbond_k=float(spec.hbond_k),
            freeze_backbone=bool(spec.freeze_backbone),
        ),
        selector=selector,
        reference_positions_nm=reference_positions_nm,
    )

    full_result = create_system(
        mol,
        glycam_params=runtime.glycam,
        selector_params_by_name=runtime.selector_params_by_name,
        connector_params_by_key=runtime.connector_params_by_key,
        parameter_provenance=runtime.source_manifest,
        nonbonded_mode="full",
        mixing_rules_cfg=mixing_rules_cfg,
    )
    prepare_system_for_minimization(
        system=full_result.system,
        mol=mol,
        restraint_spec=RuntimeRestraintSpec(
            positional_k=float(spec.positional_k),
            dihedral_k=float(spec.dihedral_k),
            hbond_k=float(spec.hbond_k),
            freeze_backbone=bool(spec.freeze_backbone),
        ),
        selector=selector,
        reference_positions_nm=reference_positions_nm,
    )
    minimization = run_two_stage_minimization(
        soft_system=soft_result.system,
        full_system=full_result.system,
        initial_positions_nm=soft_result.positions_nm,
        restraint_spec=RuntimeRestraintSpec(
            positional_k=float(spec.positional_k),
            dihedral_k=float(spec.dihedral_k),
            hbond_k=float(spec.hbond_k),
            freeze_backbone=bool(spec.freeze_backbone),
        ),
        protocol=TwoStageMinimizationProtocol(
            n_stages=int(spec.n_stages),
            soft_max_iterations=int(spec.max_iterations),
            full_max_iterations=int(spec.max_iterations),
            final_restraint_factor=0.15,
        ),
    )
    stage2_energies = list(minimization.stage2_energies_kj_mol)

    if spec.anneal_enabled and int(spec.anneal_steps) > 0:
        full_context, full_integrator = new_context(
            full_result.system,
            minimization.final_positions_nm,
        )
        set_optional_parameter(full_context, "k_pos", float(spec.positional_k) * 0.15)
        set_optional_parameter(full_context, "k_tors", float(spec.dihedral_k) * 0.15)
        set_optional_parameter(full_context, "k_hb", float(spec.hbond_k) * 0.15)
        _apply_stage2_anneal(full_context, full_integrator, spec)
        mm.LocalEnergyMinimizer.minimize(
            full_context,
            tolerance=10.0,
            maxIterations=int(spec.max_iterations),
        )
        stage2_energies.append(potential_energy_kj_mol(full_context))
        final_positions = full_context.getState(getPositions=True).getPositions(asNumpy=True)
        del full_context, full_integrator
    else:
        final_positions = minimization.final_positions_nm

    final_xyz_A = np.asarray(final_positions.value_in_unit(unit.nanometer)) * 10.0
    span = final_xyz_A.max(axis=0) - final_xyz_A.min(axis=0)

    backbone_heavy = backbone_heavy_indices(mol)
    backbone_all = backbone_all_indices(mol)
    selector_indices = selector_all_indices(mol)
    connector_indices = connector_all_indices(mol)
    backbone_drift = 0.0
    if spec.freeze_backbone and backbone_heavy:
        init_backbone_xyz = (
            np.asarray(reference_positions_nm.value_in_unit(unit.nanometer))[backbone_heavy]
            * 10.0
        )
        final_backbone_xyz = final_xyz_A[backbone_heavy]
        backbone_drift = float(np.max(np.abs(final_backbone_xyz - init_backbone_xyz)))

    out = update_rdkit_coords(mol, final_positions)
    summary: Dict[str, object] = {
        "enabled": True,
        "protocol": "two_stage_runtime",
        "stage1_nonbonded_mode": soft_result.nonbonded_mode,
        "stage1_energies_kj_mol": list(minimization.stage1_energies_kj_mol),
        "stage2_nonbonded_mode": full_result.nonbonded_mode,
        "stage2_energies_kj_mol": stage2_energies,
        "anneal_enabled": bool(spec.anneal_enabled),
        "anneal_cool_down": bool(spec.anneal_cool_down),
        "freeze_backbone": bool(spec.freeze_backbone),
        "n_backbone_atoms": len(backbone_all),
        "n_backbone_heavy_frozen": len(backbone_heavy) if spec.freeze_backbone else 0,
        "n_selector_atoms": len(selector_indices),
        "n_connector_atoms": len(connector_indices),
        "component_counts": dict(full_result.component_counts),
        "soft_exception_summary": dict(soft_result.exception_summary),
        "full_exception_summary": dict(full_result.exception_summary),
        "source_manifest": dict(full_result.source_manifest),
        "span_A": [float(value) for value in span.tolist()],
        "backbone_drift_A": float(backbone_drift),
    }
    return out, summary
