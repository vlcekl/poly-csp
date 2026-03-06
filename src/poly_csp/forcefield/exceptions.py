from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import openmm as mm

from poly_csp.topology.atom_mapping import ComponentTag


@dataclass(frozen=True)
class ScaleRule:
    scee: float
    scnb: float


_DEFAULT_RULES = {
    "backbone_backbone": ScaleRule(scee=1.0, scnb=1.0),
    "selector_selector": ScaleRule(scee=1.2, scnb=2.0),
    "cross_connector": ScaleRule(scee=1.0, scnb=1.0),
}


def _normalize_atom_map(atom_map: Mapping[int, ComponentTag | str]) -> dict[int, ComponentTag]:
    out: dict[int, ComponentTag] = {}
    for idx, tag in atom_map.items():
        if isinstance(tag, ComponentTag):
            out[int(idx)] = tag
        else:
            out[int(idx)] = ComponentTag(str(tag).strip().lower())
    return out


def _pair_rule(a: ComponentTag, b: ComponentTag, rules: dict[str, ScaleRule]) -> ScaleRule:
    if a is ComponentTag.BACKBONE and b is ComponentTag.BACKBONE:
        return rules["backbone_backbone"]
    if a is ComponentTag.SELECTOR and b is ComponentTag.SELECTOR:
        return rules["selector_selector"]
    return rules["cross_connector"]


def _parse_rules(mixing_rules_cfg: Mapping[str, object] | None) -> dict[str, ScaleRule]:
    rules = dict(_DEFAULT_RULES)
    if not mixing_rules_cfg:
        return rules

    scaling = mixing_rules_cfg.get("scaling", mixing_rules_cfg)
    if not isinstance(scaling, Mapping):
        return rules

    for key in ("backbone_backbone", "selector_selector", "cross_connector"):
        payload = scaling.get(key)
        if not isinstance(payload, Mapping):
            continue
        scee = float(payload.get("scee", rules[key].scee))
        scnb = float(payload.get("scnb", rules[key].scnb))
        rules[key] = ScaleRule(scee=scee, scnb=scnb)
    return rules


def apply_mixing_rules(
    system: mm.System,
    atom_map: Mapping[int, ComponentTag | str],
    mixing_rules_cfg: Mapping[str, object] | None = None,
) -> dict[str, int]:
    """Patch 1-4 exceptions in ``NonbondedForce`` using component-aware scales.

    This assumes the source exceptions are in GAFF-like scale (scee=1.2, scnb=2.0).
    If no ``NonbondedForce`` exists, this function is a no-op.
    """
    nonbonded = None
    for i in range(system.getNumForces()):
        force = system.getForce(i)
        if isinstance(force, mm.NonbondedForce):
            nonbonded = force
            break

    if nonbonded is None:
        return {"exceptions_seen": 0, "exceptions_patched": 0}

    tags = _normalize_atom_map(atom_map)
    rules = _parse_rules(mixing_rules_cfg)

    default_scee = 1.2
    default_scnb = 2.0
    patched = 0

    for i in range(nonbonded.getNumExceptions()):
        a, b, charge_prod, sigma, epsilon = nonbonded.getExceptionParameters(i)
        ta = tags.get(int(a), ComponentTag.BACKBONE)
        tb = tags.get(int(b), ComponentTag.BACKBONE)
        rule = _pair_rule(ta, tb, rules)

        new_charge = charge_prod
        new_epsilon = epsilon
        if rule.scee > 0:
            new_charge = charge_prod * (default_scee / float(rule.scee))
        if rule.scnb > 0:
            new_epsilon = epsilon * (default_scnb / float(rule.scnb))

        nonbonded.setExceptionParameters(i, a, b, new_charge, sigma, new_epsilon)
        patched += 1

    return {"exceptions_seen": nonbonded.getNumExceptions(), "exceptions_patched": patched}
