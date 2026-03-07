from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

from rdkit import Chem

import openmm as mm
from openmm import unit


ComponentClass = Literal["backbone", "selector", "connector"]
FinePairClass = Literal[
    "backbone_backbone",
    "selector_selector",
    "backbone_selector",
    "backbone_connector",
    "selector_connector",
    "connector_connector",
]
RuleBucket = Literal["backbone_backbone", "selector_selector", "cross_boundary"]


@dataclass(frozen=True)
class ScaleRule:
    scee: float
    scnb: float


@dataclass(frozen=True)
class ExceptionPair:
    atom_a: int
    atom_b: int
    fine_class: FinePairClass
    rule_bucket: RuleBucket

    @property
    def key(self) -> tuple[int, int]:
        return (self.atom_a, self.atom_b) if self.atom_a <= self.atom_b else (self.atom_b, self.atom_a)


_DEFAULT_RULES: dict[RuleBucket, ScaleRule] = {
    "backbone_backbone": ScaleRule(scee=1.0, scnb=1.0),
    "selector_selector": ScaleRule(scee=1.2, scnb=2.0),
    "cross_boundary": ScaleRule(scee=1.0, scnb=1.0),
}
_FULL_BASELINE_RULE = ScaleRule(scee=1.0, scnb=1.0)
_MANIFEST_SOURCES: frozenset[str] = frozenset({"backbone", "selector", "connector"})


def _component_from_atom(atom: Chem.Atom) -> ComponentClass:
    if not atom.HasProp("_poly_csp_manifest_source"):
        raise ValueError(
            f"Atom {atom.GetIdx()} is missing _poly_csp_manifest_source required for 1-4 classification."
        )
    source = atom.GetProp("_poly_csp_manifest_source").strip().lower()
    if source not in _MANIFEST_SOURCES:
        raise ValueError(
            "Unsupported manifest source for 1-4 classification: "
            f"atom {atom.GetIdx()} has source {source!r}."
        )
    return source  # type: ignore[return-value]


def _fine_pair_class(left: ComponentClass, right: ComponentClass) -> FinePairClass:
    pair = tuple(sorted((left, right)))
    if pair == ("backbone", "backbone"):
        return "backbone_backbone"
    if pair == ("selector", "selector"):
        return "selector_selector"
    if pair == ("connector", "connector"):
        return "connector_connector"
    if pair == ("backbone", "selector"):
        return "backbone_selector"
    if pair == ("backbone", "connector"):
        return "backbone_connector"
    if pair == ("connector", "selector"):
        return "selector_connector"
    raise ValueError(f"Unsupported 1-4 component pair {pair!r}.")


def _rule_bucket(fine_class: FinePairClass) -> RuleBucket:
    if fine_class == "backbone_backbone":
        return "backbone_backbone"
    if fine_class == "selector_selector":
        return "selector_selector"
    return "cross_boundary"


def _parse_rules(mixing_rules_cfg: Mapping[str, object] | None) -> dict[RuleBucket, ScaleRule]:
    rules = dict(_DEFAULT_RULES)
    if not mixing_rules_cfg:
        return rules

    scaling = mixing_rules_cfg.get("scaling", mixing_rules_cfg)
    if not isinstance(scaling, Mapping):
        return rules

    for key in ("backbone_backbone", "selector_selector", "cross_boundary"):
        payload = scaling.get(key)
        if not isinstance(payload, Mapping):
            continue
        rules[key] = ScaleRule(
            scee=float(payload.get("scee", rules[key].scee)),
            scnb=float(payload.get("scnb", rules[key].scnb)),
        )
    return rules


def _pair_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a <= b else (b, a)


def _expected_one_four_pairs(mol: Chem.Mol) -> dict[tuple[int, int], ExceptionPair]:
    n_atoms = mol.GetNumAtoms()
    distance_matrix = Chem.GetDistanceMatrix(mol)
    pairs: dict[tuple[int, int], ExceptionPair] = {}
    for src in range(n_atoms):
        for neighbor in range(src + 1, n_atoms):
            if int(round(float(distance_matrix[src, neighbor]))) != 3:
                continue
            left_component = _component_from_atom(mol.GetAtomWithIdx(src))
            right_component = _component_from_atom(mol.GetAtomWithIdx(neighbor))
            fine_class = _fine_pair_class(left_component, right_component)
            pair = ExceptionPair(
                atom_a=src,
                atom_b=neighbor,
                fine_class=fine_class,
                rule_bucket=_rule_bucket(fine_class),
            )
            pairs[pair.key] = pair
    return pairs


def _exception_index_by_pair(nonbonded: mm.NonbondedForce) -> dict[tuple[int, int], int]:
    out: dict[tuple[int, int], int] = {}
    for exception_idx in range(nonbonded.getNumExceptions()):
        atom_a, atom_b, _, _, _ = nonbonded.getExceptionParameters(exception_idx)
        out[_pair_key(int(atom_a), int(atom_b))] = int(exception_idx)
    return out


def _has_nonzero_nonbonded_exception(charge_prod, epsilon) -> bool:
    charge_abs = abs(float(charge_prod.value_in_unit(unit.elementary_charge**2)))
    epsilon_abs = abs(float(epsilon.value_in_unit(unit.kilojoule_per_mole)))
    return charge_abs > 1e-12 or epsilon_abs > 1e-12


def apply_mixing_rules(
    nonbonded: mm.NonbondedForce,
    mol: Chem.Mol,
    mixing_rules_cfg: Mapping[str, object] | None = None,
    *,
    baseline_scee: float = _FULL_BASELINE_RULE.scee,
    baseline_scnb: float = _FULL_BASELINE_RULE.scnb,
) -> dict[str, object]:
    """Patch true 1-4 exceptions in ``nonbonded`` using the canonical runtime rules."""
    if baseline_scee <= 0 or baseline_scnb <= 0:
        raise ValueError("1-4 exception baseline scales must be positive.")

    rules = _parse_rules(mixing_rules_cfg)
    expected_pairs = _expected_one_four_pairs(mol)
    by_exception_pair = _exception_index_by_pair(nonbonded)

    missing_pairs = sorted(pair for pair in expected_pairs if pair not in by_exception_pair)
    if missing_pairs:
        raise ValueError(
            "OpenMM NonbondedForce is missing expected 1-4 exceptions for pairs "
            f"{missing_pairs!r}."
        )

    patched = 0
    found_true_14 = 0
    counts_by_rule_bucket: dict[str, int] = {key: 0 for key in rules}
    counts_by_fine_pair_class: dict[str, int] = {
        "backbone_backbone": 0,
        "selector_selector": 0,
        "backbone_selector": 0,
        "backbone_connector": 0,
        "selector_connector": 0,
        "connector_connector": 0,
    }
    connector_involving_pairs = 0

    for pair_key, pair in expected_pairs.items():
        found_true_14 += 1
        counts_by_rule_bucket[pair.rule_bucket] += 1
        counts_by_fine_pair_class[pair.fine_class] += 1
        if "connector" in pair.fine_class:
            connector_involving_pairs += 1

        exception_idx = by_exception_pair[pair_key]
        atom_a, atom_b, charge_prod, sigma, epsilon = nonbonded.getExceptionParameters(exception_idx)
        target_rule = rules[pair.rule_bucket]
        new_charge = charge_prod * (float(baseline_scee) / float(target_rule.scee))
        new_epsilon = epsilon * (float(baseline_scnb) / float(target_rule.scnb))
        if (
            abs(float(baseline_scee) - float(target_rule.scee)) > 1e-12
            or abs(float(baseline_scnb) - float(target_rule.scnb)) > 1e-12
        ):
            patched += 1
        nonbonded.setExceptionParameters(exception_idx, atom_a, atom_b, new_charge, sigma, new_epsilon)

    unexpected_nonzero_pairs: list[tuple[int, int]] = []
    for pair_key, exception_idx in by_exception_pair.items():
        if pair_key in expected_pairs:
            continue
        _, _, charge_prod, _, epsilon = nonbonded.getExceptionParameters(exception_idx)
        if _has_nonzero_nonbonded_exception(charge_prod, epsilon):
            unexpected_nonzero_pairs.append(pair_key)
    if unexpected_nonzero_pairs:
        raise ValueError(
            "OpenMM NonbondedForce contains nonzero exception entries outside the expected "
            f"1-4 pair set: {unexpected_nonzero_pairs!r}."
        )

    return {
        "baseline_scee": float(baseline_scee),
        "baseline_scnb": float(baseline_scnb),
        "exceptions_seen": int(nonbonded.getNumExceptions()),
        "expected_14_pairs": int(len(expected_pairs)),
        "found_14_pairs": int(found_true_14),
        "patched_14_pairs": int(patched),
        "counts_by_rule_bucket": {
            key: int(counts_by_rule_bucket[key])
            for key in ("backbone_backbone", "selector_selector", "cross_boundary")
        },
        "counts_by_fine_pair_class": {
            key: int(counts_by_fine_pair_class[key])
            for key in (
                "backbone_backbone",
                "selector_selector",
                "backbone_selector",
                "backbone_connector",
                "selector_connector",
                "connector_connector",
            )
        },
        "connector_involving_pairs": int(connector_involving_pairs),
    }
