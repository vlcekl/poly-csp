from __future__ import annotations

import openmm as mm
from openmm import unit


def run_temperature_ramp(
    context: mm.Context,
    integrator: mm.LangevinIntegrator,
    t_start_K: float,
    t_end_K: float,
    n_steps: int,
    n_segments: int = 10,
) -> None:
    segments = max(1, int(n_segments))
    total_steps = max(0, int(n_steps))
    if total_steps == 0:
        return
    steps_per_segment = max(1, total_steps // segments)

    for seg in range(segments):
        frac = float(seg) / float(max(1, segments - 1))
        t = float(t_start_K + (t_end_K - t_start_K) * frac)
        integrator.setTemperature(t * unit.kelvin)
        integrator.step(steps_per_segment)
