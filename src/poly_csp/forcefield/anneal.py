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
    base = total_steps // segments
    remainder = total_steps % segments

    for seg in range(segments):
        steps = base + (1 if seg < remainder else 0)
        if steps <= 0:
            continue
        frac = float(seg) / float(max(1, segments - 1))
        t = float(t_start_K + (t_end_K - t_start_K) * frac)
        integrator.setTemperature(t * unit.kelvin)
        integrator.step(int(steps))


def run_heat_cool_cycle(
    context: mm.Context,
    integrator: mm.LangevinIntegrator,
    t_start_K: float,
    t_peak_K: float,
    n_steps: int,
    n_segments: int = 10,
) -> None:
    """Heat from *t_start_K* → *t_peak_K*, then cool back to *t_start_K*.

    Total step budget *n_steps* is split equally between the two ramps.
    """
    half = max(0, int(n_steps)) // 2
    if half == 0:
        return
    run_temperature_ramp(context, integrator, t_start_K, t_peak_K, half, n_segments)
    run_temperature_ramp(context, integrator, t_peak_K, t_start_K, half, n_segments)
