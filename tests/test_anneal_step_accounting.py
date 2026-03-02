from __future__ import annotations

from dataclasses import dataclass, field

from poly_csp.mm.anneal import run_temperature_ramp


@dataclass
class _DummyIntegrator:
    temperatures: list[float] = field(default_factory=list)
    steps: list[int] = field(default_factory=list)

    def setTemperature(self, value) -> None:  # noqa: N802
        self.temperatures.append(0.0)

    def step(self, n_steps: int) -> None:
        self.steps.append(int(n_steps))


def test_temperature_ramp_executes_exact_total_steps() -> None:
    integ = _DummyIntegrator()
    run_temperature_ramp(
        context=object(),  # type: ignore[arg-type]
        integrator=integ,  # type: ignore[arg-type]
        t_start_K=50.0,
        t_end_K=350.0,
        n_steps=53,
        n_segments=10,
    )
    assert sum(integ.steps) == 53
    assert len(integ.steps) == 10


def test_temperature_ramp_handles_more_segments_than_steps() -> None:
    integ = _DummyIntegrator()
    run_temperature_ramp(
        context=object(),  # type: ignore[arg-type]
        integrator=integ,  # type: ignore[arg-type]
        t_start_K=100.0,
        t_end_K=200.0,
        n_steps=3,
        n_segments=10,
    )
    assert sum(integ.steps) == 3
    assert len(integ.steps) == 3
