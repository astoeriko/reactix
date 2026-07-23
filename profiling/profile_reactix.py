"""
Profiling script for the reactix library.

Mirrors the scenario from transport-model.ipynb:
conservative tracer + first-order kinetic decay.

The script measures:
  - First-run (JIT compile + exec) vs. post-JIT execution time
  - cProfile of the compile phase and the post-JIT execution phase
  - JAX / XLA op-level trace (drag the .json.gz file into https://ui.perfetto.dev)

Output layout:
    profiling/profiling_output/
        cprofile_compile.prof / .txt
        cprofile_exec.prof    / .txt
        jax_trace/

Usage:
    python profiling/profile_reactix.py
"""

import cProfile
import io
import pstats
import time
from pathlib import Path

import jax
import jax.numpy as jnp

from reactix import (
    Advection,
    Cells,
    Dispersion,
    FixedConcentrationBoundary,
    KineticReaction,
    TransportSystem,
    declare_species,
    make_solver,
    reaction,
)

jax.config.update("jax_enable_x64", True)

OUT_DIR = Path(__file__).parent / "profiling_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _cprofile_summary(pr, n=30) -> str:
    buf = io.StringIO()
    pstats.Stats(pr, stream=buf).sort_stats("cumulative").print_stats(n)
    return buf.getvalue()


# Scenario: conservative tracer + first-order decay
# (mirrors transport-model.ipynb)

Species = declare_species(["tracer", "reactive_tracer"])
species_is_mobile = Species(tracer=True, reactive_tracer=True)

@reaction
class FirstOrderDecay(KineticReaction):
    decay_coefficient: jax.Array

    def rate(self, time, state, system):
        return self.decay_coefficient * state.reactive_tracer

    def stoichiometry(self, time, state, system):
        return {"reactive_tracer": -1}

reactions = [FirstOrderDecay(decay_coefficient=1 / 500)]

n_cells = 200
interface_areas = jnp.ones(n_cells + 1)
cells = Cells.equally_spaced(10, n_cells, interface_area=interface_areas)

dispersion = Dispersion.build(
    cells=cells,
    dispersivity=jnp.array(0.1),
    pore_diffusion=Species(
        tracer=jnp.array(1e-9 * 3600 * 24),
        reactive_tracer=jnp.array(1e-9 * 3600 * 24),
    ),
)

advection = Advection.build(limiter_type="minmod")


bcs = [
    FixedConcentrationBoundary(
        boundary="left",
        species_selector=lambda s: s.tracer,
        fixed_concentration=lambda t: jnp.array(10.0),
    ),
    FixedConcentrationBoundary(
        boundary="right",
        species_selector=lambda s: s.tracer,
        fixed_concentration=lambda t: jnp.array(3.0),
    ),
    FixedConcentrationBoundary(
        boundary="left",
        species_selector=lambda s: s.reactive_tracer,
        fixed_concentration=lambda t: jnp.array(10.0),
    ),
    FixedConcentrationBoundary(
        boundary="right",
        species_selector=lambda s: s.reactive_tracer,
        fixed_concentration=lambda t: jnp.array(3.0),
    ),
]

porosity = jnp.ones(n_cells) * 0.3
porosity = porosity.at[100:].set(0.1)

system = TransportSystem.build(
    porosity=porosity,
    discharge=lambda t: jnp.array(1 / 365) * 0.3,
    cells=cells,
    advection=advection,
    dispersion=dispersion,
    species_is_mobile=species_is_mobile,
    bcs=bcs,
    reactions=reactions,
)

t_points = jnp.linspace(0, 8000, 123)
solver = make_solver(t_max=8000, t_points=t_points, rtol=1e-3, atol=1e-3)
val0 = jnp.zeros(cells.n_cells)
state = Species(tracer=val0, reactive_tracer=val0)


def solve():
    result = solver(state, system)
    jax.block_until_ready(result.ys)


# ---------------------------------------------------------------------------
# 1. JIT compile vs. execution timing
# ---------------------------------------------------------------------------

t0 = time.perf_counter()
solve()
print(f"First run (JIT compile + exec): {time.perf_counter() - t0:.3f} s")

times = []
for _ in range(3):
    t0 = time.perf_counter()
    solve()
    times.append(time.perf_counter() - t0)
print(f"Post-JIT (3 runs): {[f'{t:.3f}s' for t in times]}  mean={sum(times) / 3:.3f}s")

# ---------------------------------------------------------------------------
# 2. cProfile — compilation phase (fresh solver, first call)
# ---------------------------------------------------------------------------

fresh_solver = make_solver(t_max=8000, t_points=t_points, rtol=1e-3, atol=1e-3)

pr_compile = cProfile.Profile()
pr_compile.enable()
result = fresh_solver(state, system)
jax.block_until_ready(result.ys)
pr_compile.disable()

pr_compile.dump_stats(str(OUT_DIR / "cprofile_compile.prof"))
compile_txt = _cprofile_summary(pr_compile)
(OUT_DIR / "cprofile_compile.txt").write_text(compile_txt)
print(f"\ncProfile (compile phase):\n{compile_txt}")

# ---------------------------------------------------------------------------
# 3. cProfile — post-JIT execution
# ---------------------------------------------------------------------------

pr_exec = cProfile.Profile()
pr_exec.enable()
solve()
pr_exec.disable()

pr_exec.dump_stats(str(OUT_DIR / "cprofile_exec.prof"))
exec_txt = _cprofile_summary(pr_exec)
(OUT_DIR / "cprofile_exec.txt").write_text(exec_txt)
print(f"cProfile (post-JIT execution):\n{exec_txt}")

# ---------------------------------------------------------------------------
# 4. JAX XLA trace
# ---------------------------------------------------------------------------

trace_dir = OUT_DIR / "jax_trace"
with jax.profiler.trace(str(trace_dir), create_perfetto_link=False):
    solve()

print(f"\nJAX trace written to {trace_dir}/")
print("Open https://ui.perfetto.dev and drag-drop the .json.gz file.")
print(f"\nAll outputs in {OUT_DIR}/")
