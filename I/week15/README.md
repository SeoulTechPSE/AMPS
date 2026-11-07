# AMPS I · Week 15 — Steady Navier–Stokes: Newton · Reynolds-Number Continuation · Lid-Driven Cavity · PCD · H(div)-Conforming DG

| File | What | When |
|---|---|---|
| `W15_lecture.html` (`.qmd` source) | Lecture slides — Parts A (theory), B (steady NS in DOLFINx 0.11), C (measurements), D (LLM, HW14) | in class |
| `W15_lab.ipynb` | Lab: Newton for the cavity and Ghia et al., Picard vs Newton, Re continuation, GMRES with a mass-matrix and a PCD Schur complement (`fenicsx-pctools`), H(div)-DG vs Taylor–Hood on Kovasznay flow, LLM demo | in class (≈ 80 min), finish at home |
| `W15_HW14.ipynb` | **HW14** — Newton, continuation, PCD, H(div) (theory); the cavity at Re = 100, 400, 1000 on three meshes; GMRES iterations along a continuation path; LLM debugging of a cavity script | due before the Week 16 session |
| `../dolfinx_0.11_api_reference.md` | DOLFINx 0.11 API reference sheet — paste it into your LLM | always |

**The session starts with a 10-minute quiz on the Week 14 lecture (closed-AI).**

**`fenicsx-pctools`.** Lab §3 and HW Q3 need the PCD preconditioner from `fenicsx-pctools`. The notebooks contain an install cell (Colab: pip is bootstrapped into the
`%%fenicsx` environment, then `pip install --no-deps`; about 20 s). Run it once per session, before the PCD cells.

**Run times** (measured locally with `-np 2`, whole notebook including compilation: lab solution 1.7 min, homework solution 4 min; Colab was 2–3 times slower
in earlier weeks, so expect roughly 4–5 minutes for the lab and about 10 minutes for the homework). Save results to `/tmp/fnx/` so that plots do not re-run
simulations, and do not save the notebook before the last cell has finished.

**Before submitting:** *Runtime → Restart session and run all*, fill in the AI-usage log, download the `.ipynb`.
An error inside a `%%fenicsx` cell does **not** stop *Run all*. An exception on one MPI rank only can leave the other rank waiting forever — if a cell hangs, interrupt it
and look for an error printed before the hang.

**Week 16: a 10-minute quiz on this week's lecture (closed-AI), then the final project presentations. There is no final exam.**
