# AMPS I · Week 13 — Advection–Diffusion: Péclet Number · Galerkin Oscillations · Upwinding · SUPG

| File | What | When |
|---|---|---|
| `W13_lecture.html` (`.qmd` source) | Lecture slides — Parts A (theory), B (SUPG in DOLFINx 0.11), C (measurements), D (LLM, HW12) | in class |
| `W13_lab.ipynb` | Lab: the Galerkin Péclet threshold in 1D, SUPG and nodal exactness, errors away from boundary layers in 2D, an interior layer (skew advection), a $\tau$ that survives $\mathbf b=0$, LLM demo | in class (≈ 80 min), finish at home |
| `W13_HW12.ipynb` | **HW12** — 1D difference equation and optimal diffusion (theory); Péclet sweep with Galerkin, upwind and SUPG on a new problem; designing $\tau$; LLM debugging of an SUPG script | due before Week 14 |
| `../dolfinx_0.11_api_reference.md` | DOLFINx 0.11 API reference sheet — paste it into your LLM | always |

**The session starts with a 10-minute quiz on the Week 12 lecture (closed-AI).**

All cells run in seconds to a minute. Measure errors **away from the layers** as well as on the whole domain, and always print the range of $u_h$.

**Before submitting:** *Runtime → Restart session and run all*, fill in the AI-usage log, download the `.ipynb`.
An error inside a `%%fenicsx` cell does **not** stop *Run all*, and `nan` in a matrix may only show up as a negative KSP reason — read every output.

**Week 14 starts with a 10-minute quiz on this week's lecture (closed-AI).**
