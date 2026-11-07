# AMPS I · Week 10 — Time-Dependent Problems: θ-Method · Stability · Space–Time Convergence · Cahn–Hilliard

| File | What | When |
|---|---|---|
| `W10_lecture.html` (`.qmd` source) | Lecture slides — Parts A (theory), B (heat equation in DOLFINx), C (stability in practice), D (Cahn–Hilliard, LLM, HW9) | in class |
| `W10_lab.ipynb` | Lab: θ-method loop with one factorisation, Gaussian benchmark, temporal rates and the spatial floor, explicit Euler limit and mass lumping, Crank–Nicolson oscillations and the Rannacher start, Cahn–Hilliard (energy, mass), LLM demo | in class (≈ 80 min), finish at home |
| `W10_HW9.ipynb` | **HW9** — amplification factor, 1D eigenvalues, energy stability; MMS with time-dependent boundary data: temporal rates, joint refinement, order reduction; explicit-Euler limit measured; LLM debugging | due before Week 11 |
| `../dolfinx_0.11_api_reference.md` | DOLFINx 0.11 API reference sheet — paste it into your LLM | always |

**The session starts with a 10-minute quiz on the Week 9 lecture (closed-AI).**

The Cahn–Hilliard cells of the lab take about a minute each on Colab.

**Before submitting:** *Runtime → Restart session and run all*, fill in the AI-usage log, download the `.ipynb`.
An error inside a `%%fenicsx` cell does **not** stop *Run all* — read every output before you submit.

**Week 11 starts with a 10-minute quiz on this week's lecture (closed-AI).**
