# AMPS I · Week 11 — Stokes Flow: Mixed Problems · Inf-Sup · Brezzi · Taylor–Hood, MINI and Unstable Pairs

| File | What | When |
|---|---|---|
| `W11_lecture.html` (`.qmd` source) | Lecture slides — Parts A (theory), B (mixed spaces in DOLFINx), C (element pairs in practice), D (LLM, HW10) | in class |
| `W11_lab.ipynb` | Lab: mixed and enriched elements, Poiseuille exactness, Taylor–Hood MMS and the interpolant pitfall, two P1–P1 failures, discrete inf-sup constants, Stokes cavity and its symmetry, LLM demo | in class (≈ 80 min), finish at home |
| `W11_HW10.ipynb` | **HW10** — saddle-point theory and Fortin's bubble; one solver for Taylor–Hood, MINI and P1–P1 (rates, pressure plot, cost); your own inf-sup computation; LLM debugging | due before Week 12 |
| `../dolfinx_0.11_api_reference.md` | DOLFINx 0.11 API reference sheet — paste it into your LLM | always |

**The session starts with a 10-minute quiz on the Week 10 lecture (closed-AI).**

The inf-sup cells (lab §5, HW Q3) build dense matrices and run with `-np 1`; they take up to a minute.

**Before submitting:** *Runtime → Restart session and run all*, fill in the AI-usage log, download the `.ipynb`.
An error inside a `%%fenicsx` cell does **not** stop *Run all*, and a singular Stokes matrix may still give KSP reason 4 — read every output and plot the pressure.

**Week 12 starts with a 10-minute quiz on this week's lecture (closed-AI).**
