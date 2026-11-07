# AMPS I · Week 12 — Solving Saddle-Point Systems: Schur Complement · Block Preconditioners · fieldsplit · Stokes and Darcy

| File | What | When |
|---|---|---|
| `W12_lecture.html` (`.qmd` source) | Lecture slides — Parts A (theory), B (DOLFINx 0.11 + PETSc), C (measurements), D (LLM, HW11) | in class |
| `W12_lab.ipynb` | Lab: spectra of the preconditioned Stokes matrix, one problem three ways (monolithic, nested, blocked), iterations vs $h$, LU vs MINRES in 2D and 3D, Darcy with L², H(div) and Schur preconditioners, LLM demo | in class (≈ 80 min), finish at home |
| `W12_HW11.ipynb` | **HW11** — preconditioned spectra (theory); 3D lid-driven Stokes cavity: LU vs two block preconditioners (iterations, time, slopes); Darcy with a high-contrast permeability; LLM debugging of a solver benchmark | due before Week 13 |
| `../dolfinx_0.11_api_reference.md` | DOLFINx 0.11 API reference sheet — paste it into your LLM | always |

**The session starts with a 10-minute quiz on the Week 11 lecture (closed-AI).**

Lab §1 builds dense matrices (`-np 1`, a few seconds). Lab §4 and HW Q2 time solvers (about 20 s and 1.5 min on the instructor's machine; longer on Colab),
and the times differ between runs — compare growth over a wide range of sizes, not the third digit.

**Before submitting:** *Runtime → Restart session and run all*, fill in the AI-usage log, download the `.ipynb`.
An error inside a `%%fenicsx` cell does **not** stop *Run all*, and a failed iterative solve does not raise an error either — print the KSP reason of every solve.

**Week 13 starts with a 10-minute quiz on this week's lecture (closed-AI).**
