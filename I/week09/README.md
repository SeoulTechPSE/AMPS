# AMPS I · Week 09 — Nonlinear Problems II: Well-posedness · Globalisation · Continuation · PETSc SNES in DOLFINx

| File | What | When |
|---|---|---|
| `W09_lecture.html` (`.qmd` source) | Lecture slides — Parts A (theory), B (SNES), C (continuation, Bratu), D (LLM, HW8) | in class |
| `W09_lab.ipynb` | Lab: `NonlinearProblem` and its diagnostics, a silent failure, line search vs initial guess, continuation in the data, Bratu's fold, a custom Newton solver that passes it, LLM demo | in class (≈ 70 min), finish at home |
| `W09_HW8.ipynb` | **HW8** — Kirchhoff transform and the exact 1D Bratu problem; $-\nabla\cdot(e^u\nabla u)=f$ with SNES (MMS, Kirchhoff cross-check, stopping tolerance); 1D Bratu fold vs the exact $\lambda^*$; LLM debugging | due before Week 10 |
| `../dolfinx_0.11_api_reference.md` | DOLFINx 0.11 API reference sheet — paste it into your LLM | always |

**The session starts with a 10-minute quiz on the Week 8 lecture (closed-AI).** Proposal feedback is handed back in class.

HW8 Q3 reuses `bratu_branch` and `fold` from lab §4 — run the lab first.

**Before submitting:** *Runtime → Restart session and run all*, fill in the AI-usage log, download the `.ipynb`.
An error inside a `%%fenicsx` cell does **not** stop *Run all*, and **a failed SNES solve does not raise an error either** — print the converged
reason after every `solve()` and read every output before you submit.

**Week 10 starts with a 10-minute quiz on this week's lecture (closed-AI).**
