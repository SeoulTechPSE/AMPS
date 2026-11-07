# AMPS I · Week 14 — Navier–Stokes: Projection Methods (Chorin, IPCS) · CFL · Channel Flow · DFG 2D-3

| File | What | When |
|---|---|---|
| `W14_lecture.html` (`.qmd` source) | Lecture slides — Parts A (theory), B (Navier–Stokes in DOLFINx 0.11), C (measurements), D (LLM, HW13) | in class |
| `W14_lab.ipynb` | Lab: channel flow and the steady state of IPCS vs Chorin, temporal order on the Taylor–Green vortex, the DFG 2D-3 cylinder on a coarse mesh (drag, lift, CFL), LLM demo | in class (≈ 80 min), finish at home |
| `W14_HW13.ipynb` | **HW13** — projection methods and CFL (theory); DFG 2D-3 on two meshes; time step: stability vs accuracy; LLM debugging of a DFG script | due before Week 15 |
| `../dolfinx_0.11_api_reference.md` | DOLFINx 0.11 API reference sheet — paste it into your LLM | always |

**The session starts with a 10-minute quiz on the Week 13 lecture (closed-AI).**

**Run times** on Colab (`-np 2`; the DFG runs measured, the rest estimated): lab §3 (DFG coarse) about 1.5 minutes; HW Q2 about 8.5 minutes (coarse 1.5 + medium 7),
Q3 about 6 minutes, Q4 roughly 5 minutes — about **20 minutes** for the homework. Start early, save results to `/tmp/fnx/` so that plots do not re-run simulations,
and do not save the notebook before the last cell has finished.

**Before submitting:** *Runtime → Restart session and run all*, fill in the AI-usage log, download the `.ipynb`.
An error inside a `%%fenicsx` cell does **not** stop *Run all*. An exception on one MPI rank only can leave the other rank waiting forever — if a cell hangs, interrupt it
and look for an error printed before the hang.

**Week 15 starts with a 10-minute quiz on this week's lecture (closed-AI).**
