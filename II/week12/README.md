# AMPS II · Week 12 — Reactive Flow: a Tubular Reactor — Dispersion, Laminar Flow, SUPG, the Graetz Problem and Navier–Stokes Coupling

| File | What | When |
|---|---|---|
| `W12_lecture.html` (`.qmd` source) | Lecture slides — Parts A, B, C, D | in class |
| `W12_lab.ipynb` | In-class lab: the exact limits in NumPy, plug flow with axial dispersion (Danckwerts or Dirichlet inlet), laminar flow without diffusion and the mixing cup, what the SUPG residual must contain, the Graetz problem and $Sh_\infty$, LLM exercise (the Graetz problem) | in class (≈ 80 min) |
| `W12_HW11.ipynb` | HW11 — the axisymmetric weak form, Danckwerts, the mixing cup and SUPG consistency (theory), second-order kinetics, SUPG variants and the outlet, Graetz with axial diffusion, Navier–Stokes in the tube and one-way coupling, Taylor–Aris dispersion, an LLM-generated reactor script with defects | due before Week 13 |
| `../dolfinx_0.11_api_reference.md` | DOLFINx 0.11 API reference sheet — **§16u** axisymmetric advection–diffusion–reaction, Danckwerts, SUPG, the mixing cup — paste it into your LLM | always |

**Open a notebook in Colab:** upload it to Google Drive → right-click → *Open with → Google Colaboratory*.
Run the setup cell first (≈ 3–5 min); re-run it after every runtime restart.

**Where code can run:** `basix`, `ufl`, `dolfinx`, `petsc4py` are used only inside `%%fenicsx` cells. Ordinary cells have NumPy and matplotlib only.
The model is one module written by the `%%writefile /tmp/fnx/reactor.py` cell — run it first. All cells run with `-np 2`.

**Before submitting:** *Runtime → Restart session and run all*, fill in the AI-usage log, download the `.ipynb`.
An error inside a `%%fenicsx` cell does **not** stop *Run all*: the traceback is printed and the next cell runs anyway. Scroll through every output before you submit.
(The first cell of HW11 Q7 runs an LLM-generated script that is *meant* to be wrong.) The lab runs in about a minute locally, HW11 in about two (longer on Colab).

**Week 13 starts with a 10-minute quiz on this week's lecture (closed-AI).**
