# AMPS II · Week 01 — Course Introduction · Linear Elasticity · Locking · Algebraic Multigrid · Reaction Forces · Hyperelasticity

| File | What | When |
|---|---|---|
| `W01_lecture.html` (`.qmd` source) | Lecture slides — Parts 0, A, B, C, D | in class |
| `W01_lab.ipynb` | In-class lab: the model-module pattern, P1 locking vs P2, CG + GAMG with and without the rigid-body near-nullspace, reaction forces two ways, a neo-Hookean cantilever with load steps, LLM exercise | in class (≈ 80 min) |
| `W01_HW1.ipynb` | HW1 — rigid-body motions and compatibility, what controls locking, multigrid, force and moment equilibrium, an LLM-generated hyperelastic script with defects | due before Week 2 |
| `../dolfinx_0.11_api_reference.md` | DOLFINx 0.11 API reference sheet — **§16 solid mechanics** (elasticity, near-nullspace + GAMG, consistent reactions, neo-Hookean + load steps) — paste it into your LLM | always |

**Open a notebook in Colab:** upload it to Google Drive → right-click → *Open with → Google Colaboratory*.
Run the setup cell first (≈ 3–5 min); re-run it after every runtime restart.

**Where code can run:** `basix`, `ufl`, `dolfinx`, `petsc4py` are used only inside `%%fenicsx` cells. Ordinary cells have NumPy and matplotlib only.
New in Course II: the model is a module written by a `%%writefile /tmp/fnx/beam.py` cell and imported by every `%%fenicsx` cell — run that cell first.

**Before submitting:** *Runtime → Restart session and run all*, fill in the AI-usage log, download the `.ipynb`.
An error inside a `%%fenicsx` cell does **not** stop *Run all*: the traceback is printed and the next cell runs anyway. A nonlinear solve that
fails is not an error at all — print the converged reason. Scroll through every output before you submit.
(The first cell of HW1 Q4 runs an LLM-generated script that is *meant* to be broken.)

**Week 2 starts with a 10-minute quiz on this week's material (closed-AI).**
