# AMPS II · Week 05 — Viscoelasticity (Prony Series, Creep, Relaxation) · Phase-Field Fracture (AT2, Staggered Solution)

| File | What | When |
|---|---|---|
| `W05_lecture.html` (`.qmd` source) | Lecture slides — Parts A, B, C, D | in class |
| `W05_lab.ipynb` | In-class lab: a Maxwell material point against exact relaxation and creep (two time schemes), a creeping thick tube against the correspondence principle, the exact 1D AT2 crack profile, the homogeneous AT2 strength, a slit specimen broken by phase field with an energy audit, LLM exercise (creep and recovery by superposition) | in class (≈ 80 min) |
| `W05_HW5.ipynb` | HW5 — time integration and creep compliance (theory), the order of the time schemes in 2D, relaxation under displacement control, branches with different Poisson ratios, mesh and length-scale study of the slit specimen, the split in compression, an LLM-generated phase-field script with defects | due before Week 6 |
| `../dolfinx_0.11_api_reference.md` | DOLFINx 0.11 API reference sheet — **§16j** viscoelasticity (Prony, one linear solve per step) · **§16k** phase field (history field, staggered loop, slit mesh) — paste it into your LLM | always |

**Open a notebook in Colab:** upload it to Google Drive → right-click → *Open with → Google Colaboratory*.
Run the setup cell first (≈ 3–5 min); re-run it after every runtime restart.

**Where code can run:** `basix`, `ufl`, `dolfinx`, `petsc4py` are used only inside `%%fenicsx` cells. Ordinary cells have NumPy and matplotlib only —
and `visco.py`, which is NumPy only and can be imported in ordinary cells.
The models are three modules written by `%%writefile /tmp/fnx/visco.py`, `/tmp/fnx/vtube.py` and `/tmp/fnx/pfield.py` cells — run all three first.

**Before submitting:** *Runtime → Restart session and run all*, fill in the AI-usage log, download the `.ipynb`.
An error inside a `%%fenicsx` cell does **not** stop *Run all*: the traceback is printed and the next cell runs anyway. A staggered step that does not converge is
not an error either: `PhaseField.step` returns `converged = False`. Scroll through every output before you submit. (The first cell of HW5 Q4 runs an
LLM-generated script that is *meant* to be wrong.) The whole HW runs in about 1.5 minutes locally (several minutes on Colab); the phase-field runs of Q3 take most of it.

**Week 6 starts with a 10-minute quiz on this week's material (closed-AI).**
