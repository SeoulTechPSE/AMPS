# AMPS II · Week 06 — Structural Dynamics (Modes, Newmark, Rayleigh Damping) · Linear Buckling (SLEPc)

| File | What | When |
|---|---|---|
| `W06_lecture.html` (`.qmd` source) | Lecture slides — Parts A, B, C, D | in class |
| `W06_lab.ipynb` | In-class lab: the Newmark family on one oscillator (stability, period error), the modes of a cantilever classified by shape, Newmark against the exact semi-discrete solution, Rayleigh damping measured by the logarithmic decrement, the buckling loads of a 3D column against Euler, LLM exercise (natural frequencies of the column) | in class (≈ 80 min) |
| `W06_HW6.ipynb` | HW6 — Newmark stability and Rayleigh damping (theory), the explicit stability limit, a numerically damped scheme, slenderness against Euler–Bernoulli, element order and slenderness in buckling, which geometric stiffness, an LLM-generated Newmark script with defects | due before Week 7 |
| `../dolfinx_0.11_api_reference.md` | DOLFINx 0.11 API reference sheet — **§16l** dynamics (mass matrix, Newmark with PETSc vectors, modes with SLEPc) · **§16m** linear buckling — paste it into your LLM | always |

**Open a notebook in Colab:** click its link on the course page or below — it opens in Colab; then *File → Save a copy in Drive* and work on that copy. (Or upload the `.ipynb` to Google Drive → right-click → *Open with → Google Colaboratory*.)
Run the setup cell first (≈ 3–5 min); re-run it after every runtime restart.

**Where code can run:** `basix`, `ufl`, `dolfinx`, `petsc4py`, `slepc4py` are used only inside `%%fenicsx` cells. Ordinary cells have NumPy and matplotlib only —
and `sdof.py`, which is NumPy only and can be imported in ordinary cells.
The models are three modules written by `%%writefile /tmp/fnx/sdof.py`, `/tmp/fnx/dyn.py` and `/tmp/fnx/buckle.py` cells — run all three first.
Cells that use `dense_modal()` (the exact solution from all modes) run with `-np 1`.

**Before submitting:** *Runtime → Restart session and run all*, fill in the AI-usage log, download the `.ipynb`.
An error inside a `%%fenicsx` cell does **not** stop *Run all*: the traceback is printed and the next cell runs anyway. Scroll through every output before you submit.
(The first cell of HW6 Q4 runs an LLM-generated script that is *meant* to be wrong.) The whole HW runs in under a minute locally (a few minutes on Colab).

**Week 7 starts with a 10-minute quiz on this week's material (closed-AI).**
