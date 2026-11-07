# AMPS II · Week 03 — Generalized Plane Strain · Thermoelasticity (One-Way and Fully Coupled) · Static Condensation

| File | What | When |
|---|---|---|
| `W03_lecture.html` (`.qmd` source) | Lecture slides — Parts A, B, C, D | in class |
| `W03_lab.ipynb` | In-class lab: the thick tube as a cross-section with one extra unknown $e_0$ (plane strain, open and closed ends against Lamé), the bordered matrix vs static condensation, a heated plate with redundant and with statically determinate supports, a Gaussian heat source, the fully coupled strip against its exact solution, LLM exercise (thermal stress in a long tube) | in class (≈ 80 min) |
| `W03_HW3.ipynb` | HW3 — thermal stress in the 2D reductions, generalized plane strain with bending (a bimetal cross-section), the coupling number for real metals, an LLM-generated coupled thermoelastic script with defects | due before Week 4 |
| `../dolfinx_0.11_api_reference.md` | DOLFINx 0.11 API reference sheet — **§16g** generalized plane strain and static condensation · **§16h** thermoelasticity, mixed $(\theta,\mathbf u)$ space — paste it into your LLM | always |

**Open a notebook in Colab:** click its link on the course page or below — it opens in Colab; then *File → Save a copy in Drive* and work on that copy. (Or upload the `.ipynb` to Google Drive → right-click → *Open with → Google Colaboratory*.)
Run the setup cell first (≈ 3–5 min); re-run it after every runtime restart.

**Where code can run:** `basix`, `ufl`, `dolfinx`, `petsc4py` are used only inside `%%fenicsx` cells. Ordinary cells have NumPy and matplotlib only.
The models are two modules written by `%%writefile /tmp/fnx/gps.py` and `%%writefile /tmp/fnx/coupled.py` cells — run both first. HW3 Q2 adds your own
module `bend.py` the same way.

**Before submitting:** *Runtime → Restart session and run all*, fill in the AI-usage log, download the `.ipynb`.
An error inside a `%%fenicsx` cell does **not** stop *Run all*: the traceback is printed and the next cell runs anyway. Scroll through every
output before you submit. (The first cell of HW3 Q4 runs an LLM-generated script that is *meant* to be wrong.) The whole HW runs in about two minutes locally
(several minutes on Colab); the transient strips of Q3 and Q4 take most of it.

**Week 4 starts with a 10-minute quiz on this week's material (closed-AI).**
