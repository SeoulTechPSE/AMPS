# AMPS II · Week 04 — J2 Plasticity: Return Mapping, Quadrature Spaces, the Consistent Tangent, Limit Loads

| File | What | When |
|---|---|---|
| `W04_lecture.html` (`.qmd` source) | Lecture slides — Parts A, B, C, D | in class |
| `W04_lab.ipynb` | In-class lab: a return mapping tested without finite elements (yield surface, finite-difference tangent, uniaxial cycle), the thick tube against Lamé and the first-yield pressure, Newton with the consistent and the elastic tangent, the limit load by pressure and by displacement control, LLM exercise (the plastic zone against Hill's solution) | in class (≈ 80 min) |
| `W04_HW4.ipynb` | HW4 — radial return and the consistent tangent (theory), hardening, autofrettage of a thick tube (residual stress, reverse yielding, cycling), an LLM-generated plasticity script with defects | due before Week 5 |
| `../dolfinx_0.11_api_reference.md` | DOLFINx 0.11 API reference sheet — **§16i** J2 plasticity: quadrature spaces, the return mapping in NumPy, an explicit Newton loop — paste it into your LLM | always |

**Open a notebook in Colab:** click its link on the course page or below — it opens in Colab; then *File → Save a copy in Drive* and work on that copy. (Or upload the `.ipynb` to Google Drive → right-click → *Open with → Google Colaboratory*.)
Run the setup cell first (≈ 3–5 min); re-run it after every runtime restart.

**Where code can run:** `basix`, `ufl`, `dolfinx`, `petsc4py` are used only inside `%%fenicsx` cells. Ordinary cells have NumPy and matplotlib only —
and `j2.py`, the material law, which is NumPy only and can be imported in ordinary cells.
The models are two modules written by `%%writefile /tmp/fnx/j2.py` and `%%writefile /tmp/fnx/plas.py` cells — run both first.

**Before submitting:** *Runtime → Restart session and run all*, fill in the AI-usage log, download the `.ipynb`.
An error inside a `%%fenicsx` cell does **not** stop *Run all*: the traceback is printed and the next cell runs anyway. A Newton step that does not converge is
not an error either: `Tube.step` returns `converged = False`. Scroll through every output before you submit. (The first cell of HW4 Q4 runs an
LLM-generated script that is *meant* to be wrong.) The whole HW runs in about 1.5 minutes locally (several minutes on Colab).

**Week 5 starts with a 10-minute quiz on this week's material (closed-AI).**
