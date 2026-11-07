# AMPS II · Week 07 — Beams: Trusses · 3D Timoshenko Frames · Finite Rotations

| File | What | When |
|---|---|---|
| `W07_lecture.html` (`.qmd` source) | Lecture slides — Parts A, B, C, D | in class |
| `W07_lab.ipynb` | In-class lab: a Warren truss against the method of joints (and what one cell too many or one rank too many does), shear locking of a Timoshenko cantilever and its cure, a frame in bending and torsion, the local frame of a portal frame, the geometrically exact rod rolled into a circle and a helix (exact answers), a large-deflection cantilever against the elastica, LLM exercise (a space truss) | in class (≈ 80 min) |
| `W07_HW7.ipynb` | HW7 — the truss singularity, the locked element and rotations about a fixed axis (theory), an indeterminate truss with lack of fit, section orientation in an L-frame, locking against slenderness, convergence of rod elements on the helix, a stiff rod and SNES's divergence test, the source's load case done right, an LLM-generated rod script with defects | due before Week 8 |
| `../dolfinx_0.11_api_reference.md` | DOLFINx 0.11 API reference sheet — **§16n** trusses and Timoshenko frames on a 1D mesh · **§16o** the geometrically exact rod — paste it into your LLM | always |

**Open a notebook in Colab:** upload it to Google Drive → right-click → *Open with → Google Colaboratory*.
Run the setup cell first (≈ 3–5 min); re-run it after every runtime restart.

**Where code can run:** `basix`, `ufl`, `dolfinx`, `petsc4py` are used only inside `%%fenicsx` cells. Ordinary cells have NumPy and matplotlib only.
The models are two modules written by `%%writefile /tmp/fnx/frame.py` and `/tmp/fnx/rod.py` cells — run both first.
**Trusses and frames run with `-np 1`:** where three or more members meet at a joint, DOLFINx 0.11 duplicates the joint when its members land on different MPI ranks
(the lab shows it). The rod is a chain and runs on two ranks.

**Before submitting:** *Runtime → Restart session and run all*, fill in the AI-usage log, download the `.ipynb`.
An error inside a `%%fenicsx` cell does **not** stop *Run all*: the traceback is printed and the next cell runs anyway. Scroll through every output before you submit.
(The first cell of HW7 Q4 runs an LLM-generated script that is *meant* to be wrong.) The whole HW runs in under a minute locally (a few minutes on Colab).

**Week 8 starts with the midterm (first half, Weeks 1–7).** There is no separate quiz; practice questions on this week's material are handed out for the review.
