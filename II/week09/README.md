# AMPS II · Week 09 — Shells on Surface Meshes · Periodic Homogenization with dolfinx_mpc

| File | What | When |
|---|---|---|
| `W09_lecture.html` (`.qmd` source) | Lecture slides — Parts A, B, C, D | in class |
| `W09_lab.ipynb` | In-class lab: a plate solved as a shell in 3D against the exact Reissner–Mindlin plate (and without the drilling term), a rotated plate, the Scordelis–Lo roof, an I-beam shell (non-manifold) against beam theory, an exact laminate with `dolfinx_mpc` and two silent traps, a hexagonal array against isotropy and the Hashin–Shtrikman bounds, LLM exercise (a fibre composite) | in class (≈ 80 min) |
| `W09_HW8.ipynb` | HW8 — drilling, the mean-strain theorem and bounding cell BCs (theory), the flat-facet error of a tube, what a clamping marker selects, the Scordelis–Lo roof on a quarter with symmetry conditions, the checkerboard singularity and a graded mesh, cell BCs and window size for a porous medium, an LLM-generated homogenization script with defects | due before Week 10 |
| `../dolfinx_0.11_api_reference.md` | DOLFINx 0.11 API reference sheet — **§16q** shells on surface meshes · **§16r** periodic cells with `dolfinx_mpc` — paste it into your LLM | always |

**Open a notebook in Colab:** upload it to Google Drive → right-click → *Open with → Google Colaboratory*.
Run the setup cell first (≈ 3–5 min); re-run it after every runtime restart. **This week the setup cell also installs `dolfinx_mpc` 0.11** (≈ 1 more minute; two packages, nothing else changes).

**Where code can run:** `basix`, `ufl`, `dolfinx`, `dolfinx_mpc`, `petsc4py` are used only inside `%%fenicsx` cells. Ordinary cells have NumPy and matplotlib only.
The models are two modules written by `%%writefile /tmp/fnx/shell.py` and `/tmp/fnx/homog.py` cells — run both first. All cells run with `-np 2`.

**Before submitting:** *Runtime → Restart session and run all*, fill in the AI-usage log, download the `.ipynb`.
An error inside a `%%fenicsx` cell does **not** stop *Run all*: the traceback is printed and the next cell runs anyway. Scroll through every output before you submit.
(The first cell of HW8 Q7 runs an LLM-generated script that is *meant* to be wrong.) The lab runs in under half a minute locally, HW8 in under a minute (a few minutes on Colab).

**Week 10 starts with a 10-minute quiz on this week's lecture (closed-AI).**
