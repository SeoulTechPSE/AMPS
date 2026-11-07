# AMPS II · Week 02 — Plane Stress · Orthotropic Materials · Stress Concentration · Axisymmetry

| File | What | When |
|---|---|---|
| `W02_lecture.html` (`.qmd` source) | Lecture slides — Parts A, B, C, D | in class |
| `W02_lab.ipynb` | In-class lab: a Gmsh plate with a hole and its area check, plane stress vs plane strain, Kirsch with straight and curved triangles, an orthotropic plate against Lekhnitskii, the Lamé cylinder in $(r,z)$ with convergence rates, LLM exercise (pressurised sphere) | in class (≈ 80 min) |
| `W02_HW2.ipynb` | HW2 — plane stress and orthotropic compliance, plate size and fibre angle (quarter vs full model), a rotating shaft and its axis, an LLM-generated pressure-vessel script with defects | due before Week 3 |
| `../dolfinx_0.11_api_reference.md` | DOLFINx 0.11 API reference sheet — **§16e** plane stress, orthotropy, Gmsh plate · **§16f** axisymmetry — paste it into your LLM | always |

**Open a notebook in Colab:** upload it to Google Drive → right-click → *Open with → Google Colaboratory*.
Run the setup cell first (≈ 3–5 min); re-run it after every runtime restart.

**Where code can run:** `basix`, `ufl`, `dolfinx`, `petsc4py`, `gmsh` are used only inside `%%fenicsx` cells. Ordinary cells have NumPy and matplotlib only.
The models are two modules written by `%%writefile /tmp/fnx/hole.py` and `%%writefile /tmp/fnx/axi.py` cells — run both first.

**Before submitting:** *Runtime → Restart session and run all*, fill in the AI-usage log, download the `.ipynb`.
An error inside a `%%fenicsx` cell does **not** stop *Run all*: the traceback is printed and the next cell runs anyway. Scroll through every
output before you submit. (The first cell of HW2 Q4 runs an LLM-generated script that is *meant* to be wrong.) The whole HW runs in about a minute locally (a few minutes on Colab).

**Week 3 starts with a 10-minute quiz on this week's material (closed-AI).**
