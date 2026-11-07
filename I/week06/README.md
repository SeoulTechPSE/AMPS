# AMPS I · Week 06 — Boundary Conditions: Mean Estimate, Pure Neumann, Trace Inequality, Partial Dirichlet, Robin; Dirichlet Lifting, Subdomains and Gmsh

| File | What | When |
|---|---|---|
| `W06_lecture.html` (`.qmd` source) | Lecture slides — Parts A, B, C, D | in class |
| `W06_lab.ipynb` | In-class lab: mean-estimate, trace and partial-Dirichlet constants, a Dirichlet condition by hand, MMS with Dirichlet + Neumann + Robin conditions, pure Neumann with a PETSc nullspace, a two-material Gmsh mesh, LLM demo | in class (≈ 80 min) |
| `W06_HW6.ipynb` | HW6 — proofs (pure Neumann, Robin, lifting), a composite wall with a convective boundary, coercivity constants for Robin and for one pinned DOF, an LLM-generated mixed-BC solver with planted bugs | due before Week 7 |
| `../dolfinx_0.11_api_reference.md` | DOLFINx 0.11 API reference sheet (§7 `LinearProblem`, §8 manual assembly + KSP, §11 parallel-safe norms) — paste it into your LLM | always |

**Open a notebook in Colab:** upload it to Google Drive → right-click → *Open with → Google Colaboratory*.
Run the setup cell first (≈ 3–5 min); re-run it after every runtime restart.

**Where code can run:** `basix`, `ufl`, `dolfinx`, `petsc4py`, `gmsh` (and SciPy for the small dense eigenvalue problems) are used only inside `%%fenicsx` cells.
Ordinary cells have NumPy and matplotlib only.

**Before submitting:** *Runtime → Restart session and run all*, fill in the AI-usage log, download the `.ipynb`.
An error inside a `%%fenicsx` cell does **not** stop *Run all*: the traceback is printed and the next cell runs anyway.
Scroll through every output before you submit. A cell that printed a traceback, and any plot that uses its results, earns no points.
(The first cell of HW6 Q4 is *meant* to fail — it is marked `EXPECTED-FAIL`.)

**Week 7 starts with a 10-minute quiz on this week's material (closed-AI).**
