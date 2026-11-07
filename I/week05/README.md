# AMPS I · Week 05 — Hilbert Spaces, Riesz Representation, Lax–Milgram; Assembly, PETSc KSP and the Method of Manufactured Solutions

| File | What | When |
|---|---|---|
| `W05_lecture.html` (`.qmd` source) | Lecture slides — Parts A, B, C, D | in class |
| `W05_lab.ipynb` | In-class lab: Gram matrix and Riesz representer, coercivity constants as eigenvalues, assembly by hand, parallel assembly and PETSc KSP, manufactured solutions and rates, LLM demo | in class (≈ 80 min) |
| `W05_HW5.ipynb` | HW5 — Lax–Milgram proofs, MMS for a variable-coefficient problem with your own assembly + KSP, coercivity constants and dual norms, an LLM-generated MMS study with planted bugs | due before Week 6 |
| `../dolfinx_0.11_api_reference.md` | DOLFINx 0.11 API reference sheet (§7 `LinearProblem`, §8 manual assembly + KSP, §11 parallel-safe norms) — paste it into your LLM | always |

**Open a notebook in Colab:** upload it to Google Drive → right-click → *Open with → Google Colaboratory*.
Run the setup cell first (≈ 3–5 min); re-run it after every runtime restart.

**Where code can run:** `basix`, `ufl`, `dolfinx`, `petsc4py` (and SciPy for the small dense eigenvalue problems) are used only inside `%%fenicsx` cells.
Ordinary cells have NumPy and matplotlib only.

**Before submitting:** *Runtime → Restart session and run all*, fill in the AI-usage log, download the `.ipynb`.
An error inside a `%%fenicsx` cell does **not** stop *Run all*: the traceback is printed and the next cell runs anyway.
Scroll through every output before you submit. A cell that printed a traceback, and any plot that uses its results, earns no points.
(The first cell of HW5 Q4 is *meant* to fail — it is marked `EXPECTED-FAIL`.)

**Week 6 starts with a 10-minute quiz on this week's material (closed-AI).**
