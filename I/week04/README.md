# AMPS I · Week 04 — Interpolation Operators, Sobolev Norms, Scaling; Functions, Interpolation and Errors in DOLFINx

| File | What | When |
|---|---|---|
| `W04_lecture.html` (`.qmd` source) | Lecture slides — Parts A, B, C, D | in class |
| `W04_lab.ipynb` | In-class lab: `fem.Function` in parallel, ways to interpolate, polynomial preservation, errors and rates, three classic mistakes, interpolation and I/O, LLM demo | in class (≈ 80 min) |
| `W04_HW4.ipynb` | HW4 — 1D interpolation estimates and scaling, convergence study (smooth and singular functions, graded meshes), harmful triangle shapes, transfer between non-matching meshes with an LLM | due before Week 5 |
| `../dolfinx_0.11_api_reference.md` | DOLFINx 0.11 API reference sheet (updated §4: vectorised callables, `cells0=`, non-matching interpolation) — paste it into your LLM | always |

**Open a notebook in Colab:** click its link on the course page or below — it opens in Colab; then *File → Save a copy in Drive* and work on that copy. (Or upload the `.ipynb` to Google Drive → right-click → *Open with → Google Colaboratory*.)
Run the setup cell first (≈ 3–5 min); re-run it after every runtime restart.

**Where code can run:** `basix`, `ufl` and `dolfinx` exist only inside `%%fenicsx` cells. Ordinary cells have NumPy and matplotlib only.
Output files (`.bp`, `.xdmf`) are written to `/tmp/fnx/`; download them from the Colab file browser to open them in ParaView.

**Before submitting:** *Runtime → Restart session and run all*, fill in the AI-usage log, download the `.ipynb`.
An error inside a `%%fenicsx` cell does **not** stop *Run all*: the traceback is printed and the next cell runs anyway.
Scroll through every output before you submit. A cell that printed a traceback, and any plot that uses its results, earns no points.

**Week 5 starts with a 10-minute quiz on this week's material (closed-AI).**
