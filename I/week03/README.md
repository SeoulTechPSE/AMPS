# AMPS I · Week 03 — From Local Elements to a Global Space: Continuity, DG Spaces, Mesh Topology, the DOF Map

| File | What | When |
|---|---|---|
| `W03_lecture.html` (`.qmd` source) | Lecture slides — Parts A, B, C, D | in class |
| `W03_lab.ipynb` | In-class lab: entities and adjacency, the sign of det J, cell-node maps P1–P3, DOF counts (CG, DG, vector, mixed), owned and ghost DOFs, LLM demo | in class (≈ 80 min) |
| `W03_HW3.ipynb` | HW3 — continuity proofs, your own cell-node map checked against DOLFINx, CG vs DG in parallel, debugging LLM-generated mesh code | due before Week 4 |
| `../dolfinx_0.11_api_reference.md` | DOLFINx 0.11 API reference sheet (new: §2a topology, geometry, owned/ghost) — paste it into your LLM | always |

**Open a notebook in Colab:** click its link on the course page or below — it opens in Colab; then *File → Save a copy in Drive* and work on that copy. (Or upload the `.ipynb` to Google Drive → right-click → *Open with → Google Colaboratory*.)
Run the setup cell first (≈ 3–5 min); re-run it after every runtime restart.

**Where code can run:** `basix`, `ufl` and `dolfinx` exist only inside `%%fenicsx` cells. Ordinary cells have NumPy and matplotlib only.
Some cells this week run on 2 MPI ranks (`%%fenicsx -np 2`): each rank prints its own lines.

**Before submitting:** *Runtime → Restart session and run all*, fill in the AI-usage log, download the `.ipynb`.
An error inside a `%%fenicsx` cell does **not** stop *Run all*: the traceback is printed and the next cell runs anyway.
Scroll through every output before you submit. A cell that printed a traceback, and any plot that uses its results, earns no points.
(HW3 Q4 contains one cell marked `# EXPECTED-FAIL`: its traceback is intended.)

**Week 4 starts with a 10-minute quiz on this week's material (closed-AI).**
