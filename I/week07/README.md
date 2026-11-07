# AMPS I · Week 07 — Weak Derivatives and Sobolev Spaces, Céa's Lemma, Elliptic Regularity, Aubin–Nitsche; Computing Convergence Rates Correctly

| File | What | When |
|---|---|---|
| `W07_lecture.html` (`.qmd` source) | Lecture slides — Parts A, B, C, D | in class |
| `W07_lab.ipynb` | In-class lab: regularity decides the rate (interpolating $r^\alpha$), four ways to compute an error, the L-shaped domain, graded meshes, LLM demo | in class (≈ 80 min) |
| `W07_HW7.ipynb` | HW7 — a rate report: proofs (weak derivatives, Aubin–Nitsche, zero error), predicted vs observed rates on a smooth and two corner problems with uniform and graded meshes, measurement pitfalls, LLM claims checked against your numbers | due before Week 8 |
| `../dolfinx_0.11_api_reference.md` | DOLFINx 0.11 API reference sheet (§11 parallel-safe norms) — paste it into your LLM | always |

**Open a notebook in Colab:** upload it to Google Drive → right-click → *Open with → Google Colaboratory*.
Run the setup cell first (≈ 3–5 min); re-run it after every runtime restart.

**Where code can run:** `basix`, `ufl`, `dolfinx` are used only inside `%%fenicsx` cells. Ordinary cells have NumPy and matplotlib only.

**Before submitting:** *Runtime → Restart session and run all*, fill in the AI-usage log, download the `.ipynb`.
An error inside a `%%fenicsx` cell does **not** stop *Run all*: the traceback is printed and the next cell runs anyway.
Scroll through every output before you submit. A cell that printed a traceback, and any plot that uses its results, earns no points.

**Week 8 starts with the midterm exam (Weeks 1–7, closed-AI, 90 minutes)** — there is no separate quiz on this week's material.
