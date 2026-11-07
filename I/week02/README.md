# AMPS I · Week 02 — Ciarlet's Finite Element, Unisolvence, Basix Elements, Lagrange Variants

| File | What | When |
|---|---|---|
| `W02_lecture.html` (`.qmd` source) | Lecture slides — Parts A, B, C, D | in class |
| `W02_lab.ipynb` | In-class lab: Vandermonde by hand, Basix elements and tabulation, conditioning, Lagrange variants, LLM demo | in class (≈ 80 min) |
| `W02_HW2.ipynb` | HW2 — unisolvence proofs, your own nodal basis vs Basix, a non-unisolvent node set, cubic Hermite with an LLM | due before Week 3 |
| `../dolfinx_0.11_api_reference.md` | DOLFINx 0.11 API reference sheet — paste it into your LLM | always |

**Open a notebook in Colab:** upload it to Google Drive → right-click → *Open with → Google Colaboratory*.
Run the setup cell first (≈ 3–5 min); re-run it after every runtime restart.

**Where code can run:** `basix`, `ufl` and `dolfinx` exist only inside `%%fenicsx` cells. Ordinary cells have NumPy and matplotlib only.

**Before submitting:** *Runtime → Restart session and run all*, fill in the AI-usage log, download the `.ipynb`.
An error inside a `%%fenicsx` cell does **not** stop *Run all*: the traceback is printed and the next cell runs anyway.
Scroll through every output before you submit. A cell that printed a traceback, and any plot that uses its results, earns no points.

**Week 3 starts with a 10-minute quiz on this week's material (closed-AI).**
