# AMPS II · Week 14 — Phase Separation and Parallel Performance: Energy-Stable Cahn–Hilliard, Flory–Huggins, Solvers, Communication

| File | What | When |
|---|---|---|
| `W14_lecture.html` (`.qmd` source) | Lecture slides — Parts A, B, C, D | in class |
| `W14_lab.ipynb` | In-class lab: growth rates and amplification factors in NumPy, each scheme against its own discrete growth rate, three schemes at large time steps, four linear solvers and the ghosts of a partition, one factorisation reused for a whole run, LLM exercise (spinodal decomposition) | in class (≈ 80 min) |
| `W14_HW13.ipynb` | HW13 — linear stability, amplification factors, convex splitting and the common tangent (theory), growth rates in time and space, the equilibrium interface, Flory–Huggins binodal compositions, solvers / memory / ghosts, accuracy per cost, an LLM-generated Cahn–Hilliard script with defects | due before Week 15 |
| `../dolfinx_0.11_api_reference.md` | DOLFINx 0.11 API reference sheet — **§16w** Cahn–Hilliard schemes, a reused factorisation, a matrix-free operator, timing and ghosts — paste it into your LLM | always |

**Open a notebook in Colab:** click its link on the course page or below — it opens in Colab; then *File → Save a copy in Drive* and work on that copy. (Or upload the `.ipynb` to Google Drive → right-click → *Open with → Google Colaboratory*.)
Run the setup cell first (≈ 3–5 min); re-run it after every runtime restart.

**Where code can run:** `basix`, `ufl`, `dolfinx`, `petsc4py` are used only inside `%%fenicsx` cells. Ordinary cells have NumPy and matplotlib only.
The model is one module written by the `%%writefile /tmp/fnx/phase.py` cell — run it first. All cells run with `-np 2`.

**Timings** depend on the machine (Colab has 2 virtual CPUs): compare ratios and iteration counts, not seconds.

**Before submitting:** *Runtime → Restart session and run all*, fill in the AI-usage log, download the `.ipynb`.
An error inside a `%%fenicsx` cell does **not** stop *Run all*: the traceback is printed and the next cell runs anyway. Scroll through every output before you submit.
(The first cell of HW13 Q7 runs an LLM-generated script that is *meant* to be wrong.) The lab runs in about half a minute locally, HW13 in about seven minutes (about 25 on Colab).

**Week 15 starts with a 10-minute quiz on this week's lecture (closed-AI), followed by the project workshop** (progress talks and code reviews, `../week15/README.md`).
