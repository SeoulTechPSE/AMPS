# AMPS II · Week 11 — Natural Convection: Boussinesq Flow, Mixed (u, p, T), Newton with Rayleigh-Number Continuation

| File | What | When |
|---|---|---|
| `W11_lecture.html` (`.qmd` source) | Lecture slides — Parts A, B, C, D | in class |
| `W11_lab.ipynb` | In-class lab: the boundary layer against the mesh, the de Vahl Davis cavity at $Ra=10^3$ with the Nusselt number two ways, continuation up to $Ra=10^6$ (and a direct start failing), Picard against Newton, LLM exercise (the cavity at $Ra=10^5$) | in class (≈ 80 min) |
| `W11_HW10.ipynb` | HW10 — scalings, the consistent flux and the symmetry (theory), a manufactured solution, the Nusselt number on four mesh families, $Nu(Ra)$, Picard's contraction and relaxation, heated from below (two branches, the onset), an LLM-generated benchmark script with defects | due before Week 12 |
| `../dolfinx_0.11_api_reference.md` | DOLFINx 0.11 API reference sheet — **§16t** coupled flow and temperature in one mixed space, continuation, consistent fluxes — paste it into your LLM | always |

**Open a notebook in Colab:** click its link on the course page or below — it opens in Colab; then *File → Save a copy in Drive* and work on that copy. (Or upload the `.ipynb` to Google Drive → right-click → *Open with → Google Colaboratory*.)
Run the setup cell first (≈ 3–5 min); re-run it after every runtime restart.

**Where code can run:** `basix`, `ufl`, `dolfinx`, `petsc4py` are used only inside `%%fenicsx` cells. Ordinary cells have NumPy and matplotlib only.
The model is one module written by the `%%writefile /tmp/fnx/bouss.py` cell — run it first. All cells run with `-np 2`.

**Before submitting:** *Runtime → Restart session and run all*, fill in the AI-usage log, download the `.ipynb`.
An error inside a `%%fenicsx` cell does **not** stop *Run all*: the traceback is printed and the next cell runs anyway. Scroll through every output before you submit.
(The first cell of HW10 Q7 runs an LLM-generated script that is *meant* to be wrong.) The lab runs in about two minutes locally, HW10 in about four (longer on Colab).

**Week 12 starts with a 10-minute quiz on this week's lecture (closed-AI).**
