# AMPS II · Week 13 — Porous Media and Conjugate Heat Transfer: Brinkman Layers, the Porous Interface, Forchheimer, Flow on a Submesh

| File | What | When |
|---|---|---|
| `W13_lecture.html` (`.qmd` source) | Lecture slides — Parts A, B, C, D | in class |
| `W13_lab.ipynb` | In-class lab: the exact channel solutions in NumPy, the Brinkman layer (flow rate against velocity), free fluid over a porous layer (interface on and between mesh lines), conjugate heat transfer with the flow on a submesh ($Nu=35/13$, the energy balance, contact conductance), LLM exercise (a porous-layer channel) | in class (≈ 80 min) |
| `W13_HW12.ipynb` | HW12 — Brinkman, the one-domain interface, the developed temperature and the consistent flux (theory), uniform against graded meshes, the Beavers–Joseph coefficient, an Ergun bed, the wall conductivity, contact resistance and the penalty, an LLM-generated heat-transfer script with defects | due before Week 14 |
| `../dolfinx_0.11_api_reference.md` | DOLFINx 0.11 API reference sheet — **§16v** Brinkman in one domain, fields between a submesh and its parent, conjugate heat transfer — paste it into your LLM | always |

**Open a notebook in Colab:** click its link on the course page or below — it opens in Colab; then *File → Save a copy in Drive* and work on that copy. (Or upload the `.ipynb` to Google Drive → right-click → *Open with → Google Colaboratory*.)
Run the setup cell first (≈ 3–5 min); re-run it after every runtime restart.

**Where code can run:** `basix`, `ufl`, `dolfinx`, `petsc4py` are used only inside `%%fenicsx` cells. Ordinary cells have NumPy and matplotlib only.
The model is one module written by the `%%writefile /tmp/fnx/porous.py` cell — run it first. All cells run with `-np 2`.

**Before submitting:** *Runtime → Restart session and run all*, fill in the AI-usage log, download the `.ipynb`.
An error inside a `%%fenicsx` cell does **not** stop *Run all*: the traceback is printed and the next cell runs anyway. Scroll through every output before you submit.
(The first cell of HW12 Q7 runs an LLM-generated script that is *meant* to be wrong.) The lab runs in about a minute locally, HW12 in about two (longer on Colab).

**Week 14 starts with a 10-minute quiz on this week's lecture (closed-AI).**
