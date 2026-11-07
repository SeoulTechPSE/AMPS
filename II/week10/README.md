# AMPS II · Week 10 — Cohesive Interfaces: Traction–Separation Laws, Submeshes, Snap-back and Path Following

| File | What | When |
|---|---|---|
| `W10_lecture.html` (`.qmd` source) | Lecture slides — Parts A, B, C, D | in class |
| `W10_lab.ipynb` | In-class lab: the exponential law in NumPy, an imperfect interface reproduced exactly (and the '+' trap with `-np 2`), a snap-back traced exactly with opening control (and displacement control jumping over it), the source's plate with two inclusions debonding between perfect bond and holes, LLM exercise (two bonded layers) | in class (≈ 80 min) |
| `W10_HW9.ipynb` | HW9 — the exponential law, snap-back and springs in series (theory), the bilinear law against the exact strip, brittleness against the strip length, springs on every facet against $h$ and $K$, counting the '+' sides, where and when the inclusions debond, an LLM-generated interface script with defects | due before Week 11 |
| `../dolfinx_0.11_api_reference.md` | DOLFINx 0.11 API reference sheet — **§16s** interfaces between submeshes, opening control — paste it into your LLM | always |

**Open a notebook in Colab:** upload it to Google Drive → right-click → *Open with → Google Colaboratory*.
Run the setup cell first (≈ 3–5 min); re-run it after every runtime restart.

**Where code can run:** `basix`, `ufl`, `dolfinx`, `petsc4py` are used only inside `%%fenicsx` cells. Ordinary cells have NumPy and matplotlib only.
The model is one module written by the `%%writefile /tmp/fnx/czm.py` cell — run it first. All cells run with `-np 2`.

**Before submitting:** *Runtime → Restart session and run all*, fill in the AI-usage log, download the `.ipynb`.
An error inside a `%%fenicsx` cell does **not** stop *Run all*: the traceback is printed and the next cell runs anyway. Scroll through every output before you submit.
(The first cell of HW9 Q7 runs an LLM-generated script that is *meant* to be wrong.) The lab runs in about a minute locally, HW9 in about two (a few minutes on Colab).

**Week 11 starts with a 10-minute quiz on this week's lecture (closed-AI).**
