# AMPS II · Week 08 — Midterm · Reissner–Mindlin Plates: Shear Locking, Reduced Integration, the Right Reference · Project Proposal

| File | What | When |
|---|---|---|
| `W08_lecture.html` (`.qmd` source) | Lecture slides (second half of the session) — Parts A, B, C, D | in class |
| `W08_lab.ipynb` | Lab: the source's locking table on quadrilaterals, reduced integration on triangles (no cure), the thickness effect against exact Reissner–Mindlin solutions, convergence rates, LLM exercise (a clamped circular plate) | in class (≈ 60 min), finish at home |
| `../projects/index.html` | **Project topics A–J** from the research of our department's laboratories (or propose your own) — read before writing the proposal | before Week 9 |
| `W08_project_proposal.ipynb` | **Term project proposal** — a topic A–J or your own problem: problem, model and weak form, numerical plan, verification plan, smoke test (`-np 2`), LLM plan, risks, timeline, AI-usage log | due before Week 9 |
| `../dolfinx_0.11_api_reference.md` | DOLFINx 0.11 API reference sheet — **§16p** Reissner–Mindlin plates — paste it into your LLM | always |

**The session starts with the midterm exam** (Weeks 1–7, 90 minutes, concepts and code reading, closed-AI, closed-book, no computer).

There is no weekly homework this week; HW8 is set in Week 9.

**Where code can run:** the model module is written by the `%%writefile /tmp/fnx/plate.py` cell — run it first. All plate cells run with `-np 2`.

**Before submitting the proposal:** *Runtime → Restart session and run all* (the smoke-test cell must run with `-np 2`), fill in the AI-usage log,
download the `.ipynb`. An error inside a `%%fenicsx` cell does **not** stop *Run all* — read every output before you submit.

**Week 9 starts with a 10-minute quiz on this week's lecture (closed-AI).**
