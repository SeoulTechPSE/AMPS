# AMPS I · Week 16 — Final Project Presentations

There is **no final exam** in Course I. The session has two parts:

1. **Quiz** on the Week 15 lecture (10 minutes, closed-AI).
2. **Final project presentations** — 12 minutes per student (8 minutes talk + 4 minutes questions, including live questions on your code).

| File | What |
|---|---|
| `W16_final_project.html` (`.qmd` source) | **Read first.** Deliverables, deadlines, report sections, code rules, presentation format, the optional inverse-problem bonus, checklist |
| `report_template/` | `report.qmd` (all required sections + the AI-usage log) and `references.bib` — copy this folder and write your report in it |
| `../projects/index.html` | The project topics A–F and their topic-specific verification |
| `W16_render_report.ipynb` | Colab notebook: renders `report.qmd` to a self-contained `report.html` with Quarto; optional `dolfinx-adjoint` install for the bonus |

**Deadlines**

| What | When |
|---|---|
| Presentation slides (PDF or HTML) | before the Week 16 session |
| `project.ipynb`, `report.qmd`, `report.html`, `figs/` | one week after the Week 16 session |

**Weight:** final report and code 25 %, presentation 10 % (with the proposal's 5 %: the term project is 40 % of the course grade).
Optional inverse-problem section: up to +2 points.

**Before submitting:** *Runtime → Restart session and run all* on `project.ipynb` (every `%%fenicsx` cell, `-np 2`), read every output, then render the report.
