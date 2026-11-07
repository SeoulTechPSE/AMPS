# AMPS II · Week 15 — Project Workshop: Progress Talks, Code Reviews and Verification at Mid-Project

| File | What | When |
|---|---|---|
| `W15_lecture.html` (`.qmd` source) | Workshop slides — the session plan, the progress talk (four slides in five minutes), verification at mid-project (predicted vs observed rates, the wrong yardstick, Richardson and the GCI, `-np 1` = `-np 2`, a run-time budget), the code review, what to cut if you are behind | in class |
| `W15_lab.ipynb` | Verification kit: `projkit.py` (NumPy-only helpers to copy into your project) and a worked example (nonlinear reaction–diffusion, Newton) — predict and measure rates, errors against the interpolant vs the exact expression, a QoI without an exact value, `-np 1` vs `-np 2`, run-time budget; §8 applies it to **your project** | in class while the others present (≈ 45 min), then on your project |
| `talk_template/` | Optional Quarto template for the progress talk (`progress_talk.qmd`, four slides; `quarto render progress_talk.qmd`) | before the session |
| `../projects/index.html` | The project topics A–J — required tasks, topic-specific verification, milestones | always |
| `../dolfinx_0.11_api_reference.md` | DOLFINx 0.11 API reference sheet — **§16x** errors against the exact expression, SNES/KSP reasons, the `projkit` helpers — paste it into your LLM | always |

**The session starts with the 10-minute quiz on Week 14 (closed-AI).** Then the workshop slides, then the progress talks in the announced order:
**10 minutes per student — a 5-minute talk, then a 5-minute code review** with your `project.ipynb` open on the screen (two questions: *show where …* and *what happens if …*).

**Upload by the evening before the session:** your talk slides (PDF or HTML) and your current `project.ipynb` **with outputs** (*Runtime → Restart session and run all* first).

**Weight:** progress talk and code review 5 % of the course grade (term project 40 % = proposal 5 + progress 5 + final report and code 20 + final presentation 10).

**There is no homework this week.** Week 16: final presentations; the final report (Quarto) and code are due one week after the Week 16 session.

**Where code can run:** `projkit.py` is NumPy only — it works in ordinary cells and in `%%fenicsx` cells. `rdx.py` (the example) uses DOLFINx and runs only in `%%fenicsx` cells. The lab runs in about one minute locally.
