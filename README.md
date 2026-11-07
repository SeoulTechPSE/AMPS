# AI-based Materials and Process Simulation I, II — student materials

Published at <https://seoultechpse.github.io/AMPS/> (GitHub Pages, branch `main`, folder `/`).

- `I/` — AMPS I, `II/` — AMPS II: the student releases, copied unchanged from `ProSim/_release/AMPS_*_student_<date>/`.
  Do not edit them here; change the sources in `ProSim/AMPS_I` or `ProSim/AMPS_II` and copy a new release.
- `index.html` — the landing page; `assets/` — logos.

Update:

```bash
cd ~/Documents/notebooks/ProSim
python3 AMPS_I/_tools/make_release.py && python3 AMPS_II/_tools/make_release.py
cd ../AMPS
rsync -a --delete ../ProSim/_release/AMPS_I_student_<date>/  I/
rsync -a --delete ../ProSim/_release/AMPS_II_student_<date>/ II/
git add -A && git commit -m "Update materials" && git push
```
