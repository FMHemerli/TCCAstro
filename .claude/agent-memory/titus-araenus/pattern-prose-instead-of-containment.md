---
name: pattern-prose-instead-of-containment
description: TCCAstro's recurring contradiction — defects found are converted into documented prose in docs/simulacao-estocastica.md while the code that produces them keeps arbitrating; and normative thresholds get crossed by single-realization numbers.
metadata:
  type: project
---

In TCCAstro, findings are habitually resolved by writing them down rather than by containing
them. `docs/simulacao-estocastica.md` grows candid subsections naming a defect ("declared here
because it wasn't", "never analysed anywhere in this document", "the criterion has the right
dimension and the wrong locality") while `src/nbody/` stays untouched and the named defect
remains the arbiter of every downstream number.

Second face of the same pattern, observed 2026-08-09 on `feature/extensoes-sobre-o-nucleo`: the
branch established that per-seed scalars from the collision pipeline are single realizations
(same seed, `max_m/M` = 0.2821 on cpu vs 0.2163 on cuda, ~25 Lyapunov times of amplification)
and in the same commit promoted a **single-seed** `f_reject_total = 6.80%` across the normative
`0.05` attention line, making a reporting obligation binding. The four-seed runner
`scripts/collision_campaign.py` was written in that same branch and carries `f_reject_max` but
no `f_reject_total` column.

**Why:** the project's documentation culture is unusually honest, and that honesty substitutes
for containment — a confessed flaw feels handled. Thresholds inherited from earlier sections are
then evaluated with whatever single number is at hand.

**How to apply:** when reviewing work here, ask which of the new numbers is a realization and
which is a property, and whether a documented defect is still deciding results. Related:
[[pattern-untested-scripts-carry-the-physics]].
