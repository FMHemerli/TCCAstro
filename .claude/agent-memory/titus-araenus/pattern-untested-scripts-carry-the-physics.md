---
name: pattern-untested-scripts-carry-the-physics
description: TCCAstro's test suite guards src/nbody/ while new physics (initial conditions, campaign protocols, closed forms) lands in scripts/ where nothing tests it — so "suite passing, core untouched" reports on the part that did not change.
metadata:
  type: project
---

`tests/` targets `src/nbody/`. Substantive physical content — JPL Keplerian initial conditions,
mass derivation, campaign protocols, derived closed forms — arrives under `scripts/`, which the
suite does not cover. A branch can therefore report "309 passing, `src/nbody/` untouched" while
every line it actually added is unverified by the suite.

**Why:** the architecture deliberately keeps the numerical core small and general, which is a
real virtue; the cost is that demonstrating the core's generality pushes new physics outward.

**How to apply:** treat "suite green, core untouched" as a statement about the unchanged part.
Ask where the new physical claims live and what checks them. Related:
[[pattern-prose-instead-of-containment]].
