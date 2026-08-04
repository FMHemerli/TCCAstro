"""
Named tolerances for the TCCAstro test suite.

Every value here is copied verbatim from docs/integradores.md (Section 6.3 unless noted
otherwise) or derived from an algebraic argument stated in that document. Nothing here is
fitted to observed test output. If a test needs a number that is not justified by a comment
at its point of use, that is a bug in the test, not license to invent one.

Where the specification does not give a number for a quantity this suite still needs (for
example: a tolerance on ||L|| for INV-5, which the document does not tabulate), the value is
marked EXTRAPOLATED with the reasoning spelled out. These are engineering judgment calls, not
values transcribed from the document, and are flagged as such wherever used.
"""

# --- Section 6.3, TOL-ACCEL: relative error of the acceleration field vs a higher-precision
# reference, in a single evaluation. "p99 measured x margin ~40".
TOL_ACCEL_FP64 = 1e-13
TOL_ACCEL_FP32 = 5e-5

# --- Section 6.3, TOL-ENERGY: relative error of U or E in a single evaluation.
TOL_ENERGY_FP64 = 1e-13
TOL_ENERGY_FP32 = 1e-5

# --- Section 6.3, TOL-MOM: ||sum_i m_i a_i|| <= N * eps_prec * max_i(m_i * ||a_i||).
# This is a formula, not a fixed number; eps_prec is the machine epsilon of the working
# precision (Section 6.3): fp64 = 2.22e-16, fp32 = 1.19e-7. The bound ratio must be <= 1.
EPS_PREC_FP64 = 2.22e-16
EPS_PREC_FP32 = 1.19e-7
TOL_MOM_RATIO_MAX = 1.0

# --- INV-2(c): operational momentum-drift bound in RUN_COLLAPSE.
TOL_MOM_DRIFT_FP64 = 1e-12
TOL_MOM_DRIFT_FP32 = 1e-4

# --- Section 6.3, TOL-IMPL-SHORT: err(r) between two implementations, same precision, t <= 0.5 t_ff.
TOL_IMPL_SHORT_FP64 = 1e-13
TOL_IMPL_SHORT_FP32 = 1e-4

# --- Section 6.3, TOL-IMPL-FULL: err(r) between two implementations, same precision, t = 3 t_ff.
# Deliberately does not exist for fp32 (Section 6.2, item 2): trajectories are decorrelated by
# the Lyapunov time (~0.12 t_ff) long before 3 t_ff in single precision.
TOL_IMPL_FULL_FP64 = 1e-5

# --- Section 6.3, TOL-XPREC: err(r) between fp32 and fp64, t <= 0.5 t_ff.
TOL_XPREC_FP32 = 1e-4

# --- Section 6.3, TOL-GRAD (also INV-1): relative error of a against -grad(U)/m by central
# difference. Two operating points are specified explicitly.
TOL_GRAD_H1E5 = 1e-8   # h = 1e-5 m, the minimum-error point (truncation O(h^2) vs round-off O(eps/h))
TOL_GRAD_H1E4 = 1e-7   # h = 1e-4 m, looser secondary check

# --- INV-3: ||L(t) - L(0)|| / ||L(0)|| in TWOBODY_ECC, 20 periods, 2000 steps/period, fp64.
INV3_L_SYMPLECTIC_EULER_MAX = 1e-12
INV3_L_VELOCITY_VERLET_MAX = 1e-12
INV3_L_RK4_MIN = 1e-13
INV3_L_RK4_MAX = 1e-9
INV3_L_EULER_MIN = 1e-2

# --- INV-3, RUN_COLLAPSE variant, normalized by L_SCALE = M_tot * R_0 * v_char.
INV3_COLLAPSE_L_VERLET_MAX = 1e-12
INV3_COLLAPSE_L_SYMPLECTIC_MAX = 1e-12
INV3_COLLAPSE_L_RK4_MIN = 1e-15
INV3_COLLAPSE_L_RK4_MAX = 1e-10
INV3_COLLAPSE_L_EULER_MIN = 1e-6

# --- INV-4: RUN_COLLAPSE energy criteria, fp64 (Section 7).
INV4_EULER_FINAL_MIN = 0.3
INV4_SYMPLECTIC_PEAK_MAX = 1e-1
INV4_VERLET_PEAK_MAX = 1e-3
INV4_RK4_PEAK_MAX = 1e-6
# "final <= peak/10" (symplectic, verlet), "final >= peak/3, negative" (rk4).
INV4_OSCILLATORY_FINAL_OVER_PEAK_MAX = 1.0 / 10.0
INV4_RK4_FINAL_OVER_PEAK_MIN = 1.0 / 3.0

# --- INV-5: two-body Kepler (eps = 0), velocity_verlet, 1 period.
INV5_RETURN_ERROR_MAX_AT_1000_STEPS = 1e-4       # relative to r_apo
INV5_RETURN_ERROR_RATIO_MIN = 20.0               # ratio between 1000 and 5000 steps/period
INV5_RETURN_ERROR_RATIO_MAX = 30.0
# EXTRAPOLATED: no explicit tolerance is given for ||L|| vs the analytic value in INV-5.
# velocity_verlet conserves L exactly by the same [T] algebraic argument used in INV-3
# (central force, sum_i r_i x a_i = 0), independent of eps; INV-3 measures this at <= 1e-12
# for a much longer run (20 periods) at eps = 0.05. One period at eps = 0 accumulates far
# fewer rounding steps, so re-using 1e-12 here is, if anything, conservative.
INV5_L_REL_TOL = 1e-12

# --- INV-6: two-body circular, softened (eps = 0.05), velocity_verlet, 2000 steps/period, 10 periods.
INV6_SEPARATION_REL_TOL = 1e-6

# --- INV-7: TWOBODY_ECC (eps = 0.05) energy-amplitude table, fp64.
INV7_SYMPLECTIC_RATIO_CENTER = 2.0
INV7_SYMPLECTIC_RATIO_TOL = 0.15
INV7_SYMPLECTIC_FINAL_OVER_AMPLITUDE_MAX = 1.0 / 50.0
INV7_VERLET_RATIO_CENTER = 4.0
INV7_VERLET_RATIO_TOL = 0.3
INV7_VERLET_FINAL_OVER_AMPLITUDE_MAX = 1.0 / 1000.0
INV7_RK4_RATIO_MIN = 12.0                         # order >= 3.6, per document's own relaxation
INV7_RK4_FINAL_OVER_AMPLITUDE_MIN = 1.0 / 3.0
INV7_EULER_FINAL_MIN = 0.2

# --- INV-8: convergence order acceptance windows, fp64, RUN_CONVERGENCE. These are the
# document's measured windows, NOT textbook order. See docs/integradores.md Section 7, INV-8.
INV8_SYMPLECTIC_EULER_PR = (0.95, 1.05)
INV8_SYMPLECTIC_EULER_PV = (1.95, 2.10)
INV8_VELOCITY_VERLET_PR = (1.95, 2.05)
INV8_VELOCITY_VERLET_PV = (1.95, 2.05)
INV8_RK4_PR = (4.5, 5.3)     # NOT (3.5, 4.5): measured 4.95, this is the open question of Sec. 7.
INV8_RK4_PV = (4.5, 5.3)
INV8_EULER_PR = (0.55, 1.20)
INV8_EULER_PV = (0.20, 1.10)

# --- INV-9: t_collapse / t_ff.
INV9_TCOLLAPSE_OVER_TFF_TOL = 0.10
INV9_RHALF_MIN_LOW = 0.30
INV9_RHALF_MIN_HIGH = 0.40
INV9_FP32_TCOLLAPSE_ABS_TOL_TFF_FRACTION = 0.02
INV9_FP32_RHALF_MIN_REL_TOL = 0.02

# --- INV-10: potential energy lower bound, exact by construction, any t, any precision.
# No numeric tolerance beyond floating round-off; enforced as U(t) >= bound * (1 - slack).
INV10_SLACK = 1e-9
