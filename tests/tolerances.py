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

# =============================================================================================
# docs/simulacao-estocastica.md -- Sections 2, 3, 5.3, 6, 7, 8 (INV-11 .. INV-17, INV-29).
# Every constant below is transcribed verbatim from the document; every acceptance window is
# either transcribed from Section 7's table or derived from an explicit statistical argument
# given in Section 6 (chi-square / KS standard-error reasoning), never fitted to observed
# output, per this file's own header rule.
# =============================================================================================

# --- Section 8: mass-spectrum constants. alpha = 2.35 (Salpeter, default), R = m_max/m_min =
# 1000 (fixed). g(alpha, R) and the derived m_min/m_max/m_big are [T] closed-form values
# (Section 2.4-2.5) computed there in 40-digit arithmetic.
MASS_ALPHA = 2.35
MASS_RATIO = 1000.0
MASS_G_FACTOR = 3.5136877959          # g(alpha, R) = <m>/m_min
MASS_MIN = 2.8460126741e8             # kg = PARTICLE_MASS / MASS_G_FACTOR
MASS_MAX = 2.8460126741e11            # kg = MASS_RATIO * MASS_MIN
MASS_BIG = 2.7509063196e10            # kg = F^-1(1 - 2/N), N = 1000
MASS_TAIL_PROB = 2.0e-3               # p = P(m > m_big) = 2/N
MASS_K_WEIGHTS = (0.37476530, 0.37514082, 0.25009388)   # Binom(N,p) renormalized on {1,2,3}
MASS_K_ACCEPT_PROB = 0.72223972       # P(K in {1,2,3}), unconditioned
MASS_COND_MEAN = 9.9239075e11         # kg, E[M_tot | K in {1,2,3}], N=1000
MASS_COND_CV = 9.2402e-2              # CV(M_tot | K in {1,2,3})
MASS_TFF_CV = 4.6201e-2               # = MASS_COND_CV / 2 (t_ff ~ M^-1/2)
MASS_SEED_DEFAULT = 20190223          # Section 8; separate RNG stream from position SEED

# --- Section 8: velocity constants, Q_DEFAULT/F_CUT_DEFAULT and the non-degeneracy coefficients
# of Section 3.4 ([T], x_c = v_cut/sigma >= 2 <=> Q <= Q_USABLE_COEFF * f_cut^2).
Q_DEFAULT = 0.25
F_CUT_DEFAULT = 0.5
Q_SUP_COEFF = 2.009056                # Q_sup(f) = Q_SUP_COEFF * f^2 (degenerate/uniform-ball limit)
Q_USABLE_COEFF = 1.532168             # Q_usable(f) = Q_USABLE_COEFF * f^2 (x_c >= X_C_MIN)
X_C_MIN = 2.0
VEL_SEED_DEFAULT = 20190224           # third, separate RNG stream

# --- Section 7 table: TOL-VIRIAL (INV-16a) and TOL-HALF-MASS (INV-29), transcribed verbatim.
TOL_VIRIAL_FP64 = 1e-12
TOL_VIRIAL_FP32 = 1e-5
TOL_HALF_MASS = 1e-14

# --- INV-11: KS test on the two sectors (body, tail) of the pooled mass sample, M=200
# realizations of N=1000 = 2e5 masses. "p_KS >= 0.01 em cada setor... detecta desvios de forma
# >~0.5% na CDF" (Section 6, INV-11).
INV11_KS_PVALUE_MIN = 0.01
# Complementary deterministic check: sample mean within 5% of the CONDITIONED theoretical mean
# (Section 2.8's 9.9239075e11/1000, NOT the unconditioned PARTICLE_MASS). Bound derived in the
# document as 3*CV/sqrt(200) = 3*0.0924/14.14 = 0.0196, rounded up with a stated 2.5x margin.
INV11_MEAN_REL_TOL = 0.05

# --- INV-12(a): structural, zero tolerance -- every one of 1000 realizations must have
# 1 <= #{m_i > m_big} <= 3.
INV12A_N_REALIZATIONS = 1000
# --- INV-12(b): chi-square (2 dof) on the k-distribution over those 1000 realizations, and the
# equivalent per-k 3-sigma binomial band explicitly given in Section 6, INV-12:
# +-0.046, +-0.046, +-0.041 for k=1,2,3 (3 * sqrt(P_k(1-P_k)/1000)).
INV12B_CHI2_PVALUE_MIN = 0.01
INV12B_PER_K_3SIGMA = (0.046, 0.046, 0.041)

# --- INV-13: chi-square (9 dof) on the slot index of the max-mass particle over 4000
# realizations, binned into 10 bins of width 100 over {0..999}. "detecta um desvio de >~10% na
# frequencia de uma caixa" (Section 6, INV-13).
INV13_N_REALIZATIONS = 4000
INV13_N_BINS = 10
INV13_CHI2_PVALUE_MIN = 0.01

# --- INV-14: closed-form m_min reproduces <m> = PARTICLE_MASS under independent quadrature.
# "|<m>_quad / PARTICLE_MASS - 1| <= 1e-10 em fp64 ... margem 50x" over truncation error
# dominated by round-off at >=1e4 Simpson nodes (Section 6, INV-14).
INV14_MEAN_REL_TOL = 1e-10
INV14_SIMPSON_NODES = 10001           # 10000 intervals (even, as composite Simpson requires)
INV14_ALPHA_SWEEP = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)   # both degenerate branches included

# --- INV-15(a): t_ff(2M)/t_ff(M) = 2^-1/2 exactly (mass doubled by an exact float factor).
INV15A_REL_TOL = 1e-14
# --- INV-15(b): ensemble CV of t_ff over 200 realizations must fall in a +-35% band around the
# predicted 4.62% (Section 2.10), which the document derives as ~5 standard errors of the
# standard-deviation estimator itself at n=200 (sd of sd ~= sd/sqrt(2n) ~= 5%).
INV15B_TFF_CV_LOW = 0.030
INV15B_TFF_CV_HIGH = 0.065
INV15B_N_REALIZATIONS = 200

# --- INV-16(a): realized virial ratio, exact by construction of lambda (Section 3.3).
INV16A_REL_TOL = TOL_VIRIAL_FP64
# --- INV-16(b): momentum residual bound, same TOL-MOM structure as Section 5.5/6.3 of
# integradores.md: ||P|| / (M_real * v_rms) <= N * eps_prec.
# --- INV-16(d): isotropy of the mass-weighted velocity dyadic tensor, aggregated over 50
# realizations of N=1000: eigenvalues in [0.30, 0.37] (Section 6, INV-16d operational form).
INV16D_EIGENVALUE_LOW = 0.30
INV16D_EIGENVALUE_HIGH = 0.37
INV16D_N_REALIZATIONS = 50

# --- INV-25-adjacent value used by INV-16(e)'s error-message check: with the vetoed combination
# (Q, f_cut) = (0.5, 0.5), Q_usable = Q_USABLE_COEFF * 0.5^2 = 0.383042 (Section 3.4 table).
INV16E_VETOED_Q = 0.5
INV16E_VETOED_F_CUT = 0.5
INV16E_Q_USABLE_AT_VETO = Q_USABLE_COEFF * INV16E_VETOED_F_CUT ** 2

# --- INV-17: Q=0 reproduces cold_sphere bit for bit. Binary, no tolerance.

# --- INV-29(a): equal masses, even N. Section 5.3 property 1 claims the mass-median INDEX rule
# "reduz-se exatamente a formula atual" -- k* = N/2 (1-based), i.e. index N//2 - 1. That claim is
# about index selection and is tested EXACTLY, on a configuration whose distances are exact binary
# fractions, by test_equal_mass_index_rule_is_exact.
#
# It is NOT a claim that a torch implementation reproduces an independent numpy reference bit for
# bit. Both compute c = sum(m*r)/sum(m) and d = |r - c| over N terms, but with different reduction
# orders, so they may differ. EXTRAPOLATED bound: a length-N floating-point sum carries relative
# error <= N * eps_fp64 = 1000 * 2.22e-16 = 2.2e-13 at the largest N tested; the norm adds a
# 3-term sum and a correctly-rounded sqrt, both negligible against that. 1e-12 is that bound with
# ~5x margin. It is not fitted to observed output: the observed difference is 2 ULP (~2.5e-16
# relative), three orders of magnitude tighter.
#
# The published values this protects (IC_R_HALF_0 = 4.881251, COLLAPSE_R_HALF_MIN = 0.3472) are
# quoted to 7 and 4 significant figures; test_observables.py guards those directly on the real
# cold_sphere IC.
INV29A_EQUAL_MASS_REL_TOL = 1e-12

# --- INV-29(b): half_mass_radius against an independent reference implementation written in
# the test (sort, accumulate, search), same structure as TOL-HALF-MASS.
INV29B_REL_TOL = TOL_HALF_MASS
INV29B_MASS_RATIO_FOR_SEPARATION = 223   # Section 5.3's worked example: differs by ~10%

# --- Section 8: derived values for the EQUAL-MASS case (M_real = TOTAL_MASS = 1e12,
# |U_0| = 6.4260397026e12 J), used as independent cross-checks for INV-16(c)'s lambda-scaled
# velocity cap. All [T] in the document.
V_ESC_SPHERE = 4.6386556804          # m/s = sqrt(2 G TOTAL_MASS / SPHERE_RADIUS)
VEL_CUT_DEFAULT = 2.319328           # m/s = F_CUT_DEFAULT * V_ESC_SPHERE
VEL_SIGMA_DEFAULT = 0.7604389        # m/s, root of sigma^2 h(v_cut/sigma) = <v^2>_target
VEL_XC_DEFAULT = 3.0500              # = VEL_CUT_DEFAULT / VEL_SIGMA_DEFAULT
VEL_RMS_DEFAULT = 1.267482           # m/s
VEL_REJECT_FRACTION = 2.5529e-2
VEL_LAMBDA_SD = 1.291e-2             # sd(lambda) = 0.5 * sqrt(2/(3N)), N=1000

# =============================================================================================
# docs/simulacao-estocastica.md -- Sections 4.1, 4.3-4.5, 6, 7, 8 (INV-18, INV-19 only).
# Collision RESOLUTION constants (MAP_B, MAP_W, FRAG_*, COH_VELOCITY_FACTOR, ...) are
# deliberately absent: INV-20..INV-24 (elastic/merger/fragmentation outcomes, E_int, L_spin) are
# out of scope for this round -- see the final report. Only what INV-18 (swept detection,
# tunneling) and INV-19 (disjoint pairing) need is transcribed here.
# =============================================================================================

# --- Section 8: contact-radius and detection-step constants.
CHI_DEFAULT = 0.1                        # [A] R_ref = CHI_DEFAULT * SOFTENING (Section 4.1)
R_REF_DEFAULT = 5.0e-3                   # m = CHI_DEFAULT * SOFTENING(=5e-2)
DT_COLLISION = 1.25e-4                   # s = DT_COLLAPSE / 4 (Section 4.4, C_coll < 1 criterion)
COLLISION_SEED_DEFAULT = 20190225        # fourth RNG stream, separate from position/mass/velocity

# --- Section 4.4: the two numbers the DT_COLLISION derivation itself rests on. U_MAX is
# explicitly marked [A] in the document ("estimado por 2 sqrt(2 G M_core / r_half,min)"), not
# [T]; transcribed verbatim (not re-derived) because Section 4.4's own worked C_coll = 0.57
# figure depends on exactly this number, and the test reproducing that derivation must use the
# same input the document used, not a different "more careful" one.
COLLISION_U_MAX_ASSUMED = 30.0                # [A] m/s, extreme relative speed estimate, Sec 4.4
COLLISION_R_SUM_LIGHTEST_PAIR = 6.58e-3       # m, R_i+R_j for the m_min-m_min pair, chi=0.1
COLLISION_C_COLL_AT_DT_COLLISION = 0.57       # [T] "C_coll,max = 30*1.25e-4/6.58e-3 = 0.57"
COLLISION_C_COLL_LIGHTEST_AT_2_5E4 = 1.14     # [T] Sec 4.4: "C_coll = 1.14 mesmo com dt=2.5e-4"

# --- Section 7 table: TOL-COURANT, TOL-REJECT, transcribed verbatim.
TOL_COURANT_MAX = 1.0                    # INV-18(b): max C_coll <= 1, a validity condition,
                                          # not a numerical-error tolerance (Section 4.4)
TOL_REJECT_MAX = 0.05                    # INV-19(d): f_reject over the whole run [A] (Section 4.5)

# --- Section 7: TOL-MASS-SUM is a FORMULA, n_events * eps_prec, not a fixed number (one
# rounding per event, Sections 4.5/4.9). eps_prec is EPS_PREC_FP64 above (already in this file,
# from docs/integradores.md Section 6.3).

# --- INV-18(a): t* closed form vs brute-force grid-search minimization (Section 6, INV-18a).
# "Comparar com minimizacao por varredura de 1e5 pontos" fixes the grid resolution, which sets
# both stated bounds: "|t*_formula - t*_grade| <= h/1e5" and, because the parabola is exactly
# quadratic (no higher-order terms) so the separation error right at the minimum is second
# order in the t-error, "| |sep|_formula/|sep|_grade - 1 | <= 1e-12".
INV18A_GRID_POINTS = 100_001             # 1e5 intervals -> spacing h/1e5, as the document states
INV18A_T_STAR_ABS_TOL_OVER_H = 1.0e-5    # |t*_formula - t*_grid| <= h/1e5
INV18A_SEP_REL_TOL = 1e-12               # round-off; parabola is exactly flat at its minimum
INV18A_N_RANDOM_CONFIGS = 200            # "200 configuracoes aleatorias" (Section 6, INV-18a)
