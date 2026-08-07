"""
docs/simulacao-estocastica.md Section 4.5 (disjoint greedy pairing) and Section 6, INV-19.

INV-19(a) -- mass conservation by the pairing structure, tested with a genuinely conflicting
candidate set (a particle claimed by more than one candidate pair), not a trivial one where
nothing conflicts.

INV-19(b) -- no slot participates in two accepted events in the same pass.

INV-19(c) -- determinism: same input gives the same event list, including when two candidates
share a particle and have bit-identical t* -- the case a missing (i, j) tie-break in the sort key
breaks silently.

INV-19(d) -- f_reject <= 0.05 over the whole run. Section 4.5 calls this a MEASURED quantity, not
a constant; if nbody.collisions does not expose it, the invariant is untestable and that is
reported as a contract/scope gap, not worked around.

Out of scope, deliberately: resolve() and collision OUTCOMES (elastic/merger/fragmentation,
E_int, L_spin -- INV-20..INV-24). The mass-conservation test below applies only a minimal,
mass-conserving TOY per-pair map to the real pair_disjoint's accepted set -- never resolve() or
any physical outcome map -- because Section 4.5's mass-conservation claim for the PAIRING itself
is purely combinatorial (disjoint accepted pairs -> per-pair maps commute and act on disjoint
mass groups), independent of which physical outcome each pair receives.

Section 9.1 gives nbody.collisions.CollisionModel's field names in prose but fixes neither
pair_disjoint's parameter list nor the type of a "candidate"/"event" record -- a real contract
gap (see final report). Closed here NOT by reading src/nbody/collisions.py (never opened), but
by `inspect.signature`/`dataclasses.fields` introspection on the live, imported objects -- the
same "declared signature is fair game, algorithm is not" rule tests/test_random_sphere.py and
_stage2_binding.py already use. As discovered this way (current round):
pair_disjoint(candidates: CollisionCandidates) -> AcceptedPairs, both dataclasses of 1-D tensors
(i, j, t_star, rel_speed, contact_radius_sum[, f_reject]) -- struct-of-arrays over ALL candidates
in one call, not a list of per-pair objects. Every call below still falls back to
_stage2_binding.py's try_calls/extract_pair_records and skips loudly on TypeError, so a future
signature change degrades to a labelled skip rather than a silent wrong pass.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

nbody = pytest.importorskip("nbody")
from nbody import config  # noqa: E402

try:
    import nbody.collisions as collisions
except ImportError:
    collisions = None

from _stage2_binding import ContractGap, extract_pair_records, try_calls  # noqa: E402

from tolerances import EPS_PREC_FP64, TOL_REJECT_MAX  # noqa: E402


# -------------------------------------------------------------------------------------------
# A genuinely conflicting candidate set: particle 1 is claimed by two candidates ((0,1) and
# (1,2)), particle 3 by two more ((2,3) and (3,4)). Sorted by t*, the greedy rule of Section 4.5
# accepts (0,1) and (2,3) and rejects (1,2) and (3,4) -- a nontrivial, hand-checkable result.
# -------------------------------------------------------------------------------------------
_CONFLICT_CANDIDATES = [
    (0.10, 0, 1),
    (0.20, 2, 3),
    (0.30, 1, 2),
    (0.40, 3, 4),
]
_CONFLICT_EXPECTED_ACCEPTED = {(0, 1), (2, 3)}
_CONFLICT_EXPECTED_REJECTED = {(1, 2), (3, 4)}
_CONFLICT_MASSES = np.array([1.0, 2.0, 3.0, 4.0, 5.0]) * config.PARTICLE_MASS


def _reference_pair_disjoint(triples):
    """
    Independent reference implementation of Section 4.5 steps 1-2 ('ordenar os candidatos pela
    chave lexicografica (t*, i, j); percorrer em ordem, aceitando o par se nenhum dos dois ja foi
    reivindicado'), written from the specification text alone -- the oracle this suite compares
    nbody.collisions.pair_disjoint against.
    """
    ordered = sorted(triples, key=lambda tij: (tij[0], tij[1], tij[2]))
    claimed = set()
    accepted, rejected = [], []
    for t_star, i, j in ordered:
        if i in claimed or j in claimed:
            rejected.append((t_star, i, j))
            continue
        accepted.append((t_star, i, j))
        claimed.add(i)
        claimed.add(j)
    return accepted, rejected


def _candidate_representations(triples):
    """
    Plausible element types for a 'candidate', tried in order. The FIRST is the struct-of-arrays
    CollisionCandidates shape discovered via dataclasses.fields (i, j, t_star as 1-D tensors,
    plus rel_speed/contact_radius_sum filled with harmless placeholder values -- pair_disjoint's
    documented job, Section 4.5 steps 1-2, uses only t*/i/j to decide acceptance). The rest are
    generic AoS fallbacks tried only if that fails, in case the type changes later.
    """
    n = len(triples)
    reps = []
    cc_cls = getattr(collisions, "CollisionCandidates", None) if collisions is not None else None
    if cc_cls is not None:
        try:
            reps.append(cc_cls(
                i=torch.tensor([t[1] for t in triples], dtype=torch.int64),
                j=torch.tensor([t[2] for t in triples], dtype=torch.int64),
                t_star=torch.tensor([t[0] for t in triples], dtype=torch.float64),
                rel_speed=torch.ones(n, dtype=torch.float64),
                contact_radius_sum=torch.full((n,), 0.01, dtype=torch.float64),
            ))
        except TypeError:
            pass
    reps.extend([
        [{"t_star": t, "i": i, "j": j} for (t, i, j) in triples],
        [{"t": t, "i": i, "j": j} for (t, i, j) in triples],
        [(t, i, j) for (t, i, j) in triples],
        [(i, j, t) for (t, i, j) in triples],
    ])
    return reps


def _call_pair_disjoint(triples):
    if collisions is None:
        pytest.skip("nbody.collisions is not importable")
    fn = getattr(collisions, "pair_disjoint", None)
    if fn is None:
        pytest.skip("nbody.collisions.pair_disjoint is not defined")

    attempts = []
    for rep in _candidate_representations(triples):
        attempts.append(((rep,), {}))
    for rep in _candidate_representations(triples):
        attempts.append(((), {"candidates": rep}))

    try:
        result, _ = try_calls(fn, attempts)
        return result
    except ContractGap as exc:
        pytest.skip(str(exc))


def _extract_accepted(result):
    # AcceptedPairs is a struct-of-arrays dataclass (i, j, t_star, ... tensors), discovered via
    # dataclasses.fields; try that shape first, then fall back to an AoS reading.
    i, j = getattr(result, "i", None), getattr(result, "j", None)
    if i is not None and j is not None:
        i_np = i.detach().cpu().numpy() if hasattr(i, "detach") else np.asarray(i)
        j_np = j.detach().cpu().numpy() if hasattr(j, "detach") else np.asarray(j)
        t_star = getattr(result, "t_star", None)
        if t_star is not None:
            t_np = t_star.detach().cpu().numpy() if hasattr(t_star, "detach") else np.asarray(t_star)
        else:
            t_np = [None] * len(i_np)
        # returned as (t_star, i, j), matching the (t_star, i, j) convention used everywhere
        # else in this module (_CONFLICT_CANDIDATES, _reference_pair_disjoint, ...).
        return [(None if t is None else float(t), int(a), int(b)) for a, b, t in zip(i_np, j_np, t_np)]

    candidates_to_try = [result]
    if isinstance(result, tuple) and len(result) > 0:
        candidates_to_try.append(result[0])
    for name in ("accepted", "events", "candidates"):
        candidates_to_try.append(getattr(result, name, None))
    for candidate in candidates_to_try:
        if candidate is None:
            continue
        try:
            return extract_pair_records(list(candidate))
        except (ContractGap, TypeError):
            continue
    raise ContractGap(
        f"could not extract accepted (i, j, t_star) records from pair_disjoint's return value "
        f"(type {type(result)!r})"
    )


def _extract_f_reject(result, n_candidates):
    direct_names = ("f_reject", "reject_fraction", "rejected_fraction")
    containers = [result]
    if isinstance(result, tuple):
        containers.extend(result)
    for container in containers:
        for name in direct_names:
            if isinstance(container, dict) and name in container:
                return float(container[name])
            if hasattr(container, name):
                return float(getattr(container, name))
    count_names = ("n_rejected", "num_rejected", "rejected_count")
    for container in containers:
        for name in count_names:
            if hasattr(container, name):
                return float(getattr(container, name)) / n_candidates
    raise ContractGap(
        f"f_reject is not exposed on pair_disjoint's return value (type {type(result)!r}); "
        f"tried direct fields {direct_names} and count fields {count_names}"
    )


# -------------------------------------------------------------------------------------------
# Reference-algorithm self-check: confirms the scenario is genuinely conflicting and that the
# documented greedy rule gives the claimed nontrivial answer, independent of any implementation.
# -------------------------------------------------------------------------------------------
class TestReferenceScenarioIsGenuinelyConflicting:
    def test_particles_1_and_3_each_appear_in_two_candidates(self):
        counts = {}
        for _, i, j in _CONFLICT_CANDIDATES:
            counts[i] = counts.get(i, 0) + 1
            counts[j] = counts.get(j, 0) + 1
        assert counts[1] == 2 and counts[3] == 2, (
            "test setup error: scenario is not genuinely conflicting"
        )

    def test_reference_greedy_algorithm_matches_hand_computed_result(self):
        accepted, rejected = _reference_pair_disjoint(_CONFLICT_CANDIDATES)
        accepted_pairs = {(min(i, j), max(i, j)) for _, i, j in accepted}
        rejected_pairs = {(min(i, j), max(i, j)) for _, i, j in rejected}
        assert accepted_pairs == _CONFLICT_EXPECTED_ACCEPTED
        assert rejected_pairs == _CONFLICT_EXPECTED_REJECTED


# -------------------------------------------------------------------------------------------
# INV-19(a): mass conservation under genuine conflicts
# -------------------------------------------------------------------------------------------
class TestINV19aMassConservationUnderGenuineConflicts:
    """
    Section 6, INV-19(a): |sum m(after)/sum m(before) - 1| <= n_events * eps_prec, per pass
    (TOL-MASS-SUM, Section 7). Section 4.5's argument is purely combinatorial: accepted pairs are
    disjoint, so per-pair maps commute and act on disjoint mass groups, and ANY mass-conserving
    per-pair map preserves the total. Tested here with a minimal toy map (merge: m_i <- m_i+m_j,
    m_j <- 0) applied to the REAL pair_disjoint's accepted set on the genuinely conflicting
    scenario above -- never resolve() or any physical outcome map (INV-20..24, out of scope).

    A pair_disjoint that lets a slot into two accepted pairs would apply the toy merge twice to
    the same mass (once already zeroed), producing an O(1) mass discrepancy -- many orders of
    magnitude above the n_events*eps_prec bound -- so this is a genuinely discriminating test,
    not a tautology.
    """

    def test_toy_merge_conserves_mass_over_a_conflicting_candidate_set(self):
        real_result = _call_pair_disjoint(_CONFLICT_CANDIDATES)
        try:
            real_accepted = _extract_accepted(real_result)
        except ContractGap as exc:
            pytest.skip(str(exc))

        real_pairs = {(min(i, j), max(i, j)) for _, i, j in real_accepted}

        m = _CONFLICT_MASSES.copy()
        total_before = float(m.sum())
        n_events = len(real_pairs)
        for i, j in real_pairs:
            merged = m[i] + m[j]
            m[i] = merged
            m[j] = 0.0
        total_after = float(m.sum())

        bound = max(n_events, 1) * EPS_PREC_FP64
        rel = abs(total_after / total_before - 1.0)
        assert rel <= bound, (
            f"mass not conserved across genuinely conflicting candidates: before={total_before}, "
            f"after={total_after}, rel dev {rel:.3e} > n_events*eps_prec={bound:.3e} "
            f"(TOL-MASS-SUM, Section 7); accepted pairs were {real_pairs}"
        )


# -------------------------------------------------------------------------------------------
# INV-19(b): disjointness
# -------------------------------------------------------------------------------------------
class TestINV19bNoSlotInTwoEvents:
    """Section 6, INV-19(b): no slot participates in two events in the same pass -- tested
    directly on the accepted-event list from the conflicting candidate scenario above."""

    def test_accepted_pairs_do_not_share_a_slot(self):
        real_result = _call_pair_disjoint(_CONFLICT_CANDIDATES)
        try:
            real_accepted = _extract_accepted(real_result)
        except ContractGap as exc:
            pytest.skip(str(exc))
        claimed = []
        for _, i, j in real_accepted:
            claimed.extend([i, j])
        assert len(claimed) == len(set(claimed)), (
            f"a slot appears in more than one accepted event: accepted={real_accepted}"
        )

    def test_every_accepted_pair_was_a_real_candidate(self):
        real_result = _call_pair_disjoint(_CONFLICT_CANDIDATES)
        try:
            real_accepted = _extract_accepted(real_result)
        except ContractGap as exc:
            pytest.skip(str(exc))
        candidate_pairs = {(min(i, j), max(i, j)) for _, i, j in _CONFLICT_CANDIDATES}
        for _, i, j in real_accepted:
            pair = (min(i, j), max(i, j))
            assert pair in candidate_pairs, (
                f"accepted pair {pair} was never offered as a candidate: "
                f"candidates were {candidate_pairs}"
            )

    def test_accepted_set_matches_the_reference_greedy_algorithm(self):
        real_result = _call_pair_disjoint(_CONFLICT_CANDIDATES)
        try:
            real_accepted = _extract_accepted(real_result)
        except ContractGap as exc:
            pytest.skip(str(exc))
        real_pairs = {(min(i, j), max(i, j)) for _, i, j in real_accepted}
        assert real_pairs == _CONFLICT_EXPECTED_ACCEPTED, (
            f"pair_disjoint's accepted set {real_pairs} differs from the reference greedy "
            f"algorithm's {_CONFLICT_EXPECTED_ACCEPTED} on a genuinely conflicting candidate set"
        )


# -------------------------------------------------------------------------------------------
# INV-19(c): determinism, including the (i, j) tie-break
# -------------------------------------------------------------------------------------------
class TestINV19cDeterminism:
    """
    Section 6, INV-19(c): same input gives an identical event list, including when t* is tied
    bit-for-bit between two candidates that share a particle -- exactly where a missing (i, j)
    tie-break makes the result depend on sort stability or presentation order (Section 4.5,
    normative: 'a chave de ordenacao inclui (i, j) precisamente para desempatar t* identicos').
    """

    def test_repeated_calls_on_the_same_input_agree(self):
        result_1 = _call_pair_disjoint(_CONFLICT_CANDIDATES)
        result_2 = _call_pair_disjoint(_CONFLICT_CANDIDATES)
        try:
            accepted_1 = _extract_accepted(result_1)
            accepted_2 = _extract_accepted(result_2)
        except ContractGap as exc:
            pytest.skip(str(exc))
        pairs_1 = {(min(i, j), max(i, j)) for _, i, j in accepted_1}
        pairs_2 = {(min(i, j), max(i, j)) for _, i, j in accepted_2}
        assert pairs_1 == pairs_2, (
            f"two calls on the identical input produced different accepted sets: "
            f"{pairs_1} vs {pairs_2}"
        )

    def test_presentation_order_does_not_affect_the_general_conflict_result(self):
        # Same candidate set as INV-19(a)/(b), fed in a different order: the accepted set must
        # not depend on it (Section 4.5 sorts by (t*, i, j) before the greedy pass, so input
        # order is not supposed to matter even without any tie).
        shuffled = [
            _CONFLICT_CANDIDATES[2], _CONFLICT_CANDIDATES[0],
            _CONFLICT_CANDIDATES[3], _CONFLICT_CANDIDATES[1],
        ]
        result_original = _call_pair_disjoint(_CONFLICT_CANDIDATES)
        result_shuffled = _call_pair_disjoint(shuffled)
        try:
            accepted_original = _extract_accepted(result_original)
            accepted_shuffled = _extract_accepted(result_shuffled)
        except ContractGap as exc:
            pytest.skip(str(exc))
        pairs_original = {(min(i, j), max(i, j)) for _, i, j in accepted_original}
        pairs_shuffled = {(min(i, j), max(i, j)) for _, i, j in accepted_shuffled}
        assert pairs_original == pairs_shuffled == _CONFLICT_EXPECTED_ACCEPTED, (
            f"presentation order changed the accepted set: original={pairs_original}, "
            f"shuffled={pairs_shuffled}, expected={_CONFLICT_EXPECTED_ACCEPTED}"
        )

    def test_tied_t_star_breaks_by_i_j_regardless_of_presentation_order(self):
        # (0, 1, t=0.1) and (0, 2, t=0.1): exactly tied t*, sharing particle 0. Lexicographic
        # (t*, i, j) puts (0, 1) first (j=1 < j=2), so it must win regardless of the order the
        # candidates are handed to pair_disjoint in.
        tied = [(0.1, 0, 1), (0.1, 0, 2)]
        reversed_tied = [(0.1, 0, 2), (0.1, 0, 1)]

        ref_forward, _ = _reference_pair_disjoint(tied)
        ref_reversed, _ = _reference_pair_disjoint(reversed_tied)
        assert {(i, j) for _, i, j in ref_forward} == {(0, 1)}
        assert {(i, j) for _, i, j in ref_reversed} == {(0, 1)}, (
            "test setup error: the reference algorithm itself is not order-independent here"
        )

        result_forward = _call_pair_disjoint(tied)
        result_reversed = _call_pair_disjoint(reversed_tied)
        try:
            accepted_forward = _extract_accepted(result_forward)
            accepted_reversed = _extract_accepted(result_reversed)
        except ContractGap as exc:
            pytest.skip(str(exc))

        pairs_forward = {(min(i, j), max(i, j)) for _, i, j in accepted_forward}
        pairs_reversed = {(min(i, j), max(i, j)) for _, i, j in accepted_reversed}

        assert pairs_forward == {(0, 1)}, (
            f"expected only (0, 1) accepted (tie broken by j=1 < j=2), got {pairs_forward}"
        )
        assert pairs_reversed == {(0, 1)}, (
            f"presentation order changed the result: forward={pairs_forward}, "
            f"reversed-input={pairs_reversed}. Section 4.5: the (i, j) tie-break in the sort key "
            f"is normative precisely to prevent this -- 'sem isso, a ordem de aceitacao depende "
            f"da estabilidade do algoritmo de ordenacao'."
        )


# -------------------------------------------------------------------------------------------
# INV-19(d): f_reject
# -------------------------------------------------------------------------------------------
class TestINV19dRejectFraction:
    """
    Section 6, INV-19(d): f_reject <= 0.05 over the whole run. Section 4.5 is explicit that this
    is a MEASURED quantity, not a constant -- if the implementation does not expose it, the
    invariant is untestable, and that is reported as a gap, not worked around by inventing an
    independent accounting outside the implementation.

    What is checked here, without resolve() (out of scope: INV-20..24 do not exist this round, so
    there is no way to run a real dt-stepped collisional campaign and accumulate f_reject 'sobre
    toda a execucao'): that f_reject, when exposed by pair_disjoint, equals
    n_rejected/n_candidates for a single pass with a KNOWN answer (2/4 = 0.5 for the conflicting
    scenario above -- deliberately far above the <=0.05 acceptance band, so this checks the
    bookkeeping formula itself, not the band). The full-execution bound is explicitly out of
    scope this round; see the final report.
    """

    def test_f_reject_is_exposed_and_matches_the_known_pass_result(self):
        result = _call_pair_disjoint(_CONFLICT_CANDIDATES)
        try:
            f_reject = _extract_f_reject(result, n_candidates=len(_CONFLICT_CANDIDATES))
        except ContractGap as exc:
            pytest.skip(f"INV-19(d) untestable: {exc}")
        expected = len(_CONFLICT_EXPECTED_REJECTED) / len(_CONFLICT_CANDIDATES)
        assert math.isclose(f_reject, expected, rel_tol=1e-9), (
            f"f_reject={f_reject} does not match n_rejected/n_candidates={expected} for the "
            f"known conflicting candidate set"
        )

    def test_reject_bound_constant_is_transcribed_correctly(self):
        # Sanity on the tolerance table transcription itself (Section 7, TOL-REJECT).
        assert TOL_REJECT_MAX == 0.05
