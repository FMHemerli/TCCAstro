from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from . import config
from ._pairwise import default_tile_size, iter_tiles
from .state import State


@dataclass(frozen=True)
class CollisionModel:
    # r_ref, m_bar keep their original relative order and stay positional-or-keyword, so the
    # pre-existing construction convention CollisionModel(r_ref, m_bar) is preserved bit for
    # bit. v_coh carries no sentinel/default (Sec. 9.1.1: "em m/s, sem sentinela; o chamador o
    # computa") and is therefore kw_only=True instead of being placed first: a plain
    # (non-kw_only) mandatory field would have to precede the two defaulted ones, which would
    # silently rebind existing positional calls (CollisionModel(x, y) would bind x -> v_coh,
    # y -> r_ref instead of raising). kw_only fields are exempt from the dataclass
    # no-default-before-default ordering rule and from positional binding entirely, so this
    # is the one placement that adds the mandatory field without touching the two that
    # already existed. seed is new too and kw_only for the same reason, though it is
    # defaulted so the choice is lower-stakes.
    r_ref: float = config.R_REF_DEFAULT
    m_bar: float = config.PARTICLE_MASS
    v_coh: float = field(kw_only=True)
    seed: int = field(default=config.COLLISION_SEED, kw_only=True)


@dataclass(frozen=True)
class CollisionCandidates:
    i: torch.Tensor
    j: torch.Tensor
    t_star: torch.Tensor
    rel_speed: torch.Tensor
    contact_radius_sum: torch.Tensor

    @property
    def n(self) -> int:
        return int(self.i.shape[0])


@dataclass(frozen=True)
class AcceptedPairs:
    i: torch.Tensor
    j: torch.Tensor
    t_star: torch.Tensor
    rel_speed: torch.Tensor
    contact_radius_sum: torch.Tensor
    f_reject: float

    @property
    def n(self) -> int:
        return int(self.i.shape[0])


def contact_radii(m: torch.Tensor, model: CollisionModel) -> torch.Tensor:
    ratio = m / model.m_bar
    return model.r_ref * ratio.pow(1.0 / 3.0)


def _empty_candidates(dtype: torch.dtype, device: torch.device) -> CollisionCandidates:
    empty_f = torch.empty(0, dtype=dtype, device=device)
    empty_i = torch.empty(0, dtype=torch.int64, device=device)
    return CollisionCandidates(
        i=empty_i,
        j=empty_i.clone(),
        t_star=empty_f,
        rel_speed=empty_f.clone(),
        contact_radius_sum=empty_f.clone(),
    )


def detect(state, dt: float, model: CollisionModel) -> CollisionCandidates:
    """Swept collision detection over [0, dt], reusing the tiling contract of _pairwise so
    the O(N^2) pass never allocates a full N x N intermediate at once. state.v is taken as
    the relative velocity to sweep with (Sec. 4.3: exact for the velocity-Verlet drift when
    v is the mid-step velocity v^(n+1/2); the caller is responsible for passing that state).
    """
    r = state.r
    v = state.v
    m = state.m
    n = r.shape[0]
    device = r.device
    dtype = r.dtype

    if n < 2:
        return _empty_candidates(dtype, device)

    live = m > 0
    radii = contact_radii(m, model)
    idx = torch.arange(n, device=device)

    tile_size = default_tile_size(n, dtype)

    i_chunks: list[torch.Tensor] = []
    j_chunks: list[torch.Tensor] = []
    t_chunks: list[torch.Tensor] = []
    speed_chunks: list[torch.Tensor] = []
    r_sum_chunks: list[torch.Tensor] = []

    for start, end in iter_tiles(n, tile_size):
        ri = r[start:end]
        vi = v[start:end]
        radii_i = radii[start:end]
        live_i = live[start:end]
        row_global = idx[start:end]

        dr = r.unsqueeze(0) - ri.unsqueeze(1)  # (tile, n, 3), dr = r_j - r_i
        dv = v.unsqueeze(0) - vi.unsqueeze(1)  # (tile, n, 3), dv = v_j - v_i

        upper = row_global.unsqueeze(1) < idx.unsqueeze(0)  # keep j > i only, once per pair
        pair_live = live_i.unsqueeze(1) & live.unsqueeze(0)

        dr_dot_dv = (dr * dv).sum(dim=-1)
        dv_sq = (dv * dv).sum(dim=-1)

        # Sec. 4.3, normative guards: (i) approaching only, dr.dv < 0 at step start; a pair
        # with dv == 0 has dr.dv == 0 exactly and is excluded by this test before the
        # degenerate division below is ever selected. (ii) |dv|^2 == 0 -> t* = 0.
        approaching = dr_dot_dv < 0.0
        candidate_mask = upper & pair_live & approaching

        # No where-guard on the division: dv_sq == 0.0 iff dv == (0, 0, 0) exactly (a sum of
        # three non-negative squares vanishes only if every term does), and IEEE754 squaring
        # of a signed zero is always +0. At such a position dr_dot_dv is itself a sum of
        # signed zeros (dr_c * 0.0), so it compares equal to, never less than, 0.0 -- meaning
        # `approaching` is already False there and candidate_mask excludes it regardless of
        # this division's result. The unguarded 0/0 = nan at that position propagates through
        # clamp, through sep = dr + t_star*dv (nan * (+0) = nan), and into sep_sq < r_sum^2
        # (any comparison against nan is False), so in_contact is also False there -- the
        # position was never going to be selected by torch.nonzero(collide) either way. At
        # every position where dv_sq != 0.0 (every real candidate) this is the same division
        # the guarded form used to perform, bit for bit -- the guard only ever changed the
        # value at positions that are provably never selected.
        raw_t_star = -dr_dot_dv / dv_sq
        t_star = raw_t_star.clamp(min=0.0, max=dt)

        sep = dr + t_star.unsqueeze(-1) * dv
        sep_sq = (sep * sep).sum(dim=-1)

        r_sum = radii_i.unsqueeze(1) + radii.unsqueeze(0)
        in_contact = sep_sq < (r_sum * r_sum)

        collide = candidate_mask & in_contact
        rows, cols = torch.nonzero(collide, as_tuple=True)
        if rows.numel() == 0:
            continue

        i_chunks.append(row_global[rows])
        j_chunks.append(cols)
        t_chunks.append(t_star[rows, cols])
        speed_chunks.append(dv_sq[rows, cols].sqrt())
        r_sum_chunks.append(r_sum[rows, cols])

    if not i_chunks:
        return _empty_candidates(dtype, device)

    return CollisionCandidates(
        i=torch.cat(i_chunks),
        j=torch.cat(j_chunks),
        t_star=torch.cat(t_chunks),
        rel_speed=torch.cat(speed_chunks),
        contact_radius_sum=torch.cat(r_sum_chunks),
    )


def pair_disjoint(candidates: CollisionCandidates) -> AcceptedPairs:
    """Sec. 4.5: sort candidates by the lexicographic key (t*, i, j) -- the (i, j) tie-break
    is normative, it is what makes the greedy acceptance reproducible across devices when
    two candidates share a bit-identical t* (e.g. a symmetric configuration). Greedily
    accept in that order, rejecting any candidate that shares a slot with an already
    accepted pair. Runs on host: the acceptance decision for candidate k depends on every
    decision before it in sorted order, so it does not vectorize, and the candidate count is
    data-dependent and far smaller than N^2, unlike the detection pass above.
    """
    n_candidates = candidates.n
    if n_candidates == 0:
        empty_f = torch.empty(0, dtype=candidates.t_star.dtype)
        empty_i = torch.empty(0, dtype=torch.int64)
        return AcceptedPairs(
            i=empty_i,
            j=empty_i.clone(),
            t_star=empty_f,
            rel_speed=empty_f.clone(),
            contact_radius_sum=empty_f.clone(),
            f_reject=0.0,
        )

    i_cpu = candidates.i.detach().to("cpu")
    j_cpu = candidates.j.detach().to("cpu")
    t_cpu = candidates.t_star.detach().to("cpu")
    speed_cpu = candidates.rel_speed.detach().to("cpu")
    r_sum_cpu = candidates.contact_radius_sum.detach().to("cpu")

    i_list = i_cpu.tolist()
    j_list = j_cpu.tolist()
    t_list = t_cpu.tolist()

    order = sorted(range(n_candidates), key=lambda k: (t_list[k], i_list[k], j_list[k]))

    claimed: set[int] = set()
    accepted_idx: list[int] = []
    n_rejected = 0
    for k in order:
        a, b = i_list[k], j_list[k]
        if a in claimed or b in claimed:
            n_rejected += 1
            continue
        claimed.add(a)
        claimed.add(b)
        accepted_idx.append(k)

    f_reject = n_rejected / n_candidates
    idx_tensor = torch.tensor(accepted_idx, dtype=torch.int64)

    return AcceptedPairs(
        i=i_cpu[idx_tensor],
        j=j_cpu[idx_tensor],
        t_star=t_cpu[idx_tensor],
        rel_speed=speed_cpu[idx_tensor],
        contact_radius_sum=r_sum_cpu[idx_tensor],
        f_reject=f_reject,
    )


@dataclass(frozen=True)
class CollisionOutcome:
    state: State
    n_elastic: int
    n_merge: int
    n_fragment: int
    delta_e_int: float
    delta_l_spin: torch.Tensor
    f_reject: float
    c_coll_max: float


def v_coh_from_state(state, radius: float) -> float:
    # Sec. 4.6 / 9.1.1: v_coh = V_CHAR = sqrt(G M_real / radius). fp64 regardless of
    # state.dtype, matching scales_from_state's convention for realization-derived scales.
    m_real = state.m.to(torch.float64).sum().item()
    return math.sqrt(config.G * m_real / radius)


def regime_probabilities(
    x: torch.Tensor,
    *,
    elastic_weight: float = config.MAP_ELASTIC_WEIGHT,
    x_clamp: float = config.MAP_X_CLAMP,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Sec. 4.7: (1/x, C, x)/Z, no exp/log. Dtype-generic: computes in whatever dtype x
    # carries, never branching on it.
    x_c = x.clamp(min=1.0 / x_clamp, max=x_clamp)
    z = (1.0 / x_c) + elastic_weight + x_c
    p_fus = (1.0 / x_c) / z
    p_el = elastic_weight / z
    p_frag = x_c / z
    return p_fus, p_el, p_frag


def _cross3(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Same component order as observables.angular_momentum.
    cx = a[1] * b[2] - a[2] * b[1]
    cy = a[2] * b[0] - a[0] * b[2]
    cz = a[0] * b[1] - a[1] * b[0]
    return torch.stack((cx, cy, cz))


def _grav_energy(m1: torch.Tensor, m2: torch.Tensor, d: torch.Tensor, softening: float) -> torch.Tensor:
    # E_grav(m, m', d) = G m m' / sqrt(d^2 + eps^2), Sec. 4.10.
    return config.G * m1 * m2 / torch.sqrt(d * d + softening * softening)


def resolve(
    state: State,
    dt: float,
    model: CollisionModel,
    accepted: AcceptedPairs,
    generator,
    softening: float,
) -> CollisionOutcome:
    """Sec. 4.9 (three outcomes), 4.9(0) (slot rule), 4.10 (E_int, L_spin), 4.7.1 (exactly
    two uniform draws per accepted event, unconditionally, before the channel branch). Pure
    with respect to long-run accumulators: returns this pass's deltas only. Completes the
    drift for every slot: non-participants get r += dt*v; participants get advanced dt - t*
    after their outcome map is applied at t*. state.v is v^(n+1/2), same convention as
    detect(). `generator` is a numpy.random.Generator (the COLLISION_SEED stream); it is
    advanced by exactly 2 * accepted.n draws, in AcceptedPairs order (already sorted by
    (t*, i, j) by pair_disjoint). With accepted.n == 0 the generator is untouched and the
    result is the plain drift r <- r + dt*v, identity-composed with the drift (Sec. 4.5,
    INV-30).
    """
    r = state.r
    v = state.v
    m = state.m
    device = r.device

    r_new = r + dt * v
    v_new = v.clone()
    m_new = m.clone()

    n_events = accepted.n

    if n_events == 0:
        return CollisionOutcome(
            state=State(r=r_new, v=v_new, m=m_new),
            n_elastic=0,
            n_merge=0,
            n_fragment=0,
            delta_e_int=0.0,
            delta_l_spin=torch.zeros(3, dtype=torch.float64, device=device),
            f_reject=accepted.f_reject,
            c_coll_max=0.0,
        )

    c_coll_max = (accepted.rel_speed * dt / accepted.contact_radius_sum).max().item()

    i_list = accepted.i.tolist()
    j_list = accepted.j.tolist()
    t_list = accepted.t_star.tolist()

    delta_e_int = 0.0
    delta_l_spin = torch.zeros(3, dtype=torch.float64, device=device)
    n_elastic = 0
    n_merge = 0
    n_fragment = 0

    for k in range(n_events):
        i_idx = i_list[k]
        j_idx = j_list[k]
        t_star = t_list[k]
        remaining = dt - t_star

        m_i = m[i_idx]
        m_j = m[j_idx]
        v_i = v[i_idx]
        v_j = v[j_idx]
        r_i = r[i_idx]
        r_j = r[j_idx]

        big_m = m_i + m_j
        mu = m_i * m_j / big_m
        u_vec = v_j - v_i
        u_sq = (u_vec * u_vec).sum()
        t_cm = 0.5 * mu * u_sq

        r_i_star = r_i + t_star * v_i
        r_j_star = r_j + t_star * v_j
        dr_star = r_j_star - r_i_star
        sep_sq = (dr_star * dr_star).sum()

        # Sec. 4.9(0.1): exact head-on encounter (zero impact parameter), sep = dr + t* dv is
        # 0/0. Not a fallback -- it is the continuous extension of n = sep/|sep|: along the
        # rectilinear approach, sep(t) = -(t* - t) u for t < t*, so
        # lim_{t -> t*-} sep(t)/|sep(t)| = -u/|u| exactly. Branch condition is exactly
        # sep_sq == 0.0, no threshold (a threshold would replace a good normal -- accurate to
        # a few ulp at any nonzero magnitude -- with the fallback, which is strictly worse).
        # dr == 0 and u == 0 simultaneously is out of contract: Sec. 4.3's dr.dv < 0 guard is
        # strict, so detect() never emits a u == 0 candidate; only a caller that bypasses
        # detect() can reach this, and there is no direction in the problem to fall back to.
        d_ij = torch.sqrt(sep_sq)
        if sep_sq.item() == 0.0:
            u_norm = torch.sqrt(u_sq)
            if u_norm.item() == 0.0:
                raise ValueError(
                    "resolve(): degenerate event (i="
                    f"{i_idx}, j={j_idx}, t*={t_star}) has both zero separation and zero "
                    "relative velocity at contact; the contact normal is undefined and out "
                    "of contract (Sec. 4.9(0.1)) -- detect() never emits such a candidate, "
                    "so this input bypassed detect()"
                )
            n_hat = -u_vec / u_norm
        else:
            n_hat = dr_star / d_ij

        # Sec. 4.6 (revision 2026-08-07 (b)): the gravitational term v_esc_eff is retracted --
        # it does not regulate runaway growth, it causes it (v_esc_eff^2 grows with the
        # merged body's own mass, driving x toward 0 and p_fus toward 1). x now depends only
        # on the relative speed at contact and the single fixed scale v_coh; mass cancels
        # completely.
        x = u_sq / (model.v_coh * model.v_coh)

        p_fus_t, p_el_t, _ = regime_probabilities(x)
        p_fus = float(p_fus_t.item())
        p_el = float(p_el_t.item())

        u1 = float(generator.random())
        u2 = float(generator.random())

        if u1 < p_fus:
            # (2) Merger, Sec. 4.9(0)/4.9.
            n_merge += 1
            r_c = (m_i * r_i_star + m_j * r_j_star) / big_m
            v_cm = (m_i * v_i + m_j * v_j) / big_m
            r_final = r_c + remaining * v_cm

            v_new[i_idx] = v_cm
            v_new[j_idx] = v_cm
            r_new[i_idx] = r_final
            r_new[j_idx] = r_final
            m_new[i_idx] = big_m
            m_new[j_idx] = 0.0

            e_grav_mutual = _grav_energy(m_i, m_j, d_ij, softening)
            event_delta_e = float((t_cm - e_grav_mutual).item())
            event_delta_l = _cross3(dr_star, u_vec) * mu

        elif u1 < p_fus + p_el:
            # (1) Elastic, Sec. 4.9(0)/4.9.
            n_elastic += 1
            u_dot_n = (u_vec * n_hat).sum()
            impulse = 2.0 * mu * u_dot_n * n_hat
            v_i_new = v_i + impulse / m_i
            v_j_new = v_j - impulse / m_j

            v_new[i_idx] = v_i_new
            v_new[j_idx] = v_j_new
            r_new[i_idx] = r_i_star + remaining * v_i_new
            r_new[j_idx] = r_j_star + remaining * v_j_new
            # masses unchanged: m_new already carries m_i, m_j via the initial .clone()

            event_delta_e = 0.0
            event_delta_l = torch.zeros(3, dtype=torch.float64, device=device)

        else:
            # (3) Fragmentation, Sec. 4.9(0)/4.9.
            n_fragment += 1
            f = config.FRAG_F_MIN + (1.0 - 2.0 * config.FRAG_F_MIN) * u2
            m_a = f * big_m
            m_b = big_m - m_a
            mu_prime = m_a * m_b / big_m
            u_prime_mag = torch.sqrt(u_sq) * torch.sqrt(mu / mu_prime)
            u_prime_vec = u_prime_mag * n_hat

            radii_new = contact_radii(torch.stack((m_a, m_b)), model)
            r_ab_sum = radii_new[0] + radii_new[1]

            r_c = (m_i * r_i_star + m_j * r_j_star) / big_m
            v_cm = (m_i * v_i + m_j * v_j) / big_m

            r_a = r_c + (m_b / big_m) * r_ab_sum * n_hat
            r_b = r_c - (m_a / big_m) * r_ab_sum * n_hat
            v_a = v_cm + (m_b / big_m) * u_prime_vec
            v_b = v_cm - (m_a / big_m) * u_prime_vec

            v_new[i_idx] = v_a
            v_new[j_idx] = v_b
            r_new[i_idx] = r_a + remaining * v_a
            r_new[j_idx] = r_b + remaining * v_b
            m_new[i_idx] = m_a
            m_new[j_idx] = m_b

            # Sec. 4.10, sign correction 2026-08-07 (b): E_int += E_grav(after) -
            # E_grav(before). Derived from the document's own general rule E_int += -(dK+dU)
            # with dK = 0: if fragmentation deepens the mutual well (E_grav_after >
            # E_grav_before), U falls, E_mec falls, and that is dissipation, so E_int must
            # rise -- this expression does; E_grav(before) - E_grav(after) does not.
            e_grav_before = _grav_energy(m_i, m_j, d_ij, softening)
            e_grav_after = _grav_energy(m_a, m_b, r_ab_sum, softening)
            event_delta_e = float((e_grav_after - e_grav_before).item())
            event_delta_l = _cross3(dr_star, u_vec) * mu

        delta_e_int += event_delta_e
        delta_l_spin = delta_l_spin + event_delta_l.to(torch.float64)

    return CollisionOutcome(
        state=State(r=r_new, v=v_new, m=m_new),
        n_elastic=n_elastic,
        n_merge=n_merge,
        n_fragment=n_fragment,
        delta_e_int=delta_e_int,
        delta_l_spin=delta_l_spin,
        f_reject=accepted.f_reject,
        c_coll_max=c_coll_max,
    )


def collision_pass(state: State, dt: float, model: CollisionModel, generator, softening: float) -> CollisionOutcome:
    """Sec. 9.1.1: detect + pair_disjoint + resolve, composed. This is what integrate() calls
    inside the velocity_verlet branch."""
    candidates = detect(state, dt, model)
    accepted = pair_disjoint(candidates)
    return resolve(state, dt, model, accepted, generator, softening)
