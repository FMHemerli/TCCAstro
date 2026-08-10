from __future__ import annotations

from dataclasses import dataclass, field

import torch

from . import config
from ._pairwise import default_tile_size, iter_tiles
from .state import State


@dataclass(frozen=True)
class CollisionModel:
    # r_ref, m_bar keep their original relative order and stay positional-or-keyword, so the
    # pre-existing construction convention CollisionModel(r_ref, m_bar) is preserved bit for
    # bit. The deterministic-contact fields (e_restitution, k_bind, chip_coeff, chip_max,
    # energy_max) and seed are kw_only with defaults, per Sec. 8/9.1.1 (revision (f)).
    r_ref: float = config.R_REF_DEFAULT
    m_bar: float = config.PARTICLE_MASS
    e_restitution: float = field(default=config.E_RESTITUTION, kw_only=True)
    k_bind: float = field(default=config.K_BIND, kw_only=True)
    chip_coeff: float = field(default=config.FRAG_CHIP_COEFF, kw_only=True)
    chip_max: float = field(default=config.FRAG_CHIP_MAX, kw_only=True)
    energy_max: float = field(default=config.FRAG_ENERGY_MAX, kw_only=True)
    seed: int = field(default=config.COLLISION_SEED, kw_only=True)

    def __post_init__(self) -> None:
        if not (0.0 < self.e_restitution <= 1.0):
            raise ValueError(
                "CollisionModel: e_restitution must lie in (0, 1], got "
                f"{self.e_restitution!r} (Sec. 4.9(A): e = 0 is forbidden)"
            )


@dataclass(frozen=True)
class CollisionCandidates:
    i: torch.Tensor
    j: torch.Tensor
    t_c: torch.Tensor
    u_n: torch.Tensor
    rel_speed: torch.Tensor
    contact_radius_sum: torch.Tensor

    @property
    def n(self) -> int:
        return int(self.i.shape[0])


@dataclass(frozen=True)
class AcceptedPairs:
    i: torch.Tensor
    j: torch.Tensor
    t_c: torch.Tensor
    u_n: torch.Tensor
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
        t_c=empty_f,
        u_n=empty_f.clone(),
        rel_speed=empty_f.clone(),
        contact_radius_sum=empty_f.clone(),
    )


def detect(state, dt: float, model: CollisionModel) -> CollisionCandidates:
    """Swept collision detection over [0, dt], reusing the tiling contract of _pairwise so
    the O(N^2) pass never allocates a full N x N intermediate at once. state.v is taken as
    the relative velocity to sweep with (Sec. 4.3: exact for the velocity-Verlet drift when
    v is the mid-step velocity v^(n+1/2); the caller is responsible for passing that state).

    Sec. 4.3, revision (f): per-pair contact time t_c is the first root of the exact-contact
    quadratic |sep(t)| = R, not the closest-approach clamp of the previous model. Guards, in
    order: (1) a = dv.dv > 0 strict; (2) dr.dv < 0 strict (approaching); (3) c = dr.dr - R^2
    <= 0 -> already overlapping, t_c = 0, D is not evaluated; (4) else D = b^2 - 4ac > 0
    strict, t_c = 2c / (-b + sqrt(D)) -- the canonical form (-b - sqrt(D)) / (2a) is
    catastrophically unstable when 4ac << b^2 and is never used; (5) 0 <= t_c <= dt.
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
    un_chunks: list[torch.Tensor] = []
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

        a = (dv * dv).sum(dim=-1)
        dr_dot_dv = (dr * dv).sum(dim=-1)
        b = 2.0 * dr_dot_dv
        dr_sq = (dr * dr).sum(dim=-1)

        r_sum = radii_i.unsqueeze(1) + radii.unsqueeze(0)
        c = dr_sq - r_sum * r_sum

        guard_a = a > 0.0
        guard_approach = dr_dot_dv < 0.0
        overlapping = c <= 0.0

        d_disc = b * b - 4.0 * a * c
        guard_disc = d_disc > 0.0
        # Masked-safe sqrt/division: entries where guard_disc is False (D <= 0, including all
        # overlapping-branch entries where c <= 0 makes D >= b^2 >= 0 trivially true, and all
        # non-candidate entries) never reach t_c_root through candidate_mask below, so garbage
        # produced by the substituted denominator here is discarded, not propagated.
        sqrt_d = torch.sqrt(torch.clamp(d_disc, min=0.0))
        denom = -b + sqrt_d
        denom_safe = torch.where(denom == 0.0, torch.ones_like(denom), denom)
        t_c_root = 2.0 * c / denom_safe

        t_c = torch.where(overlapping, torch.zeros_like(c), t_c_root)
        within_h = (t_c >= 0.0) & (t_c <= dt)

        candidate_mask = (
            upper
            & pair_live
            & guard_a
            & guard_approach
            & (overlapping | guard_disc)
            & within_h
        )

        # Contact normal at t_c (Sec. 4.4): n = sep / d_c, or the continuous extension -u/|u|
        # when d_c == 0 (head-on contact). guard_a already forces |dv| = |u| > 0 wherever
        # candidate_mask can be True, so the u/|u| branch never sees a 0/0 among accepted
        # candidates; entries where it would are masked-safe substituted and discarded below.
        sep = dr + t_c.unsqueeze(-1) * dv
        d_c = torch.sqrt((sep * sep).sum(dim=-1))
        d_c_safe = torch.where(d_c == 0.0, torch.ones_like(d_c), d_c)
        n_from_sep = sep / d_c_safe.unsqueeze(-1)
        u_norm_safe = torch.where(a == 0.0, torch.ones_like(a), torch.sqrt(a))
        n_from_u = -dv / u_norm_safe.unsqueeze(-1)
        n_hat = torch.where((d_c == 0.0).unsqueeze(-1), n_from_u, n_from_sep)
        u_n = (dv * n_hat).sum(dim=-1)

        rows, cols = torch.nonzero(candidate_mask, as_tuple=True)
        if rows.numel() == 0:
            continue

        i_chunks.append(row_global[rows])
        j_chunks.append(cols)
        t_chunks.append(t_c[rows, cols])
        un_chunks.append(u_n[rows, cols])
        speed_chunks.append(a[rows, cols].sqrt())
        r_sum_chunks.append(r_sum[rows, cols])

    if not i_chunks:
        return _empty_candidates(dtype, device)

    return CollisionCandidates(
        i=torch.cat(i_chunks),
        j=torch.cat(j_chunks),
        t_c=torch.cat(t_chunks),
        u_n=torch.cat(un_chunks),
        rel_speed=torch.cat(speed_chunks),
        contact_radius_sum=torch.cat(r_sum_chunks),
    )


def pair_disjoint(candidates: CollisionCandidates) -> AcceptedPairs:
    """Sec. 4.5: sort candidates by the lexicographic key (t_c, i, j) -- the (i, j) tie-break
    is normative, it is what makes the greedy acceptance reproducible across devices when
    two candidates share a bit-identical t_c (e.g. a symmetric configuration). Greedily
    accept in that order, rejecting any candidate that shares a slot with an already
    accepted pair. Runs on host: the acceptance decision for candidate k depends on every
    decision before it in sorted order, so it does not vectorize, and the candidate count is
    data-dependent and far smaller than N^2, unlike the detection pass above.
    """
    n_candidates = candidates.n
    if n_candidates == 0:
        empty_f = torch.empty(0, dtype=candidates.t_c.dtype)
        empty_i = torch.empty(0, dtype=torch.int64)
        return AcceptedPairs(
            i=empty_i,
            j=empty_i.clone(),
            t_c=empty_f,
            u_n=empty_f.clone(),
            rel_speed=empty_f.clone(),
            contact_radius_sum=empty_f.clone(),
            f_reject=0.0,
        )

    i_cpu = candidates.i.detach().to("cpu")
    j_cpu = candidates.j.detach().to("cpu")
    t_cpu = candidates.t_c.detach().to("cpu")
    un_cpu = candidates.u_n.detach().to("cpu")
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
        t_c=t_cpu[idx_tensor],
        u_n=un_cpu[idx_tensor],
        rel_speed=speed_cpu[idx_tensor],
        contact_radius_sum=r_sum_cpu[idx_tensor],
        f_reject=f_reject,
    )


@dataclass(frozen=True)
class CollisionOutcome:
    state: State
    n_ricochet: int
    n_merge: int
    n_erosion: int
    delta_e_int: float
    delta_l_spin: torch.Tensor
    f_reject: float
    c_coll_max: float


def _cross3(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Same component order as observables.angular_momentum.
    cx = a[1] * b[2] - a[2] * b[1]
    cy = a[2] * b[0] - a[0] * b[2]
    cz = a[0] * b[1] - a[1] * b[0]
    return torch.stack((cx, cy, cz))

def _softened(d: torch.Tensor, softening: float) -> torch.Tensor:
    return torch.sqrt(d * d + softening * softening)


def resolve(
    state: State,
    dt: float,
    model: CollisionModel,
    accepted: AcceptedPairs,
    generator,
    softening: float,
) -> CollisionOutcome:
    """Sec. 4.9 (three outcomes, deterministic gate cascade), Sec. 4.9(A) (slot rule), Sec.
    4.10 (E_int, L_spin, single accounting rule). Pure with respect to long-run accumulators:
    returns this pass's deltas only. Completes the drift for every slot: non-participants get
    r += dt*v; participants get advanced dt - t_c after their outcome map is applied at t_c.
    state.v is v^(n+1/2), same convention as detect(). `generator` is kept in the signature
    (Sec. 4.7.1, INV-32) and is not consumed: the deterministic gate cascade of revision (f)
    draws nothing.

    Sec. 4.5 item 3 (current-state accounting): within one pass, events are processed in
    accepted order (already (t_c, i, j)-sorted by pair_disjoint); the field term of an event's
    energy accounting must see positions already updated by earlier events in the same pass.
    This is tracked per slot as (r_ref, v, m, t_ref), t_ref = 0 at the start of the pass; slot
    k's position at time t is r_ref_k + (t - t_ref_k) * v_k. Only the two slots touched by an
    event get r_ref/t_ref rewritten (to their post-event position and t_c); a pair's own r_i,
    r_j, v_i, v_j at t_c are therefore always identical to what detect() saw, because
    pair_disjoint guarantees no slot is touched twice in the same pass.
    """
    r0 = state.r
    v0 = state.v
    m0 = state.m
    device = r0.device
    grav_const = config.G

    n_events = accepted.n

    if n_events == 0:
        return CollisionOutcome(
            state=State(r=r0 + dt * v0, v=v0.clone(), m=m0.clone()),
            n_ricochet=0,
            n_merge=0,
            n_erosion=0,
            delta_e_int=0.0,
            delta_l_spin=torch.zeros(3, dtype=torch.float64, device=device),
            f_reject=accepted.f_reject,
            c_coll_max=0.0,
        )

    c_coll_max = (accepted.rel_speed * dt / accepted.contact_radius_sum).max().item()

    i_list = accepted.i.tolist()
    j_list = accepted.j.tolist()
    t_list = accepted.t_c.tolist()

    r_ref = r0.clone()
    v_cur = v0.clone()
    m_cur = m0.clone()
    t_ref = torch.zeros_like(m0)

    delta_e_int = 0.0
    delta_l_spin = torch.zeros(3, dtype=torch.float64, device=device)
    n_ricochet = 0
    n_merge = 0
    n_erosion = 0

    n_slots = m0.shape[0]
    slot_idx = torch.arange(n_slots, device=device)

    for k in range(n_events):
        i_idx = i_list[k]
        j_idx = j_list[k]
        t_c = t_list[k]

        m_i = m0[i_idx]
        m_j = m0[j_idx]
        v_i = v0[i_idx]
        v_j = v0[j_idx]
        r_i = r0[i_idx]
        r_j = r0[j_idx]

        big_m = m_i + m_j
        mu = m_i * m_j / big_m
        u_vec = v_j - v_i
        u_sq = (u_vec * u_vec).sum()
        u_norm = torch.sqrt(u_sq)

        r_i_tc = r_i + t_c * v_i
        r_j_tc = r_j + t_c * v_j
        dr_tc = r_j_tc - r_i_tc
        sep_sq = (dr_tc * dr_tc).sum()
        d_c = torch.sqrt(sep_sq)

        # Sec. 4.4: n = sep/d_c, or the continuous extension -u/|u| at d_c == 0 (exact
        # head-on contact). detect()'s guard 1 (a = dv.dv > 0) forces |u| > 0 for every
        # candidate it emits, so the u == 0 branch below is unreachable through the
        # detect -> pair_disjoint -> resolve pipeline; it is retained because resolve()'s
        # own contract (Sec. 4.4 "casos degenerados") requires it to fail loudly for a
        # caller that bypasses detect() and feeds resolve() a genuinely degenerate pair.
        if d_c.item() == 0.0:
            if u_norm.item() == 0.0:
                raise ValueError(
                    "resolve(): degenerate event (i="
                    f"{i_idx}, j={j_idx}, t_c={t_c}) has both zero separation and zero "
                    "relative velocity at contact; the contact normal is undefined and out "
                    "of contract -- detect() never emits such a candidate, so this input "
                    "bypassed detect()"
                )
            n_hat = -u_vec / u_norm
        else:
            n_hat = dr_tc / d_c

        u_n = (u_vec * n_hat).sum()
        d_tilde_c = _softened(d_c, softening)

        i_is_big = bool((m_i >= m_j).item())
        if i_is_big:
            m_G, m_P = m_i, m_j
        else:
            m_G, m_P = m_j, m_i

        # Gate 1 (Sec. 4.9, fusion): |u| < sqrt(2 G M / d~_c).
        v_esc = torch.sqrt(2.0 * grav_const * big_m / d_tilde_c)
        is_fusion = bool((u_norm < v_esc).item())

        if is_fusion:
            n_merge += 1
            r_c = (m_i * r_i_tc + m_j * r_j_tc) / big_m
            v_cm = (m_i * v_i + m_j * v_j) / big_m

            if i_is_big:
                m_i_new = big_m
                m_j_new = torch.zeros_like(big_m)
            else:
                m_i_new = torch.zeros_like(big_m)
                m_j_new = big_m
            r_i_new = r_c
            r_j_new = r_c
            v_i_new = v_cm
            v_j_new = v_cm

            event_delta_e, event_delta_l = _general_accounting(
                slot_idx, r_ref, v_cur, m_cur, t_ref, t_c, i_idx, j_idx,
                m_i, m_j, v_i, v_j, r_i_tc, r_j_tc, mu,
                m_i_new, m_j_new, v_i_new, v_j_new, r_i_new, r_j_new,
                softening, grav_const,
            )
        else:
            R_P = contact_radii(m_P, model)
            e_lig = model.k_bind * grav_const * m_P * m_P / R_P
            t_n = 0.5 * mu * u_n * u_n

            # Gate 2 (Sec. 4.9, erosion): T_n > K_BIND G m_P^2 / R_P. m_P, R_P are the
            # SMALLER body's -- using the larger body's would fragment on grazing encounters.
            is_erosion = bool((t_n > e_lig).item())

            if is_erosion:
                n_erosion += 1
                q = m_G / m_P
                xi = model.chip_coeff * (t_n / e_lig - 1.0)
                f_chip = torch.clamp(xi, min=0.0, max=model.chip_max) * (1.0 - q.pow(-5.0 / 3.0))

                u_r = u_vec - (1.0 + model.e_restitution) * u_n * n_hat
                t_r = 0.5 * mu * (u_r * u_r).sum()

                e_custo = torch.minimum(f_chip * e_lig, model.energy_max * t_r)
                if bool((e_custo < f_chip * e_lig).item()):
                    f_chip = e_custo / e_lig

                m_chip = f_chip * m_P
                m_G_new = m_G + m_chip
                m_P_new = m_P - m_chip
                mu_new = m_G_new * m_P_new / big_m

                t_prime = t_r - e_custo
                u_prime = u_r * torch.sqrt(t_prime / t_r) * torch.sqrt(mu / mu_new)

                s = 1.0 if i_is_big else -1.0
                delta_shift = s * (m_chip / big_m) * dr_tc

                if i_is_big:
                    m_i_new, m_j_new = m_G_new, m_P_new
                else:
                    m_i_new, m_j_new = m_P_new, m_G_new

                r_i_new = r_i_tc + delta_shift
                r_j_new = r_j_tc + delta_shift
                v_cm = (m_i * v_i + m_j * v_j) / big_m
                v_i_new = v_cm - (m_j_new / big_m) * u_prime
                v_j_new = v_cm + (m_i_new / big_m) * u_prime

                event_delta_e, event_delta_l = _general_accounting(
                    slot_idx, r_ref, v_cur, m_cur, t_ref, t_c, i_idx, j_idx,
                    m_i, m_j, v_i, v_j, r_i_tc, r_j_tc, mu,
                    m_i_new, m_j_new, v_i_new, v_j_new, r_i_new, r_j_new,
                    softening, grav_const,
                )
            else:
                n_ricochet += 1
                impulse_dir = (1.0 + model.e_restitution) * mu * u_n * n_hat
                v_i_new = v_i + impulse_dir / m_i
                v_j_new = v_j - impulse_dir / m_j
                r_i_new = r_i_tc
                r_j_new = r_j_tc
                m_i_new = m_i
                m_j_new = m_j

                # Sec. 4.9, closed form: E_int += 0.5 mu (1 - e^2) u_n^2; Delta U = 0 exact
                # (mutual and field), Delta L_spin = 0 exact. The O(N) field sum is not run.
                event_delta_e = float(
                    (0.5 * mu * (1.0 - model.e_restitution * model.e_restitution) * u_n * u_n).item()
                )
                event_delta_l = torch.zeros(3, dtype=torch.float64, device=device)

        r_ref[i_idx] = r_i_new
        r_ref[j_idx] = r_j_new
        t_ref[i_idx] = t_c
        t_ref[j_idx] = t_c
        v_cur[i_idx] = v_i_new
        v_cur[j_idx] = v_j_new
        m_cur[i_idx] = m_i_new
        m_cur[j_idx] = m_j_new

        delta_e_int += event_delta_e
        delta_l_spin = delta_l_spin + event_delta_l.to(torch.float64)

    r_final = r_ref + (dt - t_ref).unsqueeze(-1) * v_cur

    return CollisionOutcome(
        state=State(r=r_final, v=v_cur, m=m_cur),
        n_ricochet=n_ricochet,
        n_merge=n_merge,
        n_erosion=n_erosion,
        delta_e_int=delta_e_int,
        delta_l_spin=delta_l_spin,
        f_reject=accepted.f_reject,
        c_coll_max=c_coll_max,
    )


def _general_accounting(
    slot_idx: torch.Tensor,
    r_ref: torch.Tensor,
    v_cur: torch.Tensor,
    m_cur: torch.Tensor,
    t_ref: torch.Tensor,
    t_c: float,
    i_idx: int,
    j_idx: int,
    m_i: torch.Tensor,
    m_j: torch.Tensor,
    v_i: torch.Tensor,
    v_j: torch.Tensor,
    r_i_tc: torch.Tensor,
    r_j_tc: torch.Tensor,
    mu: torch.Tensor,
    m_i_new: torch.Tensor,
    m_j_new: torch.Tensor,
    v_i_new: torch.Tensor,
    v_j_new: torch.Tensor,
    r_i_new: torch.Tensor,
    r_j_new: torch.Tensor,
    softening: float,
    grav_const: float,
) -> tuple[float, torch.Tensor]:
    """Sec. 4.10, general accounting rule, used by fusion and erosion (never ricochet, which
    has a closed form and skips this entirely). E_int += -(Delta K + Delta U); Delta U splits
    into the pair term (i, j only) and the field term (Sec. 4.5 item 3's "current state" of
    every other slot, one tensor op over the N-slot vector -- not a Python loop over k).
    """
    delta_k = (
        0.5 * (m_i_new * (v_i_new * v_i_new).sum() + m_j_new * (v_j_new * v_j_new).sum())
        - 0.5 * (m_i * (v_i * v_i).sum() + m_j * (v_j * v_j).sum())
    )

    d_before = torch.sqrt(((r_j_tc - r_i_tc) * (r_j_tc - r_i_tc)).sum())
    d_after = torch.sqrt(((r_j_new - r_i_new) * (r_j_new - r_i_new)).sum())
    d_tilde_before = _softened(d_before, softening)
    d_tilde_after = _softened(d_after, softening)

    delta_u_par = (
        -grav_const * m_i_new * m_j_new / d_tilde_after
        + grav_const * m_i * m_j / d_tilde_before
    )

    others = torch.ones(slot_idx.shape[0], dtype=torch.bool, device=slot_idx.device)
    others[i_idx] = False
    others[j_idx] = False

    pos_all = r_ref + (t_c - t_ref).unsqueeze(-1) * v_cur
    m_k = m_cur

    diff_ik = pos_all - r_i_tc.unsqueeze(0)
    diff_jk = pos_all - r_j_tc.unsqueeze(0)
    diff_ak = pos_all - r_i_new.unsqueeze(0)
    diff_bk = pos_all - r_j_new.unsqueeze(0)

    d_ik = _softened(torch.sqrt((diff_ik * diff_ik).sum(dim=-1)), softening)
    d_jk = _softened(torch.sqrt((diff_jk * diff_jk).sum(dim=-1)), softening)
    d_ak = _softened(torch.sqrt((diff_ak * diff_ak).sum(dim=-1)), softening)
    d_bk = _softened(torch.sqrt((diff_bk * diff_bk).sum(dim=-1)), softening)

    field_terms = m_k * ((m_i_new / d_ak - m_i / d_ik) + (m_j_new / d_bk - m_j / d_jk))
    field_terms = torch.where(others, field_terms, torch.zeros_like(field_terms))
    delta_u_campo = -grav_const * field_terms.sum()

    delta_u = delta_u_par + delta_u_campo
    event_delta_e = float((-(delta_k + delta_u)).item())

    dr_before = r_j_tc - r_i_tc
    dr_after = r_j_new - r_i_new
    u_before = v_j - v_i
    u_after = v_j_new - v_i_new
    # m_i_new + m_j_new == m_i + m_j exactly (mass is conserved within the pair by both the
    # fusion and erosion maps), and m_i, m_j are both > 0 (detect() only pairs live slots), so
    # this sum is strictly positive -- no guard needed.
    mu_new = m_i_new * m_j_new / (m_i_new + m_j_new)
    event_delta_l = mu * _cross3(dr_before, u_before) - mu_new * _cross3(dr_after, u_after)

    return event_delta_e, event_delta_l


def collision_pass(state: State, dt: float, model: CollisionModel, generator, softening: float) -> CollisionOutcome:
    """Sec. 9.1.1: detect + pair_disjoint + resolve, composed. This is what integrate() calls
    inside the velocity_verlet branch."""
    candidates = detect(state, dt, model)
    accepted = pair_disjoint(candidates)
    return resolve(state, dt, model, accepted, generator, softening)
