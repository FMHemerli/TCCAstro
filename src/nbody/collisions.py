from __future__ import annotations

from dataclasses import dataclass

import torch

from . import config
from ._pairwise import default_tile_size, iter_tiles


@dataclass(frozen=True)
class CollisionModel:
    r_ref: float = config.R_REF_DEFAULT
    m_bar: float = config.PARTICLE_MASS


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
        # degenerate division below is ever selected. (ii) |dv|^2 == 0 -> t* = 0, guarded
        # here unconditionally (not just on selected candidates) so the elementwise divide
        # never produces inf/nan anywhere in the tile, selected or not.
        approaching = dr_dot_dv < 0.0
        candidate_mask = upper & pair_live & approaching

        no_relative_motion = dv_sq == 0.0
        safe_dv_sq = torch.where(no_relative_motion, torch.ones_like(dv_sq), dv_sq)
        raw_t_star = torch.where(no_relative_motion, torch.zeros_like(dv_sq), -dr_dot_dv / safe_dv_sq)
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
