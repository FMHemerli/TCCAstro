from __future__ import annotations

import torch

from . import config

# Conservative count of distinct (tile, N)-shaped or (tile, N, 3)-shaped
# temporaries alive at once inside a single row-block iteration of the
# tiled pairwise kernel (diff, dsq, inv, m*inv, weighted, plus headroom for
# non-in-place intermediate copies that eager execution allocates). Counted
# in units of one fp-element per (i, j) pair, i.e. a (tile, N, 3) buffer
# counts as 3. This governs only the automatic tile_size chosen when the
# caller passes tile_size=None; it does not affect the numerical result.
_TEMPORARIES_PER_PAIR = 10


def default_tile_size(n: int, dtype: torch.dtype, memory_budget_bytes: int | None = None) -> int:
    if n <= 0:
        return 1
    budget = config.TILE_MEMORY_BUDGET_BYTES if memory_budget_bytes is None else memory_budget_bytes
    element_size = torch.finfo(dtype).bits // 8
    bytes_per_row = n * element_size * _TEMPORARIES_PER_PAIR
    tile = budget // bytes_per_row if bytes_per_row > 0 else n
    return max(1, min(int(tile), n))


def iter_tiles(n: int, tile_size: int):
    if n <= 0:
        return
    tile_size = max(1, min(tile_size, n))
    start = 0
    while start < n:
        end = min(start + tile_size, n)
        yield start, end
        start = end
