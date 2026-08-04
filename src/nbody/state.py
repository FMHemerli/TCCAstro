from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class State:
    r: torch.Tensor
    v: torch.Tensor
    m: torch.Tensor

    def __post_init__(self) -> None:
        if self.r.ndim != 2 or self.r.shape[1] != 3:
            raise ValueError(f"r must have shape (N, 3), got {tuple(self.r.shape)}")
        if self.v.ndim != 2 or self.v.shape[1] != 3:
            raise ValueError(f"v must have shape (N, 3), got {tuple(self.v.shape)}")
        if self.m.ndim != 1:
            raise ValueError(f"m must have shape (N,), got {tuple(self.m.shape)}")
        n = self.r.shape[0]
        if self.v.shape[0] != n or self.m.shape[0] != n:
            raise ValueError(
                "r, v and m must share the same N: "
                f"r has {n}, v has {self.v.shape[0]}, m has {self.m.shape[0]}"
            )
        if not (self.r.dtype == self.v.dtype == self.m.dtype):
            raise ValueError(
                "r, v and m must share the same dtype: "
                f"r={self.r.dtype}, v={self.v.dtype}, m={self.m.dtype}"
            )
        if not (self.r.device == self.v.device == self.m.device):
            raise ValueError(
                "r, v and m must share the same device: "
                f"r={self.r.device}, v={self.v.device}, m={self.m.device}"
            )

    @property
    def n(self) -> int:
        return self.r.shape[0]

    @property
    def dtype(self) -> torch.dtype:
        return self.r.dtype

    @property
    def device(self) -> torch.device:
        return self.r.device

    def to(self, *, dtype: torch.dtype | None = None, device=None) -> "State":
        return State(
            r=self.r.to(dtype=dtype, device=device),
            v=self.v.to(dtype=dtype, device=device),
            m=self.m.to(dtype=dtype, device=device),
        )
