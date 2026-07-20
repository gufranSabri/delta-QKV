"""Progress bar wrapper that's a no-op under --slurm.

tqdm's carriage-return redraws are meant for a live terminal; in a Slurm log
file they just spam thousands of lines. `set_slurm_mode(True)` (wired from
main.py's --slurm flag) makes `progress()` return the plain iterable instead.
"""

from __future__ import annotations

from typing import Iterable, Iterator, TypeVar

T = TypeVar("T")

_SLURM_MODE = False


def set_slurm_mode(enabled: bool) -> None:
    global _SLURM_MODE
    _SLURM_MODE = enabled


def is_slurm_mode() -> bool:
    return _SLURM_MODE


def progress(iterable: Iterable[T], **tqdm_kwargs) -> Iterator[T]:
    if _SLURM_MODE:
        return iter(iterable)
    from tqdm import tqdm

    return tqdm(iterable, **tqdm_kwargs)
