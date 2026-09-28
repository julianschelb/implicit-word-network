# _utils.py
"""Small internal helpers shared across modules."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TypeVar

T = TypeVar("T")


def batched(iterable: Iterable[T], size: int) -> Iterator[list[T]]:
    """Yield successive lists of at most ``size`` items (``itertools.batched`` backport)."""
    if size < 1:
        raise ValueError("size must be >= 1")
    batch: list[T] = []
    for item in iterable:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def require(module: str, *, extra: str, purpose: str) -> None:
    """Raise a helpful ImportError when an optional dependency is missing."""
    import importlib.util

    if importlib.util.find_spec(module) is None:
        raise ImportError(
            f"{purpose} requires the optional dependency {module!r}. "
            f'Install it with: pip install "implicit-word-network[{extra}]"'
        )
