from collections.abc import Callable
from typing import Any
from typing import Protocol
from typing import Sequence
from typing import TypeVar

_F = TypeVar("_F", bound=Callable)
_N = TypeVar("_N")

class _DocStringComponents(Protocol):
    _docstring_components: list[str]

def doc(
    *docstrings: str | _DocStringComponents | Callable | None, **params: Any
) -> Callable[[_F], _F]: ...
def product(seq: Sequence[_N]) -> _N: ...
def get_contiguous_span(
    shape: tuple[int, ...], strides: tuple[int, ...], itemsize: int
) -> tuple[int, int]: ...
def find_free_port() -> int: ...
def guid() -> str: ...
def download_tzdata_on_windows() -> None: ...
