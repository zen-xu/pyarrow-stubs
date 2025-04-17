# ruff: noqa: F403
import sys

from typing import Any, Final, Sequence, final, overload

from _typeshed import SupportsAdd, SupportsAllComparisons

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

from .__lib_pxi.array import *
from .__lib_pxi.benchmark import *
from .__lib_pxi.builder import *
from .__lib_pxi.compat import *
from .__lib_pxi.config import *
from .__lib_pxi.device import *
from .__lib_pxi.error import *
from .__lib_pxi.io import *
from .__lib_pxi.ipc import *
from .__lib_pxi.memory import *
from .__lib_pxi.pandas_shim import *
from .__lib_pxi.scalar import *
from .__lib_pxi.table import *
from .__lib_pxi.tensor import *
from .__lib_pxi.types import *

@final
class MonthDayNano(
    Sequence[int],
    SupportsAdd[Sequence[int], Sequence[int]],
    SupportsAllComparisons,
):
    n_fields: Final = 3
    n_unnamed_fields: Final = 0
    n_sequence_fields: Final = 3
    if sys.version_info >= (3, 10):
        __match_args__: Final = ("months", "days", "nanoseconds")
    @property
    def months(self) -> int: ...
    @property
    def days(self) -> int: ...
    @property
    def nanoseconds(self) -> int: ...
    @overload
    def __getitem__(self, index: int) -> int: ...
    @overload
    def __getitem__(self, index: slice) -> Sequence[int]: ...
    def __len__(self) -> int: ...
    def __new__(cls, iterable: tuple[int, int, int], /) -> Self: ...
    if sys.version_info >= (3, 13):
        def __replace__(self, **kwargs: Any) -> Self: ...

def cpu_count() -> int: ...
def set_cpu_count(count: int) -> None: ...
def is_threading_enabled() -> bool: ...

Type_NA: int
Type_BOOL: int
Type_UINT8: int
Type_INT8: int
Type_UINT16: int
Type_INT16: int
Type_UINT32: int
Type_INT32: int
Type_UINT64: int
Type_INT64: int
Type_HALF_FLOAT: int
Type_FLOAT: int
Type_DOUBLE: int
Type_DECIMAL128: int
Type_DECIMAL256: int
Type_DATE32: int
Type_DATE64: int
Type_TIMESTAMP: int
Type_TIME32: int
Type_TIME64: int
Type_DURATION: int
Type_INTERVAL_MONTH_DAY_NANO: int
Type_BINARY: int
Type_STRING: int
Type_LARGE_BINARY: int
Type_LARGE_STRING: int
Type_FIXED_SIZE_BINARY: int
Type_BINARY_VIEW: int
Type_STRING_VIEW: int
Type_LIST: int
Type_LARGE_LIST: int
Type_LIST_VIEW: int
Type_LARGE_LIST_VIEW: int
Type_MAP: int
Type_FIXED_SIZE_LIST: int
Type_STRUCT: int
Type_SPARSE_UNION: int
Type_DENSE_UNION: int
Type_DICTIONARY: int
Type_RUN_END_ENCODED: int
UnionMode_SPARSE: int
UnionMode_DENSE: int
