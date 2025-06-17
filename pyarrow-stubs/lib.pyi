# ruff: noqa: F403
from typing import NamedTuple

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

class MonthDayNano(NamedTuple):
    days: int
    months: int
    nanoseconds: int

def cpu_count() -> int:
    """
    Return the number of threads to use in parallel operations.

    The number of threads is determined at startup by inspecting the
    ``OMP_NUM_THREADS`` and ``OMP_THREAD_LIMIT`` environment variables.
    If neither is present, it will default to the number of hardware threads
    on the system. It can be modified at runtime by calling
    :func:`set_cpu_count()`.

    See Also
    --------
    set_cpu_count : Modify the size of this pool.
    io_thread_count : The analogous function for the I/O thread pool.
    """

def set_cpu_count(count: int) -> None:
    """
    Set the number of threads to use in parallel operations.

    Parameters
    ----------
    count : int
        The number of concurrent threads that should be used.

    See Also
    --------
    cpu_count : Get the size of this pool.
    set_io_thread_count : The analogous function for the I/O thread pool.
    """

def is_threading_enabled() -> bool:
    """
    Returns True if threading is enabled in libarrow.

    If it isn't enabled, then python shouldn't create any
    threads either, because we're probably on a system where
    threading doesn't work (e.g. Emscripten).
    """

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
