from typing import Any, Collection, Literal, Protocol, TypeAlias, TypeVar

import numpy as np

from numpy.typing import NDArray

from .__lib_pxi.array import BooleanArray, IntegerArray

ArrayLike: TypeAlias = Any
Order: TypeAlias = Literal["ascending", "descending"]
JoinType: TypeAlias = Literal[
    "left semi",
    "right semi",
    "left anti",
    "right anti",
    "inner",
    "left outer",
    "right outer",
    "full outer",
]
Compression: TypeAlias = Literal[
    "gzip", "bz2", "brotli", "lz4", "lz4_frame", "lz4_raw", "zstd", "snappy"
]
NullEncoding: TypeAlias = Literal["mask", "encode"]
NullSelectionBehavior: TypeAlias = Literal["drop", "emit_null"]
Mask: TypeAlias = list[bool | None] | NDArray[np.bool_] | BooleanArray
Indices: TypeAlias = list[int] | NDArray[np.integer] | IntegerArray

_T = TypeVar("_T")
SingleOrList: TypeAlias = list[_T] | _T

class SupportEq(Protocol):
    def __eq__(self, other) -> bool: ...

class SupportLt(Protocol):
    def __lt__(self, other) -> bool: ...

class SupportGt(Protocol):
    def __gt__(self, other) -> bool: ...

class SupportLe(Protocol):
    def __le__(self, other) -> bool: ...

class SupportGe(Protocol):
    def __ge__(self, other) -> bool: ...

FilterTuple: TypeAlias = (
    tuple[str, Literal["=", "==", "!="], SupportEq]
    | tuple[str, Literal["<"], SupportLt]
    | tuple[str, Literal[">"], SupportGt]
    | tuple[str, Literal["<="], SupportLe]
    | tuple[str, Literal[">="], SupportGe]
    | tuple[str, Literal["in", "not in"], Collection]
)

class Buffer(Protocol):
    def __buffer__(self, flags: int, /) -> memoryview: ...

class SupportPyBuffer(Protocol):
    def __buffer__(self, flags: int, /) -> memoryview: ...

class SupportArrowStream(Protocol):
    def __arrow_c_stream__(self, requested_schema=None) -> Any: ...

class SupportArrowArray(Protocol):
    def __arrow_c_array__(self, requested_schema=None) -> Any: ...

class SupportArrowDeviceArray(Protocol):
    def __arrow_c_device_array__(self, requested_schema=None, **kwargs) -> Any: ...

class SupportArrowSchema(Protocol):
    def __arrow_c_schema(self) -> Any: ...
