import enum

from typing import Any, Iterable, TypeAlias, TypedDict

from pyarrow.lib import Array, ChunkedArray

from .buffer import _PyArrowBuffer

class DtypeKind(enum.IntEnum):
    INT = 0
    UINT = 1
    FLOAT = 2
    BOOL = 20
    STRING = 21  # UTF-8
    DATETIME = 22
    CATEGORICAL = 23

Dtype: TypeAlias = tuple[DtypeKind, int, str, str]

class ColumnNullType(enum.IntEnum):
    NON_NULLABLE = 0
    USE_NAN = 1
    USE_SENTINEL = 2
    USE_BITMASK = 3
    USE_BYTEMASK = 4

class ColumnBuffers(TypedDict):
    data: tuple[_PyArrowBuffer, Dtype]
    validity: tuple[_PyArrowBuffer, Dtype] | None
    offsets: tuple[_PyArrowBuffer, Dtype] | None

class CategoricalDescription(TypedDict):
    is_ordered: bool
    is_dictionary: bool
    categories: _PyArrowColumn | None

class Endianness(enum.Enum):
    LITTLE = "<"
    BIG = ">"
    NATIVE = "="
    NA = "|"

class NoBufferPresent(Exception): ...

class _PyArrowColumn:
    def __init__(self, column: Array | ChunkedArray, allow_copy: bool = True) -> None: ...
    def size(self) -> int: ...
    @property
    def offset(self) -> int: ...
    @property
    def dtype(self) -> tuple[DtypeKind, int, str, str]: ...
    @property
    def describe_categorical(self) -> CategoricalDescription: ...
    @property
    def describe_null(self) -> tuple[ColumnNullType, Any]: ...
    @property
    def null_count(self) -> int: ...
    @property
    def metadata(self) -> dict[str, Any]: ...
    def num_chunks(self) -> int: ...
    def get_chunks(self, n_chunks: int | None = None) -> Iterable[_PyArrowColumn]: ...
    def get_buffers(self) -> ColumnBuffers: ...
