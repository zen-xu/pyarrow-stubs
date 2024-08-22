import collections.abc
import datetime as dt

from decimal import Decimal
from typing import Any, Generic, Iterator, Literal, Mapping, Self, TypeAlias, overload

import numpy as np

from pyarrow._compute import CastOptions
from pyarrow.lib import Array, Buffer, MemoryPool, MonthDayNano, Tensor, _Weakrefable
from typing_extensions import TypeVar

from . import types
from .types import _DataTypeT, _NewDataTypeT

_IsValid = TypeVar("_IsValid", default=Literal[True])
_PyType = TypeVar("_PyType")

class Scalar(_Weakrefable, Generic[_DataTypeT, _PyType, _IsValid]):
    @property
    def type(self) -> _DataTypeT: ...
    @property
    def is_valid(self) -> _IsValid: ...
    @overload
    def cast(
        self,
        target_type: None,
        safe: bool = True,
        options: CastOptions | None = None,
        memory_pool: MemoryPool | None = None,
    ) -> Self: ...
    @overload
    def cast(
        self,
        target_type: _NewDataTypeT,
        safe: bool = True,
        options: CastOptions | None = None,
        memory_pool: MemoryPool | None = None,
    ) -> Scalar[_NewDataTypeT, _PyType]: ...
    def validate(self, *, full: bool = False) -> None: ...
    def equals(self, other: Scalar) -> bool: ...
    def __hash__(self) -> int: ...
    @overload
    def as_py(self: Scalar[Any, _PyType, Literal[True]]) -> _PyType: ...
    @overload
    def as_py(self: Scalar[Any, _PyType, Literal[False]]) -> None: ...

_NULL: TypeAlias = None
NA = _NULL

class NullScalar(Scalar[types.NullType, None, _IsValid]): ...
class BooleanScalar(Scalar[types.BoolType, bool, _IsValid]): ...
class UInt8Scalar(Scalar[types.Uint8Type, int, _IsValid]): ...
class Int8Scalar(Scalar[types.Int8Type, int, _IsValid]): ...
class UInt16Scalar(Scalar[types.Uint16Type, int, _IsValid]): ...
class Int16Scalar(Scalar[types.Int16Type, int, _IsValid]): ...
class UInt32Scalar(Scalar[types.Uint32Type, int, _IsValid]): ...
class Int32Scalar(Scalar[types.Int32Type, int, _IsValid]): ...
class UInt64Scalar(Scalar[types.Uint64Type, int, _IsValid]): ...
class Int64Scalar(Scalar[types.Int64Type, int, _IsValid]): ...
class HalfFloatScalar(Scalar[types.Float16Type, float, _IsValid]): ...
class FloatScalar(Scalar[types.Float32Type, float, _IsValid]): ...
class DoubleScalar(Scalar[types.Float64Type, float, _IsValid]): ...
class Decimal128Scalar(Scalar[types.Decimal128Type, Decimal, _IsValid]): ...
class Decimal256Scalar(Scalar[types.Decimal256Type, Decimal, _IsValid]): ...
class Date32Scalar(Scalar[types.Date32Type, dt.date, _IsValid]): ...

class Date64Scalar(Scalar[types.Date64Type, dt.date, _IsValid]):
    @property
    def value(self) -> dt.date | None: ...

class Time32Scalar(Scalar[types.Time32Type, dt.time, _IsValid]):
    @property
    def value(self) -> dt.time | None: ...

class Time64Scalar(Scalar[types.Time64Type, dt.time, _IsValid]):
    @property
    def value(self) -> dt.time | None: ...

class TimestampScalar(Scalar[types.TimestampType, int, _IsValid]):
    @property
    def value(self) -> int | None: ...

class DurationScalar(Scalar[types.DurationType, dt.timedelta, _IsValid]):
    @property
    def value(self) -> dt.timedelta | None: ...

class MonthDayNanoIntervalScalar(Scalar[types.MonthDayNanoIntervalType, MonthDayNano, _IsValid]):
    @property
    def value(self) -> MonthDayNano | None: ...

class BinaryScalar(Scalar[types.BinaryType, bytes, _IsValid]):
    def as_buffer(self) -> Buffer: ...

class LargeBinaryScalar(Scalar[types.LargeBinaryType, bytes, _IsValid]):
    def as_buffer(self) -> Buffer: ...

class FixedBinaryScalar(Scalar[types.FixedSizeBinaryType, bytes, _IsValid]):
    def as_buffer(self) -> Buffer: ...

class StringScalar(Scalar[types.StringType, str, _IsValid]):
    def as_buffer(self) -> Buffer: ...

class LargeStringScalar(Scalar[types.LargeStringType, str, _IsValid]):
    def as_buffer(self) -> Buffer: ...

class BinaryViewScalar(Scalar[types.BinaryViewType, bytes, _IsValid]):
    def as_buffer(self) -> Buffer: ...

class StringViewScalar(Scalar[types.StringViewType, str, _IsValid]):
    def as_buffer(self) -> Buffer: ...

class ListScalar(Scalar[types.ListType[_DataTypeT], _PyType, _IsValid]):
    @property
    def values(self) -> Array | None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, i: int) -> Scalar[_DataTypeT, _PyType, _IsValid]: ...
    def __iter__(self) -> Iterator[Array]: ...

class FixedSizeListScalar(
    Scalar[types.FixedSizeListType[_DataTypeT, types._Size], _PyType, _IsValid]
):
    @property
    def values(self) -> Array | None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, i: int) -> Scalar[_DataTypeT, _PyType, _IsValid]: ...
    def __iter__(self) -> Iterator[Array]: ...

class LargeListScalar(Scalar[types.LargeListType[_DataTypeT], _PyType, _IsValid]):
    @property
    def values(self) -> Array | None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, i: int) -> Scalar[_DataTypeT, _PyType, _IsValid]: ...
    def __iter__(self) -> Iterator[Array]: ...

class ListViewScalar(Scalar[types.ListViewType[_DataTypeT], _PyType, _IsValid]):
    @property
    def values(self) -> Array | None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, i: int) -> Scalar[_DataTypeT, _PyType, _IsValid]: ...
    def __iter__(self) -> Iterator[Array]: ...

class LargeListViewScalar(Scalar[types.LargeListViewType[_DataTypeT], _PyType, _IsValid]):
    @property
    def values(self) -> Array | None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, i: int) -> Scalar[_DataTypeT, _PyType, _IsValid]: ...
    def __iter__(self) -> Iterator[Array]: ...

class StructScalar(
    Scalar[types.StructType, dict[str, Any], _IsValid], collections.abc.Mapping[str, Scalar]
):
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[str]: ...
    def __getitem__(self, __key: str) -> Scalar[Any, Any, _IsValid]: ...  # type: ignore[override]
    def _as_py_tuple(self) -> list[tuple[str, Any]]: ...

_K = TypeVar("_K")
_V = TypeVar("_V")

class MapScalar(Scalar[types.MapType[types._K, types._ValueT], dict[_K, _V], _IsValid]):
    @property
    def values(self) -> Array | None: ...
    def __len__(self) -> int: ...
    def __getitem__(
        self, i: int
    ) -> tuple[Scalar[types._K, Any, _IsValid], types._ValueT, Any, _IsValid]: ...
    def __iter__(self) -> Iterator[tuple[_K, _V]]: ...

class DictionaryScalar(
    Scalar[types.DictionaryType[types._IndexT, types._ValueT], dict[_K, _V], _IsValid]
):
    @property
    def index(self) -> Scalar[types._IndexT, _K, _IsValid]: ...
    @property
    def value(self) -> Scalar[types._ValueT, _V, _IsValid]: ...
    @property
    def dictionary(self) -> Array: ...

class RunEndEncodedScalar(
    Scalar[types.RunEndEncodedType[types._RunEndType, types._ValueT], tuple[int, int], _IsValid]
):
    @property
    def value(self) -> tuple[int, int] | None: ...

class UnionScalar(Scalar[types.UnionType, Any, _IsValid]):
    @property
    def value(self) -> Any | None: ...
    @property
    def type_code(self) -> str: ...

class ExtensionScalar(Scalar[types.ExtensionType, _PyType, _IsValid]):
    @property
    def value(self) -> Any | None: ...
    @staticmethod
    def from_storage(typ: types.BaseExtensionType, value) -> ExtensionScalar: ...

class FixedShapeTensorScalar(ExtensionScalar[_PyType, _IsValid]):
    def to_numpy(self) -> np.ndarray: ...
    def to_tensor(self) -> Tensor: ...

@overload
def scalar(
    value: str, *, from_pandas: bool | None = None, memory_pool: MemoryPool | None = None
) -> StringScalar: ...
@overload
def scalar(
    value: bytes, *, from_pandas: bool | None = None, memory_pool: MemoryPool | None = None
) -> BinaryScalar: ...
@overload
def scalar(  # type: ignore[overload-overlap]
    value: bool, *, from_pandas: bool | None = None, memory_pool: MemoryPool | None = None
) -> BooleanScalar: ...
@overload
def scalar(
    value: int, *, from_pandas: bool | None = None, memory_pool: MemoryPool | None = None
) -> Int64Scalar: ...
@overload
def scalar(
    value: float, *, from_pandas: bool | None = None, memory_pool: MemoryPool | None = None
) -> DoubleScalar: ...
@overload
def scalar(
    value: Decimal, *, from_pandas: bool | None = None, memory_pool: MemoryPool | None = None
) -> Decimal128Scalar: ...
@overload
def scalar(
    value: dt.date, *, from_pandas: bool | None = None, memory_pool: MemoryPool | None = None
) -> Date32Scalar: ...
@overload
def scalar(
    value: dt.time, *, from_pandas: bool | None = None, memory_pool: MemoryPool | None = None
) -> Time64Scalar: ...
@overload
def scalar(
    value: dt.timedelta, *, from_pandas: bool | None = None, memory_pool: MemoryPool | None = None
) -> DurationScalar: ...
@overload
def scalar(  # type: ignore[overload-overlap]
    value: MonthDayNano, *, from_pandas: bool | None = None, memory_pool: MemoryPool | None = None
) -> MonthDayNanoIntervalScalar: ...
@overload
def scalar(
    value: Mapping[str, Any],
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> StructScalar: ...
@overload
def scalar(
    value: list[str] | tuple[str, ...] | set[str],
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> ListScalar[types.ListType[types.StringType], list[str]]: ...
@overload
def scalar(
    value: list[bytes] | tuple[bytes, ...] | set[bytes],
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> ListScalar[types.ListType[types.BinaryType], list[bytes]]: ...
@overload
def scalar(
    value: list[int] | tuple[int, ...] | set[int],
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> ListScalar[types.ListType[types.Int64Type], list[int]]: ...
@overload
def scalar(
    value: list[bool] | tuple[bool, ...] | set[bool],
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> ListScalar[types.ListType[types.BoolType], list[bool]]: ...
@overload
def scalar(  # type: ignore[misc]
    value: list[int] | tuple[int, ...] | set[int],
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> ListScalar[types.ListType[types.Int64Type], list[int]]: ...
@overload
def scalar(
    value: list[float] | tuple[float, ...] | set[float],
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> ListScalar[types.ListType[types.Float64Type], list[float]]: ...
@overload
def scalar(
    value: list[Decimal] | tuple[Decimal, ...] | set[Decimal],
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> ListScalar[types.ListType[types.Decimal128Type], list[Decimal]]: ...
@overload
def scalar(
    value: list[dt.date] | tuple[dt.date, ...] | set[dt.date],
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> ListScalar[types.ListType[types.Date32Type], list[dt.date]]: ...
@overload
def scalar(
    value: list[dt.time] | tuple[dt.time, ...] | set[dt.time],
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> ListScalar[types.ListType[types.Time32Type], list[dt.time]]: ...
@overload
def scalar(
    value: list[dt.timedelta] | tuple[dt.timedelta, ...] | set[dt.timedelta],
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> ListScalar[types.ListType[types.DurationType], list[dt.timedelta]]: ...
@overload
def scalar(
    value: list[MonthDayNano] | tuple[MonthDayNano, ...] | set[MonthDayNano],
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> ListScalar[types.ListType[types.MonthDayNanoIntervalType], list[MonthDayNano]]: ...
@overload
def scalar(
    value: list[_V] | tuple[_V, ...] | set[_V],
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> ListScalar[Any, list[_V]]: ...
@overload
def scalar(
    value: _V,
    type: _DataTypeT,
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> Scalar[_DataTypeT, _V]: ...
