import collections.abc
import datetime as dt
import sys

from decimal import Decimal

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self
if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias
from typing import Any, Generic, Iterator, Mapping, overload

import numpy as np

from pyarrow._compute import CastOptions
from pyarrow.lib import Array, Buffer, MemoryPool, MonthDayNano, Tensor, _Weakrefable
from typing_extensions import TypeVar

from . import types
from .types import (
    _AsPyType,
    _DataType_CoT,
    _DataTypeT,
    _Time32Unit,
    _Time64Unit,
    _Tz,
    _Unit,
)

_AsPyTypeK = TypeVar("_AsPyTypeK")
_AsPyTypeV = TypeVar("_AsPyTypeV")

class Scalar(_Weakrefable, Generic[_DataType_CoT]):
    @property
    def type(self) -> _DataType_CoT: ...
    @property
    def is_valid(self) -> bool: ...
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
        target_type: _DataTypeT,
        safe: bool = True,
        options: CastOptions | None = None,
        memory_pool: MemoryPool | None = None,
    ) -> Scalar[_DataTypeT]: ...
    def validate(self, *, full: bool = False) -> None: ...
    def equals(self, other: Scalar) -> bool: ...
    def __hash__(self) -> int: ...
    @overload
    def as_py(self: Scalar[types._BasicDataType[_AsPyType]]) -> _AsPyType: ...
    @overload
    def as_py(
        self: Scalar[types.ListType[types._BasicDataType[_AsPyType]]],
    ) -> list[_AsPyType]: ...
    @overload
    def as_py(
        self: Scalar[
            types.ListType[
                types.DictionaryType[types._IndexT, types._BasicDataType[_AsPyTypeV], Any]
            ]
        ],
    ) -> list[dict[int, _AsPyTypeV]]: ...
    @overload
    def as_py(
        self: Scalar[
            types.ListType[types.DictionaryType[Any, types._BasicDataType[_AsPyTypeV], Any]],
        ],
    ) -> list[dict[Any, _AsPyTypeV]]: ...
    @overload
    def as_py(
        self: Scalar[types.ListType[types.DictionaryType[types._IndexT, Any, Any]],],
    ) -> list[dict[int, Any]]: ...
    @overload
    def as_py(
        self: Scalar[types.StructType],
    ) -> list[dict[str, Any]]: ...
    @overload
    def as_py(
        self: Scalar[
            types.MapType[types._BasicDataType[_AsPyTypeK], types._BasicDataType[_AsPyTypeV]]
        ],
    ) -> list[tuple[_AsPyTypeK, _AsPyTypeV]]: ...
    @overload
    def as_py(
        self: Scalar[types.MapType[Any, types._BasicDataType[_AsPyTypeV]]],
    ) -> list[tuple[Any, _AsPyTypeV]]: ...
    @overload
    def as_py(
        self: Scalar[types.MapType[types._BasicDataType[_AsPyTypeK], Any]],
    ) -> list[tuple[_AsPyTypeK, Any]]: ...
    @overload
    def as_py(self: Scalar[Any]) -> Any: ...

_NULL: TypeAlias = None
NA = _NULL

class NullScalar(Scalar[types.NullType]): ...
class BooleanScalar(Scalar[types.BoolType]): ...
class UInt8Scalar(Scalar[types.UInt8Type]): ...
class Int8Scalar(Scalar[types.Int8Type]): ...
class UInt16Scalar(Scalar[types.UInt16Type]): ...
class Int16Scalar(Scalar[types.Int16Type]): ...
class UInt32Scalar(Scalar[types.Uint32Type]): ...
class Int32Scalar(Scalar[types.Int32Type]): ...
class UInt64Scalar(Scalar[types.UInt64Type]): ...
class Int64Scalar(Scalar[types.Int64Type]): ...
class HalfFloatScalar(Scalar[types.Float16Type]): ...
class FloatScalar(Scalar[types.Float32Type]): ...
class DoubleScalar(Scalar[types.Float64Type]): ...
class Decimal32Scalar(Scalar[types.Decimal32Type[types._Precision, types._Scale]]): ...
class Decimal64Scalar(Scalar[types.Decimal64Type[types._Precision, types._Scale]]): ...
class Decimal128Scalar(Scalar[types.Decimal128Type[types._Precision, types._Scale]]): ...
class Decimal256Scalar(Scalar[types.Decimal256Type[types._Precision, types._Scale]]): ...
class Date32Scalar(Scalar[types.Date32Type]): ...

class Date64Scalar(Scalar[types.Date64Type]):
    @property
    def value(self) -> dt.date | None: ...

class Time32Scalar(Scalar[types.Time32Type[_Time32Unit]]):
    @property
    def value(self) -> dt.time | None: ...

class Time64Scalar(Scalar[types.Time64Type[_Time64Unit]]):
    @property
    def value(self) -> dt.time | None: ...

class TimestampScalar(Scalar[types.TimestampType[_Unit, _Tz]]):
    @property
    def value(self) -> int | None: ...

class DurationScalar(Scalar[types.DurationType[_Unit]]):
    @property
    def value(self) -> dt.timedelta | None: ...

class MonthDayNanoIntervalScalar(Scalar[types.MonthDayNanoIntervalType]):
    @property
    def value(self) -> MonthDayNano | None: ...

class BinaryScalar(Scalar[types.BinaryType]):
    def as_buffer(self) -> Buffer: ...

class LargeBinaryScalar(Scalar[types.LargeBinaryType]):
    def as_buffer(self) -> Buffer: ...

class FixedSizeBinaryScalar(Scalar[types.FixedSizeBinaryType]):
    def as_buffer(self) -> Buffer: ...

class StringScalar(Scalar[types.StringType]):
    def as_buffer(self) -> Buffer: ...

class LargeStringScalar(Scalar[types.LargeStringType]):
    def as_buffer(self) -> Buffer: ...

class BinaryViewScalar(Scalar[types.BinaryViewType]):
    def as_buffer(self) -> Buffer: ...

class StringViewScalar(Scalar[types.StringViewType]):
    def as_buffer(self) -> Buffer: ...

class ListScalar(Scalar[types.ListType[_DataType_CoT]]):
    @property
    def values(self) -> Array | None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, i: int) -> Scalar[_DataType_CoT]: ...
    def __iter__(self) -> Iterator[Array]: ...

class FixedSizeListScalar(Scalar[types.FixedSizeListType[_DataType_CoT, types._Size]]):
    @property
    def values(self) -> Array | None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, i: int) -> Scalar[_DataType_CoT]: ...
    def __iter__(self) -> Iterator[Array]: ...

class LargeListScalar(Scalar[types.LargeListType[_DataType_CoT]]):
    @property
    def values(self) -> Array | None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, i: int) -> Scalar[_DataType_CoT]: ...
    def __iter__(self) -> Iterator[Array]: ...

class ListViewScalar(Scalar[types.ListViewType[_DataType_CoT]]):
    @property
    def values(self) -> Array | None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, i: int) -> Scalar[_DataType_CoT]: ...
    def __iter__(self) -> Iterator[Array]: ...

class LargeListViewScalar(Scalar[types.LargeListViewType[_DataType_CoT]]):
    @property
    def values(self) -> Array | None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, i: int) -> Scalar[_DataType_CoT]: ...
    def __iter__(self) -> Iterator[Array]: ...

class StructScalar(Scalar[types.StructType], collections.abc.Mapping[str, Scalar]):
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[str]: ...
    def __getitem__(self, __key: str) -> Scalar[Any]: ...  # type: ignore[override]
    def _as_py_tuple(self) -> list[tuple[str, Any]]: ...

class MapScalar(Scalar[types.MapType[types._K, types._ValueT]]):
    @property
    def values(self) -> Array | None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, i: int) -> tuple[Scalar[types._K], types._ValueT, Any]: ...
    @overload
    def __iter__(
        self: Scalar[
            types.MapType[types._BasicDataType[_AsPyTypeK], types._BasicDataType[_AsPyTypeV]]
        ],
    ) -> Iterator[tuple[_AsPyTypeK, _AsPyTypeV]]: ...
    @overload
    def __iter__(
        self: Scalar[types.MapType[Any, types._BasicDataType[_AsPyTypeV]],],
    ) -> Iterator[tuple[Any, _AsPyTypeV]]: ...
    @overload
    def __iter__(
        self: Scalar[types.MapType[types._BasicDataType[_AsPyTypeK], Any],],
    ) -> Iterator[tuple[_AsPyTypeK, Any]]: ...

class DictionaryScalar(Scalar[types.DictionaryType[types._IndexT, types._BasicValueT]]):
    @property
    def index(self) -> Scalar[types._IndexT]: ...
    @property
    def value(self) -> Scalar[types._BasicValueT]: ...
    @property
    def dictionary(self) -> Array: ...

class RunEndEncodedScalar(Scalar[types.RunEndEncodedType[types._RunEndType, types._BasicValueT]]):
    @property
    def value(self) -> tuple[int, types._BasicValueT] | None: ...

class UnionScalar(Scalar[types.UnionType]):
    @property
    def value(self) -> Any | None: ...
    @property
    def type_code(self) -> str: ...

class ExtensionScalar(Scalar[types.ExtensionType]):
    @property
    def value(self) -> Any | None: ...
    @staticmethod
    def from_storage(typ: types.BaseExtensionType, value) -> ExtensionScalar: ...

class Bool8Scalar(Scalar[types.Bool8Type]): ...
class UuidScalar(Scalar[types.UuidType]): ...
class JsonScalar(Scalar[types.JsonType]): ...
class OpaqueScalar(Scalar[types.OpaqueType]): ...

class FixedShapeTensorScalar(ExtensionScalar):
    def to_numpy(self) -> np.ndarray: ...
    def to_tensor(self) -> Tensor: ...

_V = TypeVar("_V")

CollectionValue: TypeAlias = list[_V | None] | tuple[_V | None, ...] | set[_V | None]

@overload
def scalar(
    value: str, *, from_pandas: bool | None = None, memory_pool: MemoryPool | None = None
) -> StringScalar: ...
@overload
def scalar(
    value: bytes, *, from_pandas: bool | None = None, memory_pool: MemoryPool | None = None
) -> BinaryScalar: ...
@overload
def scalar(
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
    value: dt.datetime, *, from_pandas: bool | None = None, memory_pool: MemoryPool | None = None
) -> TimestampScalar: ...
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
def scalar(
    value: MonthDayNano,
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
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
    value: CollectionValue[str],
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> ListScalar[types.ListType[types.StringType]]: ...
@overload
def scalar(
    value: CollectionValue[bytes],
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> ListScalar[types.ListType[types.BinaryType]]: ...
@overload
def scalar(
    value: CollectionValue[bool],
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> ListScalar[types.ListType[types.BoolType]]: ...
@overload
def scalar(
    value: CollectionValue[int],
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> ListScalar[types.ListType[types.Int64Type]]: ...
@overload
def scalar(
    value: CollectionValue[float],
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> ListScalar[types.ListType[types.Float64Type]]: ...
@overload
def scalar(
    value: CollectionValue[Decimal],
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> ListScalar[types.ListType[types.Decimal32Type]]: ...
@overload
def scalar(
    value: CollectionValue[Decimal],
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> ListScalar[types.ListType[types.Decimal64Type]]: ...
@overload
def scalar(
    value: CollectionValue[Decimal],
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> ListScalar[types.ListType[types.Decimal128Type]]: ...
@overload
def scalar(
    value: CollectionValue[dt.datetime],
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> ListScalar[types.ListType[types.TimestampType]]: ...
@overload
def scalar(
    value: CollectionValue[dt.date],
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> ListScalar[types.ListType[types.Date32Type]]: ...
@overload
def scalar(
    value: CollectionValue[dt.time],
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> ListScalar[types.ListType[types.Time32Type]]: ...
@overload
def scalar(
    value: CollectionValue[dt.timedelta],
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> ListScalar[types.ListType[types.DurationType]]: ...
@overload
def scalar(
    value: CollectionValue[MonthDayNano],
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> ListScalar[types.ListType[types.MonthDayNanoIntervalType]]: ...
@overload
def scalar(
    value: CollectionValue,
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> ListScalar[Any]: ...
@overload
def scalar(
    value: Any,
    type: types.NullType,
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> NullScalar: ...
@overload
def scalar(
    value: Any,
    type: types.BoolType,
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> BooleanScalar: ...
@overload
def scalar(
    value: Any,
    type: types.UInt8Type,
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> UInt8Scalar: ...
@overload
def scalar(
    value: Any,
    type: types.Int8Type,
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> Int8Scalar: ...
@overload
def scalar(
    value: Any,
    type: types.UInt16Type,
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> UInt16Scalar: ...
@overload
def scalar(
    value: Any,
    type: types.Int16Type,
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> Int16Scalar: ...
@overload
def scalar(
    value: Any,
    type: types.Uint32Type,
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> UInt32Scalar: ...
@overload
def scalar(
    value: Any,
    type: types.Int32Type,
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> Int32Scalar: ...
@overload
def scalar(
    value: Any,
    type: types.UInt64Type,
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> UInt64Scalar: ...
@overload
def scalar(
    value: Any,
    type: types.Int64Type,
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> Int64Scalar: ...
@overload
def scalar(
    value: Any,
    type: types.Float16Type,
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> HalfFloatScalar: ...
@overload
def scalar(
    value: Any,
    type: types.Float32Type,
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> FloatScalar: ...
@overload
def scalar(
    value: Any,
    type: types.Float64Type,
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> DoubleScalar: ...
@overload
def scalar(
    value: Any,
    type: types.Date32Type,
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> Date32Scalar: ...
@overload
def scalar(
    value: Any,
    type: types.Date64Type,
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> Date64Scalar: ...
@overload
def scalar(
    value: Any,
    type: types.MonthDayNanoIntervalType,
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> MonthDayNanoIntervalScalar: ...
@overload
def scalar(
    value: Any,
    type: types.StringType,
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> StringScalar: ...
@overload
def scalar(
    value: Any,
    type: types.LargeStringType,
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> LargeStringScalar: ...
@overload
def scalar(
    value: Any,
    type: types.StringViewType,
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> StringViewScalar: ...
@overload
def scalar(
    value: Any,
    type: types.BinaryType,
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> BinaryScalar: ...
@overload
def scalar(
    value: Any,
    type: types.LargeBinaryType,
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> LargeBinaryScalar: ...
@overload
def scalar(
    value: Any,
    type: types.BinaryViewType,
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> BinaryViewScalar: ...
@overload
def scalar(
    value: Any,
    type: types.TimestampType[types._Unit, types._Tz],
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> TimestampScalar[types._Unit, types._Tz]: ...
@overload
def scalar(
    value: Any,
    type: types.Time32Type[types._Time32Unit],
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> Time32Scalar[types._Time32Unit]: ...
@overload
def scalar(
    value: Any,
    type: types.Time64Type[types._Time64Unit],
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> Time64Scalar[types._Time64Unit]: ...
@overload
def scalar(
    value: Any,
    type: types.DurationType[types._Unit],
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> DurationScalar[types._Unit]: ...
@overload
def scalar(
    value: Any,
    type: types.Decimal32Type[types._Precision, types._Scale],
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> Decimal32Scalar[types._Precision, types._Scale]: ...
@overload
def scalar(
    value: Any,
    type: types.Decimal64Type[types._Precision, types._Scale],
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> Decimal64Scalar[types._Precision, types._Scale]: ...
@overload
def scalar(
    value: Any,
    type: types.Decimal128Type[types._Precision, types._Scale],
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> Decimal128Scalar[types._Precision, types._Scale]: ...
@overload
def scalar(
    value: Any,
    type: types.Decimal256Type[types._Precision, types._Scale],
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> Decimal256Scalar[types._Precision, types._Scale]: ...
@overload
def scalar(
    value: Any,
    type: types.ListType[_DataTypeT],
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> ListScalar[_DataTypeT]: ...
@overload
def scalar(
    value: Any,
    type: types.LargeListType[_DataTypeT],
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> LargeListScalar[_DataTypeT]: ...
@overload
def scalar(
    value: Any,
    type: types.ListViewType[_DataTypeT],
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> ListViewScalar[_DataTypeT]: ...
@overload
def scalar(
    value: Any,
    type: types.LargeListViewType[_DataTypeT],
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> LargeListViewScalar[_DataTypeT]: ...
@overload
def scalar(
    value: Any,
    type: types.FixedSizeListType,
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> FixedSizeListScalar: ...
@overload
def scalar(
    value: Any,
    type: types.DictionaryType[types._IndexT, types._BasicValueT],
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> DictionaryScalar[types._IndexT, types._BasicValueT]: ...
@overload
def scalar(
    value: Any,
    type: types.MapType[types._K, types._ValueT],
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> MapScalar[types._K, types._ValueT]: ...
@overload
def scalar(
    value: Any,
    type: types.StringType,
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> StructScalar: ...
@overload
def scalar(
    value: Any,
    type: types.UnionType,
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> UnionScalar: ...
@overload
def scalar(
    value: Any,
    type: types.RunEndEncodedType,
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> RunEndEncodedScalar: ...
@overload
def scalar(
    value: Any,
    type: types.Bool8Type,
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> Bool8Scalar: ...
@overload
def scalar(
    value: Any,
    type: types.UuidType,
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> UuidScalar: ...
@overload
def scalar(
    value: Any,
    type: types.JsonType,
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> JsonScalar: ...
@overload
def scalar(
    value: Any,
    type: types.OpaqueType,
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> OpaqueScalar: ...
@overload
def scalar(
    value: Any,
    type: _DataTypeT,
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> Scalar[_DataTypeT]: ...

__all__ = [
    "Scalar",
    "_NULL",
    "NA",
    "NullScalar",
    "BooleanScalar",
    "UInt8Scalar",
    "Int8Scalar",
    "UInt16Scalar",
    "Int16Scalar",
    "UInt32Scalar",
    "Int32Scalar",
    "UInt64Scalar",
    "Int64Scalar",
    "HalfFloatScalar",
    "FloatScalar",
    "DoubleScalar",
    "Decimal32Scalar",
    "Decimal64Scalar",
    "Decimal128Scalar",
    "Decimal256Scalar",
    "Date32Scalar",
    "Date64Scalar",
    "Time32Scalar",
    "Time64Scalar",
    "TimestampScalar",
    "DurationScalar",
    "MonthDayNanoIntervalScalar",
    "BinaryScalar",
    "LargeBinaryScalar",
    "FixedSizeBinaryScalar",
    "StringScalar",
    "LargeStringScalar",
    "BinaryViewScalar",
    "StringViewScalar",
    "ListScalar",
    "FixedSizeListScalar",
    "LargeListScalar",
    "ListViewScalar",
    "LargeListViewScalar",
    "StructScalar",
    "MapScalar",
    "DictionaryScalar",
    "RunEndEncodedScalar",
    "UnionScalar",
    "ExtensionScalar",
    "FixedShapeTensorScalar",
    "Bool8Scalar",
    "UuidScalar",
    "JsonScalar",
    "OpaqueScalar",
    "scalar",
]
