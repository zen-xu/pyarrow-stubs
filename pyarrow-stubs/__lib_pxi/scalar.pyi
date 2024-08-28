import collections.abc
import datetime as dt

from decimal import Decimal
from typing import Any, Generic, Iterator, Literal, Mapping, Self, TypeAlias, overload

import numpy as np

from pyarrow._compute import CastOptions
from pyarrow.lib import Array, Buffer, MemoryPool, MonthDayNano, Tensor, _Weakrefable
from typing_extensions import TypeVar

from . import types
from .types import _AsPyType, _DataTypeT, _NewDataTypeT, _Time32Unit, _Time64Unit, _Tz, _Unit

_IsValid = TypeVar("_IsValid", default=Literal[True])
_AsPyTypeK = TypeVar("_AsPyTypeK")
_AsPyTypeV = TypeVar("_AsPyTypeV")

class Scalar(_Weakrefable, Generic[_DataTypeT, _IsValid]):
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
    ) -> Scalar[_NewDataTypeT, _IsValid]: ...
    def validate(self, *, full: bool = False) -> None: ...
    def equals(self, other: Scalar) -> bool: ...
    def __hash__(self) -> int: ...
    @overload
    def as_py(self: Scalar[types._BasicDataType[_AsPyType], Literal[True]]) -> _AsPyType: ...
    @overload
    def as_py(
        self: Scalar[types.ListType[types._BasicDataType[_AsPyType]], Literal[True]],
    ) -> list[_AsPyType]: ...
    @overload
    def as_py(
        self: Scalar[
            types.ListType[
                types.DictionaryType[
                    types._BasicDataType[_AsPyTypeK], types._BasicDataType[_AsPyTypeV], Any
                ]
            ],
            Literal[True],
        ],
    ) -> list[dict[_AsPyTypeK, _AsPyTypeV]]: ...
    @overload
    def as_py(
        self: Scalar[
            types.ListType[types.DictionaryType[Any, types._BasicDataType[_AsPyTypeV], Any]],
            Literal[True],
        ],
    ) -> list[dict[Any, _AsPyTypeV]]: ...
    @overload
    def as_py(
        self: Scalar[
            types.ListType[types.DictionaryType[types._BasicDataType[_AsPyTypeK], Any, Any]],
            Literal[True],
        ],
    ) -> list[dict[_AsPyTypeK, Any]]: ...
    @overload
    def as_py(
        self: Scalar[types.StructType, Literal[True]],
    ) -> list[dict[str, Any]]: ...
    @overload
    def as_py(
        self: Scalar[
            types.MapType[types._BasicDataType[_AsPyTypeK], types._BasicDataType[_AsPyTypeV]],
            Literal[True],
        ],
    ) -> list[tuple[_AsPyTypeK, _AsPyTypeV]]: ...
    @overload
    def as_py(
        self: Scalar[
            types.MapType[Any, types._BasicDataType[_AsPyTypeV]],
            Literal[True],
        ],
    ) -> list[tuple[Any, _AsPyTypeV]]: ...
    @overload
    def as_py(
        self: Scalar[
            types.MapType[types._BasicDataType[_AsPyTypeK], Any],
            Literal[True],
        ],
    ) -> list[tuple[_AsPyTypeK, Any]]: ...
    @overload
    def as_py(self: Scalar[Any, Literal[True]]) -> Any: ...
    @overload
    def as_py(self: Scalar[Any, Literal[False]]) -> None: ...

_NULL: TypeAlias = None
NA = _NULL

class NullScalar(Scalar[types.NullType, _IsValid]): ...
class BooleanScalar(Scalar[types.BoolType, _IsValid]): ...
class UInt8Scalar(Scalar[types.Uint8Type, _IsValid]): ...
class Int8Scalar(Scalar[types.Int8Type, _IsValid]): ...
class UInt16Scalar(Scalar[types.Uint16Type, _IsValid]): ...
class Int16Scalar(Scalar[types.Int16Type, _IsValid]): ...
class UInt32Scalar(Scalar[types.Uint32Type, _IsValid]): ...
class Int32Scalar(Scalar[types.Int32Type, _IsValid]): ...
class UInt64Scalar(Scalar[types.Uint64Type, _IsValid]): ...
class Int64Scalar(Scalar[types.Int64Type, _IsValid]): ...
class HalfFloatScalar(Scalar[types.Float16Type, _IsValid]): ...
class FloatScalar(Scalar[types.Float32Type, _IsValid]): ...
class DoubleScalar(Scalar[types.Float64Type, _IsValid]): ...
class Decimal128Scalar(Scalar[types.Decimal128Type, _IsValid]): ...
class Decimal256Scalar(Scalar[types.Decimal256Type, _IsValid]): ...
class Date32Scalar(Scalar[types.Date32Type, _IsValid]): ...

class Date64Scalar(Scalar[types.Date64Type, _IsValid]):
    @property
    def value(self) -> dt.date | None: ...

class Time32Scalar(Scalar[types.Time32Type[_Time32Unit], _IsValid]):
    @property
    def value(self) -> dt.time | None: ...

class Time64Scalar(Scalar[types.Time64Type[_Time64Unit], _IsValid]):
    @property
    def value(self) -> dt.time | None: ...

class TimestampScalar(Scalar[types.TimestampType[_Unit, _Tz], _IsValid]):
    @property
    def value(self) -> int | None: ...

class DurationScalar(Scalar[types.DurationType[_Unit], _IsValid]):
    @property
    def value(self) -> dt.timedelta | None: ...

class MonthDayNanoIntervalScalar(Scalar[types.MonthDayNanoIntervalType, _IsValid]):
    @property
    def value(self) -> MonthDayNano | None: ...

class BinaryScalar(Scalar[types.BinaryType, _IsValid]):
    def as_buffer(self) -> Buffer: ...

class LargeBinaryScalar(Scalar[types.LargeBinaryType, _IsValid]):
    def as_buffer(self) -> Buffer: ...

class FixedBinaryScalar(Scalar[types.FixedSizeBinaryType, _IsValid]):
    def as_buffer(self) -> Buffer: ...

class StringScalar(Scalar[types.StringType, _IsValid]):
    def as_buffer(self) -> Buffer: ...

class LargeStringScalar(Scalar[types.LargeStringType, _IsValid]):
    def as_buffer(self) -> Buffer: ...

class BinaryViewScalar(Scalar[types.BinaryViewType, _IsValid]):
    def as_buffer(self) -> Buffer: ...

class StringViewScalar(Scalar[types.StringViewType, _IsValid]):
    def as_buffer(self) -> Buffer: ...

class ListScalar(Scalar[types.ListType[_DataTypeT], _IsValid]):
    @property
    def values(self) -> Array | None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, i: int) -> Scalar[_DataTypeT, _IsValid]: ...
    def __iter__(self) -> Iterator[Array]: ...

class FixedSizeListScalar(Scalar[types.FixedSizeListType[_DataTypeT, types._Size], _IsValid]):
    @property
    def values(self) -> Array | None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, i: int) -> Scalar[_DataTypeT, _IsValid]: ...
    def __iter__(self) -> Iterator[Array]: ...

class LargeListScalar(Scalar[types.LargeListType[_DataTypeT], _IsValid]):
    @property
    def values(self) -> Array | None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, i: int) -> Scalar[_DataTypeT, _IsValid]: ...
    def __iter__(self) -> Iterator[Array]: ...

class ListViewScalar(Scalar[types.ListViewType[_DataTypeT], _IsValid]):
    @property
    def values(self) -> Array | None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, i: int) -> Scalar[_DataTypeT, _IsValid]: ...
    def __iter__(self) -> Iterator[Array]: ...

class LargeListViewScalar(Scalar[types.LargeListViewType[_DataTypeT], _IsValid]):
    @property
    def values(self) -> Array | None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, i: int) -> Scalar[_DataTypeT, _IsValid]: ...
    def __iter__(self) -> Iterator[Array]: ...

class StructScalar(Scalar[types.StructType, _IsValid], collections.abc.Mapping[str, Scalar]):
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[str]: ...
    def __getitem__(self, __key: str) -> Scalar[Any, _IsValid]: ...  # type: ignore[override]
    def _as_py_tuple(self) -> list[tuple[str, Any]]: ...

class MapScalar(Scalar[types.MapType[types._K, types._ValueT], _IsValid]):
    @property
    def values(self) -> Array | None: ...
    def __len__(self) -> int: ...
    def __getitem__(
        self, i: int
    ) -> tuple[Scalar[types._K, _IsValid], types._ValueT, Any, _IsValid]: ...
    @overload
    def __iter__(
        self: Scalar[
            types.MapType[types._BasicDataType[_AsPyTypeK], types._BasicDataType[_AsPyTypeV]],
            _IsValid,
        ],
    ) -> Iterator[tuple[_AsPyTypeK, _AsPyTypeV]]: ...
    @overload
    def __iter__(
        self: Scalar[
            types.MapType[Any, types._BasicDataType[_AsPyTypeV]],
            _IsValid,
        ],
    ) -> Iterator[tuple[Any, _AsPyTypeV]]: ...
    @overload
    def __iter__(
        self: Scalar[
            types.MapType[types._BasicDataType[_AsPyTypeK], Any],
            _IsValid,
        ],
    ) -> Iterator[tuple[_AsPyTypeK, Any]]: ...

class DictionaryScalar(Scalar[types.DictionaryType[types._IndexT, types._ValueT], _IsValid]):
    @property
    def index(self) -> Scalar[types._IndexT, _IsValid]: ...
    @property
    def value(self) -> Scalar[types._ValueT, _IsValid]: ...
    @property
    def dictionary(self) -> Array: ...

class RunEndEncodedScalar(
    Scalar[types.RunEndEncodedType[types._RunEndType, types._ValueT], _IsValid]
):
    @property
    def value(self) -> tuple[int, int] | None: ...

class UnionScalar(Scalar[types.UnionType, _IsValid]):
    @property
    def value(self) -> Any | None: ...
    @property
    def type_code(self) -> str: ...

class ExtensionScalar(Scalar[types.ExtensionType, _IsValid]):
    @property
    def value(self) -> Any | None: ...
    @staticmethod
    def from_storage(typ: types.BaseExtensionType, value) -> ExtensionScalar: ...

class FixedShapeTensorScalar(ExtensionScalar[_IsValid]):
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
def scalar(  # type: ignore[overload-overlap]
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
) -> ListScalar[types.ListType[types.Decimal128Type]]: ...
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
    value: CollectionValue[_V],
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> ListScalar[Any]: ...
@overload
def scalar(
    value: _V,
    type: _DataTypeT,
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> Scalar[_DataTypeT, _V]: ...

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
    "FixedBinaryScalar",
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
    "scalar",
]
