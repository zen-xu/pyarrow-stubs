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
from typing import Any, Generic, Iterator, Literal, Mapping, overload

import numpy as np

from pyarrow._compute import CastOptions
from pyarrow.lib import Array, Buffer, MemoryPool, MonthDayNano, Tensor, _Weakrefable
from typing_extensions import Protocol, TypeVar

from . import types
from .types import (
    _AsPyType,
    _DataTypeT,
    _Time32Unit,
    _Time64Unit,
    _Tz,
    _Unit,
)

_AsPyTypeK = TypeVar("_AsPyTypeK")
_AsPyTypeV = TypeVar("_AsPyTypeV")
_DataType_co = TypeVar("_DataType_co", bound=types.DataType, covariant=True)

class Scalar(_Weakrefable, Generic[_DataType_co]):
    """
    The base class for scalars.
    """
    @property
    def type(self) -> _DataType_co:
        """
        Data type of the Scalar object.
        """
    @property
    def is_valid(self) -> bool:
        """
        Holds a valid (non-null) value.
        """
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
    def cast(self, *args, **kwargs):
        """
        Cast scalar value to another data type.

        See :func:`pyarrow.compute.cast` for usage.

        Parameters
        ----------
        target_type : DataType, default None
            Type to cast scalar to.
        safe : boolean, default True
            Whether to check for conversion errors such as overflow.
        options : CastOptions, default None
            Additional checks pass by CastOptions
        memory_pool : MemoryPool, optional
            memory pool to use for allocations during function execution.

        Returns
        -------
        scalar : A Scalar of the given target data type.
        """
    def validate(self, *, full: bool = False) -> None:
        """
        Perform validation checks.  An exception is raised if validation fails.

        By default only cheap validation checks are run.  Pass `full=True`
        for thorough validation checks (potentially O(n)).

        Parameters
        ----------
        full : bool, default False
            If True, run expensive checks, otherwise cheap checks only.

        Raises
        ------
        ArrowInvalid
        """
    def equals(self, other: Scalar) -> bool: ...
    def __hash__(self) -> int: ...
    @overload
    def as_py(
        self: Scalar[types._BasicDataType[_AsPyType]],
        *,
        maps_as_pydicts: Literal["lossy", "strict"] | None = None,
    ) -> _AsPyType: ...
    @overload
    def as_py(
        self: Scalar[types.ListType[types._BasicDataType[_AsPyType]]],
        *,
        maps_as_pydicts: Literal["lossy", "strict"] | None = None,
    ) -> list[_AsPyType]: ...
    @overload
    def as_py(
        self: Scalar[
            types.ListType[
                types.DictionaryType[types._IndexT, types._BasicDataType[_AsPyTypeV], Any]
            ]
        ],
        *,
        maps_as_pydicts: Literal["lossy", "strict"] | None = None,
    ) -> list[dict[int, _AsPyTypeV]]: ...
    @overload
    def as_py(
        self: Scalar[
            types.ListType[types.DictionaryType[Any, types._BasicDataType[_AsPyTypeV], Any]],
        ],
        *,
        maps_as_pydicts: Literal["lossy", "strict"] | None = None,
    ) -> list[dict[Any, _AsPyTypeV]]: ...
    @overload
    def as_py(
        self: Scalar[types.ListType[types.DictionaryType[types._IndexT, Any, Any]],],
        *,
        maps_as_pydicts: Literal["lossy", "strict"] | None = None,
    ) -> list[dict[int, Any]]: ...
    @overload
    def as_py(
        self: Scalar[types.StructType],
        *,
        maps_as_pydicts: Literal["lossy", "strict"] | None = None,
    ) -> list[dict[str, Any]]: ...
    @overload
    def as_py(
        self: Scalar[
            types.MapType[types._BasicDataType[_AsPyTypeK], types._BasicDataType[_AsPyTypeV]]
        ],
        *,
        maps_as_pydicts: Literal["lossy", "strict"] | None = None,
    ) -> list[tuple[_AsPyTypeK, _AsPyTypeV]]: ...
    @overload
    def as_py(
        self: Scalar[types.MapType[Any, types._BasicDataType[_AsPyTypeV]]],
        *,
        maps_as_pydicts: Literal["lossy", "strict"] | None = None,
    ) -> list[tuple[Any, _AsPyTypeV]]: ...
    @overload
    def as_py(
        self: Scalar[types.MapType[types._BasicDataType[_AsPyTypeK], Any]],
        *,
        maps_as_pydicts: Literal["lossy", "strict"] | None = None,
    ) -> list[tuple[_AsPyTypeK, Any]]: ...
    @overload
    def as_py(
        self: Scalar[Any],
        *,
        maps_as_pydicts: Literal["lossy", "strict"] | None = None,
    ) -> Any: ...
    def as_py(self, *args, **kwargs):
        """
        Return this value as a Python representation.

        Parameters
        ----------
        maps_as_pydicts : str, optional, default `None`
            Valid values are `None`, 'lossy', or 'strict'.
            The default behavior (`None`), is to convert Arrow Map arrays to
            Python association lists (list-of-tuples) in the same order as the
            Arrow Map, as in [(key1, value1), (key2, value2), ...].

            If 'lossy' or 'strict', convert Arrow Map arrays to native Python dicts.

            If 'lossy', whenever duplicate keys are detected, a warning will be printed.
            The last seen value of a duplicate key will be in the Python dictionary.
            If 'strict', this instead results in an exception being raised when detected.
        """

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

class ListScalar(Scalar[types.ListType[_DataTypeT]]):
    @property
    def values(self) -> Array | None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, i: int) -> Scalar[_DataTypeT]: ...
    def __iter__(self) -> Iterator[Array]: ...

class FixedSizeListScalar(Scalar[types.FixedSizeListType[_DataTypeT, types._Size]]):
    @property
    def values(self) -> Array | None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, i: int) -> Scalar[_DataTypeT]: ...
    def __iter__(self) -> Iterator[Array]: ...

class LargeListScalar(Scalar[types.LargeListType[_DataTypeT]]):
    @property
    def values(self) -> Array | None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, i: int) -> Scalar[_DataTypeT]: ...
    def __iter__(self) -> Iterator[Array]: ...

class ListViewScalar(Scalar[types.ListViewType[_DataTypeT]]):
    @property
    def values(self) -> Array | None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, i: int) -> Scalar[_DataTypeT]: ...
    def __iter__(self) -> Iterator[Array]: ...

class LargeListViewScalar(Scalar[types.LargeListViewType[_DataTypeT]]):
    @property
    def values(self) -> Array | None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, i: int) -> Scalar[_DataTypeT]: ...
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
    def from_storage(typ: types.BaseExtensionType, value) -> ExtensionScalar:
        """
        Construct ExtensionScalar from type and storage value.

        Parameters
        ----------
        typ : DataType
            The extension type for the result scalar.
        value : object
            The storage value for the result scalar.

        Returns
        -------
        ext_scalar : ExtensionScalar
        """

class Bool8Scalar(Scalar[types.Bool8Type]): ...
class UuidScalar(Scalar[types.UuidType]): ...
class JsonScalar(Scalar[types.JsonType]): ...
class OpaqueScalar(Scalar[types.OpaqueType]): ...

class FixedShapeTensorScalar(ExtensionScalar):
    def to_numpy(self) -> np.ndarray:
        """
        Convert fixed shape tensor scalar to a numpy.ndarray.

        The resulting ndarray's shape matches the permuted shape of the
        fixed shape tensor scalar.
        The conversion is zero-copy.

        Returns
        -------
        numpy.ndarray
        """
    def to_tensor(self) -> Tensor:
        """
        Convert fixed shape tensor extension scalar to a pyarrow.Tensor, using shape
        and strides derived from corresponding FixedShapeTensorType.

        The conversion is zero-copy.

        Returns
        -------
        pyarrow.Tensor
            Tensor represented stored in FixedShapeTensorScalar.
        """

_V = TypeVar("_V")

class NullableCollection(Protocol[_V]):  # pyright: ignore[reportInvalidTypeVarUse]
    def __iter__(self) -> Iterator[_V] | Iterator[_V | None]: ...
    def __len__(self) -> int: ...
    def __contains__(self, item: Any, /) -> bool: ...

@overload
def scalar(
    value: str,
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> StringScalar: ...
@overload
def scalar(
    value: bytes,
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> BinaryScalar: ...
@overload
def scalar(  # pyright: ignore[reportOverlappingOverload]
    value: bool,
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> BooleanScalar: ...
@overload
def scalar(
    value: int,
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> Int64Scalar: ...
@overload
def scalar(
    value: float,
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> DoubleScalar: ...
@overload
def scalar(
    value: Decimal,
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> Decimal128Scalar: ...
@overload
def scalar(  # pyright: ignore[reportOverlappingOverload]
    value: dt.datetime,
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> TimestampScalar[Literal["us"]]: ...
@overload
def scalar(
    value: dt.date,
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> Date32Scalar: ...
@overload
def scalar(
    value: dt.time,
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> Time64Scalar[Literal["us"]]: ...
@overload
def scalar(
    value: dt.timedelta,
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> DurationScalar[Literal["us"]]: ...
@overload
def scalar(  # pyright: ignore[reportOverlappingOverload]
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
    value: NullableCollection[str],
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> ListScalar[types.ListType[types.StringType]]: ...
@overload
def scalar(
    value: NullableCollection[bytes],
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> ListScalar[types.ListType[types.BinaryType]]: ...
@overload
def scalar(
    value: NullableCollection[bool],
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> ListScalar[types.ListType[types.BoolType]]: ...
@overload
def scalar(
    value: NullableCollection[int],
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> ListScalar[types.ListType[types.Int64Type]]: ...
@overload
def scalar(
    value: NullableCollection[float],
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> ListScalar[types.ListType[types.Float64Type]]: ...
@overload
def scalar(
    value: NullableCollection[Decimal],
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> ListScalar[types.ListType[types.Decimal32Type]]: ...
@overload
def scalar(
    value: NullableCollection[dt.datetime],
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> ListScalar[types.ListType[types.TimestampType[Literal["us"]]]]: ...
@overload
def scalar(
    value: NullableCollection[dt.date],
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> ListScalar[types.ListType[types.Date32Type]]: ...
@overload
def scalar(
    value: NullableCollection[dt.time],
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> ListScalar[types.ListType[types.Time64Type[Literal["us"]]]]: ...
@overload
def scalar(
    value: NullableCollection[dt.timedelta],
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> ListScalar[types.ListType[types.DurationType[Literal["us"]]]]: ...
@overload
def scalar(
    value: NullableCollection[MonthDayNano],
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> ListScalar[types.ListType[types.MonthDayNanoIntervalType]]: ...
@overload
def scalar(
    value: NullableCollection[Any],
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
    type: types.FixedSizeListType[_DataTypeT, types._Size],
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> FixedSizeListScalar[_DataTypeT, types._Size]: ...
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
    type: types.StructType,
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
    type: types.RunEndEncodedType[types._RunEndType, types._BasicValueT],
    *,
    from_pandas: bool | None = None,
    memory_pool: MemoryPool | None = None,
) -> RunEndEncodedScalar[types._RunEndType, types._BasicValueT]: ...
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
def scalar(*args, **kwargs):
    """
    Create a pyarrow.Scalar instance from a Python object.

    Parameters
    ----------
    value : Any
        Python object coercible to arrow's type system.
    type : pyarrow.DataType
        Explicit type to attempt to coerce to, otherwise will be inferred from
        the value.
    from_pandas : bool, default None
        Use pandas's semantics for inferring nulls from values in
        ndarray-like data. Defaults to False if not passed explicitly by user,
        or True if a pandas object is passed in.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the currently-set default
        memory pool.

    Returns
    -------
    scalar : pyarrow.Scalar

    Examples
    --------
    >>> import pyarrow as pa

    >>> pa.scalar(42)
    <pyarrow.Int64Scalar: 42>

    >>> pa.scalar("string")
    <pyarrow.StringScalar: 'string'>

    >>> pa.scalar([1, 2])
    <pyarrow.ListScalar: [1, 2]>

    >>> pa.scalar([1, 2], type=pa.list_(pa.int16()))
    <pyarrow.ListScalar: [1, 2]>
    """

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
