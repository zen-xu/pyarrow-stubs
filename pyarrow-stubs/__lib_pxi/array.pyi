import datetime as dt
import sys

from collections.abc import Callable
from decimal import Decimal

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self
from typing import (
    Any,
    Generic,
    Iterable,
    Iterator,
    Literal,
    TypeVar,
    overload,
)

import numpy as np
import pandas as pd

from pandas.core.dtypes.base import ExtensionDtype
from pyarrow._compute import CastOptions
from pyarrow._stubs_typing import (
    ArrayLike,
    Indices,
    Mask,
    Order,
    SupportArrowArray,
    SupportArrowDeviceArray,
)
from pyarrow.lib import (
    Buffer,
    Device,
    MemoryManager,
    MemoryPool,
    MonthDayNano,
    Tensor,
    _Weakrefable,
)
from typing_extensions import deprecated

from . import scalar, types
from .device import DeviceAllocationType
from .scalar import NullableCollection, Scalar
from .types import (
    DataType,
    Field,
    MapType,
    _AsPyType,
    _BasicDataType,
    _BasicValueT,
    _DataTypeT,
    _IndexT,
    _RunEndType,
    _Size,
)

@overload
def array(
    values: NullableCollection[bool],
    type: None = None,
    mask: Mask | None = None,
    size: int | None = None,
    from_pandas: bool | None = None,
    safe: bool = True,
    memory_pool: MemoryPool | None = None,
) -> BooleanArray: ...
@overload
def array(
    values: NullableCollection[int],
    type: None = None,
    mask: Mask | None = None,
    size: int | None = None,
    from_pandas: bool | None = None,
    safe: bool = True,
    memory_pool: MemoryPool | None = None,
) -> Int64Array: ...
@overload
def array(
    values: NullableCollection[float],
    type: None = None,
    mask: Mask | None = None,
    size: int | None = None,
    from_pandas: bool | None = None,
    safe: bool = True,
    memory_pool: MemoryPool | None = None,
) -> DoubleArray: ...
@overload
def array(
    values: NullableCollection[Decimal],
    type: None = None,
    mask: Mask | None = None,
    size: int | None = None,
    from_pandas: bool | None = None,
    safe: bool = True,
    memory_pool: MemoryPool | None = None,
) -> Decimal128Array: ...
@overload
def array(
    values: NullableCollection[dict[str, Any]],
    type: None = None,
    mask: Mask | None = None,
    size: int | None = None,
    from_pandas: bool | None = None,
    safe: bool = True,
    memory_pool: MemoryPool | None = None,
) -> StructArray: ...
@overload
def array(
    values: NullableCollection[dt.date],
    type: None = None,
    mask: Mask | None = None,
    size: int | None = None,
    from_pandas: bool | None = None,
    safe: bool = True,
    memory_pool: MemoryPool | None = None,
) -> Date32Array: ...
@overload
def array(
    values: NullableCollection[dt.time],
    type: None = None,
    mask: Mask | None = None,
    size: int | None = None,
    from_pandas: bool | None = None,
    safe: bool = True,
    memory_pool: MemoryPool | None = None,
) -> Time64Array[Literal["us"]]: ...
@overload
def array(
    values: NullableCollection[dt.timedelta],
    type: None = None,
    mask: Mask | None = None,
    size: int | None = None,
    from_pandas: bool | None = None,
    safe: bool = True,
    memory_pool: MemoryPool | None = None,
) -> DurationArray[Literal["us"]]: ...
@overload
def array(
    values: NullableCollection[MonthDayNano],
    type: None = None,
    mask: Mask | None = None,
    size: int | None = None,
    from_pandas: bool | None = None,
    safe: bool = True,
    memory_pool: MemoryPool | None = None,
) -> MonthDayNanoIntervalArray: ...
@overload
def array(
    values: NullableCollection[str],
    type: None = None,
    mask: Mask | None = None,
    size: int | None = None,
    from_pandas: bool | None = None,
    safe: bool = True,
    memory_pool: MemoryPool | None = None,
) -> StringArray: ...
@overload
def array(
    values: NullableCollection[bytes],
    type: None = None,
    mask: Mask | None = None,
    size: int | None = None,
    from_pandas: bool | None = None,
    safe: bool = True,
    memory_pool: MemoryPool | None = None,
) -> BinaryArray: ...
@overload
def array(
    values: NullableCollection[list[Any]],
    type: None = None,
    mask: Mask | None = None,
    size: int | None = None,
    from_pandas: bool | None = None,
    safe: bool = True,
    memory_pool: MemoryPool | None = None,
) -> ListArray[Any]: ...
@overload
def array(
    values: NullableCollection[_ScalarT],
    type: None = None,
    mask: Mask | None = None,
    size: int | None = None,
    from_pandas: bool | None = None,
    safe: bool = True,
    memory_pool: MemoryPool | None = None,
) -> Array[_ScalarT]: ...
@overload
def array(
    values: Iterable[Any] | SupportArrowArray | SupportArrowDeviceArray,
    type: Literal["null"] | types.NullType,
    mask: Mask | None = None,
    size: int | None = None,
    from_pandas: bool | None = None,
    safe: bool = True,
    memory_pool: MemoryPool | None = None,
) -> NullArray: ...
@overload
def array(
    values: Iterable[Any] | SupportArrowArray | SupportArrowDeviceArray,
    type: Literal["bool", "boolean"] | types.BoolType,
    mask: Mask | None = None,
    size: int | None = None,
    from_pandas: bool | None = None,
    safe: bool = True,
    memory_pool: MemoryPool | None = None,
) -> BooleanArray: ...
@overload
def array(
    values: Iterable[Any] | SupportArrowArray | SupportArrowDeviceArray,
    type: Literal["i1", "int8"] | types.Int8Type,
    mask: Mask | None = None,
    size: int | None = None,
    from_pandas: bool | None = None,
    safe: bool = True,
    memory_pool: MemoryPool | None = None,
) -> Int8Array: ...
@overload
def array(
    values: Iterable[Any] | SupportArrowArray | SupportArrowDeviceArray,
    type: Literal["i2", "int16"] | types.Int16Type,
    mask: Mask | None = None,
    size: int | None = None,
    from_pandas: bool | None = None,
    safe: bool = True,
    memory_pool: MemoryPool | None = None,
) -> Int16Array: ...
@overload
def array(
    values: Iterable[Any] | SupportArrowArray | SupportArrowDeviceArray,
    type: Literal["i4", "int32"] | types.Int32Type,
    mask: Mask | None = None,
    size: int | None = None,
    from_pandas: bool | None = None,
    safe: bool = True,
    memory_pool: MemoryPool | None = None,
) -> Int32Array: ...
@overload
def array(
    values: Iterable[Any] | SupportArrowArray | SupportArrowDeviceArray,
    type: Literal["i8", "int64"] | types.Int64Type,
    mask: Mask | None = None,
    size: int | None = None,
    from_pandas: bool | None = None,
    safe: bool = True,
    memory_pool: MemoryPool | None = None,
) -> Int64Array: ...
@overload
def array(
    values: Iterable[Any] | SupportArrowArray | SupportArrowDeviceArray,
    type: Literal["u1", "uint8"] | types.UInt8Type,
    mask: Mask | None = None,
    size: int | None = None,
    from_pandas: bool | None = None,
    safe: bool = True,
    memory_pool: MemoryPool | None = None,
) -> UInt8Array: ...
@overload
def array(
    values: Iterable[Any] | SupportArrowArray | SupportArrowDeviceArray,
    type: Literal["u2", "uint16"] | types.UInt16Type,
    mask: Mask | None = None,
    size: int | None = None,
    from_pandas: bool | None = None,
    safe: bool = True,
    memory_pool: MemoryPool | None = None,
) -> UInt16Array: ...
@overload
def array(
    values: Iterable[Any] | SupportArrowArray | SupportArrowDeviceArray,
    type: Literal["u4", "uint32"] | types.Uint32Type,
    mask: Mask | None = None,
    size: int | None = None,
    from_pandas: bool | None = None,
    safe: bool = True,
    memory_pool: MemoryPool | None = None,
) -> UInt32Array: ...
@overload
def array(
    values: Iterable[Any] | SupportArrowArray | SupportArrowDeviceArray,
    type: Literal["u8", "uint64"] | types.UInt64Type,
    mask: Mask | None = None,
    size: int | None = None,
    from_pandas: bool | None = None,
    safe: bool = True,
    memory_pool: MemoryPool | None = None,
) -> UInt64Array: ...
@overload
def array(
    values: Iterable[Any] | SupportArrowArray | SupportArrowDeviceArray,
    type: Literal["f2", "halffloat", "float16"] | types.Float16Type,
    mask: Mask | None = None,
    size: int | None = None,
    from_pandas: bool | None = None,
    safe: bool = True,
    memory_pool: MemoryPool | None = None,
) -> HalfFloatArray: ...
@overload
def array(
    values: Iterable[Any] | SupportArrowArray | SupportArrowDeviceArray,
    type: Literal["f4", "float", "float32"] | types.Float32Type,
    mask: Mask | None = None,
    size: int | None = None,
    from_pandas: bool | None = None,
    safe: bool = True,
    memory_pool: MemoryPool | None = None,
) -> FloatArray: ...
@overload
def array(
    values: Iterable[Any] | SupportArrowArray | SupportArrowDeviceArray,
    type: Literal["f8", "double", "float64"] | types.Float64Type,
    mask: Mask | None = None,
    size: int | None = None,
    from_pandas: bool | None = None,
    safe: bool = True,
    memory_pool: MemoryPool | None = None,
) -> DoubleArray: ...
@overload
def array(
    values: Iterable[Any] | SupportArrowArray | SupportArrowDeviceArray,
    type: Literal["string", "str", "utf8"] | types.StringType,
    mask: Mask | None = None,
    size: int | None = None,
    from_pandas: bool | None = None,
    safe: bool = True,
    memory_pool: MemoryPool | None = None,
) -> StringArray: ...
@overload
def array(
    values: Iterable[Any] | SupportArrowArray | SupportArrowDeviceArray,
    type: Literal["binary"] | types.BinaryType,
    mask: Mask | None = None,
    size: int | None = None,
    from_pandas: bool | None = None,
    safe: bool = True,
    memory_pool: MemoryPool | None = None,
) -> BinaryArray: ...
@overload
def array(
    values: Iterable[Any] | SupportArrowArray | SupportArrowDeviceArray,
    type: Literal["large_string", "large_str", "large_utf8"] | types.LargeStringType,
    mask: Mask | None = None,
    size: int | None = None,
    from_pandas: bool | None = None,
    safe: bool = True,
    memory_pool: MemoryPool | None = None,
) -> LargeStringArray: ...
@overload
def array(
    values: Iterable[Any] | SupportArrowArray | SupportArrowDeviceArray,
    type: Literal["large_binary"] | types.LargeBinaryType,
    mask: Mask | None = None,
    size: int | None = None,
    from_pandas: bool | None = None,
    safe: bool = True,
    memory_pool: MemoryPool | None = None,
) -> LargeBinaryArray: ...
@overload
def array(
    values: Iterable[Any] | SupportArrowArray | SupportArrowDeviceArray,
    type: Literal["binary_view"] | types.BinaryViewType,
    mask: Mask | None = None,
    size: int | None = None,
    from_pandas: bool | None = None,
    safe: bool = True,
    memory_pool: MemoryPool | None = None,
) -> BinaryViewArray: ...
@overload
def array(
    values: Iterable[Any] | SupportArrowArray | SupportArrowDeviceArray,
    type: Literal["string_view"] | types.StringViewType,
    mask: Mask | None = None,
    size: int | None = None,
    from_pandas: bool | None = None,
    safe: bool = True,
    memory_pool: MemoryPool | None = None,
) -> StringViewArray: ...
@overload
def array(
    values: Iterable[Any] | SupportArrowArray | SupportArrowDeviceArray,
    type: Literal["date32", "date32[day]"] | types.Date32Type,
    mask: Mask | None = None,
    size: int | None = None,
    from_pandas: bool | None = None,
    safe: bool = True,
    memory_pool: MemoryPool | None = None,
) -> Date32Array: ...
@overload
def array(
    values: Iterable[Any] | SupportArrowArray | SupportArrowDeviceArray,
    type: Literal["date64", "date64[ms]"] | types.Date64Type,
    mask: Mask | None = None,
    size: int | None = None,
    from_pandas: bool | None = None,
    safe: bool = True,
    memory_pool: MemoryPool | None = None,
) -> Date64Array: ...
@overload
def array(
    values: Iterable[Any] | SupportArrowArray | SupportArrowDeviceArray,
    type: Literal["time32[s]"] | types.Time32Type[Literal["s"]],
    mask: Mask | None = None,
    size: int | None = None,
    from_pandas: bool | None = None,
    safe: bool = True,
    memory_pool: MemoryPool | None = None,
) -> Time32Array[Literal["s"]]: ...
@overload
def array(
    values: Iterable[Any] | SupportArrowArray | SupportArrowDeviceArray,
    type: Literal["time32[ms]"] | types.Time32Type[Literal["ms"]],
    mask: Mask | None = None,
    size: int | None = None,
    from_pandas: bool | None = None,
    safe: bool = True,
    memory_pool: MemoryPool | None = None,
) -> Time32Array[Literal["ms"]]: ...
@overload
def array(
    values: Iterable[Any] | SupportArrowArray | SupportArrowDeviceArray,
    type: Literal["time64[us]"] | types.Time64Type[Literal["us"]],
    mask: Mask | None = None,
    size: int | None = None,
    from_pandas: bool | None = None,
    safe: bool = True,
    memory_pool: MemoryPool | None = None,
) -> Time64Array[Literal["us"]]: ...
@overload
def array(
    values: Iterable[Any] | SupportArrowArray | SupportArrowDeviceArray,
    type: Literal["time64[ns]"] | types.Time64Type[Literal["ns"]],
    mask: Mask | None = None,
    size: int | None = None,
    from_pandas: bool | None = None,
    safe: bool = True,
    memory_pool: MemoryPool | None = None,
) -> Time64Array[Literal["ns"]]: ...
@overload
def array(
    values: Iterable[Any] | SupportArrowArray | SupportArrowDeviceArray,
    type: Literal["timestamp[s]"] | types.TimestampType[Literal["s"]],
    mask: Mask | None = None,
    size: int | None = None,
    from_pandas: bool | None = None,
    safe: bool = True,
    memory_pool: MemoryPool | None = None,
) -> TimestampArray[Literal["s"]]: ...
@overload
def array(
    values: Iterable[Any] | SupportArrowArray | SupportArrowDeviceArray,
    type: Literal["timestamp[ms]"] | types.TimestampType[Literal["ms"]],
    mask: Mask | None = None,
    size: int | None = None,
    from_pandas: bool | None = None,
    safe: bool = True,
    memory_pool: MemoryPool | None = None,
) -> TimestampArray[Literal["ms"]]: ...
@overload
def array(
    values: Iterable[Any] | SupportArrowArray | SupportArrowDeviceArray,
    type: Literal["timestamp[us]"] | types.TimestampType[Literal["us"]],
    mask: Mask | None = None,
    size: int | None = None,
    from_pandas: bool | None = None,
    safe: bool = True,
    memory_pool: MemoryPool | None = None,
) -> TimestampArray[Literal["us"]]: ...
@overload
def array(
    values: Iterable[Any] | SupportArrowArray | SupportArrowDeviceArray,
    type: Literal["duration[s]"] | types.DurationType[Literal["s"]],
    mask: Mask | None = None,
    size: int | None = None,
    from_pandas: bool | None = None,
    safe: bool = True,
    memory_pool: MemoryPool | None = None,
) -> DurationArray[Literal["s"]]: ...
@overload
def array(
    values: Iterable[Any] | SupportArrowArray | SupportArrowDeviceArray,
    type: Literal["duration[ms]"] | types.DurationType[Literal["ms"]],
    mask: Mask | None = None,
    size: int | None = None,
    from_pandas: bool | None = None,
    safe: bool = True,
    memory_pool: MemoryPool | None = None,
) -> DurationArray[Literal["ms"]]: ...
@overload
def array(
    values: Iterable[Any] | SupportArrowArray | SupportArrowDeviceArray,
    type: Literal["duration[us]"] | types.DurationType[Literal["us"]],
    mask: Mask | None = None,
    size: int | None = None,
    from_pandas: bool | None = None,
    safe: bool = True,
    memory_pool: MemoryPool | None = None,
) -> DurationArray[Literal["us"]]: ...
@overload
def array(
    values: Iterable[Any] | SupportArrowArray | SupportArrowDeviceArray,
    type: Literal["duration[ns]"] | types.DurationType[Literal["ns"]],
    mask: Mask | None = None,
    size: int | None = None,
    from_pandas: bool | None = None,
    safe: bool = True,
    memory_pool: MemoryPool | None = None,
) -> DurationArray[Literal["ns"]]: ...
@overload
def array(
    values: Iterable[Any] | SupportArrowArray | SupportArrowDeviceArray,
    type: Literal["month_day_nano_interval"] | types.MonthDayNanoIntervalType,
    mask: Mask | None = None,
    size: int | None = None,
    from_pandas: bool | None = None,
    safe: bool = True,
    memory_pool: MemoryPool | None = None,
) -> MonthDayNanoIntervalArray: ...
@overload
def array(
    values: Iterable[Any] | SupportArrowArray | SupportArrowDeviceArray,
    type: _DataTypeT,
    mask: Mask | None = None,
    size: int | None = None,
    from_pandas: bool | None = None,
    safe: bool = True,
    memory_pool: MemoryPool | None = None,
) -> Array[Scalar[_DataTypeT]]: ...
def array(*args, **kawrgs):
    """
    Create pyarrow.Array instance from a Python object.

    Parameters
    ----------
    obj : sequence, iterable, ndarray, pandas.Series, Arrow-compatible array
        If both type and size are specified may be a single use iterable. If
        not strongly-typed, Arrow type will be inferred for resulting array.
        Any Arrow-compatible array that implements the Arrow PyCapsule Protocol
        (has an ``__arrow_c_array__`` or ``__arrow_c_device_array__`` method)
        can be passed as well.
    type : pyarrow.DataType
        Explicit type to attempt to coerce to, otherwise will be inferred from
        the data.
    mask : array[bool], optional
        Indicate which values are null (True) or not null (False).
    size : int64, optional
        Size of the elements. If the input is larger than size bail at this
        length. For iterators, if size is larger than the input iterator this
        will be treated as a "max size", but will involve an initial allocation
        of size followed by a resize to the actual size (so if you know the
        exact size specifying it correctly will give you better performance).
    from_pandas : bool, default None
        Use pandas's semantics for inferring nulls from values in
        ndarray-like data. If passed, the mask tasks precedence, but
        if a value is unmasked (not-null), but still null according to
        pandas semantics, then it is null. Defaults to False if not
        passed explicitly by user, or True if a pandas object is
        passed in.
    safe : bool, default True
        Check for overflows or other unsafe conversions.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the currently-set default
        memory pool.

    Returns
    -------
    array : pyarrow.Array or pyarrow.ChunkedArray
        A ChunkedArray instead of an Array is returned if:

        - the object data overflowed binary storage.
        - the object's ``__arrow_array__`` protocol method returned a chunked
          array.

    Notes
    -----
    Timezone will be preserved in the returned array for timezone-aware data,
    else no timezone will be returned for naive timestamps.
    Internally, UTC values are stored for timezone-aware data with the
    timezone set in the data type.

    Pandas's DateOffsets and dateutil.relativedelta.relativedelta are by
    default converted as MonthDayNanoIntervalArray. relativedelta leapdays
    are ignored as are all absolute fields on both objects. datetime.timedelta
    can also be converted to MonthDayNanoIntervalArray but this requires
    passing MonthDayNanoIntervalType explicitly.

    Converting to dictionary array will promote to a wider integer type for
    indices if the number of distinct values cannot be represented, even if
    the index type was explicitly set. This means that if there are more than
    127 values the returned dictionary array's index type will be at least
    pa.int16() even if pa.int8() was passed to the function. Note that an
    explicit index type will not be demoted even if it is wider than required.

    Examples
    --------
    >>> import pandas as pd
    >>> import pyarrow as pa
    >>> pa.array(pd.Series([1, 2]))
    <pyarrow.lib.Int64Array object at ...>
    [
      1,
      2
    ]

    >>> pa.array(["a", "b", "a"], type=pa.dictionary(pa.int8(), pa.string()))
    <pyarrow.lib.DictionaryArray object at ...>
    ...
    -- dictionary:
      [
        "a",
        "b"
      ]
    -- indices:
      [
        0,
        1,
        0
      ]

    >>> import numpy as np
    >>> pa.array(pd.Series([1, 2]), mask=np.array([0, 1], dtype=bool))
    <pyarrow.lib.Int64Array object at ...>
    [
      1,
      null
    ]

    >>> arr = pa.array(range(1024), type=pa.dictionary(pa.int8(), pa.int64()))
    >>> arr.type.index_type
    DataType(int16)
    """

@overload
def asarray(values: NullableCollection[bool]) -> BooleanArray: ...
@overload
def asarray(values: NullableCollection[int]) -> Int64Array: ...
@overload
def asarray(values: NullableCollection[float]) -> DoubleArray: ...
@overload
def asarray(values: NullableCollection[Decimal]) -> Decimal128Array: ...
@overload
def asarray(values: NullableCollection[dict[str, Any]]) -> StructArray: ...
@overload
def asarray(values: NullableCollection[dt.date]) -> Date32Array: ...
@overload
def asarray(values: NullableCollection[dt.time]) -> Time64Array: ...
@overload
def asarray(values: NullableCollection[dt.timedelta]) -> DurationArray: ...
@overload
def asarray(values: NullableCollection[MonthDayNano]) -> MonthDayNanoIntervalArray: ...
@overload
def asarray(values: NullableCollection[list[Any]]) -> ListArray[Any]: ...
@overload
def asarray(
    values: Iterable[Any] | SupportArrowArray | SupportArrowDeviceArray,
    type: Literal["null"] | types.NullType,
) -> NullArray: ...
@overload
def asarray(
    values: Iterable[Any] | SupportArrowArray | SupportArrowDeviceArray,
    type: Literal["bool", "boolean"] | types.BoolType,
) -> BooleanArray: ...
@overload
def asarray(
    values: Iterable[Any] | SupportArrowArray | SupportArrowDeviceArray,
    type: Literal["i1", "int8"] | types.Int8Type,
) -> Int8Array: ...
@overload
def asarray(
    values: Iterable[Any] | SupportArrowArray | SupportArrowDeviceArray,
    type: Literal["i2", "int16"] | types.Int16Type,
) -> Int16Array: ...
@overload
def asarray(
    values: Iterable[Any] | SupportArrowArray | SupportArrowDeviceArray,
    type: Literal["i4", "int32"] | types.Int32Type,
) -> Int32Array: ...
@overload
def asarray(
    values: Iterable[Any] | SupportArrowArray | SupportArrowDeviceArray,
    type: Literal["i8", "int64"] | types.Int64Type,
) -> Int64Array: ...
@overload
def asarray(
    values: Iterable[Any] | SupportArrowArray | SupportArrowDeviceArray,
    type: Literal["u1", "uint8"] | types.UInt8Type,
) -> UInt8Array: ...
@overload
def asarray(
    values: Iterable[Any] | SupportArrowArray | SupportArrowDeviceArray,
    type: Literal["u2", "uint16"] | types.UInt16Type,
) -> UInt16Array: ...
@overload
def asarray(
    values: Iterable[Any] | SupportArrowArray | SupportArrowDeviceArray,
    type: Literal["u4", "uint32"] | types.Uint32Type,
) -> UInt32Array: ...
@overload
def asarray(
    values: Iterable[Any] | SupportArrowArray | SupportArrowDeviceArray,
    type: Literal["u8", "uint64"] | types.UInt64Type,
) -> UInt64Array: ...
@overload
def asarray(
    values: Iterable[Any] | SupportArrowArray | SupportArrowDeviceArray,
    type: Literal["f2", "halffloat", "float16"] | types.Float16Type,
) -> HalfFloatArray: ...
@overload
def asarray(
    values: Iterable[Any] | SupportArrowArray | SupportArrowDeviceArray,
    type: Literal["f4", "float", "float32"] | types.Float32Type,
) -> FloatArray: ...
@overload
def asarray(
    values: Iterable[Any] | SupportArrowArray | SupportArrowDeviceArray,
    type: Literal["f8", "double", "float64"] | types.Float64Type,
) -> DoubleArray: ...
@overload
def asarray(
    values: Iterable[Any] | SupportArrowArray | SupportArrowDeviceArray,
    type: Literal["string", "str", "utf8"] | types.StringType,
) -> StringArray: ...
@overload
def asarray(
    values: Iterable[Any] | SupportArrowArray | SupportArrowDeviceArray,
    type: Literal["binary"] | types.BinaryType,
) -> BinaryArray: ...
@overload
def asarray(
    values: Iterable[Any] | SupportArrowArray | SupportArrowDeviceArray,
    type: Literal["large_string", "large_str", "large_utf8"] | types.LargeStringType,
) -> LargeStringArray: ...
@overload
def asarray(
    values: Iterable[Any] | SupportArrowArray | SupportArrowDeviceArray,
    type: Literal["large_binary"] | types.LargeBinaryType,
) -> LargeBinaryArray: ...
@overload
def asarray(
    values: Iterable[Any] | SupportArrowArray | SupportArrowDeviceArray,
    type: Literal["binary_view"] | types.BinaryViewType,
) -> BinaryViewArray: ...
@overload
def asarray(
    values: Iterable[Any] | SupportArrowArray | SupportArrowDeviceArray,
    type: Literal["string_view"] | types.StringViewType,
) -> StringViewArray: ...
@overload
def asarray(
    values: Iterable[Any] | SupportArrowArray | SupportArrowDeviceArray,
    type: Literal["date32", "date32[day]"] | types.Date32Type,
) -> Date32Array: ...
@overload
def asarray(
    values: Iterable[Any] | SupportArrowArray | SupportArrowDeviceArray,
    type: Literal["date64", "date64[ms]"] | types.Date64Type,
) -> Date64Array: ...
@overload
def asarray(
    values: Iterable[Any] | SupportArrowArray | SupportArrowDeviceArray,
    type: Literal["time32[s]"] | types.Time32Type[Literal["s"]],
) -> Time32Array[Literal["s"]]: ...
@overload
def asarray(
    values: Iterable[Any] | SupportArrowArray | SupportArrowDeviceArray,
    type: Literal["time32[ms]"] | types.Time32Type[Literal["ms"]],
) -> Time32Array[Literal["ms"]]: ...
@overload
def asarray(
    values: Iterable[Any] | SupportArrowArray | SupportArrowDeviceArray,
    type: Literal["time64[us]"] | types.Time64Type[Literal["us"]],
) -> Time64Array[Literal["us"]]: ...
@overload
def asarray(
    values: Iterable[Any] | SupportArrowArray | SupportArrowDeviceArray,
    type: Literal["time64[ns]"] | types.Time64Type[Literal["ns"]],
) -> Time64Array[Literal["ns"]]: ...
@overload
def asarray(
    values: Iterable[Any] | SupportArrowArray | SupportArrowDeviceArray,
    type: Literal["timestamp[s]"] | types.TimestampType[Literal["s"]],
) -> TimestampArray[Literal["s"]]: ...
@overload
def asarray(
    values: Iterable[Any] | SupportArrowArray | SupportArrowDeviceArray,
    type: Literal["timestamp[ms]"] | types.TimestampType[Literal["ms"]],
) -> TimestampArray[Literal["ms"]]: ...
@overload
def asarray(
    values: Iterable[Any] | SupportArrowArray | SupportArrowDeviceArray,
    type: Literal["timestamp[us]"] | types.TimestampType[Literal["us"]],
) -> TimestampArray[Literal["us"]]: ...
@overload
def asarray(
    values: Iterable[Any] | SupportArrowArray | SupportArrowDeviceArray,
    type: Literal["timestamp[ns]"] | types.TimestampType[Literal["ns"]],
) -> TimestampArray[Literal["ns"]]: ...
@overload
def asarray(
    values: Iterable[Any] | SupportArrowArray | SupportArrowDeviceArray,
    type: Literal["duration[s]"] | types.DurationType[Literal["s"]],
) -> DurationArray[Literal["s"]]: ...
@overload
def asarray(
    values: Iterable[Any] | SupportArrowArray | SupportArrowDeviceArray,
    type: Literal["duration[ms]"] | types.DurationType[Literal["ms"]],
) -> DurationArray[Literal["ms"]]: ...
@overload
def asarray(
    values: Iterable[Any] | SupportArrowArray | SupportArrowDeviceArray,
    type: Literal["duration[us]"] | types.DurationType[Literal["us"]],
) -> DurationArray[Literal["us"]]: ...
@overload
def asarray(
    values: Iterable[Any] | SupportArrowArray | SupportArrowDeviceArray,
    type: Literal["duration[ns]"] | types.DurationType[Literal["ns"]],
) -> DurationArray[Literal["ns"]]: ...
@overload
def asarray(
    values: Iterable[Any] | SupportArrowArray | SupportArrowDeviceArray,
    type: Literal["month_day_nano_interval"] | types.MonthDayNanoIntervalType,
) -> MonthDayNanoIntervalArray: ...
@overload
def asarray(
    values: Iterable[Any] | SupportArrowArray | SupportArrowDeviceArray,
    type: _DataTypeT,
) -> Array[Scalar[_DataTypeT]]: ...
def asarray(*args, **kwargs):
    """
    Convert to pyarrow.Array, inferring type if not provided.

    Parameters
    ----------
    values : array-like
        This can be a sequence, numpy.ndarray, pyarrow.Array or
        pyarrow.ChunkedArray. If a ChunkedArray is passed, the output will be
        a ChunkedArray, otherwise the output will be a Array.
    type : string or DataType
        Explicitly construct the array with this type. Attempt to cast if
        indicated type is different.

    Returns
    -------
    arr : Array or ChunkedArray
    """

@overload
def nulls(size: int, memory_pool: MemoryPool | None = None) -> NullArray: ...
@overload
def nulls(
    size: int, type: types.NullType | None, memory_pool: MemoryPool | None = None
) -> NullArray: ...
@overload
def nulls(
    size: int, type: types.BoolType, memory_pool: MemoryPool | None = None
) -> BooleanArray: ...
@overload
def nulls(size: int, type: types.Int8Type, memory_pool: MemoryPool | None = None) -> Int8Array: ...
@overload
def nulls(
    size: int, type: types.Int16Type, memory_pool: MemoryPool | None = None
) -> Int16Array: ...
@overload
def nulls(
    size: int, type: types.Int32Type, memory_pool: MemoryPool | None = None
) -> Int32Array: ...
@overload
def nulls(
    size: int, type: types.Int64Type, memory_pool: MemoryPool | None = None
) -> Int64Array: ...
@overload
def nulls(
    size: int, type: types.UInt8Type, memory_pool: MemoryPool | None = None
) -> UInt8Array: ...
@overload
def nulls(
    size: int, type: types.UInt16Type, memory_pool: MemoryPool | None = None
) -> UInt16Array: ...
@overload
def nulls(
    size: int, type: types.Uint32Type, memory_pool: MemoryPool | None = None
) -> UInt32Array: ...
@overload
def nulls(
    size: int, type: types.UInt64Type, memory_pool: MemoryPool | None = None
) -> UInt64Array: ...
@overload
def nulls(
    size: int, type: types.Float16Type, memory_pool: MemoryPool | None = None
) -> HalfFloatArray: ...
@overload
def nulls(
    size: int, type: types.Float32Type, memory_pool: MemoryPool | None = None
) -> FloatArray: ...
@overload
def nulls(
    size: int, type: types.Float64Type, memory_pool: MemoryPool | None = None
) -> DoubleArray: ...
@overload
def nulls(
    size: int, type: types.Decimal32Type, memory_pool: MemoryPool | None = None
) -> Decimal128Array: ...
@overload
def nulls(
    size: int, type: types.Decimal64Type, memory_pool: MemoryPool | None = None
) -> Decimal128Array: ...
@overload
def nulls(
    size: int, type: types.Decimal128Type, memory_pool: MemoryPool | None = None
) -> Decimal128Array: ...
@overload
def nulls(
    size: int, type: types.Decimal256Type, memory_pool: MemoryPool | None = None
) -> Decimal256Array: ...
@overload
def nulls(
    size: int, type: types.Date32Type, memory_pool: MemoryPool | None = None
) -> Date32Array: ...
@overload
def nulls(
    size: int, type: types.Date64Type, memory_pool: MemoryPool | None = None
) -> Date64Array: ...
@overload
def nulls(
    size: int, type: types.Time32Type[types._Time32Unit], memory_pool: MemoryPool | None = None
) -> Time32Array[types._Time32Unit]: ...
@overload
def nulls(
    size: int, type: types.Time64Type[types._Time64Unit], memory_pool: MemoryPool | None = None
) -> Time64Array[types._Time64Unit]: ...
@overload
def nulls(
    size: int,
    type: types.TimestampType[types._Unit, types._Tz],
    memory_pool: MemoryPool | None = None,
) -> TimestampArray[types._Unit, types._Tz]: ...
@overload
def nulls(
    size: int, type: types.DurationType[types._Unit], memory_pool: MemoryPool | None = None
) -> DurationArray[types._Unit]: ...
@overload
def nulls(
    size: int, type: types.MonthDayNanoIntervalType, memory_pool: MemoryPool | None = None
) -> MonthDayNanoIntervalArray: ...
@overload
def nulls(
    size: int,
    type: types.BinaryType,
    memory_pool: MemoryPool | None = None,
) -> BinaryArray: ...
@overload
def nulls(
    size: int,
    type: types.LargeBinaryType,
    memory_pool: MemoryPool | None = None,
) -> LargeBinaryArray: ...
@overload
def nulls(
    size: int,
    type: types.FixedSizeBinaryType,
    memory_pool: MemoryPool | None = None,
) -> FixedSizeBinaryArray: ...
@overload
def nulls(
    size: int,
    type: types.StringType,
    memory_pool: MemoryPool | None = None,
) -> StringArray: ...
@overload
def nulls(
    size: int,
    type: types.LargeStringType,
    memory_pool: MemoryPool | None = None,
) -> LargeStringArray: ...
@overload
def nulls(
    size: int,
    type: types.BinaryViewType,
    memory_pool: MemoryPool | None = None,
) -> BinaryViewArray: ...
@overload
def nulls(
    size: int,
    type: types.StringViewType,
    memory_pool: MemoryPool | None = None,
) -> StringViewArray: ...
@overload
def nulls(
    size: int,
    type: types.LargeListType[_DataTypeT],
    memory_pool: MemoryPool | None = None,
) -> LargeListArray[_DataTypeT]: ...
@overload
def nulls(
    size: int,
    type: types.ListViewType[_DataTypeT],
    memory_pool: MemoryPool | None = None,
) -> ListViewArray[_DataTypeT]: ...
@overload
def nulls(
    size: int,
    type: types.LargeListViewType[_DataTypeT],
    memory_pool: MemoryPool | None = None,
) -> LargeListViewArray[_DataTypeT]: ...
@overload
def nulls(
    size: int,
    type: types.FixedSizeListType[_DataTypeT, _Size],
    memory_pool: MemoryPool | None = None,
) -> FixedSizeListArray[_DataTypeT, _Size]: ...
@overload
def nulls(
    size: int,
    type: types.ListType[_DataTypeT],
    memory_pool: MemoryPool | None = None,
) -> ListArray[scalar.ListScalar[_DataTypeT]]: ...
@overload
def nulls(
    size: int,
    type: types.StructType,
    memory_pool: MemoryPool | None = None,
) -> StructArray: ...
@overload
def nulls(
    size: int,
    type: types.MapType[_MapKeyT, _MapItemT],
    memory_pool: MemoryPool | None = None,
) -> MapArray[_MapKeyT, _MapItemT]: ...
@overload
def nulls(
    size: int,
    type: types.DictionaryType[_IndexT, _BasicValueT],
    memory_pool: MemoryPool | None = None,
) -> DictionaryArray[_IndexT, _BasicValueT]: ...
@overload
def nulls(
    size: int,
    type: types.RunEndEncodedType[_RunEndType, _BasicValueT],
    memory_pool: MemoryPool | None = None,
) -> RunEndEncodedArray[_RunEndType, _BasicValueT]: ...
@overload
def nulls(
    size: int,
    type: types.UnionType,
    memory_pool: MemoryPool | None = None,
) -> UnionArray: ...
@overload
def nulls(
    size: int,
    type: types.FixedShapeTensorType[types._ValueT],
    memory_pool: MemoryPool | None = None,
) -> FixedShapeTensorArray[Any]: ...
@overload
def nulls(
    size: int,
    type: types.Bool8Type,
    memory_pool: MemoryPool | None = None,
) -> Bool8Array: ...
@overload
def nulls(
    size: int,
    type: types.UuidType,
    memory_pool: MemoryPool | None = None,
) -> UuidArray[Any]: ...
@overload
def nulls(
    size: int,
    type: types.JsonType,
    memory_pool: MemoryPool | None = None,
) -> JsonArray[Any]: ...
@overload
def nulls(
    size: int,
    type: types.OpaqueType,
    memory_pool: MemoryPool | None = None,
) -> OpaqueArray[Any]: ...
@overload
def nulls(
    size: int,
    type: types.ExtensionType,
    memory_pool: MemoryPool | None = None,
) -> ExtensionArray[Any]: ...
def nulls(*args, **kwargs):
    """
    Create a strongly-typed Array instance with all elements null.

    Parameters
    ----------
    size : int
        Array length.
    type : pyarrow.DataType, default None
        Explicit type for the array. By default use NullType.
    memory_pool : MemoryPool, default None
        Arrow MemoryPool to use for allocations. Uses the default memory
        pool if not passed.

    Returns
    -------
    arr : Array

    Examples
    --------
    >>> import pyarrow as pa
    >>> pa.nulls(10)
    <pyarrow.lib.NullArray object at ...>
    10 nulls

    >>> pa.nulls(3, pa.uint32())
    <pyarrow.lib.UInt32Array object at ...>
    [
      null,
      null,
      null
    ]
    """

@overload
def repeat(
    value: None | scalar.NullScalar, size: int, memory_pool: MemoryPool | None = None
) -> NullArray: ...
@overload
def repeat(  # type: ignore[overload-overlap]
    value: bool | scalar.BooleanScalar, size: int, memory_pool: MemoryPool | None = None
) -> BooleanArray: ...
@overload
def repeat(
    value: scalar.Int8Scalar, size: int, memory_pool: MemoryPool | None = None
) -> Int8Array: ...
@overload
def repeat(
    value: scalar.Int16Scalar, size: int, memory_pool: MemoryPool | None = None
) -> Int16Array: ...
@overload
def repeat(
    value: scalar.Int32Scalar, size: int, memory_pool: MemoryPool | None = None
) -> Int32Array: ...
@overload
def repeat(
    value: int | scalar.Int64Scalar, size: int, memory_pool: MemoryPool | None = None
) -> Int64Array: ...
@overload
def repeat(
    value: scalar.UInt8Scalar, size: int, memory_pool: MemoryPool | None = None
) -> UInt8Array: ...
@overload
def repeat(
    value: scalar.UInt16Scalar, size: int, memory_pool: MemoryPool | None = None
) -> UInt16Array: ...
@overload
def repeat(
    value: scalar.UInt32Scalar, size: int, memory_pool: MemoryPool | None = None
) -> UInt32Array: ...
@overload
def repeat(
    value: scalar.UInt64Scalar, size: int, memory_pool: MemoryPool | None = None
) -> UInt64Array: ...
@overload
def repeat(
    value: scalar.HalfFloatScalar, size: int, memory_pool: MemoryPool | None = None
) -> HalfFloatArray: ...
@overload
def repeat(
    value: scalar.FloatScalar, size: int, memory_pool: MemoryPool | None = None
) -> FloatArray: ...
@overload
def repeat(
    value: float | scalar.DoubleScalar, size: int, memory_pool: MemoryPool | None = None
) -> DoubleArray: ...
@overload
def repeat(
    value: Decimal | scalar.Decimal32Scalar, size: int, memory_pool: MemoryPool | None = None
) -> Decimal32Array: ...
@overload
def repeat(
    value: scalar.Decimal64Scalar, size: int, memory_pool: MemoryPool | None = None
) -> Decimal64Array: ...
@overload
def repeat(
    value: scalar.Decimal128Scalar, size: int, memory_pool: MemoryPool | None = None
) -> Decimal128Array: ...
@overload
def repeat(
    value: scalar.Decimal256Scalar, size: int, memory_pool: MemoryPool | None = None
) -> Decimal256Array: ...
@overload
def repeat(
    value: dt.date | scalar.Date32Scalar, size: int, memory_pool: MemoryPool | None = None
) -> Date32Array: ...
@overload
def repeat(
    value: scalar.Date64Scalar, size: int, memory_pool: MemoryPool | None = None
) -> Date64Array: ...
@overload
def repeat(
    value: scalar.Time32Scalar[types._Time32Unit], size: int, memory_pool: MemoryPool | None = None
) -> Time32Array[types._Time32Unit]: ...
@overload
def repeat(
    value: dt.time | scalar.Time64Scalar[types._Time64Unit],
    size: int,
    memory_pool: MemoryPool | None = None,
) -> Time64Array[types._Time64Unit]: ...
@overload
def repeat(
    value: scalar.TimestampScalar[types._Unit, types._Tz],
    size: int,
    memory_pool: MemoryPool | None = None,
) -> TimestampArray[types._Unit, types._Tz]: ...
@overload
def repeat(
    value: dt.timedelta | scalar.DurationScalar[types._Unit],
    size: int,
    memory_pool: MemoryPool | None = None,
) -> DurationArray[types._Unit]: ...
@overload
def repeat(  # pyright: ignore[reportOverlappingOverload]
    value: MonthDayNano | scalar.MonthDayNanoIntervalScalar,
    size: int,
    memory_pool: MemoryPool | None = None,
) -> MonthDayNanoIntervalArray: ...
@overload
def repeat(
    value: bytes | scalar.BinaryScalar,
    size: int,
    memory_pool: MemoryPool | None = None,
) -> BinaryArray: ...
@overload
def repeat(
    value: scalar.LargeBinaryScalar,
    size: int,
    memory_pool: MemoryPool | None = None,
) -> LargeBinaryArray: ...
@overload
def repeat(
    value: scalar.FixedSizeBinaryScalar,
    size: int,
    memory_pool: MemoryPool | None = None,
) -> FixedSizeBinaryArray: ...
@overload
def repeat(
    value: str | scalar.StringScalar,
    size: int,
    memory_pool: MemoryPool | None = None,
) -> StringArray: ...
@overload
def repeat(
    value: scalar.LargeStringScalar,
    size: int,
    memory_pool: MemoryPool | None = None,
) -> LargeStringArray: ...
@overload
def repeat(
    value: scalar.BinaryViewScalar,
    size: int,
    memory_pool: MemoryPool | None = None,
) -> BinaryViewArray: ...
@overload
def repeat(
    value: scalar.StringViewScalar,
    size: int,
    memory_pool: MemoryPool | None = None,
) -> StringViewArray: ...
@overload
def repeat(
    value: list[Any] | tuple[Any] | scalar.ListScalar[_DataTypeT],
    size: int,
    memory_pool: MemoryPool | None = None,
) -> ListArray[scalar.ListScalar[_DataTypeT]]: ...
@overload
def repeat(
    value: scalar.FixedSizeListScalar[_DataTypeT, _Size],
    size: int,
    memory_pool: MemoryPool | None = None,
) -> FixedSizeListArray[_DataTypeT, _Size]: ...
@overload
def repeat(
    value: scalar.LargeListScalar[_DataTypeT],
    size: int,
    memory_pool: MemoryPool | None = None,
) -> LargeListArray[_DataTypeT]: ...
@overload
def repeat(
    value: scalar.ListViewScalar[_DataTypeT],
    size: int,
    memory_pool: MemoryPool | None = None,
) -> ListViewArray[_DataTypeT]: ...
@overload
def repeat(
    value: scalar.LargeListViewScalar[_DataTypeT],
    size: int,
    memory_pool: MemoryPool | None = None,
) -> LargeListViewArray[_DataTypeT]: ...
@overload
def repeat(
    value: dict[str, Any] | scalar.StructScalar,
    size: int,
    memory_pool: MemoryPool | None = None,
) -> StructArray: ...
@overload
def repeat(
    value: scalar.MapScalar[_MapKeyT, _MapItemT],
    size: int,
    memory_pool: MemoryPool | None = None,
) -> MapArray[_MapKeyT, _MapItemT]: ...
@overload
def repeat(
    value: scalar.DictionaryScalar[_IndexT, _BasicValueT],
    size: int,
    memory_pool: MemoryPool | None = None,
) -> DictionaryArray[_IndexT, _BasicValueT]: ...
@overload
def repeat(
    value: scalar.RunEndEncodedScalar[_RunEndType, _BasicValueT],
    size: int,
    memory_pool: MemoryPool | None = None,
) -> RunEndEncodedArray[_RunEndType, _BasicValueT]: ...
@overload
def repeat(
    value: scalar.UnionScalar,
    size: int,
    memory_pool: MemoryPool | None = None,
) -> UnionArray: ...
@overload
def repeat(
    value: scalar.FixedShapeTensorScalar,
    size: int,
    memory_pool: MemoryPool | None = None,
) -> FixedShapeTensorArray[Any]: ...
@overload
def repeat(
    value: scalar.Bool8Scalar,
    size: int,
    memory_pool: MemoryPool | None = None,
) -> Bool8Array: ...
@overload
def repeat(
    value: scalar.UuidScalar,
    size: int,
    memory_pool: MemoryPool | None = None,
) -> UuidArray[Any]: ...
@overload
def repeat(
    value: scalar.JsonScalar,
    size: int,
    memory_pool: MemoryPool | None = None,
) -> JsonArray[Any]: ...
@overload
def repeat(
    value: scalar.OpaqueScalar,
    size: int,
    memory_pool: MemoryPool | None = None,
) -> OpaqueArray[Any]: ...
@overload
def repeat(
    value: scalar.ExtensionScalar,
    size: int,
    memory_pool: MemoryPool | None = None,
) -> ExtensionArray[Any]: ...
def repeat(*args, **kwargs):
    """
    Create an Array instance whose slots are the given scalar.

    Parameters
    ----------
    value : Scalar-like object
        Either a pyarrow.Scalar or any python object coercible to a Scalar.
    size : int
        Number of times to repeat the scalar in the output Array.
    memory_pool : MemoryPool, default None
        Arrow MemoryPool to use for allocations. Uses the default memory
        pool if not passed.

    Returns
    -------
    arr : Array

    Examples
    --------
    >>> import pyarrow as pa
    >>> pa.repeat(10, 3)
    <pyarrow.lib.Int64Array object at ...>
    [
      10,
      10,
      10
    ]

    >>> pa.repeat([1, 2], 2)
    <pyarrow.lib.ListArray object at ...>
    [
      [
        1,
        2
      ],
      [
        1,
        2
      ]
    ]

    >>> pa.repeat("string", 3)
    <pyarrow.lib.StringArray object at ...>
    [
      "string",
      "string",
      "string"
    ]

    >>> pa.repeat(pa.scalar({"a": 1, "b": [1, 2]}), 2)
    <pyarrow.lib.StructArray object at ...>
    -- is_valid: all not null
    -- child 0 type: int64
      [
        1,
        1
      ]
    -- child 1 type: list<item: int64>
      [
        [
          1,
          2
        ],
        [
          1,
          2
        ]
      ]
    """

def infer_type(values: Iterable[Any], mask: Mask, from_pandas: bool = False) -> DataType:
    """
    Attempt to infer Arrow data type that can hold the passed Python
    sequence type in an Array object

    Parameters
    ----------
    values : array-like
        Sequence to infer type from.
    mask : ndarray (bool type), optional
        Optional exclusion mask where True marks null, False non-null.
    from_pandas : bool, default False
        Use pandas's NA/null sentinel values for type inference.

    Returns
    -------
    type : DataType
    """

class ArrayStatistics(_Weakrefable):
    """
    The class for statistics of an array.
    """
    @property
    def null_count(self) -> int:
        """
        The number of nulls.
        """
    @property
    def distinct_count(self) -> int:
        """
        The number of distinct values.
        """
    @property
    def min(self) -> Any:
        """
        The minimum value.
        """
    @property
    def is_min_exact(self) -> bool:
        """
        Whether the minimum value is an exact value or not.
        """
    @property
    def max(self) -> Any:
        """
        The maximum value.
        """

    @property
    def is_max_exact(self) -> bool:
        """
        Whether the maximum value is an exact value or not.
        """

_ConvertAs = TypeVar("_ConvertAs", pd.DataFrame, pd.Series)

class _PandasConvertible(_Weakrefable, Generic[_ConvertAs]):
    def to_pandas(
        self,
        memory_pool: MemoryPool | None = None,
        categories: list | None = None,
        strings_to_categorical: bool = False,
        zero_copy_only: bool = False,
        integer_object_nulls: bool = False,
        date_as_object: bool = True,
        timestamp_as_object: bool = False,
        use_threads: bool = True,
        deduplicate_objects: bool = True,
        ignore_metadata: bool = False,
        safe: bool = True,
        split_blocks: bool = False,
        self_destruct: bool = False,
        maps_as_pydicts: Literal["None", "lossy", "strict"] | None = None,
        types_mapper: Callable[[DataType], ExtensionDtype | None] | None = None,
        coerce_temporal_nanoseconds: bool = False,
    ) -> _ConvertAs:
        """
        Convert to a pandas-compatible NumPy array or DataFrame, as appropriate

        Parameters
        ----------
        memory_pool : MemoryPool, default None
            Arrow MemoryPool to use for allocations. Uses the default memory
            pool if not passed.
        categories : list, default empty
            List of fields that should be returned as pandas.Categorical. Only
            applies to table-like data structures.
        strings_to_categorical : bool, default False
            Encode string (UTF8) and binary types to pandas.Categorical.
        zero_copy_only : bool, default False
            Raise an ArrowException if this function call would require copying
            the underlying data.
        integer_object_nulls : bool, default False
            Cast integers with nulls to objects
        date_as_object : bool, default True
            Cast dates to objects. If False, convert to datetime64 dtype with
            the equivalent time unit (if supported). Note: in pandas version
            < 2.0, only datetime64[ns] conversion is supported.
        timestamp_as_object : bool, default False
            Cast non-nanosecond timestamps (np.datetime64) to objects. This is
            useful in pandas version 1.x if you have timestamps that don't fit
            in the normal date range of nanosecond timestamps (1678 CE-2262 CE).
            Non-nanosecond timestamps are supported in pandas version 2.0.
            If False, all timestamps are converted to datetime64 dtype.
        use_threads : bool, default True
            Whether to parallelize the conversion using multiple threads.
        deduplicate_objects : bool, default True
            Do not create multiple copies Python objects when created, to save
            on memory use. Conversion will be slower.
        ignore_metadata : bool, default False
            If True, do not use the 'pandas' metadata to reconstruct the
            DataFrame index, if present
        safe : bool, default True
            For certain data types, a cast is needed in order to store the
            data in a pandas DataFrame or Series (e.g. timestamps are always
            stored as nanoseconds in pandas). This option controls whether it
            is a safe cast or not.
        split_blocks : bool, default False
            If True, generate one internal "block" for each column when
            creating a pandas.DataFrame from a RecordBatch or Table. While this
            can temporarily reduce memory note that various pandas operations
            can trigger "consolidation" which may balloon memory use.
        self_destruct : bool, default False
            EXPERIMENTAL: If True, attempt to deallocate the originating Arrow
            memory while converting the Arrow object to pandas. If you use the
            object after calling to_pandas with this option it will crash your
            program.

            Note that you may not see always memory usage improvements. For
            example, if multiple columns share an underlying allocation,
            memory can't be freed until all columns are converted.
        maps_as_pydicts : str, optional, default `None`
            Valid values are `None`, 'lossy', or 'strict'.
            The default behavior (`None`), is to convert Arrow Map arrays to
            Python association lists (list-of-tuples) in the same order as the
            Arrow Map, as in [(key1, value1), (key2, value2), ...].

            If 'lossy' or 'strict', convert Arrow Map arrays to native Python dicts.
            This can change the ordering of (key, value) pairs, and will
            deduplicate multiple keys, resulting in a possible loss of data.

            If 'lossy', this key deduplication results in a warning printed
            when detected. If 'strict', this instead results in an exception
            being raised when detected.
        types_mapper : function, default None
            A function mapping a pyarrow DataType to a pandas ExtensionDtype.
            This can be used to override the default pandas type for conversion
            of built-in pyarrow types or in absence of pandas_metadata in the
            Table schema. The function receives a pyarrow DataType and is
            expected to return a pandas ExtensionDtype or ``None`` if the
            default conversion should be used for that type. If you have
            a dictionary mapping, you can pass ``dict.get`` as function.
        coerce_temporal_nanoseconds : bool, default False
            Only applicable to pandas version >= 2.0.
            A legacy option to coerce date32, date64, duration, and timestamp
            time units to nanoseconds when converting to pandas. This is the
            default behavior in pandas version 1.x. Set this option to True if
            you'd like to use this coercion when using pandas version >= 2.0
            for backwards compatibility (not recommended otherwise).

        Returns
        -------
        pandas.Series or pandas.DataFrame depending on type of object

        Examples
        --------
        >>> import pyarrow as pa
        >>> import pandas as pd

        Convert a Table to pandas DataFrame:

        >>> table = pa.table(
        ...     [
        ...         pa.array([2, 4, 5, 100]),
        ...         pa.array(["Flamingo", "Horse", "Brittle stars", "Centipede"]),
        ...     ],
        ...     names=["n_legs", "animals"],
        ... )
        >>> table.to_pandas()
           n_legs        animals
        0       2       Flamingo
        1       4          Horse
        2       5  Brittle stars
        3     100      Centipede
        >>> isinstance(table.to_pandas(), pd.DataFrame)
        True

        Convert a RecordBatch to pandas DataFrame:

        >>> import pyarrow as pa
        >>> n_legs = pa.array([2, 4, 5, 100])
        >>> animals = pa.array(["Flamingo", "Horse", "Brittle stars", "Centipede"])
        >>> batch = pa.record_batch([n_legs, animals], names=["n_legs", "animals"])
        >>> batch
        pyarrow.RecordBatch
        n_legs: int64
        animals: string
        ----
        n_legs: [2,4,5,100]
        animals: ["Flamingo","Horse","Brittle stars","Centipede"]
        >>> batch.to_pandas()
           n_legs        animals
        0       2       Flamingo
        1       4          Horse
        2       5  Brittle stars
        3     100      Centipede
        >>> isinstance(batch.to_pandas(), pd.DataFrame)
        True

        Convert a Chunked Array to pandas Series:

        >>> import pyarrow as pa
        >>> n_legs = pa.chunked_array([[2, 2, 4], [4, 5, 100]])
        >>> n_legs.to_pandas()
        0      2
        1      2
        2      4
        3      4
        4      5
        5    100
        dtype: int64
        >>> isinstance(n_legs.to_pandas(), pd.Series)
        True
        """

_CastAs = TypeVar("_CastAs", bound=DataType)
_Scalar_co = TypeVar("_Scalar_co", bound=Scalar, covariant=True)
_ScalarT = TypeVar("_ScalarT", bound=Scalar)

class Array(_PandasConvertible[pd.Series], Generic[_Scalar_co]):
    """
    The base class for all Arrow arrays.
    """

    def diff(self, other: Self) -> str:
        """
        Compare contents of this array against another one.

        Return a string containing the result of diffing this array
        (on the left side) against the other array (on the right side).

        Parameters
        ----------
        other : Array
            The other array to compare this array with.

        Returns
        -------
        diff : str
            A human-readable printout of the differences.

        Examples
        --------
        >>> import pyarrow as pa
        >>> left = pa.array(["one", "two", "three"])
        >>> right = pa.array(["two", None, "two-and-a-half", "three"])
        >>> print(left.diff(right))  # doctest: +SKIP

        @@ -0, +0 @@
        -"one"
        @@ -2, +1 @@
        +null
        +"two-and-a-half"
        """
    def cast(
        self,
        target_type: _CastAs,
        safe: bool = True,
        options: CastOptions | None = None,
        memory_pool: MemoryPool | None = None,
    ) -> Array[Scalar[_CastAs]]:
        """
        Cast array values to another data type

        See :func:`pyarrow.compute.cast` for usage.

        Parameters
        ----------
        target_type : DataType, default None
            Type to cast array to.
        safe : boolean, default True
            Whether to check for conversion errors such as overflow.
        options : CastOptions, default None
            Additional checks pass by CastOptions
        memory_pool : MemoryPool, optional
            memory pool to use for allocations during function execution.

        Returns
        -------
        cast : Array
        """
    def view(self, target_type: _CastAs) -> Array[Scalar[_CastAs]]:
        """
        Return zero-copy "view" of array as another data type.

        The data types must have compatible columnar buffer layouts

        Parameters
        ----------
        target_type : DataType
            Type to construct view as.

        Returns
        -------
        view : Array
        """
    def sum(self, **kwargs) -> _Scalar_co:
        """
        Sum the values in a numerical array.

        See :func:`pyarrow.compute.sum` for full usage.

        Parameters
        ----------
        **kwargs : dict, optional
            Options to pass to :func:`pyarrow.compute.sum`.

        Returns
        -------
        sum : Scalar
            A scalar containing the sum value.
        """
    @property
    def type(self: Array[Scalar[_DataTypeT]]) -> _DataTypeT: ...
    def unique(self) -> Self:
        """
        Compute distinct elements in array.

        Returns
        -------
        unique : Array
            An array of the same data type, with deduplicated elements.
        """
    def dictionary_encode(self, null_encoding: str = "mask") -> DictionaryArray:
        """
        Compute dictionary-encoded representation of array.

        See :func:`pyarrow.compute.dictionary_encode` for full usage.

        Parameters
        ----------
        null_encoding : str, default "mask"
            How to handle null entries.

        Returns
        -------
        encoded : DictionaryArray
            A dictionary-encoded version of this array.
        """
    def value_count(self) -> StructArray:
        """
        Compute counts of unique elements in array.

        Returns
        -------
        StructArray
            An array of  <input type "Values", int64 "Counts"> structs
        """
    @overload
    @staticmethod
    def from_pandas(
        obj: pd.Series | np.ndarray | ArrayLike,
        *,
        mask: Mask | None = None,
        type: _DataTypeT,
        safe: bool = True,
        memory_pool: MemoryPool | None = None,
    ) -> Array[Scalar[_DataTypeT]]: ...
    @overload
    @staticmethod
    def from_pandas(
        obj: pd.Series | np.ndarray | ArrayLike,
        *,
        mask: Mask | None = None,
        safe: bool = True,
        memory_pool: MemoryPool | None = None,
    ) -> Array[Scalar]: ...
    @staticmethod
    def from_pandas(*args, **kwargs):
        """
        Convert pandas.Series to an Arrow Array.

        This method uses Pandas semantics about what values indicate
        nulls. See pyarrow.array for more general conversion from arrays or
        sequences to Arrow arrays.

        Parameters
        ----------
        obj : ndarray, pandas.Series, array-like
        mask : array (boolean), optional
            Indicate which values are null (True) or not null (False).
        type : pyarrow.DataType
            Explicit type to attempt to coerce to, otherwise will be inferred
            from the data.
        safe : bool, default True
            Check for overflows or other unsafe conversions.
        memory_pool : pyarrow.MemoryPool, optional
            If not passed, will allocate memory from the currently-set default
            memory pool.

        Notes
        -----
        Localized timestamps will currently be returned as UTC (pandas's native
        representation). Timezone-naive data will be implicitly interpreted as
        UTC.

        Returns
        -------
        array : pyarrow.Array or pyarrow.ChunkedArray
            ChunkedArray is returned if object data overflows binary buffer.
        """
    @staticmethod
    def from_buffers(
        type: _DataTypeT,
        length: int,
        buffers: list[Buffer],
        null_count: int = -1,
        offset=0,
        children: NullableCollection[Array[Scalar[_DataTypeT]]] | None = None,
    ) -> Array[Scalar[_DataTypeT]]:
        """
        Construct an Array from a sequence of buffers.

        The concrete type returned depends on the datatype.

        Parameters
        ----------
        type : DataType
            The value type of the array.
        length : int
            The number of values in the array.
        buffers : List[Buffer]
            The buffers backing this array.
        null_count : int, default -1
            The number of null entries in the array. Negative value means that
            the null count is not known.
        offset : int, default 0
            The array's logical offset (in values, not in bytes) from the
            start of each buffer.
        children : List[Array], default None
            Nested type children with length matching type.num_fields.

        Returns
        -------
        array : Array
        """
    @property
    def null_count(self) -> int: ...
    @property
    def nbytes(self) -> int:
        """
        Total number of bytes consumed by the elements of the array.

        In other words, the sum of bytes from all buffer
        ranges referenced.

        Unlike `get_total_buffer_size` this method will account for array
        offsets.

        If buffers are shared between arrays then the shared
        portion will be counted multiple times.

        The dictionary of dictionary arrays will always be counted in their
        entirety even if the array only references a portion of the dictionary.
        """
    def get_total_buffer_size(self) -> int:
        """
        The sum of bytes in each buffer referenced by the array.

        An array may only reference a portion of a buffer.
        This method will overestimate in this case and return the
        byte size of the entire buffer.

        If a buffer is referenced multiple times then it will
        only be counted once.
        """
    def __sizeof__(self) -> int: ...
    def __iter__(self) -> Iterator[_Scalar_co]: ...
    def to_string(
        self,
        *,
        indent: int = 2,
        top_level_indent: int = 0,
        window: int = 10,
        container_window: int = 2,
        skip_new_lines: bool = False,
    ) -> str:
        """
        Render a "pretty-printed" string representation of the Array.

        Note: for data on a non-CPU device, the full array is copied to CPU
        memory.

        Parameters
        ----------
        indent : int, default 2
            How much to indent the internal items in the string to
            the right, by default ``2``.
        top_level_indent : int, default 0
            How much to indent right the entire content of the array,
            by default ``0``.
        window : int
            How many primitive items to preview at the begin and end
            of the array when the array is bigger than the window.
            The other items will be ellipsed.
        container_window : int
            How many container items (such as a list in a list array)
            to preview at the begin and end of the array when the array
            is bigger than the window.
        skip_new_lines : bool
            If the array should be rendered as a single line of text
            or if each element should be on its own line.
        """
    format = to_string
    def equals(self, other: Self) -> bool: ...
    def __len__(self) -> int: ...
    def is_null(self, *, nan_is_null: bool = False) -> BooleanArray:
        """
        Return BooleanArray indicating the null values.

        Parameters
        ----------
        nan_is_null : bool (optional, default False)
            Whether floating-point NaN values should also be considered null.

        Returns
        -------
        array : boolean Array
        """
    def is_nan(self) -> BooleanArray:
        """
        Return BooleanArray indicating the NaN values.

        Returns
        -------
        array : boolean Array
        """
    def is_valid(self) -> BooleanArray:
        """
        Return BooleanArray indicating the non-null values.
        """
    def fill_null(
        self: Array[Scalar[_BasicDataType[_AsPyType]]], fill_value: _AsPyType
    ) -> Array[Scalar[_BasicDataType[_AsPyType]]]:
        """
        See :func:`pyarrow.compute.fill_null` for usage.

        Parameters
        ----------
        fill_value : any
            The replacement value for null entries.

        Returns
        -------
        result : Array
            A new array with nulls replaced by the given value.
        """
    @overload
    def __getitem__(self, key: int) -> _Scalar_co: ...
    @overload
    def __getitem__(self, key: slice) -> Self: ...
    def __getitem__(self, key):
        """
        Slice or return value at given index

        Parameters
        ----------
        key : integer or slice
            Slices with step not equal to 1 (or None) will produce a copy
            rather than a zero-copy view

        Returns
        -------
        value : Scalar (index) or Array (slice)
        """
    def slice(self, offset: int = 0, length: int | None = None) -> Self:
        """
        Compute zero-copy slice of this array.

        Parameters
        ----------
        offset : int, default 0
            Offset from start of array to slice.
        length : int, default None
            Length of slice (default is until end of Array starting from
            offset).

        Returns
        -------
        sliced : Array
            An array with the same datatype, containing the sliced values.
        """
    def take(self, indices: Indices) -> Self:
        """
        Select values from an array.

        See :func:`pyarrow.compute.take` for full usage.

        Parameters
        ----------
        indices : Array or array-like
            The indices in the array whose values will be returned.

        Returns
        -------
        taken : Array
            An array with the same datatype, containing the taken values.
        """
    def drop_null(self) -> Self:
        """
        Remove missing values from an array.
        """
    def filter(
        self,
        mask: Mask,
        *,
        null_selection_behavior: Literal["drop", "emit_null"] = "drop",
    ) -> Self:
        """
        Select values from an array.

        See :func:`pyarrow.compute.filter` for full usage.

        Parameters
        ----------
        mask : Array or array-like
            The boolean mask to filter the array with.
        null_selection_behavior : str, default "drop"
            How nulls in the mask should be handled.

        Returns
        -------
        filtered : Array
            An array of the same type, with only the elements selected by
            the boolean mask.
        """
    @overload
    def index(
        self: Array[_ScalarT],
        value: _ScalarT,
        start: int | None = None,
        end: int | None = None,
        *,
        memory_pool: MemoryPool | None = None,
    ) -> scalar.Int64Scalar: ...
    @overload
    def index(
        self: Array[Scalar[_BasicDataType[_AsPyType]]],
        value: _AsPyType,
        start: int | None = None,
        end: int | None = None,
        *,
        memory_pool: MemoryPool | None = None,
    ) -> scalar.Int64Scalar: ...
    def index(self, *args, **kwargs):
        """
        Find the first index of a value.

        See :func:`pyarrow.compute.index` for full usage.

        Parameters
        ----------
        value : Scalar or object
            The value to look for in the array.
        start : int, optional
            The start index where to look for `value`.
        end : int, optional
            The end index where to look for `value`.
        memory_pool : MemoryPool, optional
            A memory pool for potential memory allocations.

        Returns
        -------
        index : Int64Scalar
            The index of the value in the array (-1 if not found).
        """
    def sort(self, order: Order = "ascending", **kwargs) -> Self:
        """
        Sort the Array

        Parameters
        ----------
        order : str, default "ascending"
            Which order to sort values in.
            Accepted values are "ascending", "descending".
        **kwargs : dict, optional
            Additional sorting options.
            As allowed by :class:`SortOptions`

        Returns
        -------
        result : Array
        """
    def __array__(self, dtype: np.dtype | None = None, copy: bool | None = None) -> np.ndarray: ...
    def to_numpy(self, zero_copy_only: bool = True, writable: bool = False) -> np.ndarray:
        """
        Return a NumPy view or copy of this array.

        By default, tries to return a view of this array. This is only
        supported for primitive arrays with the same memory layout as NumPy
        (i.e. integers, floating point, ..) and without any nulls.

        For the extension arrays, this method simply delegates to the
        underlying storage array.

        Parameters
        ----------
        zero_copy_only : bool, default True
            If True, an exception will be raised if the conversion to a numpy
            array would require copying the underlying data (e.g. in presence
            of nulls, or for non-primitive types).
        writable : bool, default False
            For numpy arrays created with zero copy (view on the Arrow data),
            the resulting array is not writable (Arrow data is immutable).
            By setting this to True, a copy of the array is made to ensure
            it is writable.

        Returns
        -------
        array : numpy.ndarray
        """
    def to_pylist(
        self: Array[Scalar[_BasicDataType[_AsPyType]]],
        *,
        maps_as_pydicts: Literal["lossy", "strict"] | None = None,
    ) -> list[_AsPyType | None]:
        """
        Convert to a list of native Python objects.

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

        Returns
        -------
        lst : list
        """
    tolist = to_pylist
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
    @property
    def offset(self) -> int:
        """
        A relative position into another array's data.

        The purpose is to enable zero-copy slicing. This value defaults to zero
        but must be applied on all operations with the physical storage
        buffers.
        """
    def buffers(self) -> list[Buffer | None]:
        """
        Return a list of Buffer objects pointing to this array's physical
        storage.

        To correctly interpret these buffers, you need to also apply the offset
        multiplied with the size of the stored data type.
        """
    def copy_to(self, destination: MemoryManager | Device) -> Self:
        """
        Construct a copy of the array with all buffers on destination
        device.

        This method recursively copies the array's buffers and those of its
        children onto the destination MemoryManager device and returns the
        new Array.

        Parameters
        ----------
        destination : pyarrow.MemoryManager or pyarrow.Device
            The destination device to copy the array to.

        Returns
        -------
        Array
        """
    def _export_to_c(self, out_ptr: int, out_schema_ptr: int = 0) -> None:
        """
        Export to a C ArrowArray struct, given its pointer.

        If a C ArrowSchema struct pointer is also given, the array type
        is exported to it at the same time.

        Parameters
        ----------
        out_ptr: int
            The raw pointer to a C ArrowArray struct.
        out_schema_ptr: int (optional)
            The raw pointer to a C ArrowSchema struct.

        Be careful: if you don't pass the ArrowArray struct to a consumer,
        array memory will leak.  This is a low-level function intended for
        expert users.
        """
    @classmethod
    def _import_from_c(cls, in_ptr: int, type: int | DataType) -> Self:
        """
        Import Array from a C ArrowArray struct, given its pointer
        and the imported array type.

        Parameters
        ----------
        in_ptr: int
            The raw pointer to a C ArrowArray struct.
        type: DataType or int
            Either a DataType object, or the raw pointer to a C ArrowSchema
            struct.

        This is a low-level function intended for expert users.
        """
    def __arrow_c_array__(self, requested_schema=None) -> Any:
        """
        Get a pair of PyCapsules containing a C ArrowArray representation of the object.

        Parameters
        ----------
        requested_schema : PyCapsule | None
            A PyCapsule containing a C ArrowSchema representation of a requested
            schema. PyArrow will attempt to cast the array to this data type.
            If None, the array will be returned as-is, with a type matching the
            one returned by :meth:`__arrow_c_schema__()`.

        Returns
        -------
        Tuple[PyCapsule, PyCapsule]
            A pair of PyCapsules containing a C ArrowSchema and ArrowArray,
            respectively.
        """
    @classmethod
    def _import_from_c_capsule(cls, schema_capsule, array_capsule) -> Self: ...
    def _export_to_c_device(self, out_ptr: int, out_schema_ptr: int = 0) -> None:
        """
        Export to a C ArrowDeviceArray struct, given its pointer.

        If a C ArrowSchema struct pointer is also given, the array type
        is exported to it at the same time.

        Parameters
        ----------
        out_ptr: int
            The raw pointer to a C ArrowDeviceArray struct.
        out_schema_ptr: int (optional)
            The raw pointer to a C ArrowSchema struct.

        Be careful: if you don't pass the ArrowDeviceArray struct to a consumer,
        array memory will leak.  This is a low-level function intended for
        expert users.
        """
    @classmethod
    def _import_from_c_device(cls, in_ptr: int, type: DataType | int) -> Self:
        """
        Import Array from a C ArrowDeviceArray struct, given its pointer
        and the imported array type.

        Parameters
        ----------
        in_ptr: int
            The raw pointer to a C ArrowDeviceArray struct.
        type: DataType or int
            Either a DataType object, or the raw pointer to a C ArrowSchema
            struct.

        This is a low-level function intended for expert users.
        """

    def __arrow_c_device_array__(self, requested_schema=None, **kwargs) -> Any:
        """
        Get a pair of PyCapsules containing a C ArrowDeviceArray representation
        of the object.

        Parameters
        ----------
        requested_schema : PyCapsule | None
            A PyCapsule containing a C ArrowSchema representation of a requested
            schema. PyArrow will attempt to cast the array to this data type.
            If None, the array will be returned as-is, with a type matching the
            one returned by :meth:`__arrow_c_schema__()`.
        kwargs
            Currently no additional keyword arguments are supported, but
            this method will accept any keyword with a value of ``None``
            for compatibility with future keywords.

        Returns
        -------
        Tuple[PyCapsule, PyCapsule]
            A pair of PyCapsules containing a C ArrowSchema and ArrowDeviceArray,
            respectively.
        """
    @classmethod
    def _import_from_c_device_capsule(cls, schema_capsule, array_capsule) -> Self: ...
    def __dlpack__(self, stream: int | None = None) -> Any:
        """Export a primitive array as a DLPack capsule.

        Parameters
        ----------
        stream : int, optional
            A Python integer representing a pointer to a stream. Currently not supported.
            Stream is provided by the consumer to the producer to instruct the producer
            to ensure that operations can safely be performed on the array.

        Returns
        -------
        capsule : PyCapsule
            A DLPack capsule for the array, pointing to a DLManagedTensor.
        """
    def __dlpack_device__(self) -> tuple[int, int]:
        """
        Return the DLPack device tuple this arrays resides on.

        Returns
        -------
        tuple : Tuple[int, int]
            Tuple with index specifying the type of the device (where
            CPU = 1, see cpp/src/arrow/c/dpack_abi.h) and index of the
            device which is 0 by default for CPU.
        """
    @property
    def device_type(self) -> DeviceAllocationType:
        """
        The device type where the array resides.

        Returns
        -------
        DeviceAllocationType
        """

    @property
    def is_cpu(self) -> bool:
        """
        Whether the array is CPU-accessible.
        """
    @property
    def statistics(self) -> ArrayStatistics | None:
        """
        Statistics of the array.
        """

class NullArray(Array[scalar.NullScalar]): ...

class BooleanArray(Array[scalar.BooleanScalar]):
    @property
    def false_count(self) -> int: ...
    @property
    def true_count(self) -> int: ...

class NumericArray(Array[_ScalarT]): ...
class IntegerArray(NumericArray[_ScalarT]): ...
class FloatingPointArray(NumericArray[_ScalarT]): ...
class Int8Array(IntegerArray[scalar.Int8Scalar]): ...
class UInt8Array(IntegerArray[scalar.UInt8Scalar]): ...
class Int16Array(IntegerArray[scalar.Int16Scalar]): ...
class UInt16Array(IntegerArray[scalar.UInt16Scalar]): ...
class Int32Array(IntegerArray[scalar.Int32Scalar]): ...
class UInt32Array(IntegerArray[scalar.UInt32Scalar]): ...
class Int64Array(IntegerArray[scalar.Int64Scalar]): ...
class UInt64Array(IntegerArray[scalar.UInt64Scalar]): ...
class Date32Array(NumericArray[scalar.Date32Scalar]): ...
class Date64Array(NumericArray[scalar.Date64Scalar]): ...
class TimestampArray(NumericArray[scalar.TimestampScalar[types._Unit, types._Tz]]): ...
class Time32Array(NumericArray[scalar.Time32Scalar[types._Time32Unit]]): ...
class Time64Array(NumericArray[scalar.Time64Scalar[types._Time64Unit]]): ...
class DurationArray(NumericArray[scalar.DurationScalar[types._Unit]]): ...
class MonthDayNanoIntervalArray(Array[scalar.MonthDayNanoIntervalScalar]): ...
class HalfFloatArray(FloatingPointArray[scalar.HalfFloatScalar]): ...
class FloatArray(FloatingPointArray[scalar.FloatScalar]): ...
class DoubleArray(FloatingPointArray[scalar.DoubleScalar]): ...
class FixedSizeBinaryArray(Array[scalar.FixedSizeBinaryScalar]): ...
class Decimal32Array(FixedSizeBinaryArray): ...
class Decimal64Array(FixedSizeBinaryArray): ...
class Decimal128Array(FixedSizeBinaryArray): ...
class Decimal256Array(FixedSizeBinaryArray): ...

class BaseListArray(Array[_ScalarT]):
    def flatten(self, recursive: bool = False) -> Array: ...
    def value_parent_indices(self) -> Int64Array: ...
    def value_lengths(self) -> Int32Array: ...

class ListArray(BaseListArray[_ScalarT]):
    @overload
    @classmethod
    def from_arrays(
        cls,
        offsets: Int32Array | list[int],
        values: Array[Scalar[_DataTypeT]],
        *,
        type: None = None,
        pool: MemoryPool | None = None,
        mask: Mask | None = None,
    ) -> ListArray[scalar.ListScalar[_DataTypeT]]: ...
    @overload
    @classmethod
    def from_arrays(
        cls,
        offsets: Int32Array | list[int],
        values: list[int],
        *,
        type: None = None,
        pool: MemoryPool | None = None,
        mask: Mask | None = None,
    ) -> ListArray[scalar.ListScalar[types.Int64Type]]: ...
    @overload
    @classmethod
    def from_arrays(
        cls,
        offsets: Int32Array | list[int],
        values: list[float],
        *,
        type: None = None,
        pool: MemoryPool | None = None,
        mask: Mask | None = None,
    ) -> ListArray[scalar.ListScalar[types.Float64Type]]: ...
    @overload
    @classmethod
    def from_arrays(
        cls,
        offsets: Int32Array | list[int],
        values: list[str],
        *,
        type: None = None,
        pool: MemoryPool | None = None,
        mask: Mask | None = None,
    ) -> ListArray[scalar.ListScalar[types.StringType]]: ...
    @overload
    @classmethod
    def from_arrays(
        cls,
        offsets: Int32Array | list[int],
        values: list[bytes],
        *,
        type: None = None,
        pool: MemoryPool | None = None,
        mask: Mask | None = None,
    ) -> ListArray[scalar.ListScalar[types.BinaryType]]: ...
    @overload
    @classmethod
    def from_arrays(
        cls,
        offsets: Int32Array | list[int],
        values: list,
        *,
        type: None = None,
        pool: MemoryPool | None = None,
        mask: Mask | None = None,
    ) -> ListArray: ...
    @overload
    @classmethod
    def from_arrays(
        cls,
        offsets: Int32Array | list[int],
        values: Array | list,
        *,
        type: _DataTypeT,
        pool: MemoryPool | None = None,
        mask: Mask | None = None,
    ) -> ListArray[scalar.ListScalar[_DataTypeT]]: ...
    @classmethod
    def from_arrays(cls, *args, **kwargs):
        """
        Construct ListArray from arrays of int32 offsets and values.

        Parameters
        ----------
        offsets : Array (int32 type)
        values : Array (any type)
        type : DataType, optional
            If not specified, a default ListType with the values' type is
            used.
        pool : MemoryPool, optional
        mask : Array (boolean type), optional
            Indicate which values are null (True) or not null (False).

        Returns
        -------
        list_array : ListArray

        Examples
        --------
        >>> import pyarrow as pa
        >>> values = pa.array([1, 2, 3, 4])
        >>> offsets = pa.array([0, 2, 4])
        >>> pa.ListArray.from_arrays(offsets, values)
        <pyarrow.lib.ListArray object at ...>
        [
          [
            1,
            2
          ],
          [
            3,
            4
          ]
        ]
        >>> # nulls in the offsets array become null lists
        >>> offsets = pa.array([0, None, 2, 4])
        >>> pa.ListArray.from_arrays(offsets, values)
        <pyarrow.lib.ListArray object at ...>
        [
          [
            1,
            2
          ],
          null,
          [
            3,
            4
          ]
        ]
        """
    @property
    def values(self) -> Array:
        """
        Return the underlying array of values which backs the ListArray
        ignoring the array's offset.

        If any of the list elements are null, but are backed by a
        non-empty sub-list, those elements will be included in the
        output.

        Compare with :meth:`flatten`, which returns only the non-null
        values taking into consideration the array's offset.

        Returns
        -------
        values : Array

        See Also
        --------
        ListArray.flatten : ...

        Examples
        --------

        The values include null elements from sub-lists:

        >>> import pyarrow as pa
        >>> array = pa.array([[1, 2], None, [3, 4, None, 6]])
        >>> array.values
        <pyarrow.lib.Int64Array object at ...>
        [
          1,
          2,
          3,
          4,
          null,
          6
        ]

        If an array is sliced, the slice still uses the same
        underlying data as the original array, just with an
        offset. Since values ignores the offset, the values are the
        same:

        >>> sliced = array.slice(1, 2)
        >>> sliced
        <pyarrow.lib.ListArray object at ...>
        [
          null,
          [
            3,
            4,
            null,
            6
          ]
        ]
        >>> sliced.values
        <pyarrow.lib.Int64Array object at ...>
        [
          1,
          2,
          3,
          4,
          null,
          6
        ]

        """
    @property
    def offsets(self) -> Int32Array:
        """
        Return the list offsets as an int32 array.

        The returned array will not have a validity bitmap, so you cannot
        expect to pass it to `ListArray.from_arrays` and get back the same
        list array if the original one has nulls.

        Returns
        -------
        offsets : Int32Array

        Examples
        --------
        >>> import pyarrow as pa
        >>> array = pa.array([[1, 2], None, [3, 4, 5]])
        >>> array.offsets
        <pyarrow.lib.Int32Array object at ...>
        [
          0,
          2,
          2,
          5
        ]
        """

class LargeListArray(BaseListArray[scalar.LargeListScalar[_DataTypeT]]):
    @overload
    @classmethod
    def from_arrays(
        cls,
        offsets: Int64Array,
        values: Array[Scalar[_DataTypeT]],
        *,
        type: None = None,
        pool: MemoryPool | None = None,
        mask: Mask | None = None,
    ) -> LargeListArray[_DataTypeT]: ...
    @overload
    @classmethod
    def from_arrays(
        cls,
        offsets: Int64Array,
        values: Array,
        *,
        type: _DataTypeT,
        pool: MemoryPool | None = None,
        mask: Mask | None = None,
    ) -> LargeListArray[_DataTypeT]: ...
    @classmethod
    def from_arrays(cls, *args, **kwargs):
        """
        Construct LargeListArray from arrays of int64 offsets and values.

        Parameters
        ----------
        offsets : Array (int64 type)
        values : Array (any type)
        type : DataType, optional
            If not specified, a default ListType with the values' type is
            used.
        pool : MemoryPool, optional
        mask : Array (boolean type), optional
            Indicate which values are null (True) or not null (False).

        Returns
        -------
        list_array : LargeListArray
        """
    @property
    def values(self) -> Array:
        """
        Return the underlying array of values which backs the LargeListArray
        ignoring the array's offset.

        If any of the list elements are null, but are backed by a
        non-empty sub-list, those elements will be included in the
        output.

        Compare with :meth:`flatten`, which returns only the non-null
        values taking into consideration the array's offset.

        Returns
        -------
        values : Array

        See Also
        --------
        LargeListArray.flatten : ...

        Examples
        --------

        The values include null elements from the sub-lists:

        >>> import pyarrow as pa
        >>> array = pa.array(
        ...     [[1, 2], None, [3, 4, None, 6]],
        ...     type=pa.large_list(pa.int32()),
        ... )
        >>> array.values
        <pyarrow.lib.Int32Array object at ...>
        [
          1,
          2,
          3,
          4,
          null,
          6
        ]

        If an array is sliced, the slice still uses the same
        underlying data as the original array, just with an
        offset. Since values ignores the offset, the values are the
        same:

        >>> sliced = array.slice(1, 2)
        >>> sliced
        <pyarrow.lib.LargeListArray object at ...>
        [
          null,
          [
            3,
            4,
            null,
            6
          ]
        ]
        >>> sliced.values
        <pyarrow.lib.Int32Array object at ...>
        [
          1,
          2,
          3,
          4,
          null,
          6
        ]
        """
    @property
    def offsets(self) -> Int64Array:
        """
        Return the list offsets as an int64 array.

        The returned array will not have a validity bitmap, so you cannot
        expect to pass it to `LargeListArray.from_arrays` and get back the
        same list array if the original one has nulls.

        Returns
        -------
        offsets : Int64Array
        """

class ListViewArray(BaseListArray[scalar.ListViewScalar[_DataTypeT]]):
    @overload
    @classmethod
    def from_arrays(
        cls,
        offsets: Int32Array,
        values: Array[Scalar[_DataTypeT]],
        *,
        type: None = None,
        pool: MemoryPool | None = None,
        mask: Mask | None = None,
    ) -> ListViewArray[_DataTypeT]: ...
    @overload
    @classmethod
    def from_arrays(
        cls,
        offsets: Int32Array,
        values: Array,
        *,
        type: _DataTypeT,
        pool: MemoryPool | None = None,
        mask: Mask | None = None,
    ) -> ListViewArray[_DataTypeT]: ...
    @classmethod
    def from_arrays(cls, *args, **kwargs):
        """
        Construct ListViewArray from arrays of int32 offsets, sizes, and values.

        Parameters
        ----------
        offsets : Array (int32 type)
        sizes : Array (int32 type)
        values : Array (any type)
        type : DataType, optional
            If not specified, a default ListType with the values' type is
            used.
        pool : MemoryPool, optional
        mask : Array (boolean type), optional
            Indicate which values are null (True) or not null (False).

        Returns
        -------
        list_view_array : ListViewArray

        Examples
        --------
        >>> import pyarrow as pa
        >>> values = pa.array([1, 2, 3, 4])
        >>> offsets = pa.array([0, 1, 2])
        >>> sizes = pa.array([2, 2, 2])
        >>> pa.ListViewArray.from_arrays(offsets, sizes, values)
        <pyarrow.lib.ListViewArray object at ...>
        [
          [
            1,
            2
          ],
          [
            2,
            3
          ],
          [
            3,
            4
          ]
        ]
        >>> # use a null mask to represent null values
        >>> mask = pa.array([False, True, False])
        >>> pa.ListViewArray.from_arrays(offsets, sizes, values, mask=mask)
        <pyarrow.lib.ListViewArray object at ...>
        [
          [
            1,
            2
          ],
          null,
          [
            3,
            4
          ]
        ]
        >>> # null values can be defined in either offsets or sizes arrays
        >>> # WARNING: this will result in a copy of the offsets or sizes arrays
        >>> offsets = pa.array([0, None, 2])
        >>> pa.ListViewArray.from_arrays(offsets, sizes, values)
        <pyarrow.lib.ListViewArray object at ...>
        [
          [
            1,
            2
          ],
          null,
          [
            3,
            4
          ]
        ]
        """
    @property
    def values(self) -> Array:
        """
        Return the underlying array of values which backs the ListViewArray
        ignoring the array's offset and sizes.

        The values array may be out of order and/or contain additional values
        that are not found in the logical representation of the array. The only
        guarantee is that each non-null value in the ListView Array is contiguous.

        Compare with :meth:`flatten`, which returns only the non-null
        values taking into consideration the array's order and offset.

        Returns
        -------
        values : Array

        Examples
        --------
        The values include null elements from sub-lists:

        >>> import pyarrow as pa
        >>> values = [1, 2, None, 3, 4]
        >>> offsets = [0, 0, 1]
        >>> sizes = [2, 0, 4]
        >>> array = pa.ListViewArray.from_arrays(offsets, sizes, values)
        >>> array
        <pyarrow.lib.ListViewArray object at ...>
        [
          [
            1,
            2
          ],
          [],
          [
            2,
            null,
            3,
            4
          ]
        ]
        >>> array.values
        <pyarrow.lib.Int64Array object at ...>
        [
          1,
          2,
          null,
          3,
          4
        ]
        """
    @property
    def offsets(self) -> Int32Array:
        """
        Return the list offsets as an int32 array.

        The returned array will not have a validity bitmap, so you cannot
        expect to pass it to `ListViewArray.from_arrays` and get back the same
        list array if the original one has nulls.

        Returns
        -------
        offsets : Int32Array

        Examples
        --------
        >>> import pyarrow as pa
        >>> values = [1, 2, None, 3, 4]
        >>> offsets = [0, 0, 1]
        >>> sizes = [2, 0, 4]
        >>> array = pa.ListViewArray.from_arrays(offsets, sizes, values)
        >>> array.offsets
        <pyarrow.lib.Int32Array object at ...>
        [
          0,
          0,
          1
        ]
        """
    @property
    def sizes(self) -> Int32Array:
        """
        Return the list sizes as an int32 array.

        The returned array will not have a validity bitmap, so you cannot
        expect to pass it to `ListViewArray.from_arrays` and get back the same
        list array if the original one has nulls.

        Returns
        -------
        sizes : Int32Array

        Examples
        --------
        >>> import pyarrow as pa
        >>> values = [1, 2, None, 3, 4]
        >>> offsets = [0, 0, 1]
        >>> sizes = [2, 0, 4]
        >>> array = pa.ListViewArray.from_arrays(offsets, sizes, values)
        >>> array.sizes
        <pyarrow.lib.Int32Array object at ...>
        [
          2,
          0,
          4
        ]
        """

class LargeListViewArray(BaseListArray[scalar.LargeListScalar[_DataTypeT]]):
    @overload
    @classmethod
    def from_arrays(
        cls,
        offsets: Int64Array,
        values: Array[Scalar[_DataTypeT]],
        *,
        type: None = None,
        pool: MemoryPool | None = None,
        mask: Mask | None = None,
    ) -> LargeListViewArray[_DataTypeT]: ...
    @overload
    @classmethod
    def from_arrays(
        cls,
        offsets: Int64Array,
        values: Array,
        *,
        type: _DataTypeT,
        pool: MemoryPool | None = None,
        mask: Mask | None = None,
    ) -> LargeListViewArray[_DataTypeT]: ...
    @classmethod
    def from_arrays(cls, *args, **kwargs):
        """
        Construct LargeListViewArray from arrays of int64 offsets and values.

        Parameters
        ----------
        offsets : Array (int64 type)
        sizes : Array (int64 type)
        values : Array (any type)
        type : DataType, optional
            If not specified, a default ListType with the values' type is
            used.
        pool : MemoryPool, optional
        mask : Array (boolean type), optional
            Indicate which values are null (True) or not null (False).

        Returns
        -------
        list_view_array : LargeListViewArray

        Examples
        --------
        >>> import pyarrow as pa
        >>> values = pa.array([1, 2, 3, 4])
        >>> offsets = pa.array([0, 1, 2])
        >>> sizes = pa.array([2, 2, 2])
        >>> pa.LargeListViewArray.from_arrays(offsets, sizes, values)
        <pyarrow.lib.LargeListViewArray object at ...>
        [
          [
            1,
            2
          ],
          [
            2,
            3
          ],
          [
            3,
            4
          ]
        ]
        >>> # use a null mask to represent null values
        >>> mask = pa.array([False, True, False])
        >>> pa.LargeListViewArray.from_arrays(offsets, sizes, values, mask=mask)
        <pyarrow.lib.LargeListViewArray object at ...>
        [
          [
            1,
            2
          ],
          null,
          [
            3,
            4
          ]
        ]
        >>> # null values can be defined in either offsets or sizes arrays
        >>> # WARNING: this will result in a copy of the offsets or sizes arrays
        >>> offsets = pa.array([0, None, 2])
        >>> pa.LargeListViewArray.from_arrays(offsets, sizes, values)
        <pyarrow.lib.LargeListViewArray object at ...>
        [
          [
            1,
            2
          ],
          null,
          [
            3,
            4
          ]
        ]
        """
    @property
    def values(self) -> Array:
        """
        Return the underlying array of values which backs the LargeListArray
        ignoring the array's offset.

        The values array may be out of order and/or contain additional values
        that are not found in the logical representation of the array. The only
        guarantee is that each non-null value in the ListView Array is contiguous.

        Compare with :meth:`flatten`, which returns only the non-null
        values taking into consideration the array's order and offset.

        Returns
        -------
        values : Array

        See Also
        --------
        LargeListArray.flatten : ...

        Examples
        --------

        The values include null elements from sub-lists:

        >>> import pyarrow as pa
        >>> values = [1, 2, None, 3, 4]
        >>> offsets = [0, 0, 1]
        >>> sizes = [2, 0, 4]
        >>> array = pa.LargeListViewArray.from_arrays(offsets, sizes, values)
        >>> array
        <pyarrow.lib.LargeListViewArray object at ...>
        [
          [
            1,
            2
          ],
          [],
          [
            2,
            null,
            3,
            4
          ]
        ]
        >>> array.values
        <pyarrow.lib.Int64Array object at ...>
        [
          1,
          2,
          null,
          3,
          4
        ]
        """
    @property
    def offsets(self) -> Int64Array:
        """
        Return the list view offsets as an int64 array.

        The returned array will not have a validity bitmap, so you cannot
        expect to pass it to `LargeListViewArray.from_arrays` and get back the
        same list array if the original one has nulls.

        Returns
        -------
        offsets : Int64Array

        Examples
        --------

        >>> import pyarrow as pa
        >>> values = [1, 2, None, 3, 4]
        >>> offsets = [0, 0, 1]
        >>> sizes = [2, 0, 4]
        >>> array = pa.LargeListViewArray.from_arrays(offsets, sizes, values)
        >>> array.offsets
        <pyarrow.lib.Int64Array object at ...>
        [
          0,
          0,
          1
        ]
        """
    @property
    def sizes(self) -> Int64Array:
        """
        Return the list view sizes as an int64 array.

        The returned array will not have a validity bitmap, so you cannot
        expect to pass it to `LargeListViewArray.from_arrays` and get back the
        same list array if the original one has nulls.

        Returns
        -------
        sizes : Int64Array

        Examples
        --------

        >>> import pyarrow as pa
        >>> values = [1, 2, None, 3, 4]
        >>> offsets = [0, 0, 1]
        >>> sizes = [2, 0, 4]
        >>> array = pa.LargeListViewArray.from_arrays(offsets, sizes, values)
        >>> array.sizes
        <pyarrow.lib.Int64Array object at ...>
        [
          2,
          0,
          4
        ]
        """

class FixedSizeListArray(BaseListArray[scalar.FixedSizeListScalar[_DataTypeT, _Size]]):
    @overload
    @classmethod
    def from_arrays(
        cls,
        values: Array[Scalar[_DataTypeT]],
        *,
        type: None = None,
        mask: Mask | None = None,
    ) -> FixedSizeListArray[_DataTypeT, None]: ...
    @overload
    @classmethod
    def from_arrays(
        cls,
        values: Array[Scalar[_DataTypeT]],
        limit_size: _Size,
        *,
        type: None = None,
        mask: Mask | None = None,
    ) -> FixedSizeListArray[_DataTypeT, _Size]: ...
    @classmethod
    def from_arrays(cls, *args, **kwargs):
        """
        Construct FixedSizeListArray from array of values and a list length.

        Parameters
        ----------
        values : Array (any type)
        list_size : int
            The fixed length of the lists.
        type : DataType, optional
            If not specified, a default ListType with the values' type and
            `list_size` length is used.
        mask : Array (boolean type), optional
            Indicate which values are null (True) or not null (False).


        Returns
        -------
        FixedSizeListArray

        Examples
        --------

        Create from a values array and a list size:

        >>> import pyarrow as pa
        >>> values = pa.array([1, 2, 3, 4])
        >>> arr = pa.FixedSizeListArray.from_arrays(values, 2)
        >>> arr
        <pyarrow.lib.FixedSizeListArray object at ...>
        [
          [
            1,
            2
          ],
          [
            3,
            4
          ]
        ]

        Or create from a values array, list size and matching type:

        >>> typ = pa.list_(pa.field("values", pa.int64()), 2)
        >>> arr = pa.FixedSizeListArray.from_arrays(values, type=typ)
        >>> arr
        <pyarrow.lib.FixedSizeListArray object at ...>
        [
          [
            1,
            2
          ],
          [
            3,
            4
          ]
        ]
        """
    @property
    def values(self) -> BaseListArray[scalar.ListScalar[_DataTypeT]]:
        """
        Return the underlying array of values which backs the
        FixedSizeListArray.

        Note even null elements are included.

        Compare with :meth:`flatten`, which returns only the non-null
        sub-list values.

        Returns
        -------
        values : Array

        See Also
        --------
        FixedSizeListArray.flatten : ...

        Examples
        --------
        >>> import pyarrow as pa
        >>> array = pa.array([[1, 2], None, [3, None]], type=pa.list_(pa.int32(), 2))
        >>> array.values
        <pyarrow.lib.Int32Array object at ...>
        [
          1,
          2,
          null,
          null,
          3,
          null
        ]

        """

_MapKeyT = TypeVar("_MapKeyT", bound=_BasicDataType)
_MapItemT = TypeVar("_MapItemT", bound=_BasicDataType)

class MapArray(ListArray[scalar.MapScalar[_MapKeyT, _MapItemT]]):
    @overload
    @classmethod
    def from_arrays(
        cls,
        offsets: Int64Array,
        keys: Array[Scalar[_MapKeyT]],
        items: Array[Scalar[_MapItemT]],
        *,
        type: None = None,
        pool: MemoryPool | None = None,
        mask: Mask | None = None,
    ) -> MapArray[_MapKeyT, _MapItemT]: ...
    @overload
    @classmethod
    def from_arrays(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls,
        offsets: Int64Array,
        values: Array,
        *,
        type: MapType[_MapKeyT, _MapItemT],
        pool: MemoryPool | None = None,
        mask: Mask | None = None,
    ) -> MapArray[_MapKeyT, _MapItemT]: ...
    @classmethod
    def from_arrays(cls, *args, **kwargs):  # pyright: ignore[reportIncompatibleMethodOverride]
        """
        Construct MapArray from arrays of int32 offsets and key, item arrays.

        Parameters
        ----------
        offsets : array-like or sequence (int32 type)
        keys : array-like or sequence (any type)
        items : array-like or sequence (any type)
        type : DataType, optional
            If not specified, a default MapArray with the keys' and items' type is used.
        pool : MemoryPool
        mask : Array (boolean type), optional
            Indicate which values are null (True) or not null (False).

        Returns
        -------
        map_array : MapArray

        Examples
        --------
        First, let's understand the structure of our dataset when viewed in a rectangular data model.
        The total of 5 respondents answered the question "How much did you like the movie x?".
        The value -1 in the integer array means that the value is missing. The boolean array
        represents the null bitmask corresponding to the missing values in the integer array.

        >>> import pyarrow as pa
        >>> movies_rectangular = np.ma.masked_array(
        ...     [[10, -1, -1], [8, 4, 5], [-1, 10, 3], [-1, -1, -1], [-1, -1, -1]],
        ...     [
        ...         [False, True, True],
        ...         [False, False, False],
        ...         [True, False, False],
        ...         [True, True, True],
        ...         [True, True, True],
        ...     ],
        ... )

        To represent the same data with the MapArray and from_arrays, the data is
        formed like this:

        >>> offsets = [
        ...     0,  #  -- row 1 start
        ...     1,  #  -- row 2 start
        ...     4,  #  -- row 3 start
        ...     6,  #  -- row 4 start
        ...     6,  #  -- row 5 start
        ...     6,  #  -- row 5 end
        ... ]
        >>> movies = [
        ...     "Dark Knight",  #  ---------------------------------- row 1
        ...     "Dark Knight",
        ...     "Meet the Parents",
        ...     "Superman",  #  -- row 2
        ...     "Meet the Parents",
        ...     "Superman",  #  ----------------- row 3
        ... ]
        >>> likings = [
        ...     10,  #  -------- row 1
        ...     8,
        ...     4,
        ...     5,  #  --- row 2
        ...     10,
        ...     3,  #  ------ row 3
        ... ]
        >>> pa.MapArray.from_arrays(offsets, movies, likings).to_pandas()
        0                                  [(Dark Knight, 10)]
        1    [(Dark Knight, 8), (Meet the Parents, 4), (Sup...
        2              [(Meet the Parents, 10), (Superman, 3)]
        3                                                   []
        4                                                   []
        dtype: object

        If the data in the empty rows needs to be marked as missing, it's possible
        to do so by modifying the offsets argument, so that we specify `None` as
        the starting positions of the rows we want marked as missing. The end row
        offset still has to refer to the existing value from keys (and values):

        >>> offsets = [
        ...     0,  #  ----- row 1 start
        ...     1,  #  ----- row 2 start
        ...     4,  #  ----- row 3 start
        ...     None,  #  -- row 4 start
        ...     None,  #  -- row 5 start
        ...     6,  #  ----- row 5 end
        ... ]
        >>> pa.MapArray.from_arrays(offsets, movies, likings).to_pandas()
        0                                  [(Dark Knight, 10)]
        1    [(Dark Knight, 8), (Meet the Parents, 4), (Sup...
        2              [(Meet the Parents, 10), (Superman, 3)]
        3                                                 None
        4                                                 None
        dtype: object
        """
    @property
    def keys(self) -> Array:
        """Flattened array of keys across all maps in array"""
    @property
    def items(self) -> Array:
        """Flattened array of items across all maps in array"""

class UnionArray(Array[scalar.UnionScalar]):
    @deprecated("Use fields() instead")
    def child(self, pos: int) -> Field:
        """
        DEPRECATED, use field() instead.

        Parameters
        ----------
        pos : int
            The physical index of the union child field (not its type code).

        Returns
        -------
        field : pyarrow.Field
            The given child field.
        """
    def field(self, pos: int) -> Array:
        """
        Return the given child field as an individual array.

        For sparse unions, the returned array has its offset, length,
        and null count adjusted.

        For dense unions, the returned array is unchanged.

        Parameters
        ----------
        pos : int
            The physical index of the union child field (not its type code).

        Returns
        -------
        field : Array
            The given child field.
        """
    @property
    def type_codes(self) -> Int8Array:
        """Get the type codes array."""
    @property
    def offsets(self) -> Int32Array:
        """
        Get the value offsets array (dense arrays only).

        Does not account for any slice offset.
        """
    @staticmethod
    def from_dense(
        type: Int8Array,
        value_offsets: Int32Array,
        children: NullableCollection[Array],
        field_names: list[str] | None = None,
        type_codes: Int8Array | None = None,
    ) -> UnionArray:
        """
        Construct dense UnionArray from arrays of int8 types, int32 offsets and
        children arrays

        Parameters
        ----------
        types : Array (int8 type)
        value_offsets : Array (int32 type)
        children : list
        field_names : list
        type_codes : list

        Returns
        -------
        union_array : UnionArray
        """
    @staticmethod
    def from_sparse(
        types: Int8Array,
        children: NullableCollection[Array],
        field_names: list[str] | None = None,
        type_codes: Int8Array | None = None,
    ) -> UnionArray:
        """
        Construct sparse UnionArray from arrays of int8 types and children
        arrays

        Parameters
        ----------
        types : Array (int8 type)
        children : list
        field_names : list
        type_codes : list

        Returns
        -------
        union_array : UnionArray
        """

class StringArray(Array[scalar.StringScalar]):
    @staticmethod
    def from_buffers(  # type: ignore[override]
        length: int,
        value_offsets: Buffer,
        data: Buffer,
        null_bitmap: Buffer | None = None,
        null_count: int | None = -1,
        offset: int | None = 0,
    ) -> StringArray:
        """
        Construct a StringArray from value_offsets and data buffers.
        If there are nulls in the data, also a null_bitmap and the matching
        null_count must be passed.

        Parameters
        ----------
        length : int
        value_offsets : Buffer
        data : Buffer
        null_bitmap : Buffer, optional
        null_count : int, default 0
        offset : int, default 0

        Returns
        -------
        string_array : StringArray
        """

class LargeStringArray(Array[scalar.LargeStringScalar]):
    @staticmethod
    def from_buffers(  # type: ignore[override]
        length: int,
        value_offsets: Buffer,
        data: Buffer,
        null_bitmap: Buffer | None = None,
        null_count: int | None = -1,
        offset: int | None = 0,
    ) -> StringArray:
        """
        Construct a LargeStringArray from value_offsets and data buffers.
        If there are nulls in the data, also a null_bitmap and the matching
        null_count must be passed.

        Parameters
        ----------
        length : int
        value_offsets : Buffer
        data : Buffer
        null_bitmap : Buffer, optional
        null_count : int, default 0
        offset : int, default 0

        Returns
        -------
        string_array : StringArray
        """

class StringViewArray(Array[scalar.StringViewScalar]): ...

class BinaryArray(Array[scalar.BinaryScalar]):
    @property
    def total_values_length(self) -> int:
        """
        The number of bytes from beginning to end of the data buffer addressed
        by the offsets of this BinaryArray.
        """

class LargeBinaryArray(Array[scalar.LargeBinaryScalar]):
    @property
    def total_values_length(self) -> int:
        """
        The number of bytes from beginning to end of the data buffer addressed
        by the offsets of this LargeBinaryArray.
        """

class BinaryViewArray(Array[scalar.BinaryViewScalar]): ...

class DictionaryArray(Array[scalar.DictionaryScalar[_IndexT, _BasicValueT]]):
    def dictionary_encode(self) -> Self: ...  # type: ignore[override]
    def dictionary_decode(self) -> Array[Scalar[_BasicValueT]]:
        """
        Decodes the DictionaryArray to an Array.
        """
    @property
    def indices(self) -> Array[Scalar[_IndexT]]: ...
    @property
    def dictionary(self) -> Array[Scalar[_BasicValueT]]: ...
    @staticmethod
    def from_buffers(  # type: ignore[override]
        type: _BasicValueT,
        length: int,
        buffers: list[Buffer],
        dictionary: Array | np.ndarray | pd.Series,
        null_count: int = -1,
        offset: int = 0,
    ) -> DictionaryArray[Any, _BasicValueT]:
        """
        Construct a DictionaryArray from buffers.

        Parameters
        ----------
        type : pyarrow.DataType
        length : int
            The number of values in the array.
        buffers : List[Buffer]
            The buffers backing the indices array.
        dictionary : pyarrow.Array, ndarray or pandas.Series
            The array of values referenced by the indices.
        null_count : int, default -1
            The number of null entries in the indices array. Negative value means that
            the null count is not known.
        offset : int, default 0
            The array's logical offset (in values, not in bytes) from the
            start of each buffer.

        Returns
        -------
        dict_array : DictionaryArray
        """
    @staticmethod
    def from_arrays(
        indices: Indices,
        dictionary: Array | np.ndarray | pd.Series,
        mask: np.ndarray | pd.Series | BooleanArray | None = None,
        ordered: bool = False,
        from_pandas: bool = False,
        safe: bool = True,
        memory_pool: MemoryPool | None = None,
    ) -> DictionaryArray:
        """
        Construct a DictionaryArray from indices and values.

        Parameters
        ----------
        indices : pyarrow.Array, numpy.ndarray or pandas.Series, int type
            Non-negative integers referencing the dictionary values by zero
            based index.
        dictionary : pyarrow.Array, ndarray or pandas.Series
            The array of values referenced by the indices.
        mask : ndarray or pandas.Series, bool type
            True values indicate that indices are actually null.
        ordered : bool, default False
            Set to True if the category values are ordered.
        from_pandas : bool, default False
            If True, the indices should be treated as though they originated in
            a pandas.Categorical (null encoded as -1).
        safe : bool, default True
            If True, check that the dictionary indices are in range.
        memory_pool : MemoryPool, default None
            For memory allocations, if required, otherwise uses default pool.

        Returns
        -------
        dict_array : DictionaryArray
        """

class StructArray(Array[scalar.StructScalar]):
    def field(self, index: int | str) -> Array:
        """
        Retrieves the child array belonging to field.

        Parameters
        ----------
        index : Union[int, str]
            Index / position or name of the field.

        Returns
        -------
        result : Array
        """
    def flatten(self, memory_pool: MemoryPool | None = None) -> list[Array]:
        """
        Return one individual array for each field in the struct.

        Parameters
        ----------
        memory_pool : MemoryPool, default None
            For memory allocations, if required, otherwise use default pool.

        Returns
        -------
        result : List[Array]
        """
    @staticmethod
    def from_arrays(
        arrays: Iterable[Array],
        names: list[str] | None = None,
        fields: list[Field] | None = None,
        mask=None,
        memory_pool: MemoryPool | None = None,
        type: types.StructType | None = None,
    ) -> StructArray:
        """
        Construct StructArray from collection of arrays representing
        each field in the struct.

        Either field names, field instances or a struct type must be passed.

        Parameters
        ----------
        arrays : sequence of Array
        names : List[str] (optional)
            Field names for each struct child.
        fields : List[Field] (optional)
            Field instances for each struct child.
        mask : pyarrow.Array[bool] (optional)
            Indicate which values are null (True) or not null (False).
        memory_pool : MemoryPool (optional)
            For memory allocations, if required, otherwise uses default pool.
        type : pyarrow.StructType (optional)
            Struct type for name and type of each child.

        Returns
        -------
        result : StructArray
        """
    def sort(self, order: Order = "ascending", by: str | None = None, **kwargs) -> StructArray:
        """
        Sort the StructArray

        Parameters
        ----------
        order : str, default "ascending"
            Which order to sort values in.
            Accepted values are "ascending", "descending".
        by : str or None, default None
            If to sort the array by one of its fields
            or by the whole array.
        **kwargs : dict, optional
            Additional sorting options.
            As allowed by :class:`SortOptions`

        Returns
        -------
        result : StructArray
        """

class RunEndEncodedArray(Array[scalar.RunEndEncodedScalar[_RunEndType, _BasicValueT]]):
    @overload
    @staticmethod
    def from_arrays(
        run_ends: Int16Array,
        values: Array,
        type: DataType | None = None,
    ) -> RunEndEncodedArray[types.Int16Type, _BasicValueT]: ...
    @overload
    @staticmethod
    def from_arrays(
        run_ends: Int32Array,
        values: Array,
        type: DataType | None = None,
    ) -> RunEndEncodedArray[types.Int32Type, _BasicValueT]: ...
    @overload
    @staticmethod
    def from_arrays(
        run_ends: Int64Array,
        values: Array,
        type: DataType | None = None,
    ) -> RunEndEncodedArray[types.Int64Type, _BasicValueT]: ...
    @staticmethod
    def from_arrays(*args, **kwargs):
        """
        Construct RunEndEncodedArray from run_ends and values arrays.

        Parameters
        ----------
        run_ends : Array (int16, int32, or int64 type)
            The run_ends array.
        values : Array (any type)
            The values array.
        type : pyarrow.DataType, optional
            The run_end_encoded(run_end_type, value_type) array type.

        Returns
        -------
        RunEndEncodedArray
        """
    @staticmethod
    def from_buffers(  # pyright: ignore[reportIncompatibleMethodOverride]
        type: DataType,
        length: int,
        buffers: list[Buffer],
        null_count: int = -1,
        offset=0,
        children: tuple[Array, Array] | None = None,
    ) -> RunEndEncodedArray[Any, _BasicValueT]:
        """
        Construct a RunEndEncodedArray from all the parameters that make up an
        Array.

        RunEndEncodedArrays do not have buffers, only children arrays, but this
        implementation is needed to satisfy the Array interface.

        Parameters
        ----------
        type : DataType
            The run_end_encoded(run_end_type, value_type) type.
        length : int
            The logical length of the run-end encoded array. Expected to match
            the last value of the run_ends array (children[0]) minus the offset.
        buffers : List[Buffer]
            Empty List or [None].
        null_count : int, default -1
            The number of null entries in the array. Run-end encoded arrays
            are specified to not have valid bits and null_count always equals 0.
        offset : int, default 0
            The array's logical offset (in values, not in bytes) from the
            start of each buffer.
        children : List[Array]
            Nested type children containing the run_ends and values arrays.

        Returns
        -------
        RunEndEncodedArray
        """
    @property
    def run_ends(self) -> Array[scalar.Scalar[_RunEndType]]:
        """
        An array holding the logical indexes of each run-end.

        The physical offset to the array is applied.
        """
    @property
    def values(self) -> Array[scalar.Scalar[_BasicValueT]]:
        """
        An array holding the values of each run.

        The physical offset to the array is applied.
        """
    def find_physical_offset(self) -> int:
        """
        Find the physical offset of this REE array.

        This is the offset of the run that contains the value of the first
        logical element of this array considering its offset.

        This function uses binary-search, so it has a O(log N) cost.
        """
    def find_physical_length(self) -> int:
        """
        Find the physical length of this REE array.

        The physical length of an REE is the number of physical values (and
        run-ends) necessary to represent the logical range of values from offset
        to length.

        This function uses binary-search, so it has a O(log N) cost.
        """

_ArrayT = TypeVar("_ArrayT", bound=Array)

class ExtensionArray(Array[scalar.ExtensionScalar], Generic[_ArrayT]):
    @property
    def storage(self) -> Any: ...
    @staticmethod
    def from_storage(typ: types.BaseExtensionType, storage: _ArrayT) -> ExtensionArray[_ArrayT]:
        """
        Construct ExtensionArray from type and storage array.

        Parameters
        ----------
        typ : DataType
            The extension type for the result array.
        storage : Array
            The underlying storage for the result array.

        Returns
        -------
        ext_array : ExtensionArray
        """

class JsonArray(ExtensionArray[_ArrayT]):
    """
    Concrete class for Arrow arrays of JSON data type.

    This does not guarantee that the JSON data actually
    is valid JSON.

    Examples
    --------
    Define the extension type for JSON array

    >>> import pyarrow as pa
    >>> json_type = pa.json_(pa.large_utf8())

    Create an extension array

    >>> arr = [None, '{ "id":30, "values":["a", "b"] }']
    >>> storage = pa.array(arr, pa.large_utf8())
    >>> pa.ExtensionArray.from_storage(json_type, storage)
    <pyarrow.lib.JsonArray object at ...>
    [
      null,
      "{ "id":30, "values":["a", "b"] }"
    ]
    """

class UuidArray(ExtensionArray[_ArrayT]): ...

class FixedShapeTensorArray(ExtensionArray[_ArrayT]):
    """
    Concrete class for fixed shape tensor extension arrays.

    Examples
    --------
    Define the extension type for tensor array

    >>> import pyarrow as pa
    >>> tensor_type = pa.fixed_shape_tensor(pa.int32(), [2, 2])

    Create an extension array

    >>> arr = [[1, 2, 3, 4], [10, 20, 30, 40], [100, 200, 300, 400]]
    >>> storage = pa.array(arr, pa.list_(pa.int32(), 4))
    >>> pa.ExtensionArray.from_storage(tensor_type, storage)
    <pyarrow.lib.FixedShapeTensorArray object at ...>
    [
      [
        1,
        2,
        3,
        4
      ],
      [
        10,
        20,
        30,
        40
      ],
      [
        100,
        200,
        300,
        400
      ]
    ]
    """

    def to_numpy_ndarray(self) -> np.ndarray:
        """
        Convert fixed shape tensor extension array to a multi-dimensional numpy.ndarray.

        The resulting ndarray will have (ndim + 1) dimensions.
        The size of the first dimension will be the length of the fixed shape tensor array
        and the rest of the dimensions will match the permuted shape of the fixed
        shape tensor.

        The conversion is zero-copy.

        Returns
        -------
        numpy.ndarray
            Ndarray representing tensors in the fixed shape tensor array concatenated
            along the first dimension.
        """
    def to_tensor(self) -> Tensor:
        """
        Convert fixed shape tensor extension array to a pyarrow.Tensor.

        The resulting Tensor will have (ndim + 1) dimensions.
        The size of the first dimension will be the length of the fixed shape tensor array
        and the rest of the dimensions will match the permuted shape of the fixed
        shape tensor.

        The conversion is zero-copy.

        Returns
        -------
        pyarrow.Tensor
            Tensor representing tensors in the fixed shape tensor array concatenated
            along the first dimension.
        """

    @classmethod
    def from_numpy_ndarray(cls, obj: np.ndarray) -> Self:
        """
        Convert numpy tensors (ndarrays) to a fixed shape tensor extension array.
        The first dimension of ndarray will become the length of the fixed
        shape tensor array.
        If input array data is not contiguous a copy will be made.

        Parameters
        ----------
        obj : numpy.ndarray

        Examples
        --------
        >>> import pyarrow as pa
        >>> import numpy as np
        >>> arr = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]], dtype=np.float32)
        >>> pa.FixedShapeTensorArray.from_numpy_ndarray(arr)
        <pyarrow.lib.FixedShapeTensorArray object at ...>
        [
          [
            1,
            2,
            3,
            4,
            5,
            6
          ],
          [
            1,
            2,
            3,
            4,
            5,
            6
          ]
        ]
        """

class OpaqueArray(ExtensionArray[_ArrayT]):
    """
    Concrete class for opaque extension arrays.

    Examples
    --------
    Define the extension type for an opaque array

    >>> import pyarrow as pa
    >>> opaque_type = pa.opaque(
    ...     pa.binary(),
    ...     type_name="geometry",
    ...     vendor_name="postgis",
    ... )

    Create an extension array

    >>> arr = [None, b"data"]
    >>> storage = pa.array(arr, pa.binary())
    >>> pa.ExtensionArray.from_storage(opaque_type, storage)
    <pyarrow.lib.OpaqueArray object at ...>
    [
      null,
      64617461
    ]
    """

class Bool8Array(ExtensionArray):
    """
    Concrete class for bool8 extension arrays.

    Examples
    --------
    Define the extension type for an bool8 array

    >>> import pyarrow as pa
    >>> bool8_type = pa.bool8()

    Create an extension array

    >>> arr = [-1, 0, 1, 2, None]
    >>> storage = pa.array(arr, pa.int8())
    >>> pa.ExtensionArray.from_storage(bool8_type, storage)
    <pyarrow.lib.Bool8Array object at ...>
    [
      -1,
      0,
      1,
      2,
      null
    ]
    """

    def to_numpy(self, zero_copy_only: bool = ..., writable: bool = ...) -> np.ndarray:
        """
        Return a NumPy bool view or copy of this array.

        By default, tries to return a view of this array. This is only
        supported for arrays without any nulls.

        Parameters
        ----------
        zero_copy_only : bool, default True
            If True, an exception will be raised if the conversion to a numpy
            array would require copying the underlying data (e.g. in presence
            of nulls).
        writable : bool, default False
            For numpy arrays created with zero copy (view on the Arrow data),
            the resulting array is not writable (Arrow data is immutable).
            By setting this to True, a copy of the array is made to ensure
            it is writable.

        Returns
        -------
        array : numpy.ndarray
        """
    @classmethod
    def from_storage(cls, storage: Int8Array) -> Self:  # type: ignore[override]
        """
        Construct Bool8Array from Int8Array storage.

        Parameters
        ----------
        storage : Int8Array
            The underlying storage for the result array.

        Returns
        -------
        bool8_array : Bool8Array
        """
    @classmethod
    def from_numpy(cls, obj: np.ndarray) -> Self:
        """
        Convert numpy array to a bool8 extension array without making a copy.
        The input array must be 1-dimensional, with either bool_ or int8 dtype.

        Parameters
        ----------
        obj : numpy.ndarray

        Returns
        -------
        bool8_array : Bool8Array

        Examples
        --------
        >>> import pyarrow as pa
        >>> import numpy as np
        >>> arr = np.array([True, False, True], dtype=np.bool_)
        >>> pa.Bool8Array.from_numpy(arr)
        <pyarrow.lib.Bool8Array object at ...>
        [
          1,
          0,
          1
        ]
        """

def concat_arrays(arrays: Iterable[_ArrayT], memory_pool: MemoryPool | None = None) -> _ArrayT:
    """
    Concatenate the given arrays.

    The contents of the input arrays are copied into the returned array.

    Raises
    ------
    ArrowInvalid
        If not all of the arrays have the same type.

    Parameters
    ----------
    arrays : iterable of pyarrow.Array
        Arrays to concatenate, must be identically typed.
    memory_pool : MemoryPool, default None
        For memory allocations. If None, the default pool is used.

    Examples
    --------
    >>> import pyarrow as pa
    >>> arr1 = pa.array([2, 4, 5, 100])
    >>> arr2 = pa.array([2, 4])
    >>> pa.concat_arrays([arr1, arr2])
    <pyarrow.lib.Int64Array object at ...>
    [
      2,
      4,
      5,
      100,
      2,
      4
    ]

    """

def _empty_array(type: _DataTypeT) -> Array[scalar.Scalar[_DataTypeT]]:
    """
    Create empty array of the given type.
    """

__all__ = [
    "array",
    "asarray",
    "nulls",
    "repeat",
    "infer_type",
    "_PandasConvertible",
    "Array",
    "NullArray",
    "BooleanArray",
    "NumericArray",
    "IntegerArray",
    "FloatingPointArray",
    "Int8Array",
    "UInt8Array",
    "Int16Array",
    "UInt16Array",
    "Int32Array",
    "UInt32Array",
    "Int64Array",
    "UInt64Array",
    "Date32Array",
    "Date64Array",
    "TimestampArray",
    "Time32Array",
    "Time64Array",
    "DurationArray",
    "MonthDayNanoIntervalArray",
    "HalfFloatArray",
    "FloatArray",
    "DoubleArray",
    "FixedSizeBinaryArray",
    "Decimal32Array",
    "Decimal64Array",
    "Decimal128Array",
    "Decimal256Array",
    "BaseListArray",
    "ListArray",
    "LargeListArray",
    "ListViewArray",
    "LargeListViewArray",
    "FixedSizeListArray",
    "MapArray",
    "UnionArray",
    "StringArray",
    "LargeStringArray",
    "StringViewArray",
    "BinaryArray",
    "LargeBinaryArray",
    "BinaryViewArray",
    "DictionaryArray",
    "StructArray",
    "RunEndEncodedArray",
    "ExtensionArray",
    "Bool8Array",
    "UuidArray",
    "JsonArray",
    "OpaqueArray",
    "FixedShapeTensorArray",
    "concat_arrays",
    "_empty_array",
]
