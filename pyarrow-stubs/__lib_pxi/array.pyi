from collections.abc import Callable
from typing import Any, Generic, Iterator, Literal, Self, TypeAlias, TypeVar, overload

import numpy as np
import pandas as pd

from pandas.core.dtypes.base import ExtensionDtype
from pyarrow._compute import CastOptions
from pyarrow._stubs_typing import ArrayLike, Order
from pyarrow.lib import Buffer, MemoryPool, _Weakrefable

from . import scalar
from .device import DeviceAllocationType
from .scalar import Scalar
from .types import DataType, MapType, _AsPyType, _BasicDataType, _DataTypeT

_ConvertAs = TypeVar("_ConvertAs", pd.DataFrame, pd.Series)
_SchemaCapsule: TypeAlias = Any
_ArrayCapsule: TypeAlias = Any

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
    ) -> _ConvertAs: ...

_CastAs = TypeVar("_CastAs", bound=DataType)
_ScalarT = TypeVar("_ScalarT", bound=Scalar)

class Array(_PandasConvertible[pd.Series], Generic[_ScalarT]):
    def diff(self, other: Self) -> str: ...
    def cast(
        self,
        target_type: _CastAs,
        safe: bool = True,
        options: CastOptions | None = None,
        memory_pool: MemoryPool | None = None,
    ) -> Array[Scalar[_CastAs]]: ...
    def view(self, target_type: _CastAs) -> Array[Scalar[_CastAs]]: ...
    def sum(self, **kwargs) -> _ScalarT: ...
    def unique(self) -> Self: ...
    def dictionary_encode(self, null_encoding: str = "mask") -> DictionaryArray: ...
    @overload
    @staticmethod
    def from_pandas(
        obj: pd.Series | np.ndarray | ArrayLike,
        *,
        mask: bool | None = None,
        type: _DataTypeT,
        safe: bool = True,
        memory_pool: MemoryPool | None = None,
    ) -> Array[Scalar[_DataTypeT]]: ...
    @overload
    @staticmethod
    def from_pandas(
        obj: pd.Series | np.ndarray | ArrayLike,
        *,
        mask: bool | None = None,
        safe: bool = True,
        memory_pool: MemoryPool | None = None,
    ) -> Array[Scalar]: ...
    @staticmethod
    def from_buffers(
        type: _DataTypeT,
        length: int,
        buffers: list[Buffer],
        null_count: int = -1,
        offset=0,
        children: list[Array[Scalar[_DataTypeT]]] | None = None,
    ) -> Array[Scalar[_DataTypeT]]: ...
    @property
    def null_count(self) -> int: ...
    @property
    def nbytes(self) -> int: ...
    def get_total_buffer_size(self) -> int: ...
    def __sizeof__(self) -> int: ...
    def __iter__(self) -> Iterator[_ScalarT]: ...
    def to_string(
        self,
        *,
        indent: int = 2,
        top_level_indent: int = 0,
        window: int = 10,
        container_window: int = 2,
        skip_new_lines: bool = False,
    ) -> str: ...
    format = to_string
    def equals(self, other: Self) -> bool: ...
    def __len__(self) -> int: ...
    def is_null(self, *, nan_is_null: bool = False) -> BooleanArray: ...
    def is_nan(self) -> BooleanArray: ...
    def is_valid(self) -> BooleanArray: ...
    def fill_null(
        self: Array[Scalar[_BasicDataType[_AsPyType], Any]], fill_value: _AsPyType
    ) -> Array[Scalar[_BasicDataType[_AsPyType], Any]]: ...
    @overload
    def __getitem__(self, key: int) -> _ScalarT: ...
    @overload
    def __getitem__(self, key: slice) -> Self: ...
    def slice(self, offset: int = 0, length: int | None = None) -> Self: ...
    def take(self, indices: list[int] | Array | ArrayLike) -> Self: ...
    def drop_null(self) -> Self: ...
    def filter(
        self,
        mask: list[bool] | Array | ArrayLike,
        *,
        null_selection_behavior: Literal["drop", "emit_null"] = "drop",
    ) -> Self: ...
    @overload
    def index(
        self,
        value: _ScalarT,
        start: int | None = None,
        end: int | None = None,
        *,
        memory_pool: MemoryPool | None = None,
    ) -> scalar.Int64Scalar: ...
    @overload
    def index(
        self: Array[Scalar[_BasicDataType[_AsPyType], Any]],
        value: _AsPyType,
        start: int | None = None,
        end: int | None = None,
        *,
        memory_pool: MemoryPool | None = None,
    ) -> scalar.Int64Scalar: ...
    def sort(self, order: Order = "ascending", **kwargs) -> Self: ...
    def __array__(self, dtype: np.dtype | None = None, copy: bool | None = None) -> np.ndarray: ...
    def to_numpy(self, zero_copy_only: bool = True, writable: bool = False) -> np.ndarray: ...
    def to_pylist(self: Array[Scalar[_BasicDataType[_AsPyType], Any]]) -> list[_AsPyType]: ...
    tolist = to_pylist
    def validate(self, *, full: bool = False) -> None: ...
    @property
    def offset(self) -> int: ...
    def buffers(self) -> list[Buffer | None]: ...
    def _export_to_c(self, out_ptr: int, out_schema_ptr: int = 0) -> None: ...
    @classmethod
    def _import_from_c(cls, in_ptr: int, type: int | DataType) -> Self: ...
    def __arrow_c_array__(self, requested_schema=None) -> tuple[_SchemaCapsule, _ArrayCapsule]: ...
    @classmethod
    def _import_from_c_capsule(
        cls, schema_capsule: _SchemaCapsule, array_capsule: _ArrayCapsule
    ) -> Self: ...
    def _export_to_c_device(self, out_ptr: int, out_schema_ptr: int = 0) -> None: ...
    @classmethod
    def _import_from_c_device(cls, in_ptr: int, type: DataType | int) -> Self: ...
    def __arrow_c_device_array__(
        self, requested_schema=None, **kwargs
    ) -> tuple[_SchemaCapsule, _ArrayCapsule]: ...
    @classmethod
    def _import_from_c_device_capsule(
        cls, schema_capsule: _SchemaCapsule, array_capsule: _ArrayCapsule
    ) -> Self: ...
    def __dlpack__(self, stream: int | None = None) -> Any: ...
    def __dlpack_device__(self) -> tuple[int, int]: ...
    @property
    def device_type(self) -> DeviceAllocationType: ...
    @property
    def is_cpu(self) -> bool: ...

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
class TimestampArray(NumericArray[scalar.TimestampScalar]): ...
class Time32Array(NumericArray[scalar.Time32Scalar]): ...
class Time64Array(NumericArray[scalar.Time64Scalar]): ...
class DurationArray(NumericArray[scalar.DurationScalar]): ...
class MonthDayNanoIntervalArray(Array[scalar.MonthDayNanoIntervalScalar]): ...
class HalfFloatArray(FloatingPointArray[scalar.HalfFloatScalar]): ...
class FloatArray(FloatingPointArray[scalar.FloatScalar]): ...
class DoubleArray(FloatingPointArray[scalar.DoubleScalar]): ...
class FixedSizeBinaryArray(Array[scalar.FixedBinaryScalar]): ...
class Decimal128Array(FixedSizeBinaryArray): ...
class Decimal256Array(FixedSizeBinaryArray): ...

class BaseListArray(Array[_ScalarT]):
    def flatten(self, recursive: bool = False) -> Array: ...
    def value_parent_indices(self) -> Int64Array: ...
    def value_lengths(self) -> Int32Array: ...

_ListValueT = TypeVar("_ListValueT", bound=_BasicDataType)

class ListArray(BaseListArray[_ScalarT]):
    @overload
    @classmethod
    def from_arrays(
        cls,
        offsets: Int32Array,
        values: Array[Scalar[_ListValueT]],
        *,
        type: None = None,
        pool: MemoryPool | None = None,
        mask: bool | None = None,
    ) -> ListArray[scalar.ListScalar[_ListValueT]]: ...
    @overload
    @classmethod
    def from_arrays(
        cls,
        offsets: Int32Array,
        values: Array,
        *,
        type: _DataTypeT,
        pool: MemoryPool | None = None,
        mask: bool | None = None,
    ) -> ListArray[scalar.ListScalar[_DataTypeT]]: ...
    @property
    def values(self) -> Array: ...
    @property
    def offsets(self) -> Int32Array: ...

class LargeListArray(BaseListArray[_ScalarT]):
    @overload
    @classmethod
    def from_arrays(
        cls,
        offsets: Int64Array,
        values: Array[Scalar[_ListValueT]],
        *,
        type: None = None,
        pool: MemoryPool | None = None,
        mask: bool | None = None,
    ) -> LargeListArray[scalar.ListScalar[_ListValueT]]: ...
    @overload
    @classmethod
    def from_arrays(
        cls,
        offsets: Int64Array,
        values: Array,
        *,
        type: _DataTypeT,
        pool: MemoryPool | None = None,
        mask: bool | None = None,
    ) -> LargeListArray[scalar.ListScalar[_DataTypeT]]: ...
    @property
    def values(self) -> Array: ...
    @property
    def offsets(self) -> Int64Array: ...

class ListViewArray(BaseListArray[_ScalarT]):
    @overload
    @classmethod
    def from_arrays(
        cls,
        offsets: Int32Array,
        values: Array[Scalar[_ListValueT]],
        *,
        type: None = None,
        pool: MemoryPool | None = None,
        mask: bool | None = None,
    ) -> ListViewArray[scalar.ListScalar[_ListValueT]]: ...
    @overload
    @classmethod
    def from_arrays(
        cls,
        offsets: Int32Array,
        values: Array,
        *,
        type: _DataTypeT,
        pool: MemoryPool | None = None,
        mask: bool | None = None,
    ) -> ListViewArray[scalar.ListScalar[_DataTypeT]]: ...
    @property
    def values(self) -> Array: ...
    @property
    def offsets(self) -> Int32Array: ...
    @property
    def sizes(self) -> Int32Array: ...

class LargeListViewArray(BaseListArray[_ScalarT]):
    @overload
    @classmethod
    def from_arrays(
        cls,
        offsets: Int64Array,
        values: Array[Scalar[_ListValueT]],
        *,
        type: None = None,
        pool: MemoryPool | None = None,
        mask: bool | None = None,
    ) -> LargeListArray[scalar.ListScalar[_ListValueT]]: ...
    @overload
    @classmethod
    def from_arrays(
        cls,
        offsets: Int64Array,
        values: Array,
        *,
        type: _DataTypeT,
        pool: MemoryPool | None = None,
        mask: bool | None = None,
    ) -> LargeListArray[scalar.ListScalar[_DataTypeT]]: ...
    @property
    def values(self) -> Array: ...
    @property
    def offsets(self) -> Int64Array: ...
    @property
    def sizes(self) -> Int64Array: ...

_MapKeyT = TypeVar("_MapKeyT", bound=_BasicDataType)
_MapItemT = TypeVar("_MapItemT", bound=_BasicDataType)

class MapArray(ListArray[_ScalarT]):
    @overload  # type: ignore[override]
    @classmethod
    def from_arrays(
        cls,
        offsets: Int64Array,
        keys: Array[Scalar[_MapKeyT]],
        items: Array[Scalar[_MapItemT]],
        *,
        type: None = None,
        pool: MemoryPool | None = None,
        mask: bool | None = None,
    ) -> MapArray[scalar.MapScalar[_MapKeyT, _MapItemT]]: ...
    @overload
    @classmethod
    def from_arrays(
        cls,
        offsets: Int64Array,
        values: Array,
        *,
        type: MapType[_MapKeyT, _MapItemT],
        pool: MemoryPool | None = None,
        mask: bool | None = None,
    ) -> MapArray[scalar.MapScalar[_MapKeyT, _MapItemT]]: ...
    @property
    def keys(self) -> Array: ...
    @property
    def items(self) -> Array: ...

class DictionaryArray: ...
