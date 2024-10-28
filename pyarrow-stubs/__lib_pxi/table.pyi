# mypy: disable-error-code="overload-overlap,type-arg,misc"

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
from typing import (
    Any,
    Generator,
    Generic,
    Iterable,
    Iterator,
    Literal,
    Mapping,
    Sequence,
    TypeVar,
    overload,
)

import numpy as np
import pandas as pd

from pyarrow._compute import CastOptions, FunctionOptions
from pyarrow._stubs_typing import (
    Indices,
    Mask,
    NullEncoding,
    NullSelectionBehavior,
    Order,
    SupportArrowArray,
    SupportArrowDeviceArray,
    SupportArrowStream,
)
from pyarrow.compute import Expression
from pyarrow.interchange.dataframe import _PyArrowDataFrame
from pyarrow.lib import Field, MemoryPool, MonthDayNano, Schema

from . import scalar
from .array import Array, NullableIterable, StructArray, _CastAs, _PandasConvertible
from .device import DeviceAllocationType
from .io import Buffer
from .ipc import RecordBatchReader
from .scalar import Int64Scalar, Scalar
from .tensor import Tensor
from .types import DataType, _AsPyType, _BasicDataType, _DataTypeT

_Scalar_CoT = TypeVar("_Scalar_CoT", bound=Scalar, covariant=True)

class ChunkedArray(_PandasConvertible[pd.Series], Generic[_Scalar_CoT]):
    @property
    def data(self) -> Self: ...
    @property
    def type(self) -> DataType: ...
    def length(self) -> int: ...
    __len__ = length
    def to_string(
        self,
        *,
        indent: int = 0,
        window: int = 5,
        container_window: int = 2,
        skip_new_lines: bool = False,
    ) -> str: ...
    format = to_string
    def validate(self, *, full: bool = False) -> None: ...
    @property
    def null_count(self) -> int: ...
    @property
    def nbytes(self) -> int: ...
    def get_total_buffer_size(self) -> int: ...
    def __sizeof__(self) -> int: ...
    @overload
    def __getitem__(self, key: slice) -> Self: ...
    @overload
    def __getitem__(self, key: int) -> _Scalar_CoT: ...
    def getitem(self, i: int) -> Scalar: ...
    def is_null(self, *, nan_is_null: bool = False) -> ChunkedArray[scalar.BooleanScalar]: ...
    def is_nan(self) -> ChunkedArray[scalar.BooleanScalar]: ...
    def is_valid(self) -> ChunkedArray[scalar.BooleanScalar]: ...
    def fill_null(self, fill_value: Scalar[_DataTypeT]) -> Self: ...
    def equals(self, other: Self) -> bool: ...
    def to_numpy(self, zero_copy_only: bool = False) -> np.ndarray: ...
    def __array__(self, dtype: np.dtype | None = None, copy: bool | None = None) -> np.ndarray: ...
    @overload
    def cast(
        self,
        target_type: None = None,
        safe: bool | None = None,
        options: CastOptions | None = None,
    ) -> Self: ...
    @overload
    def cast(
        self, target_type: _CastAs, safe: bool | None = None, options: CastOptions | None = None
    ) -> ChunkedArray[Scalar[_CastAs]]: ...
    def dictionary_encode(self, null_encoding: NullEncoding = "mask") -> Self: ...
    def flatten(self, memory_pool: MemoryPool | None = None) -> list[ChunkedArray[Any]]: ...
    def combine_chunks(self, memory_pool: MemoryPool | None = None) -> Array[_Scalar_CoT]: ...
    def unique(self) -> ChunkedArray[_Scalar_CoT]: ...
    def value_counts(self) -> StructArray: ...
    def slice(self, offset: int = 0, length: int | None = None) -> Self: ...
    def filter(self, mask: Mask, null_selection_behavior: NullSelectionBehavior = "drop"): ...
    @overload
    def index(
        self: ChunkedArray[Scalar[_BasicDataType[_AsPyType]]],
        value: Scalar[_DataTypeT] | _AsPyType,
        start: int | None = None,
        end: int | None = None,
        *,
        memory_pool: MemoryPool | None = None,
    ) -> Int64Scalar: ...
    @overload
    def index(
        self,
        value: Scalar[_DataTypeT],
        start: int | None = None,
        end: int | None = None,
        *,
        memory_pool: MemoryPool | None = None,
    ) -> Int64Scalar: ...
    def take(self, indices: Indices) -> Self: ...
    def drop_null(self) -> Self: ...
    def sort(self, order: Order = "ascending", **kwargs) -> Self: ...
    def unify_dictionaries(self, memory_pool: MemoryPool | None = None) -> Self: ...
    @property
    def num_chunks(self) -> int: ...
    def chunk(self, i: int) -> ChunkedArray[_Scalar_CoT]: ...
    @property
    def chunks(self) -> list[Array[_Scalar_CoT]]: ...
    def iterchunks(self) -> Generator[Array[_Scalar_CoT], None, None]: ...
    def __iter__(self) -> Iterator[Array[_Scalar_CoT]]: ...
    def to_pylist(
        self: ChunkedArray[Scalar[_BasicDataType[_AsPyType]]],
    ) -> list[_AsPyType | None]: ...
    def __arrow_c_stream__(self, requested_schema=None) -> Any: ...
    @classmethod
    def _import_from_c_capsule(cls, stream) -> Self: ...

@overload  # type: ignore[overload-overlap]
def chunked_array(
    values: NullableIterable[bool],
    type: None = None,
) -> ChunkedArray[scalar.BooleanScalar]: ...
@overload
def chunked_array(
    values: NullableIterable[int],
    type: None = None,
) -> ChunkedArray[scalar.Int64Scalar]: ...
@overload
def chunked_array(
    values: NullableIterable[float],
    type: None = None,
) -> ChunkedArray[scalar.DoubleScalar]: ...
@overload
def chunked_array(
    values: NullableIterable[Decimal],
    type: None = None,
) -> ChunkedArray[scalar.Decimal128Scalar]: ...
@overload
def chunked_array(
    values: NullableIterable[dict[str, Any]],
    type: None = None,
) -> ChunkedArray[scalar.StructScalar]: ...
@overload
def chunked_array(
    values: NullableIterable[dt.datetime],
    type: None = None,
) -> ChunkedArray[scalar.TimestampScalar]: ...
@overload
def chunked_array(
    values: NullableIterable[dt.date],
    type: None = None,
) -> ChunkedArray[scalar.Date32Scalar]: ...
@overload
def chunked_array(
    values: NullableIterable[dt.time],
    type: None = None,
) -> ChunkedArray[scalar.Time64Scalar]: ...
@overload
def chunked_array(
    values: NullableIterable[dt.timedelta],
    type: None = None,
) -> ChunkedArray[scalar.DurationScalar]: ...
@overload
def chunked_array(
    values: NullableIterable[MonthDayNano],
    type: None = None,
) -> ChunkedArray[scalar.MonthDayNanoIntervalScalar]: ...
@overload
def chunked_array(
    values: NullableIterable[str],
    type: None = None,
) -> ChunkedArray[scalar.StringScalar]: ...
@overload
def chunked_array(
    values: NullableIterable[bytearray],
    type: None = None,
) -> ChunkedArray[scalar.BinaryScalar]: ...
@overload
def chunked_array(
    values: NullableIterable[list],
    type: None = None,
) -> ChunkedArray[scalar.ListScalar]: ...
@overload
def chunked_array(
    values: NullableIterable[_Scalar_CoT],
    type: None = None,
) -> ChunkedArray[_Scalar_CoT]: ...
@overload
def chunked_array(
    values: Iterable | SupportArrowStream | SupportArrowArray,
    type: _DataTypeT,
) -> ChunkedArray[Scalar[_DataTypeT]]: ...
@overload
def chunked_array(
    values: Iterable | SupportArrowStream | SupportArrowArray,
    type: Literal["null"],
) -> ChunkedArray[scalar.NullScalar]: ...
@overload
def chunked_array(
    values: Iterable | SupportArrowStream | SupportArrowArray,
    type: Literal["bool", "boolean"],
) -> ChunkedArray[scalar.BooleanScalar]: ...
@overload
def chunked_array(
    values: Iterable | SupportArrowStream | SupportArrowArray,
    type: Literal["i1", "int8"],
) -> ChunkedArray[scalar.Int8Scalar]: ...
@overload
def chunked_array(
    values: Iterable | SupportArrowStream | SupportArrowArray,
    type: Literal["i2", "int16"],
) -> ChunkedArray[scalar.Int16Scalar]: ...
@overload
def chunked_array(
    values: Iterable | SupportArrowStream | SupportArrowArray,
    type: Literal["i4", "int32"],
) -> ChunkedArray[scalar.Int32Scalar]: ...
@overload
def chunked_array(
    values: Iterable | SupportArrowStream | SupportArrowArray,
    type: Literal["i8", "int64"],
) -> ChunkedArray[scalar.Int64Scalar]: ...
@overload
def chunked_array(
    values: Iterable | SupportArrowStream | SupportArrowArray,
    type: Literal["u1", "uint8"],
) -> ChunkedArray[scalar.UInt8Scalar]: ...
@overload
def chunked_array(
    values: Iterable | SupportArrowStream | SupportArrowArray,
    type: Literal["u2", "uint16"],
) -> ChunkedArray[scalar.UInt16Scalar]: ...
@overload
def chunked_array(
    values: Iterable | SupportArrowStream | SupportArrowArray,
    type: Literal["u4", "uint32"],
) -> ChunkedArray[scalar.UInt32Scalar]: ...
@overload
def chunked_array(
    values: Iterable | SupportArrowStream | SupportArrowArray,
    type: Literal["u8", "uint64"],
) -> ChunkedArray[scalar.UInt64Scalar]: ...
@overload
def chunked_array(
    values: Iterable | SupportArrowStream | SupportArrowArray,
    type: Literal["f2", "halffloat", "float16"],
) -> ChunkedArray[scalar.HalfFloatScalar]: ...
@overload
def chunked_array(
    values: Iterable | SupportArrowStream | SupportArrowArray,
    type: Literal["f4", "float", "float32"],
) -> ChunkedArray[scalar.FloatScalar]: ...
@overload
def chunked_array(
    values: Iterable | SupportArrowStream | SupportArrowArray,
    type: Literal["f8", "double", "float64"],
) -> ChunkedArray[scalar.DoubleScalar]: ...
@overload
def chunked_array(
    values: Iterable | SupportArrowStream | SupportArrowArray,
    type: Literal["string", "str", "utf8"],
) -> ChunkedArray[scalar.StringScalar]: ...
@overload
def chunked_array(
    values: Iterable | SupportArrowStream | SupportArrowArray,
    type: Literal["binary"],
) -> ChunkedArray[scalar.BinaryScalar]: ...
@overload
def chunked_array(
    values: Iterable | SupportArrowStream | SupportArrowArray,
    type: Literal["large_string", "large_str", "large_utf8"],
) -> ChunkedArray[scalar.LargeStringScalar]: ...
@overload
def chunked_array(
    values: Iterable | SupportArrowStream | SupportArrowArray,
    type: Literal["large_binary"],
) -> ChunkedArray[scalar.LargeBinaryScalar]: ...
@overload
def chunked_array(
    values: Iterable | SupportArrowStream | SupportArrowArray,
    type: Literal["binary_view"],
) -> ChunkedArray[scalar.BinaryViewScalar]: ...
@overload
def chunked_array(
    values: Iterable | SupportArrowStream | SupportArrowArray,
    type: Literal["string_view"],
) -> ChunkedArray[scalar.StringViewScalar]: ...
@overload
def chunked_array(
    values: Iterable | SupportArrowStream | SupportArrowArray,
    type: Literal["date32", "date32[day]"],
) -> ChunkedArray[scalar.Date32Scalar]: ...
@overload
def chunked_array(
    values: Iterable | SupportArrowStream | SupportArrowArray,
    type: Literal["date64", "date64[ms]"],
) -> ChunkedArray[scalar.Date64Scalar]: ...
@overload
def chunked_array(
    values: Iterable | SupportArrowStream | SupportArrowArray,
    type: Literal["time32[s]"],
) -> ChunkedArray[scalar.Time32Scalar[Literal["s"]]]: ...
@overload
def chunked_array(
    values: Iterable | SupportArrowStream | SupportArrowArray,
    type: Literal["time32[ms]"],
) -> ChunkedArray[scalar.Time32Scalar[Literal["ms"]]]: ...
@overload
def chunked_array(
    values: Iterable | SupportArrowStream | SupportArrowArray,
    type: Literal["time64[us]"],
) -> ChunkedArray[scalar.Time64Scalar[Literal["us"]]]: ...
@overload
def chunked_array(
    values: Iterable | SupportArrowStream | SupportArrowArray,
    type: Literal["time64[ns]"],
) -> ChunkedArray[scalar.Time64Scalar[Literal["ns"]]]: ...
@overload
def chunked_array(
    values: Iterable | SupportArrowStream | SupportArrowArray,
    type: Literal["timestamp[s]"],
) -> ChunkedArray[scalar.TimestampScalar[Literal["s"]]]: ...
@overload
def chunked_array(
    values: Iterable | SupportArrowStream | SupportArrowArray,
    type: Literal["timestamp[ms]"],
) -> ChunkedArray[scalar.TimestampScalar[Literal["ms"]]]: ...
@overload
def chunked_array(
    values: Iterable | SupportArrowStream | SupportArrowArray,
    type: Literal["timestamp[us]"],
) -> ChunkedArray[scalar.TimestampScalar[Literal["us"]]]: ...
@overload
def chunked_array(
    values: Iterable | SupportArrowStream | SupportArrowArray,
    type: Literal["timestamp[ns]"],
) -> ChunkedArray[scalar.TimestampScalar[Literal["ns"]]]: ...
@overload
def chunked_array(
    values: Iterable | SupportArrowStream | SupportArrowArray,
    type: Literal["duration[s]"],
) -> ChunkedArray[scalar.DurationScalar[Literal["s"]]]: ...
@overload
def chunked_array(
    values: Iterable | SupportArrowStream | SupportArrowArray,
    type: Literal["duration[ms]"],
) -> ChunkedArray[scalar.DurationScalar[Literal["ms"]]]: ...
@overload
def chunked_array(
    values: Iterable | SupportArrowStream | SupportArrowArray,
    type: Literal["duration[us]"],
) -> ChunkedArray[scalar.DurationScalar[Literal["us"]]]: ...
@overload
def chunked_array(
    values: Iterable | SupportArrowStream | SupportArrowArray,
    type: Literal["duration[ns]"],
) -> ChunkedArray[scalar.DurationScalar[Literal["ns"]]]: ...
@overload
def chunked_array(
    values: Iterable | SupportArrowStream | SupportArrowArray,
    type: Literal["month_day_nano_interval"],
) -> ChunkedArray[scalar.MonthDayNanoIntervalScalar]: ...

_ColumnT = TypeVar("_ColumnT", bound=Array | ChunkedArray)

class _Tabular(_PandasConvertible[pd.DataFrame], Generic[_ColumnT]):
    def __array__(self, dtype: np.dtype | None = None, copy: bool | None = None) -> np.ndarray: ...
    def __dataframe__(
        self, nan_as_null: bool = False, allow_copy: bool = True
    ) -> _PyArrowDataFrame: ...
    @overload
    def __getitem__(self, key: int | str) -> _ColumnT: ...
    @overload
    def __getitem__(self, key: slice) -> Self: ...
    def __len__(self) -> int: ...
    def column(self, i: int | str) -> _ColumnT: ...
    @property
    def column_names(self) -> list[str]: ...
    @property
    def columns(self) -> list[_ColumnT]: ...
    def drop_null(self) -> Self: ...
    def field(self, i: int | str) -> Field: ...
    @classmethod
    def from_pydict(
        cls,
        mapping: Mapping[str, ChunkedArray | Array | list | np.ndarray],
        schema: Schema | None = None,
        metadata: Mapping | None = None,
    ) -> Self: ...
    @classmethod
    def from_pylist(
        cls,
        mapping: Sequence[Mapping[str, Any]],
        schema: Schema | None = None,
        metadata: Mapping | None = None,
    ) -> Self: ...
    def itercolumns(self) -> Generator[_ColumnT, None, None]: ...
    @property
    def num_columns(self) -> int: ...
    @property
    def num_rows(self) -> int: ...
    @property
    def shape(self) -> tuple[int, int]: ...
    @property
    def schema(self) -> Schema: ...
    @property
    def nbytes(self) -> int: ...
    def sort_by(self, sorting: str | list[tuple[str, Order]], **kwargs) -> Self: ...
    def take(self, indices: Indices) -> Self: ...
    def filter(
        self, mask: Mask | Expression, null_selection_behavior: NullSelectionBehavior = "drop"
    ) -> Self: ...
    def to_pydict(self) -> dict[str, list]: ...
    def to_pylist(self) -> list[dict[str, Any]]: ...
    def to_string(self, *, show_metadata: bool = False, preview_cols: int = 0) -> str: ...
    def remove_column(self, i: int) -> Self: ...
    def drop_columns(self, columns: str | list[str]) -> Self: ...
    def add_column(
        self, i: int, field_: str | Field, column: ChunkedArray | Array | list
    ) -> Self: ...
    def append_column(self, field_: str | Field, column: ChunkedArray | Array | list) -> Self: ...

class RecordBatch(_Tabular[Array]):
    def validate(self, *, full: bool = False) -> None: ...
    def replace_schema_metadata(self, metadata: dict | None = None) -> Self: ...
    def get_total_buffer_size(self) -> int: ...
    def __sizeof__(self) -> int: ...
    def set_column(self, i: int, field_: str | Field, column: Array | list) -> Self: ...
    @overload
    def rename_columns(self, names: list[str]) -> Self: ...
    @overload
    def rename_columns(self, names: dict[str, str]) -> Self: ...
    def serialize(self, memory_pool: MemoryPool | None = None) -> Buffer: ...
    def slice(self, offset: int = 0, length: int | None = None) -> Self: ...
    def equals(self, other: Self, check_metadata: bool = False) -> bool: ...
    def select(self, columns: list[str] | Indices) -> Self: ...
    def cast(
        self, target_schema: Schema, safe: bool | None = None, options: CastOptions | None = None
    ) -> Self: ...
    @classmethod
    def from_arrays(
        cls,
        arrays: list[Array] | list[ChunkedArray],
        names: list[str] | None = None,
        schema: Schema | None = None,
        metadata: Mapping | None = None,
    ) -> Self: ...
    @classmethod
    def from_pandas(
        cls,
        df: pd.DataFrame,
        schema: Schema | None = None,
        preserve_index: bool | None = None,
        nthreads: int | None = None,
        columns: list[str] | None = None,
    ) -> Self: ...
    @classmethod
    def from_struct_array(
        cls, struct_array: StructArray | ChunkedArray[scalar.StructScalar]
    ) -> Self: ...
    def to_struct_array(self) -> StructArray: ...
    def to_tensor(
        self,
        null_to_nan: bool = False,
        row_major: bool = True,
        memory_pool: MemoryPool | None = None,
    ) -> Tensor: ...
    def _export_to_c(self, out_ptr: int, out_schema_ptr: int = 0): ...
    @classmethod
    def _import_from_c(cls, in_ptr: int, schema: Schema) -> Self: ...
    def __arrow_c_array__(self, requested_schema=None): ...
    def __arrow_c_stream__(self, requested_schema=None): ...
    @classmethod
    def _import_from_c_capsule(cls, schema_capsule, array_capsule) -> Self: ...
    def _export_to_c_device(self, out_ptr: int, out_schema_ptr: int = 0) -> None: ...
    @classmethod
    def _import_from_c_device(cls, in_ptr: int, schema: Schema) -> Self: ...
    def __arrow_c_device_array__(self, requested_schema=None, **kwargs): ...
    @classmethod
    def _import_from_c_device_capsule(cls, schema_capsule, array_capsule) -> Self: ...
    @property
    def device_type(self) -> DeviceAllocationType: ...
    @property
    def is_cpu(self) -> bool: ...

def table_to_blocks(options, table: Table, categories, extension_columns): ...

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

class Table(_Tabular[ChunkedArray]):
    def validate(self, *, full=False) -> None: ...
    def slice(self, offset=0, length=None) -> Self: ...
    def select(self, columns: list[str] | Indices) -> Self: ...
    def replace_schema_metadata(self, metadata: dict | None = None) -> Self: ...
    def flatten(self, memory_pool: MemoryPool | None = None) -> Self: ...
    def combine_chunks(self, memory_pool: MemoryPool | None = None) -> Self: ...
    def unify_dictionaries(self, memory_pool: MemoryPool | None = None) -> Self: ...
    def equals(self, other: Self, check_metadata: bool = False) -> Self: ...
    def cast(
        self, target_schema: Schema, safe: bool | None = None, options: CastOptions | None = None
    ) -> Self: ...
    @classmethod
    def from_pandas(
        cls,
        df: pd.DataFrame,
        schema: Schema | None = None,
        preserve_index: bool | None = None,
        nthreads: int | None = None,
        columns: list[str] | None = None,
        safe: bool = True,
    ) -> Self: ...
    @classmethod
    def from_arrays(
        cls,
        arrays: list[Array] | list[ChunkedArray],
        names: list[str] | None = None,
        schema: Schema | None = None,
        metadata: Mapping | None = None,
    ) -> Self: ...
    @classmethod
    def from_struct_array(
        cls, struct_array: StructArray | ChunkedArray[scalar.StructScalar]
    ) -> Self: ...
    def to_struct_array(
        self, max_chunksize: int | None = None
    ) -> ChunkedArray[scalar.StructScalar]: ...
    @classmethod
    def from_batches(
        cls, batches: Iterable[RecordBatch], schema: Schema | None = None
    ) -> Self: ...
    def to_batches(self, max_chunksize: int | None = None) -> list[RecordBatch]: ...
    def to_reader(self, max_chunksize: int | None = None) -> RecordBatchReader: ...
    def get_total_buffer_size(self) -> int: ...
    def __sizeof__(self) -> int: ...
    def set_column(self, i: int, field_: str | Field, column: Array | list) -> Self: ...
    @overload
    def rename_columns(self, names: list[str]) -> Self: ...
    @overload
    def rename_columns(self, names: dict[str, str]) -> Self: ...
    def drop(self, columns: str | list[str]) -> Self: ...
    def group_by(self, keys: str | list[str], use_threads: bool = True) -> TableGroupBy: ...
    def join(
        self,
        right_table: Self,
        keys: str | list[str],
        right_keys: str | list[str] | None = None,
        join_type: JoinType = "left outer",
        left_suffix: str | None = None,
        right_suffix: str | None = None,
        coalesce_keys: bool = True,
        use_threads: bool = True,
    ) -> Self: ...
    def join_asof(
        self,
        right_table: Self,
        on: str,
        by: str | list[str],
        tolerance: int,
        right_on: str | list[str] | None = None,
        right_by: str | list[str] | None = None,
    ) -> Self: ...
    def __arrow_c_stream__(self, requested_schema=None): ...

def record_batch(
    data: dict[str, list | Array]
    | list[Array]
    | pd.DataFrame
    | SupportArrowArray
    | SupportArrowDeviceArray,
    names: list[str] | None = None,
    schema: Schema | None = None,
    metadata: Mapping | None = None,
) -> RecordBatch: ...
@overload
def table(
    data: dict[str, list | Array],
    schema: Schema | None = None,
    metadata: Mapping | None = None,
    nthreads: int | None = None,
) -> Table: ...
@overload
def table(
    data: list[Array | ChunkedArray]
    | pd.DataFrame
    | SupportArrowArray
    | SupportArrowStream
    | SupportArrowDeviceArray,
    names: list[str] | None = None,
    schema: Schema | None = None,
    metadata: Mapping | None = None,
    nthreads: int | None = None,
) -> Table: ...
def concat_tables(
    tables: list[Table],
    memory_pool: MemoryPool | None = None,
    promote_options: Literal["none", "default", "permissive"] = "none",
    **kwargs,
) -> Table: ...

class TableGroupBy:
    keys: str | list[str]
    def __init__(self, table: Table, keys: str | list[str], use_threads: bool = True): ...
    def aggregate(
        self, aggregations: list[tuple[str, str]] | list[tuple[str, str, FunctionOptions]]
    ) -> Table: ...
    def _table(self) -> Table: ...
    @property
    def _use_threads(self) -> bool: ...

__all__ = [
    "ChunkedArray",
    "chunked_array",
    "_Tabular",
    "RecordBatch",
    "table_to_blocks",
    "Table",
    "record_batch",
    "table",
    "concat_tables",
    "TableGroupBy",
]
