from typing import Any, Protocol, TypeAlias

from pyarrow.lib import Array, Buffer, DataType, DictionaryArray, RecordBatch, Table

from .column import (
    ColumnBuffers,
    ColumnNullType,
    Dtype,
    DtypeKind,
)

class DataFrameObject(Protocol):
    def __dataframe__(self, nan_as_null: bool = False, allow_copy: bool = True) -> Any: ...

ColumnObject: TypeAlias = Any

def from_dataframe(df: DataFrameObject, allow_copy=True) -> Table: ...
def protocol_df_chunk_to_pyarrow(df: DataFrameObject, allow_copy: bool = True) -> RecordBatch: ...
def column_to_array(col: ColumnObject, allow_copy: bool = True) -> Array: ...
def bool_column_to_array(col: ColumnObject, allow_copy: bool = True) -> Array: ...
def categorical_column_to_dictionary(
    col: ColumnObject, allow_copy: bool = True
) -> DictionaryArray: ...
def parse_datetime_format_str(format_str: str) -> tuple[str, str]: ...
def map_date_type(data_type: tuple[DtypeKind, int, str, str]) -> DataType: ...
def buffers_to_array(
    buffers: ColumnBuffers,
    data_type: tuple[DtypeKind, int, str, str],
    length: int,
    describe_null: ColumnNullType,
    offset: int = 0,
    allow_copy: bool = True,
) -> Array: ...
def validity_buffer_from_mask(
    validity_buff: Buffer,
    validity_dtype: Dtype,
    describe_null: ColumnNullType,
    length: int,
    offset: int = 0,
    allow_copy: bool = True,
) -> Buffer: ...
def validity_buffer_nan_sentinel(
    data_pa_buffer: Buffer,
    data_type: Dtype,
    describe_null: ColumnNullType,
    length: int,
    offset: int = 0,
    allow_copy: bool = True,
) -> Buffer: ...
