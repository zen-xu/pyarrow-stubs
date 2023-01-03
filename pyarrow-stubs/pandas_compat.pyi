from typing import (
    Any,
    Callable,
)

import numpy as np
import pandas as pd
from pandas.core.internals import BlockManager
from pyarrow.lib import (
    Array,
    DataType,
    Schema,
    Table,
    _ArrowType,
    frombytes as frombytes,
)
from typing_extensions import TypedDict

class _SerializedDict(TypedDict):
    blocks: list[Any]
    axes: list[Any]

def get_logical_type_map() -> dict[_ArrowType, str]: ...
def get_logical_type(arrow_type: _ArrowType) -> str: ...
def get_logical_type_from_numpy(pandas_collection: pd.Series | pd.Index) -> str: ...
def get_extension_dtype_info(
    column: pd.Series | pd.Index,
) -> tuple[str, dict[str, Any] | None]: ...
def get_column_metadata(
    column: pd.Series | pd.Index, name: str, arrow_type: DataType, field_name: str
) -> dict[str, Any]: ...
def construct_metadata(
    columns_to_convert: list[pd.Series],
    df: pd.DataFrame,
    column_names: list[str],
    index_levels: list[pd.Index],
    index_descriptors: list[dict],
    preserve_index: bool,
    types: list[DataType],
) -> dict[bytes, bytes]: ...
def dataframe_to_types(
    df: pd.DataFrame, preserve_index: bool, columns: list[str] | None = ...
) -> tuple[list[str], list[DataType], dict[bytes, bytes]]: ...
def dataframe_to_arrays(
    df: pd.DataFrame,
    schema: Schema,
    preserve_index: bool,
    nthreads: int = ...,
    columns: list[str] | None = ...,
    safe: bool = ...,
) -> tuple[Array, Schema, int | None]: ...
def get_datetimetz_type(
    values: pd.Series | pd.Index, dtype: np.dtype, type_: DataType | None
) -> tuple[pd.Series | pd.Index, DataType]: ...
def dataframe_to_serialized_dict(frame: pd.DataFrame) -> _SerializedDict: ...
def serialized_dict_to_dataframe(data: _SerializedDict) -> pd.DataFrame: ...
def make_datetimetz(tz: str) -> pd.DatetimeTZDtype: ...
def table_to_blockmanager(
    options: dict,
    table: Table,
    categories: list[str] | None = ...,
    ignore_metadata: bool = ...,
    types_mapper: Callable[[DataType], np.generic] | None = ...,
) -> BlockManager: ...
def make_tz_aware(series: pd.Series, tz: str) -> pd.Series: ...
