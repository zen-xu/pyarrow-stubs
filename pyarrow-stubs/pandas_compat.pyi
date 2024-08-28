from typing import Any, TypedDict, TypeVar

import pandas as pd

from pandas import DatetimeTZDtype

from .lib import Array, DataType, Schema, Table

_T = TypeVar("_T")

def get_logical_type_map() -> dict[int, str]: ...
def get_logical_type(arrow_type: DataType) -> str: ...
def get_logical_type_from_numpy(pandas_collection) -> str: ...
def get_extension_dtype_info(column) -> tuple[str, dict[str, Any]]: ...

class _ColumnMetadata(TypedDict):
    name: str
    field_name: str
    pandas_type: int
    numpy_type: str
    metadata: dict | None

def get_column_metadata(
    column: pd.Series | pd.Index, name: str, arrow_type: DataType, field_name: str
) -> _ColumnMetadata: ...
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
    df: pd.DataFrame, preserve_index: bool | None, columns: list[str] | None = None
) -> tuple[list[str], list[DataType], dict[bytes, bytes]]: ...
def dataframe_to_arrays(
    df: pd.DataFrame,
    schema: Schema,
    preserve_index: bool | None,
    nthreads: int = 1,
    columns: list[str] | None = None,
    safe: bool = True,
) -> tuple[Array, Schema, int]: ...
def get_datetimetz_type(values: _T, dtype, type_) -> tuple[_T, DataType]: ...
def make_datetimetz(unit: str, tz: str) -> DatetimeTZDtype: ...
def table_to_dataframe(
    options, table: Table, categories=None, ignore_metadata: bool = False, types_mapper=None
) -> pd.DataFrame: ...
def make_tz_aware(series: pd.Series, tz: str) -> pd.Series: ...
