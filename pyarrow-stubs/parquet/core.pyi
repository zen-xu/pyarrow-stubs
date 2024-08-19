from typing import Collection, Literal, Protocol, TypeAlias

from pyarrow._compute import Expression
from pyarrow._parquet import (
    ColumnChunkMetaData,
    ColumnSchema,
    FileDecryptionProperties,
    FileEncryptionProperties,
    FileMetaData,
    ParquetLogicalType,
    ParquetReader,
    ParquetSchema,
    RowGroupMetaData,
    SortingColumn,
    Statistics,
)
from typing_extensions import deprecated

__all__ = (
    "ColumnChunkMetaData",
    "ColumnSchema",
    "FileDecryptionProperties",
    "FileEncryptionProperties",
    "FileMetaData",
    "ParquetDataset",
    "ParquetFile",
    "ParquetLogicalType",
    "ParquetReader",
    "ParquetSchema",
    "ParquetWriter",
    "RowGroupMetaData",
    "SortingColumn",
    "Statistics",
    "read_metadata",
    "read_pandas",
    "read_schema",
    "read_table",
    "write_metadata",
    "write_table",
    "write_to_dataset",
    "_filters_to_expression",
    "filters_to_expression",
)

class SupportEq(Protocol):
    def __eq__(self, other) -> bool: ...

class SupportLt(Protocol):
    def __lt__(self, other) -> bool: ...

class SupportGt(Protocol):
    def __gt__(self, other) -> bool: ...

class SupportLe(Protocol):
    def __le__(self, other) -> bool: ...

class SupportGe(Protocol):
    def __ge__(self, other) -> bool: ...

_Filter: TypeAlias = (
    tuple[str, Literal["=", "==", "!="], SupportEq]
    | tuple[str, Literal["<"], SupportLt]
    | tuple[str, Literal[">"], SupportGt]
    | tuple[str, Literal["<="], SupportLe]
    | tuple[str, Literal[">="], SupportGe]
    | tuple[str, Literal["in", "not in"], Collection]
)

def filters_to_expression(filters: list[_Filter | list[_Filter]]) -> Expression: ...
@deprecated("use filters_to_expression")
def _filters_to_expression(filters: list[_Filter | list[_Filter]]) -> Expression: ...
