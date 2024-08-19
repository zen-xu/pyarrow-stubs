from typing import Collection, Literal, Protocol, TypeAlias

from pyarrow._compute import Expression
from pyarrow._parquet import ColumnChunkMetaData as ColumnChunkMetaData
from pyarrow._parquet import ColumnSchema as ColumnSchema
from pyarrow._parquet import FileDecryptionProperties as FileDecryptionProperties
from pyarrow._parquet import FileEncryptionProperties as FileEncryptionProperties
from pyarrow._parquet import FileMetaData as FileMetaData
from pyarrow._parquet import ParquetLogicalType as ParquetLogicalType
from pyarrow._parquet import ParquetReader as ParquetReader
from pyarrow._parquet import ParquetSchema as ParquetSchema
from pyarrow._parquet import RowGroupMetaData as RowGroupMetaData
from pyarrow._parquet import SortingColumn as SortingColumn
from pyarrow._parquet import Statistics as Statistics

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
