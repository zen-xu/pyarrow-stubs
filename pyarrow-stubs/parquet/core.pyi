from io import IOBase
from os import PathLike
import pathlib
from typing import (
    Callable,
    Generator,
    Generic,
    TypeVar,
)

from _typeshed import Incomplete
import pyarrow
from pyarrow import (
    Array,
    NativeFile,
    RecordBatch,
    Schema,
    Table,
)
from pyarrow._parquet import (
    ColumnChunkMetaData as ColumnChunkMetaData,
    ColumnSchema as ColumnSchema,
    FileDecryptionProperties as FileDecryptionProperties,
    FileEncryptionProperties as FileEncryptionProperties,
    FileMetaData as FileMetaData,
    ParquetLogicalType as ParquetLogicalType,
    ParquetReader as ParquetReader,
    ParquetSchema as ParquetSchema,
    RowGroupMetaData as RowGroupMetaData,
    Statistics as Statistics,
)
from pyarrow.compute import Expression
from pyarrow.dataset import Partitioning
from pyarrow.fs import FileSystem
from typing_extensions import (
    Literal,
    TypeAlias,
)

def filters_to_expression(
    filters: list[tuple[str, str, str] | list[tuple[str, str, str]]]
) -> Expression: ...

class ParquetFile:
    reader: ParquetReader
    common_metadata: FileMetaData | None
    def __init__(
        self,
        source: str | PathLike | pyarrow.NativeFile | IOBase,
        *,
        metadata: FileMetaData | None = ...,
        common_metadata: FileMetaData | None = ...,
        read_dictionary: list[str] | None = ...,
        memory_map: bool = ...,
        buffer_size: int = ...,
        pre_buffer: bool = ...,
        coerce_int96_timestamp_unit: Literal["ms", "ns"] | None = ...,
        decryption_properties: FileDecryptionProperties | None = ...,
        thrift_string_size_limit: int | None = ...,
        thrift_container_size_limit: int | None = ...,
    ) -> None: ...
    def __enter__(self) -> ParquetFile: ...
    def __exit__(self, *args, **kwargs) -> None: ...
    @property
    def metadata(self) -> FileMetaData | None: ...
    @property
    def schema(self) -> ParquetSchema: ...
    @property
    def schema_arrow(self) -> pyarrow.Schema: ...
    @property
    def num_row_groups(self) -> int: ...
    def close(self, force: bool = ...) -> None: ...
    @property
    def closed(self) -> bool: ...
    def read_row_group(
        self,
        i: int,
        columns: list[str] | None = ...,
        use_threads: bool = ...,
        use_pandas_metadata: bool = ...,
    ) -> pyarrow.Table: ...
    def read_row_groups(
        self,
        row_groups: list[str],
        columns: list[str] | None = ...,
        use_threads: bool = ...,
        use_pandas_metadata: bool = ...,
    ) -> pyarrow.Table: ...
    def iter_batches(
        self,
        batch_size: int = ...,
        row_groups: list[str] | None = ...,
        columns: list[str] | None = ...,
        use_threads: bool = ...,
        use_pandas_metadata: bool = ...,
    ) -> Generator[pyarrow.RecordBatch, None, None]: ...
    def read(
        self,
        columns: list[str] | None = ...,
        use_threads: bool = ...,
        use_pandas_metadata: bool = ...,
    ) -> pyarrow.Table: ...
    def scan_contents(
        self, columns: list[int] | None = ..., batch_size: int = ...
    ) -> int: ...

_COMPRESSION: TypeAlias = Literal["NONE", "SNAPPY", "GZIP", "BROTLI", "LZ4", "ZSTD"]

class ParquetWriter:
    flavor: Literal["spark"] | None
    schema_changed: bool
    schema: pyarrow.Schema
    where: str | PathLike | IOBase
    file_handle: Incomplete
    writer: Incomplete
    is_open: bool
    def __init__(
        self,
        where: str | PathLike | IOBase,
        schema: pyarrow.Schema,
        filesystem: FileSystem | None = ...,
        flavor: Literal["spark"] | None = ...,
        version: str = ...,
        use_dictionary: bool | list[str] = ...,
        compression: _COMPRESSION | dict[str, _COMPRESSION] = ...,
        write_statistics: bool | list[bool] = ...,
        use_deprecated_int96_timestamps: bool | None = ...,
        compression_level: int | dict[str, int] | None = ...,
        use_byte_stream_split: bool | list[str] = ...,
        column_encoding: str | dict[str, str] | None = ...,
        writer_engine_version: str | None = ...,
        data_page_version: Literal["1.0", "2.0"] = ...,
        use_compliant_nested_type: bool = ...,
        encryption_properties: FileEncryptionProperties | None = ...,
        write_batch_size: int | None = ...,
        dictionary_pagesize_limit: int | None = ...,
        **options,
    ) -> None: ...
    def __del__(self) -> None: ...
    def __enter__(self) -> ParquetWriter: ...
    def __exit__(self, *args, **kwargs): ...
    def write(
        self,
        table_or_batch: Table | RecordBatch,
        row_group_size: int | None = ...,
    ) -> None: ...
    def write_batch(
        self, batch: RecordBatch, row_group_size: int | None = ...
    ) -> None: ...
    def write_table(self, table: Table, row_group_size: int | None = ...) -> None: ...
    def close(self) -> None: ...

class ParquetDatasetPiece:
    def __init__(
        self,
        path: str | pathlib.Path,
        open_file_func: Callable = ...,
        file_options: dict | None = ...,
        row_group: int | None = ...,
        partition_keys: list[tuple[str, str]] | None = ...,
    ) -> None: ...
    def __eq__(self, other) -> bool: ...
    def get_metadata(self) -> FileMetaData: ...
    def open(self) -> ParquetFile: ...
    def read(
        self,
        columns: list[str] | None = ...,
        use_threads: bool = ...,
        partitions: ParquetPartitions | None = ...,
        file: IOBase | None = ...,
        use_pandas_metadata: bool = ...,
    ) -> Table: ...

_K = TypeVar("_K")

class PartitionSet(Generic[_K]):
    name: str
    keys: list[_K]
    key_indices: dict[_K, int]
    def __init__(self, name: str, keys: list[_K] | None = ...) -> None: ...
    def get_index(self, key: _K) -> int: ...
    @property
    def dictionary(self) -> Array: ...
    @property
    def is_sorted(self) -> bool: ...

_PPK = TypeVar("_PPK", str, int)

class ParquetPartitions(Generic[_PPK]):
    levels: list[PartitionSet[_PPK]]
    partition_names: set[str]
    def __init__(self) -> None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, i): ...
    def equals(self, other: ParquetPartitions) -> bool: ...
    def __eq__(self, other) -> bool: ...
    def get_index(self, level: int, name: str, key: _PPK) -> int: ...
    def filter_accepts_partition(self, part_key, filter, level: int) -> bool: ...

class ParquetManifest:
    filesystem: Incomplete
    open_file_func: Incomplete
    pathsep: Incomplete
    dirpath: Incomplete
    partition_scheme: Incomplete
    partitions: Incomplete
    pieces: Incomplete
    common_metadata_path: Incomplete
    metadata_path: Incomplete
    def __init__(
        self,
        dirpath,
        open_file_func: Incomplete | None = ...,
        filesystem: Incomplete | None = ...,
        pathsep: str = ...,
        partition_scheme: str = ...,
        metadata_nthreads: int = ...,
    ) -> None: ...

class _ParquetDatasetMetadata: ...

class ParquetDataset:
    paths: list[str]
    split_row_groups: bool

    def __new__(
        cls,
        path_or_paths: str | list[str] | None = ...,
        filesystem: FileSystem | None = ...,
        schema: Schema | None = ...,
        metadata: FileMetaData | None = ...,
        split_row_groups: bool = ...,
        validate_schema: bool = ...,
        filters: list[tuple[str, str, str] | list[tuple[str, str, str]]] | None = ...,
        metadata_nthreads: int | None = ...,
        read_dictionary: list[str] | None = ...,
        memory_map: bool = ...,
        buffer_size: int = ...,
        partitioning: str = ...,
        use_legacy_dataset: bool | None = ...,
        pre_buffer: bool = ...,
        coerce_int96_timestamp_unit: Literal["ms", "ns"] | None = ...,
        thrift_string_size_limit: int | None = ...,
        thrift_container_size_limit: int | None = ...,
    ): ...
    def equals(self, other) -> bool: ...
    def __eq__(self, other) -> bool: ...
    def validate_schemas(self) -> None: ...
    def read(
        self,
        columns: list[str] | None = ...,
        use_threads: bool = ...,
        use_pandas_metadata: bool = ...,
    ) -> Table: ...
    def read_pandas(self, **kwargs) -> Table: ...
    @property
    def pieces(self): ...
    @property
    def partitions(self): ...
    @property
    def schema(self): ...
    @property
    def memory_map(self): ...
    @property
    def read_dictionary(self): ...
    @property
    def buffer_size(self): ...
    @property
    def fs(self): ...
    @property
    def metadata(self): ...
    @property
    def metadata_path(self): ...
    @property
    def common_metadata_path(self): ...
    @property
    def common_metadata(self): ...
    @property
    def fragments(self) -> None: ...
    @property
    def files(self) -> None: ...
    @property
    def filesystem(self) -> None: ...
    @property
    def partitioning(self) -> None: ...

class _ParquetDatasetV2:
    def __init__(
        self,
        path_or_paths: str | list[str],
        filesystem: FileSystem | None = ...,
        *,
        filters: list[tuple[str, str, str] | list[tuple[str, str, str]]] | None = ...,
        partitioning: str = ...,
        read_dictionary: list[str] | None = ...,
        buffer_size: int | None = ...,
        memory_map: bool = ...,
        ignore_prefixes: list[str] | None = ...,
        pre_buffer: bool = ...,
        coerce_int96_timestamp_unit: Literal["ms", "ns"] | None = ...,
        schema: Schema | None = ...,
        decryption_properties: FileDecryptionProperties | None = ...,
        thrift_string_size_limit: Incomplete | None = ...,
        thrift_container_size_limit: Incomplete | None = ...,
        **kwargs,
    ) -> None: ...
    @property
    def schema(self) -> Schema: ...
    def read(
        self,
        columns: list[str] | None = ...,
        use_threads: bool = ...,
        use_pandas_metadata: bool = ...,
    ) -> Table: ...
    def read_pandas(self, **kwargs) -> Table: ...
    @property
    def pieces(self): ...
    @property
    def fragments(self): ...
    @property
    def files(self): ...
    @property
    def filesystem(self): ...
    @property
    def partitioning(self): ...

def read_table(
    source: str | NativeFile | IOBase,
    *,
    columns: list[str] | None = ...,
    use_threads: bool = ...,
    metadata: FileMetaData | None = ...,
    schema: Schema | None = ...,
    use_pandas_metadata: bool = ...,
    memory_map: bool = ...,
    read_dictionary: list[str] | None = ...,
    filesystem: FileSystem | None = ...,
    filters: list[tuple[str, str, str] | list[tuple[str, str, str]]] | None = ...,
    buffer_size: int = ...,
    partitioning: str = ...,
    use_legacy_dataset: bool = ...,
    ignore_prefixes: list[str] | None = ...,
    pre_buffer: bool = ...,
    coerce_int96_timestamp_unit: Literal["ms", "ns"] | None = ...,
    decryption_properties: FileDecryptionProperties | None = ...,
    thrift_string_size_limit: int | None = ...,
    thrift_container_size_limit: int | None = ...,
) -> Table: ...
def read_pandas(
    source: str | NativeFile | IOBase, columns: list[str] | None = ..., **kwargs
) -> Table: ...
def write_table(
    table: Table,
    where: str | NativeFile,
    row_group_size: int | None = ...,
    version: str = ...,
    use_dictionary: bool | list[str] = ...,
    compression: str = ...,
    write_statistics: bool = ...,
    use_deprecated_int96_timestamps: bool | None = ...,
    coerce_timestamps: str | None = ...,
    allow_truncated_timestamps: bool = ...,
    data_page_size: int | None = ...,
    flavor: Literal["spark"] | None = ...,
    filesystem: FileSystem | None = ...,
    compression_level: int | dict[str, int] | None = ...,
    use_byte_stream_split: bool = ...,
    column_encoding: str | dict[str, str] | None = ...,
    data_page_version: str = ...,
    use_compliant_nested_type: bool = ...,
    encryption_properties: FileEncryptionProperties | None = ...,
    write_batch_size: int | None = ...,
    dictionary_pagesize_limit: int | None = ...,
    **kwargs,
) -> None: ...
def write_to_dataset(
    table: Table,
    root_path: str | pathlib.Path,
    partition_cols: list[str] | None = ...,
    partition_filename_cb: Callable | None = ...,
    filesystem: FileSystem | None = ...,
    use_legacy_dataset: bool | None = ...,
    schema: Schema | None = ...,
    partitioning: list[str] | Partitioning | None = ...,
    basename_template: str | None = ...,
    use_threads: bool | None = ...,
    file_visitor: Callable | None = ...,
    existing_data_behavior: Literal["overwrite_or_ignore", "error", "delete_matching"]
    | None = ...,
    **kwargs,
) -> None: ...
def write_metadata(
    schema: Schema,
    where: str | NativeFile,
    metadata_collector: list | None = ...,
    **kwargs,
) -> None: ...
def read_metadata(
    where,
    memory_map: bool = ...,
    decryption_properties: FileDecryptionProperties | None = ...,
    filesystem: Incomplete | None = ...,
): ...
def read_schema(
    where: str | IOBase,
    memory_map: bool = ...,
    decryption_properties: FileDecryptionProperties | None = ...,
    filesystem: FileSystem | None = ...,
) -> FileMetaData: ...

# Names in __all__ with no definition:
#   _filters_to_expression
