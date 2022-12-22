from io import IOBase
from os import PathLike
from typing import (
    Generator,
    Literal,
)

from _typeshed import Incomplete
import pyarrow
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
from pyarrow.fs import FileSystem

def filters_to_expression(
    filters: list[tuple[str, ...]] | list[tuple[tuple[str, ...], ...]]
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

_COMPRESSION = Literal["NONE", "SNAPPY", "GZIP", "BROTLI", "LZ4", "ZSTD"]

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
    def __enter__(self): ...
    def __exit__(self, *args, **kwargs): ...
    def write(
        self, table_or_batch, row_group_size: Incomplete | None = ...
    ) -> None: ...
    def write_batch(self, batch, row_group_size: Incomplete | None = ...) -> None: ...
    def write_table(self, table, row_group_size: Incomplete | None = ...) -> None: ...
    def close(self) -> None: ...

class ParquetDatasetPiece:
    def __init__(
        self,
        path,
        open_file_func=...,
        file_options: Incomplete | None = ...,
        row_group: Incomplete | None = ...,
        partition_keys: Incomplete | None = ...,
    ) -> None: ...
    def __eq__(self, other): ...
    def get_metadata(self): ...
    def open(self): ...
    def read(
        self,
        columns: Incomplete | None = ...,
        use_threads: bool = ...,
        partitions: Incomplete | None = ...,
        file: Incomplete | None = ...,
        use_pandas_metadata: bool = ...,
    ): ...

class PartitionSet:
    name: Incomplete
    keys: Incomplete
    key_indices: Incomplete
    def __init__(self, name, keys: Incomplete | None = ...) -> None: ...
    def get_index(self, key): ...
    @property
    def dictionary(self): ...
    @property
    def is_sorted(self): ...

class ParquetPartitions:
    levels: Incomplete
    partition_names: Incomplete
    def __init__(self) -> None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, i): ...
    def equals(self, other): ...
    def __eq__(self, other): ...
    def get_index(self, level, name, key): ...
    def filter_accepts_partition(self, part_key, filter, level): ...

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
    __doc__: Incomplete
    def __new__(
        cls,
        path_or_paths: Incomplete | None = ...,
        filesystem: Incomplete | None = ...,
        schema: Incomplete | None = ...,
        metadata: Incomplete | None = ...,
        split_row_groups: bool = ...,
        validate_schema: bool = ...,
        filters: Incomplete | None = ...,
        metadata_nthreads: Incomplete | None = ...,
        read_dictionary: Incomplete | None = ...,
        memory_map: bool = ...,
        buffer_size: int = ...,
        partitioning: str = ...,
        use_legacy_dataset: Incomplete | None = ...,
        pre_buffer: bool = ...,
        coerce_int96_timestamp_unit: Incomplete | None = ...,
        thrift_string_size_limit: Incomplete | None = ...,
        thrift_container_size_limit: Incomplete | None = ...,
    ): ...
    paths: Incomplete
    split_row_groups: Incomplete
    def __init__(
        self,
        path_or_paths,
        filesystem: Incomplete | None = ...,
        schema: Incomplete | None = ...,
        metadata: Incomplete | None = ...,
        split_row_groups: bool = ...,
        validate_schema: bool = ...,
        filters: Incomplete | None = ...,
        metadata_nthreads: Incomplete | None = ...,
        read_dictionary: Incomplete | None = ...,
        memory_map: bool = ...,
        buffer_size: int = ...,
        partitioning: str = ...,
        use_legacy_dataset: bool = ...,
        pre_buffer: bool = ...,
        coerce_int96_timestamp_unit: Incomplete | None = ...,
        thrift_string_size_limit: Incomplete | None = ...,
        thrift_container_size_limit: Incomplete | None = ...,
    ) -> None: ...
    def equals(self, other): ...
    def __eq__(self, other): ...
    def validate_schemas(self) -> None: ...
    def read(
        self,
        columns: Incomplete | None = ...,
        use_threads: bool = ...,
        use_pandas_metadata: bool = ...,
    ): ...
    def read_pandas(self, **kwargs): ...
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
        path_or_paths,
        filesystem: Incomplete | None = ...,
        *,
        filters: Incomplete | None = ...,
        partitioning: str = ...,
        read_dictionary: Incomplete | None = ...,
        buffer_size: Incomplete | None = ...,
        memory_map: bool = ...,
        ignore_prefixes: Incomplete | None = ...,
        pre_buffer: bool = ...,
        coerce_int96_timestamp_unit: Incomplete | None = ...,
        schema: Incomplete | None = ...,
        decryption_properties: Incomplete | None = ...,
        thrift_string_size_limit: Incomplete | None = ...,
        thrift_container_size_limit: Incomplete | None = ...,
        **kwargs,
    ) -> None: ...
    @property
    def schema(self): ...
    def read(
        self,
        columns: Incomplete | None = ...,
        use_threads: bool = ...,
        use_pandas_metadata: bool = ...,
    ): ...
    def read_pandas(self, **kwargs): ...
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
    source,
    *,
    columns: Incomplete | None = ...,
    use_threads: bool = ...,
    metadata: Incomplete | None = ...,
    schema: Incomplete | None = ...,
    use_pandas_metadata: bool = ...,
    memory_map: bool = ...,
    read_dictionary: Incomplete | None = ...,
    filesystem: Incomplete | None = ...,
    filters: Incomplete | None = ...,
    buffer_size: int = ...,
    partitioning: str = ...,
    use_legacy_dataset: bool = ...,
    ignore_prefixes: Incomplete | None = ...,
    pre_buffer: bool = ...,
    coerce_int96_timestamp_unit: Incomplete | None = ...,
    decryption_properties: Incomplete | None = ...,
    thrift_string_size_limit: Incomplete | None = ...,
    thrift_container_size_limit: Incomplete | None = ...,
): ...
def read_pandas(source, columns: Incomplete | None = ..., **kwargs): ...
def write_table(
    table,
    where,
    row_group_size: Incomplete | None = ...,
    version: str = ...,
    use_dictionary: bool = ...,
    compression: str = ...,
    write_statistics: bool = ...,
    use_deprecated_int96_timestamps: Incomplete | None = ...,
    coerce_timestamps: Incomplete | None = ...,
    allow_truncated_timestamps: bool = ...,
    data_page_size: Incomplete | None = ...,
    flavor: Incomplete | None = ...,
    filesystem: Incomplete | None = ...,
    compression_level: Incomplete | None = ...,
    use_byte_stream_split: bool = ...,
    column_encoding: Incomplete | None = ...,
    data_page_version: str = ...,
    use_compliant_nested_type: bool = ...,
    encryption_properties: Incomplete | None = ...,
    write_batch_size: Incomplete | None = ...,
    dictionary_pagesize_limit: Incomplete | None = ...,
    **kwargs,
) -> None: ...
def write_to_dataset(
    table,
    root_path,
    partition_cols: Incomplete | None = ...,
    partition_filename_cb: Incomplete | None = ...,
    filesystem: Incomplete | None = ...,
    use_legacy_dataset: Incomplete | None = ...,
    schema: Incomplete | None = ...,
    partitioning: Incomplete | None = ...,
    basename_template: Incomplete | None = ...,
    use_threads: Incomplete | None = ...,
    file_visitor: Incomplete | None = ...,
    existing_data_behavior: Incomplete | None = ...,
    **kwargs,
) -> None: ...
def write_metadata(
    schema, where, metadata_collector: Incomplete | None = ..., **kwargs
) -> None: ...
def read_metadata(
    where,
    memory_map: bool = ...,
    decryption_properties: Incomplete | None = ...,
    filesystem: Incomplete | None = ...,
): ...
def read_schema(
    where,
    memory_map: bool = ...,
    decryption_properties: Incomplete | None = ...,
    filesystem: Incomplete | None = ...,
): ...

# Names in __all__ with no definition:
#   _filters_to_expression
