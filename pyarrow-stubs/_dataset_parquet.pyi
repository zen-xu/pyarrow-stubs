from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Iterable, TypedDict

from ._compute import Expression
from ._dataset import (
    DatasetFactory,
    FileFormat,
    FileFragment,
    FileWriteOptions,
    Fragment,
    FragmentScanOptions,
    Partitioning,
    PartitioningFactory,
)
from ._dataset_parquet_encryption import ParquetDecryptionConfig
from ._fs import FileSystem
from ._parquet import FileDecryptionProperties, FileMetaData
from .lib import CacheOptions, Schema, _Weakrefable

parquet_encryption_enabled: bool

class ParquetFileFormat(FileFormat):
    def __init__(
        self,
        read_options: ParquetReadOptions,
        default_fragment_scan_options: ParquetFragmentScanOptions,
        **kwargs,
    ) -> None: ...
    @property
    def read_options(self) -> ParquetReadOptions: ...
    def make_write_options(self) -> ParquetFileWriteOptions: ...  # type: ignore[override]
    def equals(self, other: ParquetFileFormat) -> bool: ...
    @property
    def default_extname(self) -> str: ...
    def make_fragment(
        self,
        file: IO | Path | str,
        filesystem: FileSystem | None = None,
        partition_expression: Expression | None = None,
        row_groups: Iterable[int] | None = None,
        *,
        file_size: int | None = None,
    ) -> Fragment: ...

class _NameStats(TypedDict):
    min: Any
    max: Any

class RowGroupInfo:
    id: int
    metadata: FileMetaData
    schema: Schema

    def __init__(self, id: int, metadata: FileMetaData, schema: Schema) -> None: ...
    @property
    def num_rows(self) -> int: ...
    @property
    def total_byte_size(self) -> int: ...
    @property
    def statistics(self) -> dict[str, _NameStats]: ...

class ParquetFileFragment(FileFragment):
    def ensure_complete_metadata(self) -> None: ...
    @property
    def row_groups(self) -> list[RowGroupInfo]: ...
    @property
    def metadata(self) -> FileMetaData: ...
    @property
    def num_row_groups(self) -> int: ...
    def split_by_row_group(
        self, filter: Expression | None = None, schema: Schema | None = None
    ) -> list[Fragment]: ...
    def subset(
        self,
        filter: Expression | None = None,
        schema: Schema | None = None,
        row_group_ids: list[int] | None = None,
    ) -> ParquetFileFormat: ...

class ParquetReadOptions(_Weakrefable):
    def __init__(
        self, dictionary_columns: list[str] | None, coerce_int96_timestamp_unit: str | None = None
    ) -> None: ...
    @property
    def coerce_int96_timestamp_unit(self) -> str: ...
    @coerce_int96_timestamp_unit.setter
    def coerce_int96_timestamp_unit(self, unit: str) -> None: ...
    def equals(self, other: ParquetReadOptions) -> bool: ...

class ParquetFileWriteOptions(FileWriteOptions):
    def update(self, **kwargs) -> None: ...
    def _set_properties(self) -> None: ...
    def _set_arrow_properties(self) -> None: ...
    def _set_encryption_config(self) -> None: ...

@dataclass(kw_only=True)
class ParquetFragmentScanOptions(FragmentScanOptions):
    use_buffered_stream: bool = False
    buffer_size: int = 8192
    pre_buffer: bool = True
    cache_options: CacheOptions | None = None
    thrift_string_size_limit: int | None = None
    thrift_container_size_limit: int | None = None
    decryption_config: ParquetDecryptionConfig | None = None
    decryption_properties: FileDecryptionProperties | None = None
    page_checksum_verification: bool = False

    def equals(self, other: ParquetFragmentScanOptions) -> bool: ...

@dataclass
class ParquetFactoryOptions(_Weakrefable):
    partition_base_dir: str | None = None
    partitioning: Partitioning | PartitioningFactory | None = None
    validate_column_chunk_paths: bool = False

class ParquetDatasetFactory(DatasetFactory):
    def __init__(
        self,
        metadata_path: str,
        filesystem: FileSystem,
        format: FileFormat,
        options: ParquetFactoryOptions | None = None,
    ) -> None: ...
