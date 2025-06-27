from dataclasses import dataclass
from typing import IO, Any, Iterable, TypedDict

from _typeshed import StrPath

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
from ._fs import SupportedFileSystem
from ._parquet import FileDecryptionProperties, FileMetaData
from .lib import CacheOptions, Schema, _Weakrefable

parquet_encryption_enabled: bool

class ParquetFileFormat(FileFormat):
    """
    FileFormat for Parquet

    Parameters
    ----------
    read_options : ParquetReadOptions
        Read options for the file.
    default_fragment_scan_options : ParquetFragmentScanOptions
        Scan Options for the file.
    **kwargs : dict
        Additional options for read option or scan option
    """
    def __init__(
        self,
        read_options: ParquetReadOptions | None = None,
        default_fragment_scan_options: ParquetFragmentScanOptions | None = None,
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
        file: StrPath | IO,
        filesystem: SupportedFileSystem | None = None,
        partition_expression: Expression | None = None,
        row_groups: Iterable[int] | None = None,
        *,
        file_size: int | None = None,
    ) -> Fragment:
        """
        Make a FileFragment from a given file.

        Parameters
        ----------
        file : file-like object, path-like or str
            The file or file path to make a fragment from.
        filesystem : Filesystem, optional
            If `filesystem` is given, `file` must be a string and specifies
            the path of the file to read from the filesystem.
        partition_expression : Expression, optional
            An expression that is guaranteed true for all rows in the fragment.  Allows
            fragment to be potentially skipped while scanning with a filter.
        row_groups : Iterable, optional
            The indices of the row groups to include
        file_size : int, optional
            The size of the file in bytes. Can improve performance with high-latency filesystems
            when file size needs to be known before reading.

        Returns
        -------
        fragment : Fragment
            The file fragment
        """

class _NameStats(TypedDict):
    min: Any
    max: Any

class RowGroupInfo:
    """
    A wrapper class for RowGroup information

    Parameters
    ----------
    id : integer
        The group ID.
    metadata : FileMetaData
        The rowgroup metadata.
    schema : Schema
        Schema of the rows.
    """

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
    """A Fragment representing a parquet file."""

    def ensure_complete_metadata(self) -> None: ...
    @property
    def row_groups(self) -> list[RowGroupInfo]: ...
    @property
    def metadata(self) -> FileMetaData: ...
    @property
    def num_row_groups(self) -> int:
        """
        Return the number of row groups viewed by this fragment (not the
        number of row groups in the origin file).
        """
    def split_by_row_group(
        self, filter: Expression | None = None, schema: Schema | None = None
    ) -> list[Fragment]:
        """
        Split the fragment into multiple fragments.

        Yield a Fragment wrapping each row group in this ParquetFileFragment.
        Row groups will be excluded whose metadata contradicts the optional
        filter.

        Parameters
        ----------
        filter : Expression, default None
            Only include the row groups which satisfy this predicate (using
            the Parquet RowGroup statistics).
        schema : Schema, default None
            Schema to use when filtering row groups. Defaults to the
            Fragment's physical schema

        Returns
        -------
        A list of Fragments
        """
    def subset(
        self,
        filter: Expression | None = None,
        schema: Schema | None = None,
        row_group_ids: list[int] | None = None,
    ) -> ParquetFileFormat:
        """
        Create a subset of the fragment (viewing a subset of the row groups).

        Subset can be specified by either a filter predicate (with optional
        schema) or by a list of row group IDs. Note that when using a filter,
        the resulting fragment can be empty (viewing no row groups).

        Parameters
        ----------
        filter : Expression, default None
            Only include the row groups which satisfy this predicate (using
            the Parquet RowGroup statistics).
        schema : Schema, default None
            Schema to use when filtering row groups. Defaults to the
            Fragment's physical schema
        row_group_ids : list of ints
            The row group IDs to include in the subset. Can only be specified
            if `filter` is None.

        Returns
        -------
        ParquetFileFragment
        """

class ParquetReadOptions(_Weakrefable):
    """
    Parquet format specific options for reading.

    Parameters
    ----------
    dictionary_columns : list of string, default None
        Names of columns which should be dictionary encoded as
        they are read
    coerce_int96_timestamp_unit : str, default None
        Cast timestamps that are stored in INT96 format to a particular
        resolution (e.g. 'ms'). Setting to None is equivalent to 'ns'
        and therefore INT96 timestamps will be inferred as timestamps
        in nanoseconds
    """
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
    """
    Scan-specific options for Parquet fragments.

    Parameters
    ----------
    use_buffered_stream : bool, default False
        Read files through buffered input streams rather than loading entire
        row groups at once. This may be enabled to reduce memory overhead.
        Disabled by default.
    buffer_size : int, default 8192
        Size of buffered stream, if enabled. Default is 8KB.
    pre_buffer : bool, default True
        If enabled, pre-buffer the raw Parquet data instead of issuing one
        read per column chunk. This can improve performance on high-latency
        filesystems (e.g. S3, GCS) by coalescing and issuing file reads in
        parallel using a background I/O thread pool.
        Set to False if you want to prioritize minimal memory usage
        over maximum speed.
    cache_options : pyarrow.CacheOptions, default None
        Cache options used when pre_buffer is enabled. The default values should
        be good for most use cases. You may want to adjust these for example if
        you have exceptionally high latency to the file system.
    thrift_string_size_limit : int, default None
        If not None, override the maximum total string size allocated
        when decoding Thrift structures. The default limit should be
        sufficient for most Parquet files.
    thrift_container_size_limit : int, default None
        If not None, override the maximum total size of containers allocated
        when decoding Thrift structures. The default limit should be
        sufficient for most Parquet files.
    decryption_config : pyarrow.dataset.ParquetDecryptionConfig, default None
        If not None, use the provided ParquetDecryptionConfig to decrypt the
        Parquet file.
    decryption_properties : pyarrow.parquet.FileDecryptionProperties, default None
        If not None, use the provided FileDecryptionProperties to decrypt encrypted
        Parquet file.
    page_checksum_verification : bool, default False
        If True, verify the page checksum for each page read from the file.
    """

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
    """
    Influences the discovery of parquet dataset.

    Parameters
    ----------
    partition_base_dir : str, optional
        For the purposes of applying the partitioning, paths will be
        stripped of the partition_base_dir. Files not matching the
        partition_base_dir prefix will be skipped for partitioning discovery.
        The ignored files will still be part of the Dataset, but will not
        have partition information.
    partitioning : Partitioning, PartitioningFactory, optional
        The partitioning scheme applied to fragments, see ``Partitioning``.
    validate_column_chunk_paths : bool, default False
        Assert that all ColumnChunk paths are consistent. The parquet spec
        allows for ColumnChunk data to be stored in multiple files, but
        ParquetDatasetFactory supports only a single file with all ColumnChunk
        data. If this flag is set construction of a ParquetDatasetFactory will
        raise an error if ColumnChunk data is not resident in a single file.
    """

    partition_base_dir: str | None = None
    partitioning: Partitioning | PartitioningFactory | None = None
    validate_column_chunk_paths: bool = False

class ParquetDatasetFactory(DatasetFactory):
    """
    Create a ParquetDatasetFactory from a Parquet `_metadata` file.

    Parameters
    ----------
    metadata_path : str
        Path to the `_metadata` parquet metadata-only file generated with
        `pyarrow.parquet.write_metadata`.
    filesystem : pyarrow.fs.FileSystem
        Filesystem to read the metadata_path from, and subsequent parquet
        files.
    format : ParquetFileFormat
        Parquet format options.
    options : ParquetFactoryOptions, optional
        Various flags influencing the discovery of filesystem paths.
    """
    def __init__(
        self,
        metadata_path: str,
        filesystem: SupportedFileSystem,
        format: FileFormat,
        options: ParquetFactoryOptions | None = None,
    ) -> None: ...
