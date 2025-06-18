import sys

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self
from typing import IO, Literal

from _typeshed import StrPath

from . import _orc
from ._fs import SupportedFileSystem
from .lib import KeyValueMetadata, NativeFile, RecordBatch, Schema, Table

class ORCFile:
    """
    Reader interface for a single ORC file

    Parameters
    ----------
    source : str or pyarrow.NativeFile
        Readable source. For passing Python file objects or byte buffers,
        see pyarrow.io.PythonFileInterface or pyarrow.io.BufferReader.
    """

    reader: _orc.ORCReader
    def __init__(self, source: StrPath | NativeFile | IO) -> None: ...
    @property
    def metadata(self) -> KeyValueMetadata:
        """The file metadata, as an arrow KeyValueMetadata"""
    @property
    def schema(self) -> Schema:
        """The file schema, as an arrow schema"""
    @property
    def nrows(self) -> int:
        """The number of rows in the file"""
    @property
    def nstripes(self) -> int:
        """The number of stripes in the file"""
    @property
    def file_version(self) -> str:
        """Format version of the ORC file, must be 0.11 or 0.12"""
    @property
    def software_version(self) -> str:
        """Software instance and version that wrote this file"""
    @property
    def compression(self) -> Literal["UNCOMPRESSED", "ZLIB", "SNAPPY", "LZ4", "ZSTD"]:
        """Compression codec of the file"""
    @property
    def compression_size(self) -> int:
        """Number of bytes to buffer for the compression codec in the file"""
    @property
    def writer(self) -> str:
        """Name of the writer that wrote this file.
        If the writer is unknown then its Writer ID
        (a number) is returned"""
    @property
    def writer_version(self) -> str:
        """Version of the writer"""
    @property
    def row_index_stride(self) -> int:
        """Number of rows per an entry in the row index or 0
        if there is no row index"""
    @property
    def nstripe_statistics(self) -> int:
        """Number of stripe statistics"""
    @property
    def content_length(self) -> int:
        """Length of the data stripes in the file in bytes"""
    @property
    def stripe_statistics_length(self) -> int:
        """The number of compressed bytes in the file stripe statistics"""
    @property
    def file_footer_length(self) -> int:
        """The number of compressed bytes in the file footer"""
    @property
    def file_postscript_length(self) -> int:
        """The number of bytes in the file postscript"""
    @property
    def file_length(self) -> int:
        """The number of bytes in the file"""
    def read_stripe(self, n: int, columns: list[str] | None = None) -> RecordBatch:
        """Read a single stripe from the file.

        Parameters
        ----------
        n : int
            The stripe index
        columns : list
            If not None, only these columns will be read from the stripe. A
            column name may be a prefix of a nested field, e.g. 'a' will select
            'a.b', 'a.c', and 'a.d.e'

        Returns
        -------
        pyarrow.RecordBatch
            Content of the stripe as a RecordBatch.
        """
    def read(self, columns: list[str] | None = None) -> Table:
        """Read the whole file.

        Parameters
        ----------
        columns : list
            If not None, only these columns will be read from the file. A
            column name may be a prefix of a nested field, e.g. 'a' will select
            'a.b', 'a.c', and 'a.d.e'. Output always follows the
            ordering of the file and not the `columns` list.

        Returns
        -------
        pyarrow.Table
            Content of the file as a Table.
        """

class ORCWriter:
    """
    Writer interface for a single ORC file

    Parameters
    ----------
    where : str or pyarrow.io.NativeFile
        Writable target. For passing Python file objects or byte buffers,
        see pyarrow.io.PythonFileInterface, pyarrow.io.BufferOutputStream
        or pyarrow.io.FixedSizeBufferWriter.
    file_version : {"0.11", "0.12"}, default "0.12"
        Determine which ORC file version to use.
        `Hive 0.11 / ORC v0 <https://orc.apache.org/specification/ORCv0/>`_
        is the older version
        while `Hive 0.12 / ORC v1 <https://orc.apache.org/specification/ORCv1/>`_
        is the newer one.
    batch_size : int, default 1024
        Number of rows the ORC writer writes at a time.
    stripe_size : int, default 64 * 1024 * 1024
        Size of each ORC stripe in bytes.
    compression : string, default 'uncompressed'
        The compression codec.
        Valid values: {'UNCOMPRESSED', 'SNAPPY', 'ZLIB', 'LZ4', 'ZSTD'}
        Note that LZ0 is currently not supported.
    compression_block_size : int, default 64 * 1024
        Size of each compression block in bytes.
    compression_strategy : string, default 'speed'
        The compression strategy i.e. speed vs size reduction.
        Valid values: {'SPEED', 'COMPRESSION'}
    row_index_stride : int, default 10000
        The row index stride i.e. the number of rows per
        an entry in the row index.
    padding_tolerance : double, default 0.0
        The padding tolerance.
    dictionary_key_size_threshold : double, default 0.0
        The dictionary key size threshold. 0 to disable dictionary encoding.
        1 to always enable dictionary encoding.
    bloom_filter_columns : None, set-like or list-like, default None
        Columns that use the bloom filter.
    bloom_filter_fpp : double, default 0.05
        Upper limit of the false-positive rate of the bloom filter.
    """

    writer: _orc.ORCWriter
    is_open: bool
    def __init__(
        self,
        where: StrPath | NativeFile | IO,
        *,
        file_version: str = "0.12",
        batch_size: int = 1024,
        stripe_size: int = 64 * 1024 * 1024,
        compression: Literal["UNCOMPRESSED", "ZLIB", "SNAPPY", "LZ4", "ZSTD"] = "UNCOMPRESSED",
        compression_block_size: int = 65536,
        compression_strategy: Literal["COMPRESSION", "SPEED"] = "SPEED",
        row_index_stride: int = 10000,
        padding_tolerance: float = 0.0,
        dictionary_key_size_threshold: float = 0.0,
        bloom_filter_columns: list[int] | None = None,
        bloom_filter_fpp: float = 0.05,
    ): ...
    def __enter__(self) -> Self: ...
    def __exit__(self, *args, **kwargs) -> None: ...
    def write(self, table: Table) -> None:
        """
        Write the table into an ORC file. The schema of the table must
        be equal to the schema used when opening the ORC file.

        Parameters
        ----------
        table : pyarrow.Table
            The table to be written into the ORC file
        """
    def close(self) -> None:
        """
        Close the ORC file
        """

def read_table(
    source: StrPath | NativeFile | IO,
    columns: list[str] | None = None,
    filesystem: SupportedFileSystem | None = None,
) -> Table:
    """
    Read a Table from an ORC file.

    Parameters
    ----------
    source : str, pyarrow.NativeFile, or file-like object
        If a string passed, can be a single file name. For file-like objects,
        only read a single file. Use pyarrow.BufferReader to read a file
        contained in a bytes or buffer-like object.
    columns : list
        If not None, only these columns will be read from the file. A column
        name may be a prefix of a nested field, e.g. 'a' will select 'a.b',
        'a.c', and 'a.d.e'. Output always follows the ordering of the file and
        not the `columns` list. If empty, no columns will be read. Note
        that the table will still have the correct num_rows set despite having
        no columns.
    filesystem : FileSystem, default None
        If nothing passed, will be inferred based on path.
        Path will try to be found in the local on-disk filesystem otherwise
        it will be parsed as an URI to determine the filesystem.
    """

def write_table(
    table: Table,
    where: StrPath | NativeFile | IO,
    *,
    file_version: str = "0.12",
    batch_size: int = 1024,
    stripe_size: int = 64 * 1024 * 1024,
    compression: Literal["UNCOMPRESSED", "ZLIB", "SNAPPY", "LZ4", "ZSTD"] = "UNCOMPRESSED",
    compression_block_size: int = 65536,
    compression_strategy: Literal["COMPRESSION", "SPEED"] = "SPEED",
    row_index_stride: int = 10000,
    padding_tolerance: float = 0.0,
    dictionary_key_size_threshold: float = 0.0,
    bloom_filter_columns: list[int] | None = None,
    bloom_filter_fpp: float = 0.05,
) -> None:
    """
    Write a table into an ORC file.

    Parameters
    ----------
    table : pyarrow.lib.Table
        The table to be written into the ORC file
    where : str or pyarrow.io.NativeFile
        Writable target. For passing Python file objects or byte buffers,
        see pyarrow.io.PythonFileInterface, pyarrow.io.BufferOutputStream
        or pyarrow.io.FixedSizeBufferWriter.
    file_version : {"0.11", "0.12"}, default "0.12"
        Determine which ORC file version to use.
        `Hive 0.11 / ORC v0 <https://orc.apache.org/specification/ORCv0/>`_
        is the older version
        while `Hive 0.12 / ORC v1 <https://orc.apache.org/specification/ORCv1/>`_
        is the newer one.
    batch_size : int, default 1024
        Number of rows the ORC writer writes at a time.
    stripe_size : int, default 64 * 1024 * 1024
        Size of each ORC stripe in bytes.
    compression : string, default 'uncompressed'
        The compression codec.
        Valid values: {'UNCOMPRESSED', 'SNAPPY', 'ZLIB', 'LZ4', 'ZSTD'}
        Note that LZ0 is currently not supported.
    compression_block_size : int, default 64 * 1024
        Size of each compression block in bytes.
    compression_strategy : string, default 'speed'
        The compression strategy i.e. speed vs size reduction.
        Valid values: {'SPEED', 'COMPRESSION'}
    row_index_stride : int, default 10000
        The row index stride i.e. the number of rows per
        an entry in the row index.
    padding_tolerance : double, default 0.0
        The padding tolerance.
    dictionary_key_size_threshold : double, default 0.0
        The dictionary key size threshold. 0 to disable dictionary encoding.
        1 to always enable dictionary encoding.
    bloom_filter_columns : None, set-like or list-like, default None
        Columns that use the bloom filter.
    bloom_filter_fpp : double, default 0.05
        Upper limit of the false-positive rate of the bloom filter.
    """
