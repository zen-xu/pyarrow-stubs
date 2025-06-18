import sys

from pathlib import Path

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self
from typing import IO, Callable, Iterator, Literal, Sequence

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

from pyarrow import _parquet
from pyarrow._compute import Expression
from pyarrow._fs import FileSystem, SupportedFileSystem
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
from pyarrow._stubs_typing import FilterTuple, SingleOrList
from pyarrow.dataset import ParquetFileFragment, Partitioning
from pyarrow.lib import NativeFile, RecordBatch, Schema, Table
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

def filters_to_expression(filters: list[FilterTuple | list[FilterTuple]]) -> Expression:
    """
    Check if filters are well-formed and convert to an ``Expression``.

    Parameters
    ----------
    filters : List[Tuple] or List[List[Tuple]]

    Notes
    -----
    See internal ``pyarrow._DNF_filter_doc`` attribute for more details.

    Examples
    --------

    >>> filters_to_expression([("foo", "==", "bar")])
    <pyarrow.compute.Expression (foo == "bar")>

    Returns
    -------
    pyarrow.compute.Expression
        An Expression representing the filters
    """

@deprecated("use filters_to_expression")
def _filters_to_expression(filters: list[FilterTuple | list[FilterTuple]]) -> Expression: ...

_Compression: TypeAlias = Literal["gzip", "bz2", "brotli", "lz4", "zstd", "snappy", "none"]

class ParquetFile:
    """
    Reader interface for a single Parquet file.

    Parameters
    ----------
    source : str, pathlib.Path, pyarrow.NativeFile, or file-like object
        Readable source. For passing bytes or buffer-like file containing a
        Parquet file, use pyarrow.BufferReader.
    metadata : FileMetaData, default None
        Use existing metadata object, rather than reading from file.
    common_metadata : FileMetaData, default None
        Will be used in reads for pandas schema metadata if not found in the
        main file's metadata, no other uses at the moment.
    read_dictionary : list
        List of column names to read directly as DictionaryArray.
    memory_map : bool, default False
        If the source is a file path, use a memory map to read file, which can
        improve performance in some environments.
    buffer_size : int, default 0
        If positive, perform read buffering when deserializing individual
        column chunks. Otherwise IO calls are unbuffered.
    pre_buffer : bool, default False
        Coalesce and issue file reads in parallel to improve performance on
        high-latency filesystems (e.g. S3). If True, Arrow will use a
        background I/O thread pool.
    coerce_int96_timestamp_unit : str, default None
        Cast timestamps that are stored in INT96 format to a particular
        resolution (e.g. 'ms'). Setting to None is equivalent to 'ns'
        and therefore INT96 timestamps will be inferred as timestamps
        in nanoseconds.
    decryption_properties : FileDecryptionProperties, default None
        File decryption properties for Parquet Modular Encryption.
    thrift_string_size_limit : int, default None
        If not None, override the maximum total string size allocated
        when decoding Thrift structures. The default limit should be
        sufficient for most Parquet files.
    thrift_container_size_limit : int, default None
        If not None, override the maximum total size of containers allocated
        when decoding Thrift structures. The default limit should be
        sufficient for most Parquet files.
    filesystem : FileSystem, default None
        If nothing passed, will be inferred based on path.
        Path will try to be found in the local on-disk filesystem otherwise
        it will be parsed as an URI to determine the filesystem.
    page_checksum_verification : bool, default False
        If True, verify the checksum for each page read from the file.

    Examples
    --------

    Generate an example PyArrow Table and write it to Parquet file:

    >>> import pyarrow as pa
    >>> table = pa.table(
    ...     {
    ...         "n_legs": [2, 2, 4, 4, 5, 100],
    ...         "animal": ["Flamingo", "Parrot", "Dog", "Horse", "Brittle stars", "Centipede"],
    ...     }
    ... )

    >>> import pyarrow.parquet as pq
    >>> pq.write_table(table, "example.parquet")

    Create a ``ParquetFile`` object from the Parquet file:

    >>> parquet_file = pq.ParquetFile("example.parquet")

    Read the data:

    >>> parquet_file.read()
    pyarrow.Table
    n_legs: int64
    animal: string
    ----
    n_legs: [[2,2,4,4,5,100]]
    animal: [["Flamingo","Parrot","Dog","Horse","Brittle stars","Centipede"]]

    Create a ParquetFile object with "animal" column as DictionaryArray:

    >>> parquet_file = pq.ParquetFile("example.parquet", read_dictionary=["animal"])
    >>> parquet_file.read()
    pyarrow.Table
    n_legs: int64
    animal: dictionary<values=string, indices=int32, ordered=0>
    ----
    n_legs: [[2,2,4,4,5,100]]
    animal: [  -- dictionary:
    ["Flamingo","Parrot",...,"Brittle stars","Centipede"]  -- indices:
    [0,1,2,3,4,5]]
    """

    reader: ParquetReader
    common_metadata: FileMetaData

    def __init__(
        self,
        source: str | Path | NativeFile | IO,
        *,
        metadata: FileMetaData | None = None,
        common_metadata: FileMetaData | None = None,
        read_dictionary: list[str] | None = None,
        memory_map: bool = False,
        buffer_size: int = 0,
        pre_buffer: bool = False,
        coerce_int96_timestamp_unit: str | None = None,
        decryption_properties: FileDecryptionProperties | None = None,
        thrift_string_size_limit: int | None = None,
        thrift_container_size_limit: int | None = None,
        filesystem: SupportedFileSystem | None = None,
        page_checksum_verification: bool = False,
    ): ...
    def __enter__(self) -> Self: ...
    def __exit__(self, *args, **kwargs) -> None: ...
    @property
    def metadata(self) -> FileMetaData:
        """
        Return the Parquet metadata.
        """
    @property
    def schema(self) -> ParquetSchema:
        """
        Return the Parquet schema, unconverted to Arrow types
        """
    @property
    def schema_arrow(self) -> Schema:
        """
        Return the inferred Arrow schema, converted from the whole Parquet
        file's schema

        Examples
        --------
        Generate an example Parquet file:

        >>> import pyarrow as pa
        >>> table = pa.table(
        ...     {
        ...         "n_legs": [2, 2, 4, 4, 5, 100],
        ...         "animal": ["Flamingo", "Parrot", "Dog", "Horse", "Brittle stars", "Centipede"],
        ...     }
        ... )
        >>> import pyarrow.parquet as pq
        >>> pq.write_table(table, "example.parquet")
        >>> parquet_file = pq.ParquetFile("example.parquet")

        Read the Arrow schema:

        >>> parquet_file.schema_arrow
        n_legs: int64
        animal: string
        """
    @property
    def num_row_groups(self) -> int:
        """
        Return the number of row groups of the Parquet file.

        Examples
        --------
        >>> import pyarrow as pa
        >>> table = pa.table(
        ...     {
        ...         "n_legs": [2, 2, 4, 4, 5, 100],
        ...         "animal": ["Flamingo", "Parrot", "Dog", "Horse", "Brittle stars", "Centipede"],
        ...     }
        ... )
        >>> import pyarrow.parquet as pq
        >>> pq.write_table(table, "example.parquet")
        >>> parquet_file = pq.ParquetFile("example.parquet")

        >>> parquet_file.num_row_groups
        1
        """
    def close(self, force: bool = False) -> None: ...
    @property
    def closed(self) -> bool: ...
    def read_row_group(
        self,
        i: int,
        columns: list | None = None,
        use_threads: bool = True,
        use_pandas_metadata: bool = False,
    ) -> Table:
        """
        Read a single row group from a Parquet file.

        Parameters
        ----------
        i : int
            Index of the individual row group that we want to read.
        columns : list
            If not None, only these columns will be read from the row group. A
            column name may be a prefix of a nested field, e.g. 'a' will select
            'a.b', 'a.c', and 'a.d.e'.
        use_threads : bool, default True
            Perform multi-threaded column reads.
        use_pandas_metadata : bool, default False
            If True and file has custom pandas schema metadata, ensure that
            index columns are also loaded.

        Returns
        -------
        pyarrow.table.Table
            Content of the row group as a table (of columns)

        Examples
        --------
        >>> import pyarrow as pa
        >>> table = pa.table(
        ...     {
        ...         "n_legs": [2, 2, 4, 4, 5, 100],
        ...         "animal": ["Flamingo", "Parrot", "Dog", "Horse", "Brittle stars", "Centipede"],
        ...     }
        ... )
        >>> import pyarrow.parquet as pq
        >>> pq.write_table(table, "example.parquet")
        >>> parquet_file = pq.ParquetFile("example.parquet")

        >>> parquet_file.read_row_group(0)
        pyarrow.Table
        n_legs: int64
        animal: string
        ----
        n_legs: [[2,2,4,4,5,100]]
        animal: [["Flamingo","Parrot",...,"Brittle stars","Centipede"]]
        """
    def read_row_groups(
        self,
        row_groups: list,
        columns: list | None = None,
        use_threads: bool = True,
        use_pandas_metadata: bool = False,
    ) -> Table:
        """
        Read a multiple row groups from a Parquet file.

        Parameters
        ----------
        row_groups : list
            Only these row groups will be read from the file.
        columns : list
            If not None, only these columns will be read from the row group. A
            column name may be a prefix of a nested field, e.g. 'a' will select
            'a.b', 'a.c', and 'a.d.e'.
        use_threads : bool, default True
            Perform multi-threaded column reads.
        use_pandas_metadata : bool, default False
            If True and file has custom pandas schema metadata, ensure that
            index columns are also loaded.

        Returns
        -------
        pyarrow.table.Table
            Content of the row groups as a table (of columns).

        Examples
        --------
        >>> import pyarrow as pa
        >>> table = pa.table(
        ...     {
        ...         "n_legs": [2, 2, 4, 4, 5, 100],
        ...         "animal": ["Flamingo", "Parrot", "Dog", "Horse", "Brittle stars", "Centipede"],
        ...     }
        ... )
        >>> import pyarrow.parquet as pq
        >>> pq.write_table(table, "example.parquet")
        >>> parquet_file = pq.ParquetFile("example.parquet")

        >>> parquet_file.read_row_groups([0, 0])
        pyarrow.Table
        n_legs: int64
        animal: string
        ----
        n_legs: [[2,2,4,4,5,...,2,4,4,5,100]]
        animal: [["Flamingo","Parrot","Dog",...,"Brittle stars","Centipede"]]
        """
    def iter_batches(
        self,
        batch_size: int = 65536,
        row_groups: list | None = None,
        columns: list | None = None,
        use_threads: bool = True,
        use_pandas_metadata: bool = False,
    ) -> Iterator[RecordBatch]:
        """
        Read streaming batches from a Parquet file.

        Parameters
        ----------
        batch_size : int, default 64K
            Maximum number of records to yield per batch. Batches may be
            smaller if there aren't enough rows in the file.
        row_groups : list
            Only these row groups will be read from the file.
        columns : list
            If not None, only these columns will be read from the file. A
            column name may be a prefix of a nested field, e.g. 'a' will select
            'a.b', 'a.c', and 'a.d.e'.
        use_threads : boolean, default True
            Perform multi-threaded column reads.
        use_pandas_metadata : boolean, default False
            If True and file has custom pandas schema metadata, ensure that
            index columns are also loaded.

        Yields
        ------
        pyarrow.RecordBatch
            Contents of each batch as a record batch

        Examples
        --------
        Generate an example Parquet file:

        >>> import pyarrow as pa
        >>> table = pa.table(
        ...     {
        ...         "n_legs": [2, 2, 4, 4, 5, 100],
        ...         "animal": ["Flamingo", "Parrot", "Dog", "Horse", "Brittle stars", "Centipede"],
        ...     }
        ... )
        >>> import pyarrow.parquet as pq
        >>> pq.write_table(table, "example.parquet")
        >>> parquet_file = pq.ParquetFile("example.parquet")
        >>> for i in parquet_file.iter_batches():
        ...     print("RecordBatch")
        ...     print(i.to_pandas())
        RecordBatch
           n_legs         animal
        0       2       Flamingo
        1       2         Parrot
        2       4            Dog
        3       4          Horse
        4       5  Brittle stars
        5     100      Centipede
        """
    def read(
        self,
        columns: list | None = None,
        use_threads: bool = True,
        use_pandas_metadata: bool = False,
    ) -> Table:
        """
        Read a Table from Parquet format.

        Parameters
        ----------
        columns : list
            If not None, only these columns will be read from the file. A
            column name may be a prefix of a nested field, e.g. 'a' will select
            'a.b', 'a.c', and 'a.d.e'.
        use_threads : bool, default True
            Perform multi-threaded column reads.
        use_pandas_metadata : bool, default False
            If True and file has custom pandas schema metadata, ensure that
            index columns are also loaded.

        Returns
        -------
        pyarrow.table.Table
            Content of the file as a table (of columns).

        Examples
        --------
        Generate an example Parquet file:

        >>> import pyarrow as pa
        >>> table = pa.table(
        ...     {
        ...         "n_legs": [2, 2, 4, 4, 5, 100],
        ...         "animal": ["Flamingo", "Parrot", "Dog", "Horse", "Brittle stars", "Centipede"],
        ...     }
        ... )
        >>> import pyarrow.parquet as pq
        >>> pq.write_table(table, "example.parquet")
        >>> parquet_file = pq.ParquetFile("example.parquet")

        Read a Table:

        >>> parquet_file.read(columns=["animal"])
        pyarrow.Table
        animal: string
        ----
        animal: [["Flamingo","Parrot",...,"Brittle stars","Centipede"]]
        """
    def scan_contents(self, columns: list | None = None, batch_size: int = 65536) -> int:
        """
        Read contents of file for the given columns and batch size.

        Notes
        -----
        This function's primary purpose is benchmarking.
        The scan is executed on a single thread.

        Parameters
        ----------
        columns : list of integers, default None
            Select columns to read, if None scan all columns.
        batch_size : int, default 64K
            Number of rows to read at a time internally.

        Returns
        -------
        num_rows : int
            Number of rows in file

        Examples
        --------
        >>> import pyarrow as pa
        >>> table = pa.table(
        ...     {
        ...         "n_legs": [2, 2, 4, 4, 5, 100],
        ...         "animal": ["Flamingo", "Parrot", "Dog", "Horse", "Brittle stars", "Centipede"],
        ...     }
        ... )
        >>> import pyarrow.parquet as pq
        >>> pq.write_table(table, "example.parquet")
        >>> parquet_file = pq.ParquetFile("example.parquet")

        >>> parquet_file.scan_contents()
        6
        """

class ParquetWriter:
    """
    Class for incrementally building a Parquet file for Arrow tables.

    Parameters
    ----------
    where : path or file-like object
    schema : pyarrow.Schema
    version : {"1.0", "2.4", "2.6"}, default "2.6"
        Determine which Parquet logical types are available for use, whether the
        reduced set from the Parquet 1.x.x format or the expanded logical types
        added in later format versions.
        Files written with version='2.4' or '2.6' may not be readable in all
        Parquet implementations, so version='1.0' is likely the choice that
        maximizes file compatibility.
        UINT32 and some logical types are only available with version '2.4'.
        Nanosecond timestamps are only available with version '2.6'.
        Other features such as compression algorithms or the new serialized
        data page format must be enabled separately (see 'compression' and
        'data_page_version').
    use_dictionary : bool or list, default True
        Specify if we should use dictionary encoding in general or only for
        some columns.
        When encoding the column, if the dictionary size is too large, the
        column will fallback to ``PLAIN`` encoding. Specially, ``BOOLEAN`` type
        doesn't support dictionary encoding.
    compression : str or dict, default 'snappy'
        Specify the compression codec, either on a general basis or per-column.
        Valid values: {'NONE', 'SNAPPY', 'GZIP', 'BROTLI', 'LZ4', 'ZSTD'}.
    write_statistics : bool or list, default True
        Specify if we should write statistics in general (default is True) or only
        for some columns.
    use_deprecated_int96_timestamps : bool, default None
        Write timestamps to INT96 Parquet format. Defaults to False unless enabled
        by flavor argument. This take priority over the coerce_timestamps option.
    coerce_timestamps : str, default None
        Cast timestamps to a particular resolution. If omitted, defaults are chosen
        depending on `version`. For ``version='1.0'`` and ``version='2.4'``,
        nanoseconds are cast to microseconds ('us'), while for
        ``version='2.6'`` (the default), they are written natively without loss
        of resolution.  Seconds are always cast to milliseconds ('ms') by default,
        as Parquet does not have any temporal type with seconds resolution.
        If the casting results in loss of data, it will raise an exception
        unless ``allow_truncated_timestamps=True`` is given.
        Valid values: {None, 'ms', 'us'}
    allow_truncated_timestamps : bool, default False
        Allow loss of data when coercing timestamps to a particular
        resolution. E.g. if microsecond or nanosecond data is lost when coercing to
        'ms', do not raise an exception. Passing ``allow_truncated_timestamp=True``
        will NOT result in the truncation exception being ignored unless
        ``coerce_timestamps`` is not None.
    data_page_size : int, default None
        Set a target threshold for the approximate encoded size of data
        pages within a column chunk (in bytes). If None, use the default data page
        size of 1MByte.
    flavor : {'spark'}, default None
        Sanitize schema or set other compatibility options to work with
        various target systems.
    filesystem : FileSystem, default None
        If nothing passed, will be inferred from `where` if path-like, else
        `where` is already a file-like object so no filesystem is needed.
    compression_level : int or dict, default None
        Specify the compression level for a codec, either on a general basis or
        per-column. If None is passed, arrow selects the compression level for
        the compression codec in use. The compression level has a different
        meaning for each codec, so you have to read the documentation of the
        codec you are using.
        An exception is thrown if the compression codec does not allow specifying
        a compression level.
    use_byte_stream_split : bool or list, default False
        Specify if the byte_stream_split encoding should be used in general or
        only for some columns. If both dictionary and byte_stream_stream are
        enabled, then dictionary is preferred.
        The byte_stream_split encoding is valid for integer, floating-point
        and fixed-size binary data types (including decimals); it should be
        combined with a compression codec so as to achieve size reduction.
    column_encoding : string or dict, default None
        Specify the encoding scheme on a per column basis.
        Can only be used when ``use_dictionary`` is set to False, and
        cannot be used in combination with ``use_byte_stream_split``.
        Currently supported values: {'PLAIN', 'BYTE_STREAM_SPLIT',
        'DELTA_BINARY_PACKED', 'DELTA_LENGTH_BYTE_ARRAY', 'DELTA_BYTE_ARRAY'}.
        Certain encodings are only compatible with certain data types.
        Please refer to the encodings section of `Reading and writing Parquet
        files <https://arrow.apache.org/docs/cpp/parquet.html#encodings>`_.
    data_page_version : {"1.0", "2.0"}, default "1.0"
        The serialized Parquet data page format version to write, defaults to
        1.0. This does not impact the file schema logical types and Arrow to
        Parquet type casting behavior; for that use the "version" option.
    use_compliant_nested_type : bool, default True
        Whether to write compliant Parquet nested type (lists) as defined
        `here <https://github.com/apache/parquet-format/blob/master/
        LogicalTypes.md#nested-types>`_, defaults to ``True``.
        For ``use_compliant_nested_type=True``, this will write into a list
        with 3-level structure where the middle level, named ``list``,
        is a repeated group with a single field named ``element``::

            <list-repetition> group <name> (LIST) {
                repeated group list {
                        <element-repetition> <element-type> element;
                }
            }

        For ``use_compliant_nested_type=False``, this will also write into a list
        with 3-level structure, where the name of the single field of the middle
        level ``list`` is taken from the element name for nested columns in Arrow,
        which defaults to ``item``::

            <list-repetition> group <name> (LIST) {
                repeated group list {
                    <element-repetition> <element-type> item;
                }
            }
    encryption_properties : FileEncryptionProperties, default None
        File encryption properties for Parquet Modular Encryption.
        If None, no encryption will be done.
        The encryption properties can be created using:
        ``CryptoFactory.file_encryption_properties()``.
    write_batch_size : int, default None
        Number of values to write to a page at a time. If None, use the default of
        1024. ``write_batch_size`` is complementary to ``data_page_size``. If pages
        are exceeding the ``data_page_size`` due to large column values, lowering
        the batch size can help keep page sizes closer to the intended size.
    dictionary_pagesize_limit : int, default None
        Specify the dictionary page size limit per row group. If None, use the
        default 1MB.
    store_schema : bool, default True
        By default, the Arrow schema is serialized and stored in the Parquet
        file metadata (in the "ARROW:schema" key). When reading the file,
        if this key is available, it will be used to more faithfully recreate
        the original Arrow data. For example, for tz-aware timestamp columns
        it will restore the timezone (Parquet only stores the UTC values without
        timezone), or columns with duration type will be restored from the int64
        Parquet column.
    write_page_index : bool, default False
        Whether to write a page index in general for all columns.
        Writing statistics to the page index disables the old method of writing
        statistics to each data page header. The page index makes statistics-based
        filtering more efficient than the page header, as it gathers all the
        statistics for a Parquet file in a single place, avoiding scattered I/O.
        Note that the page index is not yet used on the read size by PyArrow.
    write_page_checksum : bool, default False
        Whether to write page checksums in general for all columns.
        Page checksums enable detection of data corruption, which might occur during
        transmission or in the storage.
    sorting_columns : Sequence of SortingColumn, default None
        Specify the sort order of the data being written. The writer does not sort
        the data nor does it verify that the data is sorted. The sort order is
        written to the row group metadata, which can then be used by readers.
    store_decimal_as_integer : bool, default False
        Allow decimals with 1 <= precision <= 18 to be stored as integers.
        In Parquet, DECIMAL can be stored in any of the following physical types:
        - int32: for 1 <= precision <= 9.
        - int64: for 10 <= precision <= 18.
        - fixed_len_byte_array: precision is limited by the array size.
            Length n can store <= floor(log_10(2^(8*n - 1) - 1)) base-10 digits.
        - binary: precision is unlimited. The minimum number of bytes to store the
            unscaled value is used.

        By default, this is DISABLED and all decimal types annotate fixed_len_byte_array.
        When enabled, the writer will use the following physical types to store decimals:
        - int32: for 1 <= precision <= 9.
        - int64: for 10 <= precision <= 18.
        - fixed_len_byte_array: for precision > 18.

        As a consequence, decimal columns stored in integer types are more compact.
    writer_engine_version : unused
    **options : dict
        If options contains a key `metadata_collector` then the
        corresponding value is assumed to be a list (or any object with
        `.append` method) that will be filled with the file metadata instance
        of the written file.

    Examples
    --------
    Generate an example PyArrow Table and RecordBatch:

    >>> import pyarrow as pa
    >>> table = pa.table(
    ...     {
    ...         "n_legs": [2, 2, 4, 4, 5, 100],
    ...         "animal": ["Flamingo", "Parrot", "Dog", "Horse", "Brittle stars", "Centipede"],
    ...     }
    ... )
    >>> batch = pa.record_batch(
    ...     [
    ...         [2, 2, 4, 4, 5, 100],
    ...         ["Flamingo", "Parrot", "Dog", "Horse", "Brittle stars", "Centipede"],
    ...     ],
    ...     names=["n_legs", "animal"],
    ... )

    create a ParquetWriter object:

    >>> import pyarrow.parquet as pq
    >>> writer = pq.ParquetWriter("example.parquet", table.schema)

    and write the Table into the Parquet file:

    >>> writer.write_table(table)
    >>> writer.close()

    >>> pq.read_table("example.parquet").to_pandas()
       n_legs         animal
    0       2       Flamingo
    1       2         Parrot
    2       4            Dog
    3       4          Horse
    4       5  Brittle stars
    5     100      Centipede

    create a ParquetWriter object for the RecordBatch:

    >>> writer2 = pq.ParquetWriter("example2.parquet", batch.schema)

    and write the RecordBatch into the Parquet file:

    >>> writer2.write_batch(batch)
    >>> writer2.close()

    >>> pq.read_table("example2.parquet").to_pandas()
       n_legs         animal
    0       2       Flamingo
    1       2         Parrot
    2       4            Dog
    3       4          Horse
    4       5  Brittle stars
    5     100      Centipede
    """

    flavor: str
    schema_changed: bool
    schema: ParquetSchema
    where: str | Path | IO
    file_handler: NativeFile | None
    writer: _parquet.ParquetWriter
    is_open: bool

    def __init__(
        self,
        where: str | Path | IO | NativeFile,
        schema: Schema,
        filesystem: SupportedFileSystem | None = None,
        flavor: str | None = None,
        version: Literal["1.0", "2.4", "2.6"] = ...,
        use_dictionary: bool = True,
        compression: _Compression | dict[str, _Compression] = "snappy",
        write_statistics: bool | list = True,
        use_deprecated_int96_timestamps: bool | None = None,
        compression_level: int | dict | None = None,
        use_byte_stream_split: bool | list = False,
        column_encoding: str | dict | None = None,
        writer_engine_version=None,
        data_page_version: Literal["1.0", "2.0"] = ...,
        use_compliant_nested_type: bool = True,
        encryption_properties: FileEncryptionProperties | None = None,
        write_batch_size: int | None = None,
        dictionary_pagesize_limit: int | None = None,
        store_schema: bool = True,
        write_page_index: bool = False,
        write_page_checksum: bool = False,
        sorting_columns: Sequence[SortingColumn] | None = None,
        store_decimal_as_integer: bool = False,
        **options,
    ) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(self, *args, **kwargs) -> Literal[False]: ...
    def write(
        self, table_or_batch: RecordBatch | Table, row_group_size: int | None = None
    ) -> None:
        """
        Write RecordBatch or Table to the Parquet file.

        Parameters
        ----------
        table_or_batch : {RecordBatch, Table}
        row_group_size : int, default None
            Maximum number of rows in each written row group. If None,
            the row group size will be the minimum of the input
            table or batch length and 1024 * 1024.
        """
    def write_batch(self, batch: RecordBatch, row_group_size: int | None = None) -> None:
        """
        Write RecordBatch to the Parquet file.

        Parameters
        ----------
        batch : RecordBatch
        row_group_size : int, default None
            Maximum number of rows in written row group. If None, the
            row group size will be the minimum of the RecordBatch
            size and 1024 * 1024.  If set larger than 64Mi then 64Mi
            will be used instead.
        """
    def write_table(self, table: Table, row_group_size: int | None = None) -> None:
        """
        Write Table to the Parquet file.

        Parameters
        ----------
        table : Table
        row_group_size : int, default None
            Maximum number of rows in each written row group. If None,
            the row group size will be the minimum of the Table size
            and 1024 * 1024.  If set larger than 64Mi then 64Mi will
            be used instead.

        """
    def close(self) -> None:
        """
        Close the connection to the Parquet file.
        """
    def add_key_value_metadata(self, key_value_metadata: dict[str, str]) -> None:
        """
        Add key-value metadata to the file.
        This will overwrite any existing metadata with the same key.

        Parameters
        ----------
        key_value_metadata : dict
            Keys and values must be string-like / coercible to bytes.
        """

class ParquetDataset:
    """
    Encapsulates details of reading a complete Parquet dataset possibly
    consisting of multiple files and partitions in subdirectories.

    Parameters
    ----------
    path_or_paths : str or List[str]
        A directory name, single file name, or list of file names.
    filesystem : FileSystem, default None
        If nothing passed, will be inferred based on path.
        Path will try to be found in the local on-disk filesystem otherwise
        it will be parsed as an URI to determine the filesystem.
    schema : pyarrow.parquet.Schema
        Optionally provide the Schema for the Dataset, in which case it will
        not be inferred from the source.
    filters : pyarrow.compute.Expression or List[Tuple] or List[List[Tuple]], default None
        Rows which do not match the filter predicate will be removed from scanned
        data. Partition keys embedded in a nested directory structure will be
        exploited to avoid loading files at all if they contain no matching rows.
        Within-file level filtering and different partitioning schemes are supported.

        Predicates are expressed using an ``Expression`` or using
        the disjunctive normal form (DNF), like ``[[('x', '=', 0), ...], ...]``.
        DNF allows arbitrary boolean logical combinations of single column predicates.
        The innermost tuples each describe a single column predicate. The list of inner
        predicates is interpreted as a conjunction (AND), forming a more selective and
        multiple column predicate. Finally, the most outer list combines these filters
        as a disjunction (OR).

        Predicates may also be passed as List[Tuple]. This form is interpreted
        as a single conjunction. To express OR in predicates, one must
        use the (preferred) List[List[Tuple]] notation.

        Each tuple has format: (``key``, ``op``, ``value``) and compares the
        ``key`` with the ``value``.
        The supported ``op`` are:  ``=`` or ``==``, ``!=``, ``<``, ``>``, ``<=``,
        ``>=``, ``in`` and ``not in``. If the ``op`` is ``in`` or ``not in``, the
        ``value`` must be a collection such as a ``list``, a ``set`` or a
        ``tuple``.

        Examples:

        Using the ``Expression`` API:

        .. code-block:: python

            import pyarrow.compute as pc
            pc.field('x') = 0
            pc.field('y').isin(['a', 'b', 'c'])
            ~pc.field('y').isin({'a', 'b'})

        Using the DNF format:

        .. code-block:: python

            ("x", "=", 0)
            ("y", "in", ["a", "b", "c"])
            ("z", "not in", {"a", "b"})


    read_dictionary : list, default None
        List of names or column paths (for nested types) to read directly
        as DictionaryArray. Only supported for BYTE_ARRAY storage. To read
        a flat column as dictionary-encoded pass the column name. For
        nested types, you must pass the full column "path", which could be
        something like level1.level2.list.item. Refer to the Parquet
        file's schema to obtain the paths.
    memory_map : bool, default False
        If the source is a file path, use a memory map to read file, which can
        improve performance in some environments.
    buffer_size : int, default 0
        If positive, perform read buffering when deserializing individual
        column chunks. Otherwise IO calls are unbuffered.
    partitioning : pyarrow.dataset.Partitioning or str or list of str, default "hive"
        The partitioning scheme for a partitioned dataset. The default of "hive"
        assumes directory names with key=value pairs like "/year=2009/month=11".
        In addition, a scheme like "/2009/11" is also supported, in which case
        you need to specify the field names or a full schema. See the
        ``pyarrow.dataset.partitioning()`` function for more details.
    ignore_prefixes : list, optional
        Files matching any of these prefixes will be ignored by the
        discovery process.
        This is matched to the basename of a path.
        By default this is ['.', '_'].
        Note that discovery happens only if a directory is passed as source.
    pre_buffer : bool, default True
        Coalesce and issue file reads in parallel to improve performance on
        high-latency filesystems (e.g. S3, GCS). If True, Arrow will use a
        background I/O thread pool. If using a filesystem layer that itself
        performs readahead (e.g. fsspec's S3FS), disable readahead for best
        results. Set to False if you want to prioritize minimal memory usage
        over maximum speed.
    coerce_int96_timestamp_unit : str, default None
        Cast timestamps that are stored in INT96 format to a particular resolution
        (e.g. 'ms'). Setting to None is equivalent to 'ns' and therefore INT96
        timestamps will be inferred as timestamps in nanoseconds.
    decryption_properties : FileDecryptionProperties or None
        File-level decryption properties.
        The decryption properties can be created using
        ``CryptoFactory.file_decryption_properties()``.
    thrift_string_size_limit : int, default None
        If not None, override the maximum total string size allocated
        when decoding Thrift structures. The default limit should be
        sufficient for most Parquet files.
    thrift_container_size_limit : int, default None
        If not None, override the maximum total size of containers allocated
        when decoding Thrift structures. The default limit should be
        sufficient for most Parquet files.
    page_checksum_verification : bool, default False
        If True, verify the page checksum for each page read from the file.

    Examples
    --------
    Generate an example PyArrow Table and write it to a partitioned dataset:

    >>> import pyarrow as pa
    >>> table = pa.table(
    ...     {
    ...         "year": [2020, 2022, 2021, 2022, 2019, 2021],
    ...         "n_legs": [2, 2, 4, 4, 5, 100],
    ...         "animal": ["Flamingo", "Parrot", "Dog", "Horse", "Brittle stars", "Centipede"],
    ...     }
    ... )
    >>> import pyarrow.parquet as pq
    >>> pq.write_to_dataset(table, root_path="dataset_v2", partition_cols=["year"])

    create a ParquetDataset object from the dataset source:

    >>> dataset = pq.ParquetDataset("dataset_v2/")

    and read the data:

    >>> dataset.read().to_pandas()
       n_legs         animal  year
    0       5  Brittle stars  2019
    1       2       Flamingo  2020
    2       4            Dog  2021
    3     100      Centipede  2021
    4       2         Parrot  2022
    5       4          Horse  2022

    create a ParquetDataset object with filter:

    >>> dataset = pq.ParquetDataset("dataset_v2/", filters=[("n_legs", "=", 4)])
    >>> dataset.read().to_pandas()
       n_legs animal  year
    0       4    Dog  2021
    1       4  Horse  2022
    """
    def __init__(
        self,
        path_or_paths: SingleOrList[str]
        | SingleOrList[Path]
        | SingleOrList[NativeFile]
        | SingleOrList[IO],
        filesystem: SupportedFileSystem | None = None,
        schema: Schema | None = None,
        *,
        filters: Expression | FilterTuple | list[FilterTuple] | None = None,
        read_dictionary: list[str] | None = None,
        memory_map: bool = False,
        buffer_size: int = 0,
        partitioning: str | list[str] | Partitioning | None = "hive",
        ignore_prefixes: list[str] | None = None,
        pre_buffer: bool = True,
        coerce_int96_timestamp_unit: str | None = None,
        decryption_properties: FileDecryptionProperties | None = None,
        thrift_string_size_limit: int | None = None,
        thrift_container_size_limit: int | None = None,
        page_checksum_verification: bool = False,
    ): ...
    def equals(self, other: ParquetDataset) -> bool: ...
    @property
    def schema(self) -> Schema:
        """
        Schema of the Dataset.

        Examples
        --------
        Generate an example dataset:

        >>> import pyarrow as pa
        >>> table = pa.table(
        ...     {
        ...         "year": [2020, 2022, 2021, 2022, 2019, 2021],
        ...         "n_legs": [2, 2, 4, 4, 5, 100],
        ...         "animal": ["Flamingo", "Parrot", "Dog", "Horse", "Brittle stars", "Centipede"],
        ...     }
        ... )
        >>> import pyarrow.parquet as pq
        >>> pq.write_to_dataset(table, root_path="dataset_v2_schema", partition_cols=["year"])
        >>> dataset = pq.ParquetDataset("dataset_v2_schema/")

        Read the schema:

        >>> dataset.schema
        n_legs: int64
        animal: string
        year: dictionary<values=int32, indices=int32, ordered=0>
        """
    def read(
        self,
        columns: list[str] | None = None,
        use_threads: bool = True,
        use_pandas_metadata: bool = False,
    ) -> Table:
        """
        Read (multiple) Parquet files as a single pyarrow.Table.

        Parameters
        ----------
        columns : List[str]
            Names of columns to read from the dataset. The partition fields
            are not automatically included.
        use_threads : bool, default True
            Perform multi-threaded column reads.
        use_pandas_metadata : bool, default False
            If True and file has custom pandas schema metadata, ensure that
            index columns are also loaded.

        Returns
        -------
        pyarrow.Table
            Content of the file as a table (of columns).

        Examples
        --------
        Generate an example dataset:

        >>> import pyarrow as pa
        >>> table = pa.table(
        ...     {
        ...         "year": [2020, 2022, 2021, 2022, 2019, 2021],
        ...         "n_legs": [2, 2, 4, 4, 5, 100],
        ...         "animal": ["Flamingo", "Parrot", "Dog", "Horse", "Brittle stars", "Centipede"],
        ...     }
        ... )
        >>> import pyarrow.parquet as pq
        >>> pq.write_to_dataset(table, root_path="dataset_v2_read", partition_cols=["year"])
        >>> dataset = pq.ParquetDataset("dataset_v2_read/")

        Read the dataset:

        >>> dataset.read(columns=["n_legs"])
        pyarrow.Table
        n_legs: int64
        ----
        n_legs: [[5],[2],[4,100],[2,4]]
        """
    def read_pandas(self, **kwargs) -> Table:
        """
        Read dataset including pandas metadata, if any. Other arguments passed
        through to :func:`read`, see docstring for further details.

        Parameters
        ----------
        **kwargs : optional
            Additional options for :func:`read`

        Examples
        --------
        Generate an example parquet file:

        >>> import pyarrow as pa
        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...     {
        ...         "year": [2020, 2022, 2021, 2022, 2019, 2021],
        ...         "n_legs": [2, 2, 4, 4, 5, 100],
        ...         "animal": ["Flamingo", "Parrot", "Dog", "Horse", "Brittle stars", "Centipede"],
        ...     }
        ... )
        >>> table = pa.Table.from_pandas(df)
        >>> import pyarrow.parquet as pq
        >>> pq.write_table(table, "table_V2.parquet")
        >>> dataset = pq.ParquetDataset("table_V2.parquet")

        Read the dataset with pandas metadata:

        >>> dataset.read_pandas(columns=["n_legs"])
        pyarrow.Table
        n_legs: int64
        ----
        n_legs: [[2,2,4,4,5,100]]

        >>> dataset.read_pandas(columns=["n_legs"]).schema.pandas_metadata
        {'index_columns': [{'kind': 'range', 'name': None, 'start': 0, ...}
        """
    @property
    def fragments(self) -> list[ParquetFileFragment]:
        """
        A list of the Dataset source fragments or pieces with absolute
        file paths.

        Examples
        --------
        Generate an example dataset:

        >>> import pyarrow as pa
        >>> table = pa.table(
        ...     {
        ...         "year": [2020, 2022, 2021, 2022, 2019, 2021],
        ...         "n_legs": [2, 2, 4, 4, 5, 100],
        ...         "animal": ["Flamingo", "Parrot", "Dog", "Horse", "Brittle stars", "Centipede"],
        ...     }
        ... )
        >>> import pyarrow.parquet as pq
        >>> pq.write_to_dataset(table, root_path="dataset_v2_fragments", partition_cols=["year"])
        >>> dataset = pq.ParquetDataset("dataset_v2_fragments/")

        List the fragments:

        >>> dataset.fragments
        [<pyarrow.dataset.ParquetFileFragment path=dataset_v2_fragments/...
        """
    @property
    def files(self) -> list[str]:
        """
        A list of absolute Parquet file paths in the Dataset source.

        Examples
        --------
        Generate an example dataset:

        >>> import pyarrow as pa
        >>> table = pa.table(
        ...     {
        ...         "year": [2020, 2022, 2021, 2022, 2019, 2021],
        ...         "n_legs": [2, 2, 4, 4, 5, 100],
        ...         "animal": ["Flamingo", "Parrot", "Dog", "Horse", "Brittle stars", "Centipede"],
        ...     }
        ... )
        >>> import pyarrow.parquet as pq
        >>> pq.write_to_dataset(table, root_path="dataset_v2_files", partition_cols=["year"])
        >>> dataset = pq.ParquetDataset("dataset_v2_files/")

        List the files:

        >>> dataset.files
        ['dataset_v2_files/year=2019/...-0.parquet', ...
        """
    @property
    def filesystem(self) -> FileSystem:
        """
        The filesystem type of the Dataset source.
        """
    @property
    def partitioning(self) -> Partitioning:
        """
        The partitioning of the Dataset source, if discovered.
        """

def read_table(
    source: SingleOrList[str] | SingleOrList[Path] | SingleOrList[NativeFile] | SingleOrList[IO],
    *,
    columns: list | None = None,
    use_threads: bool = True,
    schema: Schema | None = None,
    use_pandas_metadata: bool = False,
    read_dictionary: list[str] | None = None,
    memory_map: bool = False,
    buffer_size: int = 0,
    partitioning: str | list[str] | Partitioning | None = "hive",
    filesystem: SupportedFileSystem | None = None,
    filters: Expression | FilterTuple | list[FilterTuple] | None = None,
    ignore_prefixes: list[str] | None = None,
    pre_buffer: bool = True,
    coerce_int96_timestamp_unit: str | None = None,
    decryption_properties: FileDecryptionProperties | None = None,
    thrift_string_size_limit: int | None = None,
    thrift_container_size_limit: int | None = None,
    page_checksum_verification: bool = False,
) -> Table:
    """
    Read a Table from Parquet format

    Parameters
    ----------
    source : str, pyarrow.NativeFile, or file-like object
        If a string passed, can be a single file name or directory name. For
        file-like objects, only read a single file. Use pyarrow.BufferReader to
        read a file contained in a bytes or buffer-like object.
    columns : list
        If not None, only these columns will be read from the file. A column
        name may be a prefix of a nested field, e.g. 'a' will select 'a.b',
        'a.c', and 'a.d.e'. If empty, no columns will be read. Note
        that the table will still have the correct num_rows set despite having
        no columns.
    use_threads : bool, default True
        Perform multi-threaded column reads.
    schema : Schema, optional
        Optionally provide the Schema for the parquet dataset, in which case it
        will not be inferred from the source.
    use_pandas_metadata : bool, default False
        If True and file has custom pandas schema metadata, ensure that
        index columns are also loaded.
    read_dictionary : list, default None
        List of names or column paths (for nested types) to read directly
        as DictionaryArray. Only supported for BYTE_ARRAY storage. To read
        a flat column as dictionary-encoded pass the column name. For
        nested types, you must pass the full column "path", which could be
        something like level1.level2.list.item. Refer to the Parquet
        file's schema to obtain the paths.
    memory_map : bool, default False
        If the source is a file path, use a memory map to read file, which can
        improve performance in some environments.
    buffer_size : int, default 0
        If positive, perform read buffering when deserializing individual
        column chunks. Otherwise IO calls are unbuffered.
    partitioning : pyarrow.dataset.Partitioning or str or list of str, default "hive"
        The partitioning scheme for a partitioned dataset. The default of "hive"
        assumes directory names with key=value pairs like "/year=2009/month=11".
        In addition, a scheme like "/2009/11" is also supported, in which case
        you need to specify the field names or a full schema. See the
        ``pyarrow.dataset.partitioning()`` function for more details.
    filesystem : FileSystem, default None
        If nothing passed, will be inferred based on path.
        Path will try to be found in the local on-disk filesystem otherwise
        it will be parsed as an URI to determine the filesystem.
    filters : pyarrow.compute.Expression or List[Tuple] or List[List[Tuple]], default None
        Rows which do not match the filter predicate will be removed from scanned
        data. Partition keys embedded in a nested directory structure will be
        exploited to avoid loading files at all if they contain no matching rows.
        Within-file level filtering and different partitioning schemes are supported.

        Predicates are expressed using an ``Expression`` or using
        the disjunctive normal form (DNF), like ``[[('x', '=', 0), ...], ...]``.
        DNF allows arbitrary boolean logical combinations of single column predicates.
        The innermost tuples each describe a single column predicate. The list of inner
        predicates is interpreted as a conjunction (AND), forming a more selective and
        multiple column predicate. Finally, the most outer list combines these filters
        as a disjunction (OR).

        Predicates may also be passed as List[Tuple]. This form is interpreted
        as a single conjunction. To express OR in predicates, one must
        use the (preferred) List[List[Tuple]] notation.

        Each tuple has format: (``key``, ``op``, ``value``) and compares the
        ``key`` with the ``value``.
        The supported ``op`` are:  ``=`` or ``==``, ``!=``, ``<``, ``>``, ``<=``,
        ``>=``, ``in`` and ``not in``. If the ``op`` is ``in`` or ``not in``, the
        ``value`` must be a collection such as a ``list``, a ``set`` or a
        ``tuple``.

        Examples:

        Using the ``Expression`` API:

        .. code-block:: python

            import pyarrow.compute as pc
            pc.field('x') = 0
            pc.field('y').isin(['a', 'b', 'c'])
            ~pc.field('y').isin({'a', 'b'})

        Using the DNF format:

        .. code-block:: python

            ("x", "=", 0)
            ("y", "in", ["a", "b", "c"])
            ("z", "not in", {"a", "b"})


    ignore_prefixes : list, optional
        Files matching any of these prefixes will be ignored by the
        discovery process.
        This is matched to the basename of a path.
        By default this is ['.', '_'].
        Note that discovery happens only if a directory is passed as source.
    pre_buffer : bool, default True
        Coalesce and issue file reads in parallel to improve performance on
        high-latency filesystems (e.g. S3). If True, Arrow will use a
        background I/O thread pool. If using a filesystem layer that itself
        performs readahead (e.g. fsspec's S3FS), disable readahead for best
        results.
    coerce_int96_timestamp_unit : str, default None
        Cast timestamps that are stored in INT96 format to a particular
        resolution (e.g. 'ms'). Setting to None is equivalent to 'ns'
        and therefore INT96 timestamps will be inferred as timestamps
        in nanoseconds.
    decryption_properties : FileDecryptionProperties or None
        File-level decryption properties.
        The decryption properties can be created using
        ``CryptoFactory.file_decryption_properties()``.
    thrift_string_size_limit : int, default None
        If not None, override the maximum total string size allocated
        when decoding Thrift structures. The default limit should be
        sufficient for most Parquet files.
    thrift_container_size_limit : int, default None
        If not None, override the maximum total size of containers allocated
        when decoding Thrift structures. The default limit should be
        sufficient for most Parquet files.
    page_checksum_verification : bool, default False
        If True, verify the checksum for each page read from the file.

    Returns
    -------
    pyarrow.Table
        Content of the file as a table (of columns)


    Examples
    --------

    Generate an example PyArrow Table and write it to a partitioned dataset:

    >>> import pyarrow as pa
    >>> table = pa.table(
    ...     {
    ...         "year": [2020, 2022, 2021, 2022, 2019, 2021],
    ...         "n_legs": [2, 2, 4, 4, 5, 100],
    ...         "animal": ["Flamingo", "Parrot", "Dog", "Horse", "Brittle stars", "Centipede"],
    ...     }
    ... )
    >>> import pyarrow.parquet as pq
    >>> pq.write_to_dataset(table, root_path="dataset_name_2", partition_cols=["year"])

    Read the data:

    >>> pq.read_table("dataset_name_2").to_pandas()
       n_legs         animal  year
    0       5  Brittle stars  2019
    1       2       Flamingo  2020
    2       4            Dog  2021
    3     100      Centipede  2021
    4       2         Parrot  2022
    5       4          Horse  2022


    Read only a subset of columns:

    >>> pq.read_table("dataset_name_2", columns=["n_legs", "animal"])
    pyarrow.Table
    n_legs: int64
    animal: string
    ----
    n_legs: [[5],[2],[4,100],[2,4]]
    animal: [["Brittle stars"],["Flamingo"],["Dog","Centipede"],["Parrot","Horse"]]

    Read a subset of columns and read one column as DictionaryArray:

    >>> pq.read_table("dataset_name_2", columns=["n_legs", "animal"], read_dictionary=["animal"])
    pyarrow.Table
    n_legs: int64
    animal: dictionary<values=string, indices=int32, ordered=0>
    ----
    n_legs: [[5],[2],[4,100],[2,4]]
    animal: [  -- dictionary:
    ["Brittle stars"]  -- indices:
    [0],  -- dictionary:
    ["Flamingo"]  -- indices:
    [0],  -- dictionary:
    ["Dog","Centipede"]  -- indices:
    [0,1],  -- dictionary:
    ["Parrot","Horse"]  -- indices:
    [0,1]]

    Read the table with filter:

    >>> pq.read_table(
    ...     "dataset_name_2", columns=["n_legs", "animal"], filters=[("n_legs", "<", 4)]
    ... ).to_pandas()
       n_legs    animal
    0       2  Flamingo
    1       2    Parrot

    Read data from a single Parquet file:

    >>> pq.write_table(table, "example.parquet")
    >>> pq.read_table("dataset_name_2").to_pandas()
       n_legs         animal  year
    0       5  Brittle stars  2019
    1       2       Flamingo  2020
    2       4            Dog  2021
    3     100      Centipede  2021
    4       2         Parrot  2022
    5       4          Horse  2022
    """

def read_pandas(
    source: str | Path | NativeFile | IO, columns: list | None = None, **kwargs
) -> Table:
    """

    Read a Table from Parquet format, also reading DataFrame
    index values if known in the file metadata

    Parameters
    ----------
    source : str, pyarrow.NativeFile, or file-like object
        If a string passed, can be a single file name or directory name. For
        file-like objects, only read a single file. Use pyarrow.BufferReader to
        read a file contained in a bytes or buffer-like object.
    columns : list
        If not None, only these columns will be read from the file. A column
        name may be a prefix of a nested field, e.g. 'a' will select 'a.b',
        'a.c', and 'a.d.e'. If empty, no columns will be read. Note
        that the table will still have the correct num_rows set despite having
        no columns.
    use_threads : bool, default True
        Perform multi-threaded column reads.
    schema : Schema, optional
        Optionally provide the Schema for the parquet dataset, in which case it
        will not be inferred from the source.
    read_dictionary : list, default None
        List of names or column paths (for nested types) to read directly
        as DictionaryArray. Only supported for BYTE_ARRAY storage. To read
        a flat column as dictionary-encoded pass the column name. For
        nested types, you must pass the full column "path", which could be
        something like level1.level2.list.item. Refer to the Parquet
        file's schema to obtain the paths.
    memory_map : bool, default False
        If the source is a file path, use a memory map to read file, which can
        improve performance in some environments.
    buffer_size : int, default 0
        If positive, perform read buffering when deserializing individual
        column chunks. Otherwise IO calls are unbuffered.
    partitioning : pyarrow.dataset.Partitioning or str or list of str, default "hive"
        The partitioning scheme for a partitioned dataset. The default of "hive"
        assumes directory names with key=value pairs like "/year=2009/month=11".
        In addition, a scheme like "/2009/11" is also supported, in which case
        you need to specify the field names or a full schema. See the
        ``pyarrow.dataset.partitioning()`` function for more details.
    **kwargs
        additional options for :func:`read_table`
    filesystem : FileSystem, default None
        If nothing passed, will be inferred based on path.
        Path will try to be found in the local on-disk filesystem otherwise
        it will be parsed as an URI to determine the filesystem.
    filters : pyarrow.compute.Expression or List[Tuple] or List[List[Tuple]], default None
        Rows which do not match the filter predicate will be removed from scanned
        data. Partition keys embedded in a nested directory structure will be
        exploited to avoid loading files at all if they contain no matching rows.
        Within-file level filtering and different partitioning schemes are supported.

        Predicates are expressed using an ``Expression`` or using
        the disjunctive normal form (DNF), like ``[[('x', '=', 0), ...], ...]``.
        DNF allows arbitrary boolean logical combinations of single column predicates.
        The innermost tuples each describe a single column predicate. The list of inner
        predicates is interpreted as a conjunction (AND), forming a more selective and
        multiple column predicate. Finally, the most outer list combines these filters
        as a disjunction (OR).

        Predicates may also be passed as List[Tuple]. This form is interpreted
        as a single conjunction. To express OR in predicates, one must
        use the (preferred) List[List[Tuple]] notation.

        Each tuple has format: (``key``, ``op``, ``value``) and compares the
        ``key`` with the ``value``.
        The supported ``op`` are:  ``=`` or ``==``, ``!=``, ``<``, ``>``, ``<=``,
        ``>=``, ``in`` and ``not in``. If the ``op`` is ``in`` or ``not in``, the
        ``value`` must be a collection such as a ``list``, a ``set`` or a
        ``tuple``.

        Examples:

        Using the ``Expression`` API:

        .. code-block:: python

            import pyarrow.compute as pc
            pc.field('x') = 0
            pc.field('y').isin(['a', 'b', 'c'])
            ~pc.field('y').isin({'a', 'b'})

        Using the DNF format:

        .. code-block:: python

            ("x", "=", 0)
            ("y", "in", ["a", "b", "c"])
            ("z", "not in", {"a", "b"})


    ignore_prefixes : list, optional
        Files matching any of these prefixes will be ignored by the
        discovery process.
        This is matched to the basename of a path.
        By default this is ['.', '_'].
        Note that discovery happens only if a directory is passed as source.
    pre_buffer : bool, default True
        Coalesce and issue file reads in parallel to improve performance on
        high-latency filesystems (e.g. S3). If True, Arrow will use a
        background I/O thread pool. If using a filesystem layer that itself
        performs readahead (e.g. fsspec's S3FS), disable readahead for best
        results.
    coerce_int96_timestamp_unit : str, default None
        Cast timestamps that are stored in INT96 format to a particular
        resolution (e.g. 'ms'). Setting to None is equivalent to 'ns'
        and therefore INT96 timestamps will be inferred as timestamps
        in nanoseconds.
    decryption_properties : FileDecryptionProperties or None
        File-level decryption properties.
        The decryption properties can be created using
        ``CryptoFactory.file_decryption_properties()``.
    thrift_string_size_limit : int, default None
        If not None, override the maximum total string size allocated
        when decoding Thrift structures. The default limit should be
        sufficient for most Parquet files.
    thrift_container_size_limit : int, default None
        If not None, override the maximum total size of containers allocated
        when decoding Thrift structures. The default limit should be
        sufficient for most Parquet files.
    page_checksum_verification : bool, default False
        If True, verify the checksum for each page read from the file.

    Returns
    -------
    pyarrow.Table
        Content of the file as a Table of Columns, including DataFrame
        indexes as columns
    """

def write_table(
    table: Table,
    where: str | Path | NativeFile | IO,
    row_group_size: int | None = None,
    version: Literal["1.0", "2.4", "2.6"] = "2.6",
    use_dictionary: bool = True,
    compression: _Compression | dict[str, _Compression] = "snappy",
    write_statistics: bool | list = True,
    use_deprecated_int96_timestamps: bool | None = None,
    coerce_timestamps: str | None = None,
    allow_truncated_timestamps: bool = False,
    data_page_size: int | None = None,
    flavor: str | None = None,
    filesystem: SupportedFileSystem | None = None,
    compression_level: int | dict | None = None,
    use_byte_stream_split: bool = False,
    column_encoding: str | dict | None = None,
    data_page_version: Literal["1.0", "2.0"] = ...,
    use_compliant_nested_type: bool = True,
    encryption_properties: FileEncryptionProperties | None = None,
    write_batch_size: int | None = None,
    dictionary_pagesize_limit: int | None = None,
    store_schema: bool = True,
    write_page_index: bool = False,
    write_page_checksum: bool = False,
    sorting_columns: Sequence[SortingColumn] | None = None,
    store_decimal_as_integer: bool = False,
    **kwargs,
) -> None:
    """

    Write a Table to Parquet format.

    Parameters
    ----------
    table : pyarrow.Table
    where : string or pyarrow.NativeFile
    row_group_size : int
        Maximum number of rows in each written row group. If None, the
        row group size will be the minimum of the Table size and
        1024 * 1024.
    version : {"1.0", "2.4", "2.6"}, default "2.6"
        Determine which Parquet logical types are available for use, whether the
        reduced set from the Parquet 1.x.x format or the expanded logical types
        added in later format versions.
        Files written with version='2.4' or '2.6' may not be readable in all
        Parquet implementations, so version='1.0' is likely the choice that
        maximizes file compatibility.
        UINT32 and some logical types are only available with version '2.4'.
        Nanosecond timestamps are only available with version '2.6'.
        Other features such as compression algorithms or the new serialized
        data page format must be enabled separately (see 'compression' and
        'data_page_version').
    use_dictionary : bool or list, default True
        Specify if we should use dictionary encoding in general or only for
        some columns.
        When encoding the column, if the dictionary size is too large, the
        column will fallback to ``PLAIN`` encoding. Specially, ``BOOLEAN`` type
        doesn't support dictionary encoding.
    compression : str or dict, default 'snappy'
        Specify the compression codec, either on a general basis or per-column.
        Valid values: {'NONE', 'SNAPPY', 'GZIP', 'BROTLI', 'LZ4', 'ZSTD'}.
    write_statistics : bool or list, default True
        Specify if we should write statistics in general (default is True) or only
        for some columns.
    use_deprecated_int96_timestamps : bool, default None
        Write timestamps to INT96 Parquet format. Defaults to False unless enabled
        by flavor argument. This take priority over the coerce_timestamps option.
    coerce_timestamps : str, default None
        Cast timestamps to a particular resolution. If omitted, defaults are chosen
        depending on `version`. For ``version='1.0'`` and ``version='2.4'``,
        nanoseconds are cast to microseconds ('us'), while for
        ``version='2.6'`` (the default), they are written natively without loss
        of resolution.  Seconds are always cast to milliseconds ('ms') by default,
        as Parquet does not have any temporal type with seconds resolution.
        If the casting results in loss of data, it will raise an exception
        unless ``allow_truncated_timestamps=True`` is given.
        Valid values: {None, 'ms', 'us'}
    allow_truncated_timestamps : bool, default False
        Allow loss of data when coercing timestamps to a particular
        resolution. E.g. if microsecond or nanosecond data is lost when coercing to
        'ms', do not raise an exception. Passing ``allow_truncated_timestamp=True``
        will NOT result in the truncation exception being ignored unless
        ``coerce_timestamps`` is not None.
    data_page_size : int, default None
        Set a target threshold for the approximate encoded size of data
        pages within a column chunk (in bytes). If None, use the default data page
        size of 1MByte.
    flavor : {'spark'}, default None
        Sanitize schema or set other compatibility options to work with
        various target systems.
    filesystem : FileSystem, default None
        If nothing passed, will be inferred from `where` if path-like, else
        `where` is already a file-like object so no filesystem is needed.
    compression_level : int or dict, default None
        Specify the compression level for a codec, either on a general basis or
        per-column. If None is passed, arrow selects the compression level for
        the compression codec in use. The compression level has a different
        meaning for each codec, so you have to read the documentation of the
        codec you are using.
        An exception is thrown if the compression codec does not allow specifying
        a compression level.
    use_byte_stream_split : bool or list, default False
        Specify if the byte_stream_split encoding should be used in general or
        only for some columns. If both dictionary and byte_stream_stream are
        enabled, then dictionary is preferred.
        The byte_stream_split encoding is valid for integer, floating-point
        and fixed-size binary data types (including decimals); it should be
        combined with a compression codec so as to achieve size reduction.
    column_encoding : string or dict, default None
        Specify the encoding scheme on a per column basis.
        Can only be used when ``use_dictionary`` is set to False, and
        cannot be used in combination with ``use_byte_stream_split``.
        Currently supported values: {'PLAIN', 'BYTE_STREAM_SPLIT',
        'DELTA_BINARY_PACKED', 'DELTA_LENGTH_BYTE_ARRAY', 'DELTA_BYTE_ARRAY'}.
        Certain encodings are only compatible with certain data types.
        Please refer to the encodings section of `Reading and writing Parquet
        files <https://arrow.apache.org/docs/cpp/parquet.html#encodings>`_.
    data_page_version : {"1.0", "2.0"}, default "1.0"
        The serialized Parquet data page format version to write, defaults to
        1.0. This does not impact the file schema logical types and Arrow to
        Parquet type casting behavior; for that use the "version" option.
    use_compliant_nested_type : bool, default True
        Whether to write compliant Parquet nested type (lists) as defined
        `here <https://github.com/apache/parquet-format/blob/master/
        LogicalTypes.md#nested-types>`_, defaults to ``True``.
        For ``use_compliant_nested_type=True``, this will write into a list
        with 3-level structure where the middle level, named ``list``,
        is a repeated group with a single field named ``element``::

            <list-repetition> group <name> (LIST) {
                repeated group list {
                      <element-repetition> <element-type> element;
                }
            }

        For ``use_compliant_nested_type=False``, this will also write into a list
        with 3-level structure, where the name of the single field of the middle
        level ``list`` is taken from the element name for nested columns in Arrow,
        which defaults to ``item``::

            <list-repetition> group <name> (LIST) {
                repeated group list {
                    <element-repetition> <element-type> item;
                }
            }
    encryption_properties : FileEncryptionProperties, default None
        File encryption properties for Parquet Modular Encryption.
        If None, no encryption will be done.
        The encryption properties can be created using:
        ``CryptoFactory.file_encryption_properties()``.
    write_batch_size : int, default None
        Number of values to write to a page at a time. If None, use the default of
        1024. ``write_batch_size`` is complementary to ``data_page_size``. If pages
        are exceeding the ``data_page_size`` due to large column values, lowering
        the batch size can help keep page sizes closer to the intended size.
    dictionary_pagesize_limit : int, default None
        Specify the dictionary page size limit per row group. If None, use the
        default 1MB.
    store_schema : bool, default True
        By default, the Arrow schema is serialized and stored in the Parquet
        file metadata (in the "ARROW:schema" key). When reading the file,
        if this key is available, it will be used to more faithfully recreate
        the original Arrow data. For example, for tz-aware timestamp columns
        it will restore the timezone (Parquet only stores the UTC values without
        timezone), or columns with duration type will be restored from the int64
        Parquet column.
    write_page_index : bool, default False
        Whether to write a page index in general for all columns.
        Writing statistics to the page index disables the old method of writing
        statistics to each data page header. The page index makes statistics-based
        filtering more efficient than the page header, as it gathers all the
        statistics for a Parquet file in a single place, avoiding scattered I/O.
        Note that the page index is not yet used on the read size by PyArrow.
    write_page_checksum : bool, default False
        Whether to write page checksums in general for all columns.
        Page checksums enable detection of data corruption, which might occur during
        transmission or in the storage.
    sorting_columns : Sequence of SortingColumn, default None
        Specify the sort order of the data being written. The writer does not sort
        the data nor does it verify that the data is sorted. The sort order is
        written to the row group metadata, which can then be used by readers.
    store_decimal_as_integer : bool, default False
        Allow decimals with 1 <= precision <= 18 to be stored as integers.
        In Parquet, DECIMAL can be stored in any of the following physical types:
        - int32: for 1 <= precision <= 9.
        - int64: for 10 <= precision <= 18.
        - fixed_len_byte_array: precision is limited by the array size.
          Length n can store <= floor(log_10(2^(8*n - 1) - 1)) base-10 digits.
        - binary: precision is unlimited. The minimum number of bytes to store the
          unscaled value is used.

        By default, this is DISABLED and all decimal types annotate fixed_len_byte_array.
        When enabled, the writer will use the following physical types to store decimals:
        - int32: for 1 <= precision <= 9.
        - int64: for 10 <= precision <= 18.
        - fixed_len_byte_array: for precision > 18.

        As a consequence, decimal columns stored in integer types are more compact.

    **kwargs : optional
        Additional options for ParquetWriter

    Examples
    --------
    Generate an example PyArrow Table:

    >>> import pyarrow as pa
    >>> table = pa.table(
    ...     {
    ...         "n_legs": [2, 2, 4, 4, 5, 100],
    ...         "animal": ["Flamingo", "Parrot", "Dog", "Horse", "Brittle stars", "Centipede"],
    ...     }
    ... )

    and write the Table into Parquet file:

    >>> import pyarrow.parquet as pq
    >>> pq.write_table(table, "example.parquet")

    Defining row group size for the Parquet file:

    >>> pq.write_table(table, "example.parquet", row_group_size=3)

    Defining row group compression (default is Snappy):

    >>> pq.write_table(table, "example.parquet", compression="none")

    Defining row group compression and encoding per-column:

    >>> pq.write_table(
    ...     table,
    ...     "example.parquet",
    ...     compression={"n_legs": "snappy", "animal": "gzip"},
    ...     use_dictionary=["n_legs", "animal"],
    ... )

    Defining column encoding per-column:

    >>> pq.write_table(
    ...     table, "example.parquet", column_encoding={"animal": "PLAIN"}, use_dictionary=False
    ... )
    """

def write_to_dataset(
    table: Table,
    root_path: str | Path,
    partition_cols: list[str] | None = None,
    filesystem: SupportedFileSystem | None = None,
    schema: Schema | None = None,
    partitioning: Partitioning | list[str] | None = None,
    basename_template: str | None = None,
    use_threads: bool | None = None,
    file_visitor: Callable[[str], None] | None = None,
    existing_data_behavior: Literal["overwrite_or_ignore", "error", "delete_matching"]
    | None = None,
    **kwargs,
) -> None:
    """
    Wrapper around dataset.write_dataset for writing a Table to
    Parquet format by partitions.
    For each combination of partition columns and values,
    a subdirectories are created in the following
        manner:

        root_dir/
          group1=value1
            group2=value1
              <uuid>.parquet
            group2=value2
              <uuid>.parquet
          group1=valueN
            group2=value1
              <uuid>.parquet
            group2=valueN
              <uuid>.parquet

    Parameters
    ----------
    table : pyarrow.Table
    root_path : str, pathlib.Path
        The root directory of the dataset.
    partition_cols : list,
        Column names by which to partition the dataset.
        Columns are partitioned in the order they are given.
    filesystem : FileSystem, default None
        If nothing passed, will be inferred based on path.
        Path will try to be found in the local on-disk filesystem otherwise
        it will be parsed as an URI to determine the filesystem.
    schema : Schema, optional
        This Schema of the dataset.
    partitioning : Partitioning or list[str], optional
        The partitioning scheme specified with the
        ``pyarrow.dataset.partitioning()`` function or a list of field names.
        When providing a list of field names, you can use
        ``partitioning_flavor`` to drive which partitioning type should be
        used.
    basename_template : str, optional
        A template string used to generate basenames of written data files.
        The token '{i}' will be replaced with an automatically incremented
        integer. If not specified, it defaults to "guid-{i}.parquet".
    use_threads : bool, default True
        Write files in parallel. If enabled, then maximum parallelism will be
        used determined by the number of available CPU cores.
    file_visitor : function
        If set, this function will be called with a WrittenFile instance
        for each file created during the call.  This object will have both
        a path attribute and a metadata attribute.

        The path attribute will be a string containing the path to
        the created file.

        The metadata attribute will be the parquet metadata of the file.
        This metadata will have the file path attribute set and can be used
        to build a _metadata file.  The metadata attribute will be None if
        the format is not parquet.

        Example visitor which simple collects the filenames created::

            visited_paths = []

            def file_visitor(written_file):
                visited_paths.append(written_file.path)

    existing_data_behavior : 'overwrite_or_ignore' | 'error' | 'delete_matching'
        Controls how the dataset will handle data that already exists in
        the destination. The default behaviour is 'overwrite_or_ignore'.

        'overwrite_or_ignore' will ignore any existing data and will
        overwrite files with the same name as an output file.  Other
        existing files will be ignored.  This behavior, in combination
        with a unique basename_template for each write, will allow for
        an append workflow.

        'error' will raise an error if any data exists in the destination.

        'delete_matching' is useful when you are writing a partitioned
        dataset.  The first time each partition directory is encountered
        the entire directory will be deleted.  This allows you to overwrite
        old partitions completely.
    **kwargs : dict,
        Used as additional kwargs for :func:`pyarrow.dataset.write_dataset`
        function for matching kwargs, and remainder to
        :func:`pyarrow.dataset.ParquetFileFormat.make_write_options`.
        See the docstring of :func:`write_table` and
        :func:`pyarrow.dataset.write_dataset` for the available options.
        Using `metadata_collector` in kwargs allows one to collect the
        file metadata instances of dataset pieces. The file paths in the
        ColumnChunkMetaData will be set relative to `root_path`.

    Examples
    --------
    Generate an example PyArrow Table:

    >>> import pyarrow as pa
    >>> table = pa.table(
    ...     {
    ...         "year": [2020, 2022, 2021, 2022, 2019, 2021],
    ...         "n_legs": [2, 2, 4, 4, 5, 100],
    ...         "animal": ["Flamingo", "Parrot", "Dog", "Horse", "Brittle stars", "Centipede"],
    ...     }
    ... )

    and write it to a partitioned dataset:

    >>> import pyarrow.parquet as pq
    >>> pq.write_to_dataset(table, root_path="dataset_name_3", partition_cols=["year"])
    >>> pq.ParquetDataset("dataset_name_3").files
    ['dataset_name_3/year=2019/...-0.parquet', ...

    Write a single Parquet file into the root folder:

    >>> pq.write_to_dataset(table, root_path="dataset_name_4")
    >>> pq.ParquetDataset("dataset_name_4/").files
    ['dataset_name_4/...-0.parquet']
    """

def write_metadata(
    schema: Schema,
    where: str | NativeFile,
    metadata_collector: list[FileMetaData] | None = None,
    filesystem: SupportedFileSystem | None = None,
    **kwargs,
) -> None:
    """
    Write metadata-only Parquet file from schema. This can be used with
    `write_to_dataset` to generate `_common_metadata` and `_metadata` sidecar
    files.

    Parameters
    ----------
    schema : pyarrow.Schema
    where : string or pyarrow.NativeFile
    metadata_collector : list
        where to collect metadata information.
    filesystem : FileSystem, default None
        If nothing passed, will be inferred from `where` if path-like, else
        `where` is already a file-like object so no filesystem is needed.
    **kwargs : dict,
        Additional kwargs for ParquetWriter class. See docstring for
        `ParquetWriter` for more information.

    Examples
    --------
    Generate example data:

    >>> import pyarrow as pa
    >>> table = pa.table(
    ...     {
    ...         "n_legs": [2, 2, 4, 4, 5, 100],
    ...         "animal": ["Flamingo", "Parrot", "Dog", "Horse", "Brittle stars", "Centipede"],
    ...     }
    ... )

    Write a dataset and collect metadata information.

    >>> metadata_collector = []
    >>> import pyarrow.parquet as pq
    >>> pq.write_to_dataset(table, "dataset_metadata", metadata_collector=metadata_collector)

    Write the `_common_metadata` parquet file without row groups statistics.

    >>> pq.write_metadata(table.schema, "dataset_metadata/_common_metadata")

    Write the `_metadata` parquet file with row groups statistics.

    >>> pq.write_metadata(
    ...     table.schema, "dataset_metadata/_metadata", metadata_collector=metadata_collector
    ... )
    """

def read_metadata(
    where: str | Path | IO | NativeFile,
    memory_map: bool = False,
    decryption_properties: FileDecryptionProperties | None = None,
    filesystem: SupportedFileSystem | None = None,
) -> FileMetaData:
    """
    Read FileMetaData from footer of a single Parquet file.

    Parameters
    ----------
    where : str (file path) or file-like object
    memory_map : bool, default False
        Create memory map when the source is a file path.
    decryption_properties : FileDecryptionProperties, default None
        Decryption properties for reading encrypted Parquet files.
    filesystem : FileSystem, default None
        If nothing passed, will be inferred based on path.
        Path will try to be found in the local on-disk filesystem otherwise
        it will be parsed as an URI to determine the filesystem.

    Returns
    -------
    metadata : FileMetaData
        The metadata of the Parquet file

    Examples
    --------
    >>> import pyarrow as pa
    >>> import pyarrow.parquet as pq
    >>> table = pa.table({"n_legs": [4, 5, 100], "animal": ["Dog", "Brittle stars", "Centipede"]})
    >>> pq.write_table(table, "example.parquet")

    >>> pq.read_metadata("example.parquet")
    <pyarrow._parquet.FileMetaData object at ...>
      created_by: parquet-cpp-arrow version ...
      num_columns: 2
      num_rows: 3
      num_row_groups: 1
      format_version: 2.6
      serialized_size: ...
    """

def read_schema(
    where: str | Path | IO | NativeFile,
    memory_map: bool = False,
    decryption_properties: FileDecryptionProperties | None = None,
    filesystem: SupportedFileSystem | None = None,
) -> Schema:
    """
    Read effective Arrow schema from Parquet file metadata.

    Parameters
    ----------
    where : str (file path) or file-like object
    memory_map : bool, default False
        Create memory map when the source is a file path.
    decryption_properties : FileDecryptionProperties, default None
        Decryption properties for reading encrypted Parquet files.
    filesystem : FileSystem, default None
        If nothing passed, will be inferred based on path.
        Path will try to be found in the local on-disk filesystem otherwise
        it will be parsed as an URI to determine the filesystem.

    Returns
    -------
    schema : pyarrow.Schema
        The schema of the Parquet file

    Examples
    --------
    >>> import pyarrow as pa
    >>> import pyarrow.parquet as pq
    >>> table = pa.table({"n_legs": [4, 5, 100], "animal": ["Dog", "Brittle stars", "Centipede"]})
    >>> pq.write_table(table, "example.parquet")

    >>> pq.read_schema("example.parquet")
    n_legs: int64
    animal: string
    """
