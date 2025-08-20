import sys

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self
from typing import (
    IO,
    Any,
    Callable,
    Generic,
    Iterator,
    Literal,
    NamedTuple,
    TypeVar,
    overload,
)

from _typeshed import StrPath

from . import _csv, _json, _parquet, lib
from ._fs import FileSelector, FileSystem, SupportedFileSystem
from ._stubs_typing import Indices, JoinType, Order
from .acero import ExecNodeOptions
from .compute import Expression
from .ipc import IpcWriteOptions, RecordBatchReader

class Dataset(lib._Weakrefable):
    """
    Collection of data fragments and potentially child datasets.

    Arrow Datasets allow you to query against data that has been split across
    multiple files. This sharding of data may indicate partitioning, which
    can accelerate queries that only touch some partitions (files).
    """

    @property
    def partition_expression(self) -> Expression:
        """
        An Expression which evaluates to true for all data viewed by this
        Dataset.
        """
    def replace_schema(self, schema: lib.Schema) -> None:
        """
        Return a copy of this Dataset with a different schema.

        The copy will view the same Fragments. If the new schema is not
        compatible with the original dataset's schema then an error will
        be raised.

        Parameters
        ----------
        schema : Schema
            The new dataset schema.
        """
    def get_fragments(self, filter: Expression | None = None):
        """Returns an iterator over the fragments in this dataset.

        Parameters
        ----------
        filter : Expression, default None
            Return fragments matching the optional filter, either using the
            partition_expression or internal information like Parquet's
            statistics.

        Returns
        -------
        fragments : iterator of Fragment
        """
    def scanner(
        self,
        columns: list[str] | None = None,
        filter: Expression | None = None,
        batch_size: int = ...,
        batch_readahead: int = 16,
        fragment_readahead: int = 4,
        fragment_scan_options: FragmentScanOptions | None = None,
        use_threads: bool = True,
        cache_metadata: bool = True,
        memory_pool: lib.MemoryPool | None = None,
    ) -> Scanner:
        """
        Build a scan operation against the dataset.

        Data is not loaded immediately. Instead, this produces a Scanner,
        which exposes further operations (e.g. loading all data as a
        table, counting rows).

        See the :meth:`Scanner.from_dataset` method for further information.

        Parameters
        ----------
        columns : list of str, default None
            The columns to project. This can be a list of column names to
            include (order and duplicates will be preserved), or a dictionary
            with {new_column_name: expression} values for more advanced
            projections.

            The list of columns or expressions may use the special fields
            `__batch_index` (the index of the batch within the fragment),
            `__fragment_index` (the index of the fragment within the dataset),
            `__last_in_fragment` (whether the batch is last in fragment), and
            `__filename` (the name of the source file or a description of the
            source fragment).

            The columns will be passed down to Datasets and corresponding data
            fragments to avoid loading, copying, and deserializing columns
            that will not be required further down the compute chain.
            By default all of the available columns are projected. Raises
            an exception if any of the referenced column names does not exist
            in the dataset's Schema.
        filter : Expression, default None
            Scan will return only the rows matching the filter.
            If possible the predicate will be pushed down to exploit the
            partition information or internal metadata found in the data
            source, e.g. Parquet statistics. Otherwise filters the loaded
            RecordBatches before yielding them.
        batch_size : int, default 131_072
            The maximum row count for scanned record batches. If scanned
            record batches are overflowing memory then this method can be
            called to reduce their size.
        batch_readahead : int, default 16
            The number of batches to read ahead in a file. This might not work
            for all file formats. Increasing this number will increase
            RAM usage but could also improve IO utilization.
        fragment_readahead : int, default 4
            The number of files to read ahead. Increasing this number will increase
            RAM usage but could also improve IO utilization.
        fragment_scan_options : FragmentScanOptions, default None
            Options specific to a particular scan and fragment type, which
            can change between different scans of the same dataset.
        use_threads : bool, default True
            If enabled, then maximum parallelism will be used determined by
            the number of available CPU cores.
        cache_metadata : bool, default True
            If enabled, metadata may be cached when scanning to speed up
            repeated scans.
        memory_pool : MemoryPool, default None
            For memory allocations, if required. If not specified, uses the
            default pool.

        Returns
        -------
        scanner : Scanner

        Examples
        --------
        >>> import pyarrow as pa
        >>> table = pa.table(
        ...     {
        ...         "year": [2020, 2022, 2021, 2022, 2019, 2021],
        ...         "n_legs": [2, 2, 4, 4, 5, 100],
        ...         "animal": ["Flamingo", "Parrot", "Dog", "Horse", "Brittle stars", "Centipede"],
        ...     }
        ... )
        >>>
        >>> import pyarrow.parquet as pq
        >>> pq.write_table(table, "dataset_scanner.parquet")

        >>> import pyarrow.dataset as ds
        >>> dataset = ds.dataset("dataset_scanner.parquet")

        Selecting a subset of the columns:

        >>> dataset.scanner(columns=["year", "n_legs"]).to_table()
        pyarrow.Table
        year: int64
        n_legs: int64
        ----
        year: [[2020,2022,2021,2022,2019,2021]]
        n_legs: [[2,2,4,4,5,100]]

        Projecting selected columns using an expression:

        >>> dataset.scanner(
        ...     columns={
        ...         "n_legs_uint": ds.field("n_legs").cast("uint8"),
        ...     }
        ... ).to_table()
        pyarrow.Table
        n_legs_uint: uint8
        ----
        n_legs_uint: [[2,2,4,4,5,100]]

        Filtering rows while scanning:

        >>> dataset.scanner(filter=ds.field("year") > 2020).to_table()
        pyarrow.Table
        year: int64
        n_legs: int64
        animal: string
        ----
        year: [[2022,2021,2022,2021]]
        n_legs: [[2,4,4,100]]
        animal: [["Parrot","Dog","Horse","Centipede"]]
        """
    def to_batches(
        self,
        columns: list[str] | None = None,
        filter: Expression | None = None,
        batch_size: int = ...,
        batch_readahead: int = 16,
        fragment_readahead: int = 4,
        fragment_scan_options: FragmentScanOptions | None = None,
        use_threads: bool = True,
        cache_metadata: bool = True,
        memory_pool: lib.MemoryPool | None = None,
    ) -> Iterator[lib.RecordBatch]:
        """
        Read the dataset as materialized record batches.

        Parameters
        ----------
        columns : list of str, default None
            The columns to project. This can be a list of column names to
            include (order and duplicates will be preserved), or a dictionary
            with {new_column_name: expression} values for more advanced
            projections.

            The list of columns or expressions may use the special fields
            `__batch_index` (the index of the batch within the fragment),
            `__fragment_index` (the index of the fragment within the dataset),
            `__last_in_fragment` (whether the batch is last in fragment), and
            `__filename` (the name of the source file or a description of the
            source fragment).

            The columns will be passed down to Datasets and corresponding data
            fragments to avoid loading, copying, and deserializing columns
            that will not be required further down the compute chain.
            By default all of the available columns are projected. Raises
            an exception if any of the referenced column names does not exist
            in the dataset's Schema.
        filter : Expression, default None
            Scan will return only the rows matching the filter.
            If possible the predicate will be pushed down to exploit the
            partition information or internal metadata found in the data
            source, e.g. Parquet statistics. Otherwise filters the loaded
            RecordBatches before yielding them.
        batch_size : int, default 131_072
            The maximum row count for scanned record batches. If scanned
            record batches are overflowing memory then this method can be
            called to reduce their size.
        batch_readahead : int, default 16
            The number of batches to read ahead in a file. This might not work
            for all file formats. Increasing this number will increase
            RAM usage but could also improve IO utilization.
        fragment_readahead : int, default 4
            The number of files to read ahead. Increasing this number will increase
            RAM usage but could also improve IO utilization.
        fragment_scan_options : FragmentScanOptions, default None
            Options specific to a particular scan and fragment type, which
            can change between different scans of the same dataset.
        use_threads : bool, default True
            If enabled, then maximum parallelism will be used determined by
            the number of available CPU cores.
        cache_metadata : bool, default True
            If enabled, metadata may be cached when scanning to speed up
            repeated scans.
        memory_pool : MemoryPool, default None
            For memory allocations, if required. If not specified, uses the
            default pool.

        Returns
        -------
        record_batches : iterator of RecordBatch
        """
    def to_table(
        self,
        columns: list[str] | dict[str, Expression] | None = None,
        filter: Expression | None = None,
        batch_size: int = ...,
        batch_readahead: int = 16,
        fragment_readahead: int = 4,
        fragment_scan_options: FragmentScanOptions | None = None,
        use_threads: bool = True,
        cache_metadata: bool = True,
        memory_pool: lib.MemoryPool | None = None,
    ) -> lib.Table:
        """
        Read the dataset to an Arrow table.

        Note that this method reads all the selected data from the dataset
        into memory.

        Parameters
        ----------
        columns : list of str, default None
            The columns to project. This can be a list of column names to
            include (order and duplicates will be preserved), or a dictionary
            with {new_column_name: expression} values for more advanced
            projections.

            The list of columns or expressions may use the special fields
            `__batch_index` (the index of the batch within the fragment),
            `__fragment_index` (the index of the fragment within the dataset),
            `__last_in_fragment` (whether the batch is last in fragment), and
            `__filename` (the name of the source file or a description of the
            source fragment).

            The columns will be passed down to Datasets and corresponding data
            fragments to avoid loading, copying, and deserializing columns
            that will not be required further down the compute chain.
            By default all of the available columns are projected. Raises
            an exception if any of the referenced column names does not exist
            in the dataset's Schema.
        filter : Expression, default None
            Scan will return only the rows matching the filter.
            If possible the predicate will be pushed down to exploit the
            partition information or internal metadata found in the data
            source, e.g. Parquet statistics. Otherwise filters the loaded
            RecordBatches before yielding them.
        batch_size : int, default 131_072
            The maximum row count for scanned record batches. If scanned
            record batches are overflowing memory then this method can be
            called to reduce their size.
        batch_readahead : int, default 16
            The number of batches to read ahead in a file. This might not work
            for all file formats. Increasing this number will increase
            RAM usage but could also improve IO utilization.
        fragment_readahead : int, default 4
            The number of files to read ahead. Increasing this number will increase
            RAM usage but could also improve IO utilization.
        fragment_scan_options : FragmentScanOptions, default None
            Options specific to a particular scan and fragment type, which
            can change between different scans of the same dataset.
        use_threads : bool, default True
            If enabled, then maximum parallelism will be used determined by
            the number of available CPU cores.
        cache_metadata : bool, default True
            If enabled, metadata may be cached when scanning to speed up
            repeated scans.
        memory_pool : MemoryPool, default None
            For memory allocations, if required. If not specified, uses the
            default pool.

        Returns
        -------
        table : Table
        """
    def take(
        self,
        indices: Indices,
        columns: list[str] | None = None,
        filter: Expression | None = None,
        batch_size: int = ...,
        batch_readahead: int = 16,
        fragment_readahead: int = 4,
        fragment_scan_options: FragmentScanOptions | None = None,
        use_threads: bool = True,
        cache_metadata: bool = True,
        memory_pool: lib.MemoryPool | None = None,
    ) -> lib.Table:
        """
        Select rows of data by index.

        Parameters
        ----------
        indices : Array or array-like
            indices of rows to select in the dataset.
        columns : list of str, default None
            The columns to project. This can be a list of column names to
            include (order and duplicates will be preserved), or a dictionary
            with {new_column_name: expression} values for more advanced
            projections.

            The list of columns or expressions may use the special fields
            `__batch_index` (the index of the batch within the fragment),
            `__fragment_index` (the index of the fragment within the dataset),
            `__last_in_fragment` (whether the batch is last in fragment), and
            `__filename` (the name of the source file or a description of the
            source fragment).

            The columns will be passed down to Datasets and corresponding data
            fragments to avoid loading, copying, and deserializing columns
            that will not be required further down the compute chain.
            By default all of the available columns are projected. Raises
            an exception if any of the referenced column names does not exist
            in the dataset's Schema.
        filter : Expression, default None
            Scan will return only the rows matching the filter.
            If possible the predicate will be pushed down to exploit the
            partition information or internal metadata found in the data
            source, e.g. Parquet statistics. Otherwise filters the loaded
            RecordBatches before yielding them.
        batch_size : int, default 131_072
            The maximum row count for scanned record batches. If scanned
            record batches are overflowing memory then this method can be
            called to reduce their size.
        batch_readahead : int, default 16
            The number of batches to read ahead in a file. This might not work
            for all file formats. Increasing this number will increase
            RAM usage but could also improve IO utilization.
        fragment_readahead : int, default 4
            The number of files to read ahead. Increasing this number will increase
            RAM usage but could also improve IO utilization.
        fragment_scan_options : FragmentScanOptions, default None
            Options specific to a particular scan and fragment type, which
            can change between different scans of the same dataset.
        use_threads : bool, default True
            If enabled, then maximum parallelism will be used determined by
            the number of available CPU cores.
        cache_metadata : bool, default True
            If enabled, metadata may be cached when scanning to speed up
            repeated scans.
        memory_pool : MemoryPool, default None
            For memory allocations, if required. If not specified, uses the
            default pool.

        Returns
        -------
        table : Table
        """
    def head(
        self,
        num_rows: int,
        columns: list[str] | None = None,
        filter: Expression | None = None,
        batch_size: int = ...,
        batch_readahead: int = 16,
        fragment_readahead: int = 4,
        fragment_scan_options: FragmentScanOptions | None = None,
        use_threads: bool = True,
        cache_metadata: bool = True,
        memory_pool: lib.MemoryPool | None = None,
    ) -> lib.Table:
        """
        Load the first N rows of the dataset.

        Parameters
        ----------
        num_rows : int
            The number of rows to load.
        columns : list of str, default None
            The columns to project. This can be a list of column names to
            include (order and duplicates will be preserved), or a dictionary
            with {new_column_name: expression} values for more advanced
            projections.

            The list of columns or expressions may use the special fields
            `__batch_index` (the index of the batch within the fragment),
            `__fragment_index` (the index of the fragment within the dataset),
            `__last_in_fragment` (whether the batch is last in fragment), and
            `__filename` (the name of the source file or a description of the
            source fragment).

            The columns will be passed down to Datasets and corresponding data
            fragments to avoid loading, copying, and deserializing columns
            that will not be required further down the compute chain.
            By default all of the available columns are projected. Raises
            an exception if any of the referenced column names does not exist
            in the dataset's Schema.
        filter : Expression, default None
            Scan will return only the rows matching the filter.
            If possible the predicate will be pushed down to exploit the
            partition information or internal metadata found in the data
            source, e.g. Parquet statistics. Otherwise filters the loaded
            RecordBatches before yielding them.
        batch_size : int, default 131_072
            The maximum row count for scanned record batches. If scanned
            record batches are overflowing memory then this method can be
            called to reduce their size.
        batch_readahead : int, default 16
            The number of batches to read ahead in a file. This might not work
            for all file formats. Increasing this number will increase
            RAM usage but could also improve IO utilization.
        fragment_readahead : int, default 4
            The number of files to read ahead. Increasing this number will increase
            RAM usage but could also improve IO utilization.
        fragment_scan_options : FragmentScanOptions, default None
            Options specific to a particular scan and fragment type, which
            can change between different scans of the same dataset.
        use_threads : bool, default True
            If enabled, then maximum parallelism will be used determined by
            the number of available CPU cores.
        cache_metadata : bool, default True
            If enabled, metadata may be cached when scanning to speed up
            repeated scans.
        memory_pool : MemoryPool, default None
            For memory allocations, if required. If not specified, uses the
            default pool.

        Returns
        -------
        table : Table
        """
    def count_rows(
        self,
        filter: Expression | None = None,
        batch_size: int = ...,
        batch_readahead: int = 16,
        fragment_readahead: int = 4,
        fragment_scan_options: FragmentScanOptions | None = None,
        use_threads: bool = True,
        cache_metadata: bool = True,
        memory_pool: lib.MemoryPool | None = None,
    ) -> int:
        """
        Count rows matching the scanner filter.

        Parameters
        ----------
        filter : Expression, default None
            Scan will return only the rows matching the filter.
            If possible the predicate will be pushed down to exploit the
            partition information or internal metadata found in the data
            source, e.g. Parquet statistics. Otherwise filters the loaded
            RecordBatches before yielding them.
        batch_size : int, default 131_072
            The maximum row count for scanned record batches. If scanned
            record batches are overflowing memory then this method can be
            called to reduce their size.
        batch_readahead : int, default 16
            The number of batches to read ahead in a file. This might not work
            for all file formats. Increasing this number will increase
            RAM usage but could also improve IO utilization.
        fragment_readahead : int, default 4
            The number of files to read ahead. Increasing this number will increase
            RAM usage but could also improve IO utilization.
        fragment_scan_options : FragmentScanOptions, default None
            Options specific to a particular scan and fragment type, which
            can change between different scans of the same dataset.
        use_threads : bool, default True
            If enabled, then maximum parallelism will be used determined by
            the number of available CPU cores.
        cache_metadata : bool, default True
            If enabled, metadata may be cached when scanning to speed up
            repeated scans.
        memory_pool : MemoryPool, default None
            For memory allocations, if required. If not specified, uses the
            default pool.

        Returns
        -------
        count : int
        """
    @property
    def schema(self) -> lib.Schema:
        """The common schema of the full Dataset"""
    def filter(self, expression: Expression) -> Self:
        """
        Apply a row filter to the dataset.

        Parameters
        ----------
        expression : Expression
            The filter that should be applied to the dataset.

        Returns
        -------
        Dataset
        """
    def sort_by(self, sorting: str | list[tuple[str, Order]], **kwargs) -> InMemoryDataset:
        """
        Sort the Dataset by one or multiple columns.

        Parameters
        ----------
        sorting : str or list[tuple(name, order)]
            Name of the column to use to sort (ascending), or
            a list of multiple sorting conditions where
            each entry is a tuple with column name
            and sorting order ("ascending" or "descending")
        **kwargs : dict, optional
            Additional sorting options.
            As allowed by :class:`SortOptions`

        Returns
        -------
        InMemoryDataset
            A new dataset sorted according to the sort keys.
        """
    def join(
        self,
        right_dataset: Dataset,
        keys: str | list[str],
        right_keys: str | list[str] | None = None,
        join_type: JoinType = "left outer",
        left_suffix: str | None = None,
        right_suffix: str | None = None,
        coalesce_keys: bool = True,
        use_threads: bool = True,
    ) -> InMemoryDataset:
        """
        Perform a join between this dataset and another one.

        Result of the join will be a new dataset, where further
        operations can be applied.

        Parameters
        ----------
        right_dataset : dataset
            The dataset to join to the current one, acting as the right dataset
            in the join operation.
        keys : str or list[str]
            The columns from current dataset that should be used as keys
            of the join operation left side.
        right_keys : str or list[str], default None
            The columns from the right_dataset that should be used as keys
            on the join operation right side.
            When ``None`` use the same key names as the left dataset.
        join_type : str, default "left outer"
            The kind of join that should be performed, one of
            ("left semi", "right semi", "left anti", "right anti",
            "inner", "left outer", "right outer", "full outer")
        left_suffix : str, default None
            Which suffix to add to right column names. This prevents confusion
            when the columns in left and right datasets have colliding names.
        right_suffix : str, default None
            Which suffix to add to the left column names. This prevents confusion
            when the columns in left and right datasets have colliding names.
        coalesce_keys : bool, default True
            If the duplicated keys should be omitted from one of the sides
            in the join result.
        use_threads : bool, default True
            Whenever to use multithreading or not.

        Returns
        -------
        InMemoryDataset
        """
    def join_asof(
        self,
        right_dataset: Dataset,
        on: str,
        by: str | list[str],
        tolerance: int,
        right_on: str | list[str] | None = None,
        right_by: str | list[str] | None = None,
    ) -> InMemoryDataset:
        """
        Perform an asof join between this dataset and another one.

        This is similar to a left-join except that we match on nearest key rather
        than equal keys. Both datasets must be sorted by the key. This type of join
        is most useful for time series data that are not perfectly aligned.

        Optionally match on equivalent keys with "by" before searching with "on".

        Result of the join will be a new Dataset, where further
        operations can be applied.

        Parameters
        ----------
        right_dataset : dataset
            The dataset to join to the current one, acting as the right dataset
            in the join operation.
        on : str
            The column from current dataset that should be used as the "on" key
            of the join operation left side.

            An inexact match is used on the "on" key, i.e. a row is considered a
            match if and only if left_on - tolerance <= right_on <= left_on.

            The input table must be sorted by the "on" key. Must be a single
            field of a common type.

            Currently, the "on" key must be an integer, date, or timestamp type.
        by : str or list[str]
            The columns from current dataset that should be used as the keys
            of the join operation left side. The join operation is then done
            only for the matches in these columns.
        tolerance : int
            The tolerance for inexact "on" key matching. A right row is considered
            a match with the left row `right.on - left.on <= tolerance`. The
            `tolerance` may be:

            - negative, in which case a past-as-of-join occurs;
            - or positive, in which case a future-as-of-join occurs;
            - or zero, in which case an exact-as-of-join occurs.

            The tolerance is interpreted in the same units as the "on" key.
        right_on : str or list[str], default None
            The columns from the right_dataset that should be used as the on key
            on the join operation right side.
            When ``None`` use the same key name as the left dataset.
        right_by : str or list[str], default None
            The columns from the right_dataset that should be used as by keys
            on the join operation right side.
            When ``None`` use the same key names as the left dataset.

        Returns
        -------
        InMemoryDataset
        """

class InMemoryDataset(Dataset):
    """
    A Dataset wrapping in-memory data.

    Parameters
    ----------
    source : RecordBatch, Table, list, tuple
        The data for this dataset. Can be a RecordBatch, Table, list of
        RecordBatch/Table, iterable of RecordBatch, or a RecordBatchReader
        If an iterable is provided, the schema must also be provided.
    schema : Schema, optional
        Only required if passing an iterable as the source
    """

class UnionDataset(Dataset):
    """
    A Dataset wrapping child datasets.

    Children's schemas must agree with the provided schema.

    Parameters
    ----------
    schema : Schema
        A known schema to conform to.
    children : list of Dataset
        One or more input children
    """

    @property
    def children(self) -> list[Dataset]: ...

class FileSystemDataset(Dataset):
    """
    A Dataset of file fragments.

    A FileSystemDataset is composed of one or more FileFragment.

    Parameters
    ----------
    fragments : list[Fragments]
        List of fragments to consume.
    schema : Schema
        The top-level schema of the Dataset.
    format : FileFormat
        File format of the fragments, currently only ParquetFileFormat,
        IpcFileFormat, CsvFileFormat, and JsonFileFormat are supported.
    filesystem : FileSystem
        FileSystem of the fragments.
    root_partition : Expression, optional
        The top-level partition of the DataDataset.
    """

    def __init__(
        self,
        fragments: list[Fragment],
        schema: lib.Schema,
        format: FileFormat,
        filesystem: SupportedFileSystem | None = None,
        root_partition: Expression | None = None,
    ) -> None: ...
    @classmethod
    def from_paths(
        cls,
        paths: list[str],
        schema: lib.Schema | None = None,
        format: FileFormat | None = None,
        filesystem: SupportedFileSystem | None = None,
        partitions: list[Expression] | None = None,
        root_partition: Expression | None = None,
    ) -> FileSystemDataset:
        """
        A Dataset created from a list of paths on a particular filesystem.

        Parameters
        ----------
        paths : list of str
            List of file paths to create the fragments from.
        schema : Schema
            The top-level schema of the DataDataset.
        format : FileFormat
            File format to create fragments from, currently only
            ParquetFileFormat, IpcFileFormat, CsvFileFormat, and JsonFileFormat are supported.
        filesystem : FileSystem
            The filesystem which files are from.
        partitions : list[Expression], optional
            Attach additional partition information for the file paths.
        root_partition : Expression, optional
            The top-level partition of the DataDataset.
        """
    @property
    def filesystem(self) -> FileSystem: ...
    @property
    def partitioning(self) -> Partitioning | None:
        """
        The partitioning of the Dataset source, if discovered.

        If the FileSystemDataset is created using the ``dataset()`` factory
        function with a partitioning specified, this will return the
        finalized Partitioning object from the dataset discovery. In all
        other cases, this returns None.
        """
    @property
    def files(self) -> list[str]:
        """List of the files"""
    @property
    def format(self) -> FileFormat:
        """The FileFormat of this source."""

class FileWriteOptions(lib._Weakrefable):
    @property
    def format(self) -> FileFormat: ...

class FileFormat(lib._Weakrefable):
    def inspect(
        self, file: StrPath | IO, filesystem: SupportedFileSystem | None = None
    ) -> lib.Schema:
        """
        Infer the schema of a file.

        Parameters
        ----------
        file : file-like object, path-like or str
            The file or file path to infer a schema from.
        filesystem : Filesystem, optional
            If `filesystem` is given, `file` must be a string and specifies
            the path of the file to read from the filesystem.

        Returns
        -------
        schema : Schema
            The schema inferred from the file
        """
    def make_fragment(
        self,
        file: StrPath | IO,
        filesystem: SupportedFileSystem | None = None,
        partition_expression: Expression | None = None,
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
        file_size : int, optional
            The size of the file in bytes. Can improve performance with high-latency filesystems
            when file size needs to be known before reading.

        Returns
        -------
        fragment : Fragment
            The file fragment
        """
    def make_write_options(self) -> FileWriteOptions: ...
    @property
    def default_extname(self) -> str: ...
    @property
    def default_fragment_scan_options(self) -> FragmentScanOptions: ...
    @default_fragment_scan_options.setter
    def default_fragment_scan_options(self, options: FragmentScanOptions) -> None: ...

class Fragment(lib._Weakrefable):
    """Fragment of data from a Dataset."""
    @property
    def physical_schema(self) -> lib.Schema:
        """Return the physical schema of this Fragment. This schema can be
        different from the dataset read schema."""
    @property
    def partition_expression(self) -> Expression:
        """An Expression which evaluates to true for all data viewed by this
        Fragment.
        """
    def scanner(
        self,
        schema: lib.Schema | None = None,
        columns: list[str] | None = None,
        filter: Expression | None = None,
        batch_size: int = ...,
        batch_readahead: int = 16,
        fragment_readahead: int = 4,
        fragment_scan_options: FragmentScanOptions | None = None,
        use_threads: bool = True,
        cache_metadata: bool = True,
        memory_pool: lib.MemoryPool | None = None,
    ) -> Scanner:
        """
        Build a scan operation against the fragment.

        Data is not loaded immediately. Instead, this produces a Scanner,
        which exposes further operations (e.g. loading all data as a
        table, counting rows).

        Parameters
        ----------
        schema : Schema
            Schema to use for scanning. This is used to unify a Fragment to
            its Dataset's schema. If not specified this will use the
            Fragment's physical schema which might differ for each Fragment.
        columns : list of str, default None
            The columns to project. This can be a list of column names to
            include (order and duplicates will be preserved), or a dictionary
            with {new_column_name: expression} values for more advanced
            projections.

            The list of columns or expressions may use the special fields
            `__batch_index` (the index of the batch within the fragment),
            `__fragment_index` (the index of the fragment within the dataset),
            `__last_in_fragment` (whether the batch is last in fragment), and
            `__filename` (the name of the source file or a description of the
            source fragment).

            The columns will be passed down to Datasets and corresponding data
            fragments to avoid loading, copying, and deserializing columns
            that will not be required further down the compute chain.
            By default all of the available columns are projected. Raises
            an exception if any of the referenced column names does not exist
            in the dataset's Schema.
        filter : Expression, default None
            Scan will return only the rows matching the filter.
            If possible the predicate will be pushed down to exploit the
            partition information or internal metadata found in the data
            source, e.g. Parquet statistics. Otherwise filters the loaded
            RecordBatches before yielding them.
        batch_size : int, default 131_072
            The maximum row count for scanned record batches. If scanned
            record batches are overflowing memory then this method can be
            called to reduce their size.
        batch_readahead : int, default 16
            The number of batches to read ahead in a file. This might not work
            for all file formats. Increasing this number will increase
            RAM usage but could also improve IO utilization.
        fragment_readahead : int, default 4
            The number of files to read ahead. Increasing this number will increase
            RAM usage but could also improve IO utilization.
        fragment_scan_options : FragmentScanOptions, default None
            Options specific to a particular scan and fragment type, which
            can change between different scans of the same dataset.
        use_threads : bool, default True
            If enabled, then maximum parallelism will be used determined by
            the number of available CPU cores.
        cache_metadata : bool, default True
            If enabled, metadata may be cached when scanning to speed up
            repeated scans.
        memory_pool : MemoryPool, default None
            For memory allocations, if required. If not specified, uses the
            default pool.

        Returns
        -------
        scanner : Scanner
        """
    def to_batches(
        self,
        schema: lib.Schema | None = None,
        columns: list[str] | None = None,
        filter: Expression | None = None,
        batch_size: int = ...,
        batch_readahead: int = 16,
        fragment_readahead: int = 4,
        fragment_scan_options: FragmentScanOptions | None = None,
        use_threads: bool = True,
        cache_metadata: bool = True,
        memory_pool: lib.MemoryPool | None = None,
    ) -> Iterator[lib.RecordBatch]:
        """
        Read the fragment as materialized record batches.

        Parameters
        ----------
        schema : Schema, optional
            Concrete schema to use for scanning.
        columns : list of str, default None
            The columns to project. This can be a list of column names to
            include (order and duplicates will be preserved), or a dictionary
            with {new_column_name: expression} values for more advanced
            projections.

            The list of columns or expressions may use the special fields
            `__batch_index` (the index of the batch within the fragment),
            `__fragment_index` (the index of the fragment within the dataset),
            `__last_in_fragment` (whether the batch is last in fragment), and
            `__filename` (the name of the source file or a description of the
            source fragment).

            The columns will be passed down to Datasets and corresponding data
            fragments to avoid loading, copying, and deserializing columns
            that will not be required further down the compute chain.
            By default all of the available columns are projected. Raises
            an exception if any of the referenced column names does not exist
            in the dataset's Schema.
        filter : Expression, default None
            Scan will return only the rows matching the filter.
            If possible the predicate will be pushed down to exploit the
            partition information or internal metadata found in the data
            source, e.g. Parquet statistics. Otherwise filters the loaded
            RecordBatches before yielding them.
        batch_size : int, default 131_072
            The maximum row count for scanned record batches. If scanned
            record batches are overflowing memory then this method can be
            called to reduce their size.
        batch_readahead : int, default 16
            The number of batches to read ahead in a file. This might not work
            for all file formats. Increasing this number will increase
            RAM usage but could also improve IO utilization.
        fragment_readahead : int, default 4
            The number of files to read ahead. Increasing this number will increase
            RAM usage but could also improve IO utilization.
        fragment_scan_options : FragmentScanOptions, default None
            Options specific to a particular scan and fragment type, which
            can change between different scans of the same dataset.
        use_threads : bool, default True
            If enabled, then maximum parallelism will be used determined by
            the number of available CPU cores.
        cache_metadata : bool, default True
            If enabled, metadata may be cached when scanning to speed up
            repeated scans.
        memory_pool : MemoryPool, default None
            For memory allocations, if required. If not specified, uses the
            default pool.

        Returns
        -------
        record_batches : iterator of RecordBatch
        """
    def to_table(
        self,
        schema: lib.Schema | None = None,
        columns: list[str] | None = None,
        filter: Expression | None = None,
        batch_size: int = ...,
        batch_readahead: int = 16,
        fragment_readahead: int = 4,
        fragment_scan_options: FragmentScanOptions | None = None,
        use_threads: bool = True,
        cache_metadata: bool = True,
        memory_pool: lib.MemoryPool | None = None,
    ) -> lib.Table:
        """
        Convert this Fragment into a Table.

        Use this convenience utility with care. This will serially materialize
        the Scan result in memory before creating the Table.

        Parameters
        ----------
        schema : Schema, optional
            Concrete schema to use for scanning.
        columns : list of str, default None
            The columns to project. This can be a list of column names to
            include (order and duplicates will be preserved), or a dictionary
            with {new_column_name: expression} values for more advanced
            projections.

            The list of columns or expressions may use the special fields
            `__batch_index` (the index of the batch within the fragment),
            `__fragment_index` (the index of the fragment within the dataset),
            `__last_in_fragment` (whether the batch is last in fragment), and
            `__filename` (the name of the source file or a description of the
            source fragment).

            The columns will be passed down to Datasets and corresponding data
            fragments to avoid loading, copying, and deserializing columns
            that will not be required further down the compute chain.
            By default all of the available columns are projected. Raises
            an exception if any of the referenced column names does not exist
            in the dataset's Schema.
        filter : Expression, default None
            Scan will return only the rows matching the filter.
            If possible the predicate will be pushed down to exploit the
            partition information or internal metadata found in the data
            source, e.g. Parquet statistics. Otherwise filters the loaded
            RecordBatches before yielding them.
        batch_size : int, default 131_072
            The maximum row count for scanned record batches. If scanned
            record batches are overflowing memory then this method can be
            called to reduce their size.
        batch_readahead : int, default 16
            The number of batches to read ahead in a file. This might not work
            for all file formats. Increasing this number will increase
            RAM usage but could also improve IO utilization.
        fragment_readahead : int, default 4
            The number of files to read ahead. Increasing this number will increase
            RAM usage but could also improve IO utilization.
        fragment_scan_options : FragmentScanOptions, default None
            Options specific to a particular scan and fragment type, which
            can change between different scans of the same dataset.
        use_threads : bool, default True
            If enabled, then maximum parallelism will be used determined by
            the number of available CPU cores.
        cache_metadata : bool, default True
            If enabled, metadata may be cached when scanning to speed up
            repeated scans.
        memory_pool : MemoryPool, default None
            For memory allocations, if required. If not specified, uses the
            default pool.

        Returns
        -------
        table : Table
        """
    def take(
        self,
        indices: Indices,
        columns: list[str] | None = None,
        filter: Expression | None = None,
        batch_size: int = ...,
        batch_readahead: int = 16,
        fragment_readahead: int = 4,
        fragment_scan_options: FragmentScanOptions | None = None,
        use_threads: bool = True,
        cache_metadata: bool = True,
        memory_pool: lib.MemoryPool | None = None,
    ) -> lib.Table:
        """
        Select rows of data by index.

        Parameters
        ----------
        indices : Array or array-like
            The indices of row to select in the dataset.
        columns : list of str, default None
            The columns to project. This can be a list of column names to
            include (order and duplicates will be preserved), or a dictionary
            with {new_column_name: expression} values for more advanced
            projections.

            The list of columns or expressions may use the special fields
            `__batch_index` (the index of the batch within the fragment),
            `__fragment_index` (the index of the fragment within the dataset),
            `__last_in_fragment` (whether the batch is last in fragment), and
            `__filename` (the name of the source file or a description of the
            source fragment).

            The columns will be passed down to Datasets and corresponding data
            fragments to avoid loading, copying, and deserializing columns
            that will not be required further down the compute chain.
            By default all of the available columns are projected. Raises
            an exception if any of the referenced column names does not exist
            in the dataset's Schema.
        filter : Expression, default None
            Scan will return only the rows matching the filter.
            If possible the predicate will be pushed down to exploit the
            partition information or internal metadata found in the data
            source, e.g. Parquet statistics. Otherwise filters the loaded
            RecordBatches before yielding them.
        batch_size : int, default 131_072
            The maximum row count for scanned record batches. If scanned
            record batches are overflowing memory then this method can be
            called to reduce their size.
        batch_readahead : int, default 16
            The number of batches to read ahead in a file. This might not work
            for all file formats. Increasing this number will increase
            RAM usage but could also improve IO utilization.
        fragment_readahead : int, default 4
            The number of files to read ahead. Increasing this number will increase
            RAM usage but could also improve IO utilization.
        fragment_scan_options : FragmentScanOptions, default None
            Options specific to a particular scan and fragment type, which
            can change between different scans of the same dataset.
        use_threads : bool, default True
            If enabled, then maximum parallelism will be used determined by
            the number of available CPU cores.
        cache_metadata : bool, default True
            If enabled, metadata may be cached when scanning to speed up
            repeated scans.
        memory_pool : MemoryPool, default None
            For memory allocations, if required. If not specified, uses the
            default pool.

        Returns
        -------
        Table
        """
    def head(
        self,
        num_rows: int,
        columns: list[str] | None = None,
        filter: Expression | None = None,
        batch_size: int = ...,
        batch_readahead: int = 16,
        fragment_readahead: int = 4,
        fragment_scan_options: FragmentScanOptions | None = None,
        use_threads: bool = True,
        cache_metadata: bool = True,
        memory_pool: lib.MemoryPool | None = None,
    ) -> lib.Table:
        """
        Load the first N rows of the fragment.

        Parameters
        ----------
        num_rows : int
            The number of rows to load.
        columns : list of str, default None
            The columns to project. This can be a list of column names to
            include (order and duplicates will be preserved), or a dictionary
            with {new_column_name: expression} values for more advanced
            projections.

            The list of columns or expressions may use the special fields
            `__batch_index` (the index of the batch within the fragment),
            `__fragment_index` (the index of the fragment within the dataset),
            `__last_in_fragment` (whether the batch is last in fragment), and
            `__filename` (the name of the source file or a description of the
            source fragment).

            The columns will be passed down to Datasets and corresponding data
            fragments to avoid loading, copying, and deserializing columns
            that will not be required further down the compute chain.
            By default all of the available columns are projected. Raises
            an exception if any of the referenced column names does not exist
            in the dataset's Schema.
        filter : Expression, default None
            Scan will return only the rows matching the filter.
            If possible the predicate will be pushed down to exploit the
            partition information or internal metadata found in the data
            source, e.g. Parquet statistics. Otherwise filters the loaded
            RecordBatches before yielding them.
        batch_size : int, default 131_072
            The maximum row count for scanned record batches. If scanned
            record batches are overflowing memory then this method can be
            called to reduce their size.
        batch_readahead : int, default 16
            The number of batches to read ahead in a file. This might not work
            for all file formats. Increasing this number will increase
            RAM usage but could also improve IO utilization.
        fragment_readahead : int, default 4
            The number of files to read ahead. Increasing this number will increase
            RAM usage but could also improve IO utilization.
        fragment_scan_options : FragmentScanOptions, default None
            Options specific to a particular scan and fragment type, which
            can change between different scans of the same dataset.
        use_threads : bool, default True
            If enabled, then maximum parallelism will be used determined by
            the number of available CPU cores.
        cache_metadata : bool, default True
            If enabled, metadata may be cached when scanning to speed up
            repeated scans.
        memory_pool : MemoryPool, default None
            For memory allocations, if required. If not specified, uses the
            default pool.

        Returns
        -------
        Table
        """
    def count_rows(
        self,
        columns: list[str] | None = None,
        filter: Expression | None = None,
        batch_size: int = ...,
        batch_readahead: int = 16,
        fragment_readahead: int = 4,
        fragment_scan_options: FragmentScanOptions | None = None,
        use_threads: bool = True,
        cache_metadata: bool = True,
        memory_pool: lib.MemoryPool | None = None,
    ) -> int:
        """
        Count rows matching the scanner filter.

        Parameters
        ----------
        filter : Expression, default None
            Scan will return only the rows matching the filter.
            If possible the predicate will be pushed down to exploit the
            partition information or internal metadata found in the data
            source, e.g. Parquet statistics. Otherwise filters the loaded
            RecordBatches before yielding them.
        batch_size : int, default 131_072
            The maximum row count for scanned record batches. If scanned
            record batches are overflowing memory then this method can be
            called to reduce their size.
        batch_readahead : int, default 16
            The number of batches to read ahead in a file. This might not work
            for all file formats. Increasing this number will increase
            RAM usage but could also improve IO utilization.
        fragment_readahead : int, default 4
            The number of files to read ahead. Increasing this number will increase
            RAM usage but could also improve IO utilization.
        fragment_scan_options : FragmentScanOptions, default None
            Options specific to a particular scan and fragment type, which
            can change between different scans of the same dataset.
        use_threads : bool, default True
            If enabled, then maximum parallelism will be used determined by
            the number of available CPU cores.
        cache_metadata : bool, default True
            If enabled, metadata may be cached when scanning to speed up
            repeated scans.
        memory_pool : MemoryPool, default None
            For memory allocations, if required. If not specified, uses the
            default pool.

        Returns
        -------
        count : int
        """

class FileFragment(Fragment):
    """A Fragment representing a data file."""

    def open(self) -> lib.NativeFile:
        """
        Open a NativeFile of the buffer or file viewed by this fragment.
        """
    @property
    def path(self) -> str:
        """
        The path of the data file viewed by this fragment, if it views a
        file. If instead it views a buffer, this will be "<Buffer>".
        """
    @property
    def filesystem(self) -> FileSystem:
        """
        The FileSystem containing the data file viewed by this fragment, if
        it views a file. If instead it views a buffer, this will be None.
        """
    @property
    def buffer(self) -> lib.Buffer:
        """
        The buffer viewed by this fragment, if it views a buffer. If
        instead it views a file, this will be None.
        """
    @property
    def format(self) -> FileFormat:
        """
        The format of the data file viewed by this fragment.
        """

class FragmentScanOptions(lib._Weakrefable):
    """Scan options specific to a particular fragment and scan operation."""

    @property
    def type_name(self) -> str: ...

class IpcFileWriteOptions(FileWriteOptions):
    @property
    def write_options(self) -> IpcWriteOptions: ...
    @write_options.setter
    def write_options(self, write_options: IpcWriteOptions) -> None: ...

class IpcFileFormat(FileFormat):
    def equals(self, other: IpcFileFormat) -> bool: ...
    def make_write_options(self, **kwargs) -> IpcFileWriteOptions: ...
    @property
    def default_extname(self) -> str: ...

class FeatherFileFormat(IpcFileFormat): ...

class CsvFileFormat(FileFormat):
    """
    FileFormat for CSV files.

    Parameters
    ----------
    parse_options : pyarrow.csv.ParseOptions
        Options regarding CSV parsing.
    default_fragment_scan_options : CsvFragmentScanOptions
        Default options for fragments scan.
    convert_options : pyarrow.csv.ConvertOptions
        Options regarding value conversion.
    read_options : pyarrow.csv.ReadOptions
        General read options.
    """
    def __init__(
        self,
        parse_options: _csv.ParseOptions | None = None,
        default_fragment_scan_options: CsvFragmentScanOptions | None = None,
        convert_options: _csv.ConvertOptions | None = None,
        read_options: _csv.ReadOptions | None = None,
    ) -> None: ...
    def make_write_options(self) -> _csv.WriteOptions: ...  # type: ignore[override]
    @property
    def parse_options(self) -> _csv.ParseOptions: ...
    @parse_options.setter
    def parse_options(self, parse_options: _csv.ParseOptions) -> None: ...
    def equals(self, other: CsvFileFormat) -> bool: ...

class CsvFragmentScanOptions(FragmentScanOptions):
    """
    Scan-specific options for CSV fragments.

    Parameters
    ----------
    convert_options : pyarrow.csv.ConvertOptions
        Options regarding value conversion.
    read_options : pyarrow.csv.ReadOptions
        General read options.
    """

    convert_options: _csv.ConvertOptions
    read_options: _csv.ReadOptions

    def __init__(
        self, convert_options: _csv.ConvertOptions, read_options: _csv.ReadOptions
    ) -> None: ...
    def equals(self, other: CsvFragmentScanOptions) -> bool: ...

class CsvFileWriteOptions(FileWriteOptions):
    write_options: _csv.WriteOptions

class JsonFileFormat(FileFormat):
    """
    FileFormat for JSON files.

    Parameters
    ----------
    default_fragment_scan_options : JsonFragmentScanOptions
        Default options for fragments scan.
    parse_options : pyarrow.json.ParseOptions
        Options regarding json parsing.
    read_options : pyarrow.json.ReadOptions
        General read options.
    """
    def __init__(
        self,
        default_fragment_scan_options: JsonFragmentScanOptions | None = None,
        parse_options: _json.ParseOptions | None = None,
        read_options: _json.ReadOptions | None = None,
    ) -> None: ...
    def equals(self, other: JsonFileFormat) -> bool: ...

class JsonFragmentScanOptions(FragmentScanOptions):
    """
    Scan-specific options for JSON fragments.

    Parameters
    ----------
    parse_options : pyarrow.json.ParseOptions
        Options regarding JSON parsing.
    read_options : pyarrow.json.ReadOptions
        General read options.
    """

    parse_options: _json.ParseOptions
    read_options: _json.ReadOptions
    def __init__(
        self, parse_options: _json.ParseOptions, read_options: _json.ReadOptions
    ) -> None: ...
    def equals(self, other: JsonFragmentScanOptions) -> bool: ...

class Partitioning(lib._Weakrefable):
    def parse(self, path: str) -> Expression:
        """
        Parse a path into a partition expression.

        Parameters
        ----------
        path : str

        Returns
        -------
        pyarrow.dataset.Expression
        """
    def format(self, expr: Expression) -> tuple[str, str]:
        """
        Convert a filter expression into a tuple of (directory, filename) using
        the current partitioning scheme

        Parameters
        ----------
        expr : pyarrow.dataset.Expression

        Returns
        -------
        tuple[str, str]

        Examples
        --------

        Specify the Schema for paths like "/2009/June":

        >>> import pyarrow as pa
        >>> import pyarrow.dataset as ds
        >>> import pyarrow.compute as pc
        >>> part = ds.partitioning(pa.schema([("year", pa.int16()), ("month", pa.string())]))
        >>> part.format((pc.field("year") == 1862) & (pc.field("month") == "Jan"))
        ('1862/Jan', '')
        """
    @property
    def schema(self) -> lib.Schema:
        """The arrow Schema attached to the partitioning."""

class PartitioningFactory(lib._Weakrefable):
    @property
    def type_name(self) -> str: ...

class KeyValuePartitioning(Partitioning):
    @property
    def dictionaries(self) -> list[lib.Array | None]:
        """
        The unique values for each partition field, if available.

        Those values are only available if the Partitioning object was
        created through dataset discovery from a PartitioningFactory, or
        if the dictionaries were manually specified in the constructor.
        If no dictionary field is available, this returns an empty list.
        """

class DirectoryPartitioning(KeyValuePartitioning):
    """
    A Partitioning based on a specified Schema.

    The DirectoryPartitioning expects one segment in the file path for each
    field in the schema (all fields are required to be present).
    For example given schema<year:int16, month:int8> the path "/2009/11" would
    be parsed to ("year"_ == 2009 and "month"_ == 11).

    Parameters
    ----------
    schema : Schema
        The schema that describes the partitions present in the file path.
    dictionaries : dict[str, Array]
        If the type of any field of `schema` is a dictionary type, the
        corresponding entry of `dictionaries` must be an array containing
        every value which may be taken by the corresponding column or an
        error will be raised in parsing.
    segment_encoding : str, default "uri"
        After splitting paths into segments, decode the segments. Valid
        values are "uri" (URI-decode segments) and "none" (leave as-is).

    Returns
    -------
    DirectoryPartitioning

    Examples
    --------
    >>> from pyarrow.dataset import DirectoryPartitioning
    >>> partitioning = DirectoryPartitioning(
    ...     pa.schema([("year", pa.int16()), ("month", pa.int8())])
    ... )
    >>> print(partitioning.parse("/2009/11/"))
    ((year == 2009) and (month == 11))
    """

    @staticmethod
    def discover(
        field_names: list[str] | None = None,
        infer_dictionary: bool = False,
        max_partition_dictionary_size: int = 0,
        schema: lib.Schema | None = None,
        segment_encoding: Literal["uri", "none"] = "uri",
    ) -> PartitioningFactory:
        """
        Discover a DirectoryPartitioning.

        Parameters
        ----------
        field_names : list of str
            The names to associate with the values from the subdirectory names.
            If schema is given, will be populated from the schema.
        infer_dictionary : bool, default False
            When inferring a schema for partition fields, yield dictionary
            encoded types instead of plain types. This can be more efficient
            when materializing virtual columns, and Expressions parsed by the
            finished Partitioning will include dictionaries of all unique
            inspected values for each field.
        max_partition_dictionary_size : int, default 0
            Synonymous with infer_dictionary for backwards compatibility with
            1.0: setting this to -1 or None is equivalent to passing
            infer_dictionary=True.
        schema : Schema, default None
            Use this schema instead of inferring a schema from partition
            values. Partition values will be validated against this schema
            before accumulation into the Partitioning's dictionary.
        segment_encoding : str, default "uri"
            After splitting paths into segments, decode the segments. Valid
            values are "uri" (URI-decode segments) and "none" (leave as-is).

        Returns
        -------
        PartitioningFactory
            To be used in the FileSystemFactoryOptions.
        """
    def __init__(
        self,
        schema: lib.Schema,
        dictionaries: dict[str, lib.Array] | None = None,
        segment_encoding: Literal["uri", "none"] = "uri",
    ) -> None: ...

class HivePartitioning(KeyValuePartitioning):
    """
    A Partitioning for "/$key=$value/" nested directories as found in
    Apache Hive.

    Multi-level, directory based partitioning scheme originating from
    Apache Hive with all data files stored in the leaf directories. Data is
    partitioned by static values of a particular column in the schema.
    Partition keys are represented in the form $key=$value in directory names.
    Field order is ignored, as are missing or unrecognized field names.

    For example, given schema<year:int16, month:int8, day:int8>, a possible
    path would be "/year=2009/month=11/day=15".

    Parameters
    ----------
    schema : Schema
        The schema that describes the partitions present in the file path.
    dictionaries : dict[str, Array]
        If the type of any field of `schema` is a dictionary type, the
        corresponding entry of `dictionaries` must be an array containing
        every value which may be taken by the corresponding column or an
        error will be raised in parsing.
    null_fallback : str, default "__HIVE_DEFAULT_PARTITION__"
        If any field is None then this fallback will be used as a label
    segment_encoding : str, default "uri"
        After splitting paths into segments, decode the segments. Valid
        values are "uri" (URI-decode segments) and "none" (leave as-is).

    Returns
    -------
    HivePartitioning

    Examples
    --------
    >>> from pyarrow.dataset import HivePartitioning
    >>> partitioning = HivePartitioning(pa.schema([("year", pa.int16()), ("month", pa.int8())]))
    >>> print(partitioning.parse("/year=2009/month=11/"))
    ((year == 2009) and (month == 11))

    """
    def __init__(
        self,
        schema: lib.Schema,
        dictionaries: dict[str, lib.Array] | None = None,
        null_fallback: str = "__HIVE_DEFAULT_PARTITION__",
        segment_encoding: Literal["uri", "none"] = "uri",
    ) -> None: ...
    @staticmethod
    def discover(
        infer_dictionary: bool = False,
        max_partition_dictionary_size: int = 0,
        null_fallback="__HIVE_DEFAULT_PARTITION__",
        schema: lib.Schema | None = None,
        segment_encoding: Literal["uri", "none"] = "uri",
    ) -> PartitioningFactory:
        """
        Discover a HivePartitioning.

        Parameters
        ----------
        infer_dictionary : bool, default False
            When inferring a schema for partition fields, yield dictionary
            encoded types instead of plain. This can be more efficient when
            materializing virtual columns, and Expressions parsed by the
            finished Partitioning will include dictionaries of all unique
            inspected values for each field.
        max_partition_dictionary_size : int, default 0
            Synonymous with infer_dictionary for backwards compatibility with
            1.0: setting this to -1 or None is equivalent to passing
            infer_dictionary=True.
        null_fallback : str, default "__HIVE_DEFAULT_PARTITION__"
            When inferring a schema for partition fields this value will be
            replaced by null.  The default is set to __HIVE_DEFAULT_PARTITION__
            for compatibility with Spark
        schema : Schema, default None
            Use this schema instead of inferring a schema from partition
            values. Partition values will be validated against this schema
            before accumulation into the Partitioning's dictionary.
        segment_encoding : str, default "uri"
            After splitting paths into segments, decode the segments. Valid
            values are "uri" (URI-decode segments) and "none" (leave as-is).

        Returns
        -------
        PartitioningFactory
            To be used in the FileSystemFactoryOptions.
        """

class FilenamePartitioning(KeyValuePartitioning):
    """
    A Partitioning based on a specified Schema.

    The FilenamePartitioning expects one segment in the file name for each
    field in the schema (all fields are required to be present) separated
    by '_'. For example given schema<year:int16, month:int8> the name
    ``"2009_11_"`` would be parsed to ("year" == 2009 and "month" == 11).

    Parameters
    ----------
    schema : Schema
        The schema that describes the partitions present in the file path.
    dictionaries : dict[str, Array]
        If the type of any field of `schema` is a dictionary type, the
        corresponding entry of `dictionaries` must be an array containing
        every value which may be taken by the corresponding column or an
        error will be raised in parsing.
    segment_encoding : str, default "uri"
        After splitting paths into segments, decode the segments. Valid
        values are "uri" (URI-decode segments) and "none" (leave as-is).

    Returns
    -------
    FilenamePartitioning

    Examples
    --------
    >>> from pyarrow.dataset import FilenamePartitioning
    >>> partitioning = FilenamePartitioning(
    ...     pa.schema([("year", pa.int16()), ("month", pa.int8())])
    ... )
    >>> print(partitioning.parse("2009_11_data.parquet"))
    ((year == 2009) and (month == 11))
    """

    def __init__(
        self,
        schema: lib.Schema,
        dictionaries: dict[str, lib.Array] | None = None,
        segment_encoding: Literal["uri", "none"] = "uri",
    ) -> None: ...
    @staticmethod
    def discover(
        field_names: list[str] | None = None,
        infer_dictionary: bool = False,
        schema: lib.Schema | None = None,
        segment_encoding: Literal["uri", "none"] = "uri",
    ) -> PartitioningFactory:
        """
        Discover a FilenamePartitioning.

        Parameters
        ----------
        field_names : list of str
            The names to associate with the values from the subdirectory names.
            If schema is given, will be populated from the schema.
        infer_dictionary : bool, default False
            When inferring a schema for partition fields, yield dictionary
            encoded types instead of plain types. This can be more efficient
            when materializing virtual columns, and Expressions parsed by the
            finished Partitioning will include dictionaries of all unique
            inspected values for each field.
        schema : Schema, default None
            Use this schema instead of inferring a schema from partition
            values. Partition values will be validated against this schema
            before accumulation into the Partitioning's dictionary.
        segment_encoding : str, default "uri"
            After splitting paths into segments, decode the segments. Valid
            values are "uri" (URI-decode segments) and "none" (leave as-is).

        Returns
        -------
        PartitioningFactory
            To be used in the FileSystemFactoryOptions.
        """

class DatasetFactory(lib._Weakrefable):
    """
    DatasetFactory is used to create a Dataset, inspect the Schema
    of the fragments contained in it, and declare a partitioning.
    """

    root_partition: Expression
    def finish(self, schema: lib.Schema | None = None) -> Dataset:
        """
        Create a Dataset using the inspected schema or an explicit schema
        (if given).

        Parameters
        ----------
        schema : Schema, default None
            The schema to conform the source to.  If None, the inspected
            schema is used.

        Returns
        -------
        Dataset
        """
    def inspect(self) -> lib.Schema:
        """
        Inspect all data fragments and return a common Schema.

        Returns
        -------
        Schema
        """
    def inspect_schemas(self) -> list[lib.Schema]: ...

class FileSystemFactoryOptions(lib._Weakrefable):
    """
    Influences the discovery of filesystem paths.

    Parameters
    ----------
    partition_base_dir : str, optional
        For the purposes of applying the partitioning, paths will be
        stripped of the partition_base_dir. Files not matching the
        partition_base_dir prefix will be skipped for partitioning discovery.
        The ignored files will still be part of the Dataset, but will not
        have partition information.
    partitioning : Partitioning/PartitioningFactory, optional
       Apply the Partitioning to every discovered Fragment. See Partitioning or
       PartitioningFactory documentation.
    exclude_invalid_files : bool, optional (default True)
        If True, invalid files will be excluded (file format specific check).
        This will incur IO for each files in a serial and single threaded
        fashion. Disabling this feature will skip the IO, but unsupported
        files may be present in the Dataset (resulting in an error at scan
        time).
    selector_ignore_prefixes : list, optional
        When discovering from a Selector (and not from an explicit file list),
        ignore files and directories matching any of these prefixes.
        By default this is ['.', '_'].
    """

    partitioning: Partitioning
    partitioning_factory: PartitioningFactory
    partition_base_dir: str
    exclude_invalid_files: bool
    selector_ignore_prefixes: list[str]

    def __init__(
        self,
        artition_base_dir: str | None = None,
        partitioning: Partitioning | PartitioningFactory | None = None,
        exclude_invalid_files: bool = True,
        selector_ignore_prefixes: list[str] | None = None,
    ) -> None: ...

class FileSystemDatasetFactory(DatasetFactory):
    """
    Create a DatasetFactory from a list of paths with schema inspection.

    Parameters
    ----------
    filesystem : pyarrow.fs.FileSystem
        Filesystem to discover.
    paths_or_selector : pyarrow.fs.FileSelector or list of path-likes
        Either a Selector object or a list of path-like objects.
    format : FileFormat
        Currently only ParquetFileFormat and IpcFileFormat are supported.
    options : FileSystemFactoryOptions, optional
        Various flags influencing the discovery of filesystem paths.
    """

    def __init__(
        self,
        filesystem: SupportedFileSystem,
        paths_or_selector: FileSelector,
        format: FileFormat,
        options: FileSystemFactoryOptions | None = None,
    ) -> None: ...

class UnionDatasetFactory(DatasetFactory):
    """
    Provides a way to inspect/discover a Dataset's expected schema before
    materialization.

    Parameters
    ----------
    factories : list of DatasetFactory
    """
    def __init__(self, factories: list[DatasetFactory]) -> None: ...

_RecordBatchT = TypeVar("_RecordBatchT", bound=lib.RecordBatch)

class RecordBatchIterator(lib._Weakrefable, Generic[_RecordBatchT]):
    """An iterator over a sequence of record batches."""
    def __iter__(self) -> Self: ...
    def __next__(self) -> _RecordBatchT: ...

class TaggedRecordBatch(NamedTuple):
    """
    A combination of a record batch and the fragment it came from.

    Parameters
    ----------
    record_batch : RecordBatch
        The record batch.
    fragment : Fragment
        Fragment of the record batch.
    """

    record_batch: lib.RecordBatch
    fragment: Fragment

class TaggedRecordBatchIterator(lib._Weakrefable):
    """An iterator over a sequence of record batches with fragments."""
    def __iter__(self) -> Self: ...
    def __next__(self) -> TaggedRecordBatch: ...

class Scanner(lib._Weakrefable):
    """A materialized scan operation with context and options bound.

    A scanner is the class that glues the scan tasks, data fragments and data
    sources together.
    """
    @staticmethod
    def from_dataset(
        dataset: Dataset,
        *,
        columns: list[str] | dict[str, Expression] | None = None,
        filter: Expression | None = None,
        batch_size: int = ...,
        batch_readahead: int = 16,
        fragment_readahead: int = 4,
        fragment_scan_options: FragmentScanOptions | None = None,
        use_threads: bool = True,
        cache_metadata: bool = True,
        memory_pool: lib.MemoryPool | None = None,
    ) -> Scanner:
        """
        Create Scanner from Dataset,

        Parameters
        ----------
        dataset : Dataset
            Dataset to scan.
        columns : list[str] or dict[str, Expression], default None
            The columns to project. This can be a list of column names to
            include (order and duplicates will be preserved), or a dictionary
            with {new_column_name: expression} values for more advanced
            projections.

            The list of columns or expressions may use the special fields
            `__batch_index` (the index of the batch within the fragment),
            `__fragment_index` (the index of the fragment within the dataset),
            `__last_in_fragment` (whether the batch is last in fragment), and
            `__filename` (the name of the source file or a description of the
            source fragment).

            The columns will be passed down to Datasets and corresponding data
            fragments to avoid loading, copying, and deserializing columns
            that will not be required further down the compute chain.
            By default all of the available columns are projected. Raises
            an exception if any of the referenced column names does not exist
            in the dataset's Schema.
        filter : Expression, default None
            Scan will return only the rows matching the filter.
            If possible the predicate will be pushed down to exploit the
            partition information or internal metadata found in the data
            source, e.g. Parquet statistics. Otherwise filters the loaded
            RecordBatches before yielding them.
        batch_size : int, default 131_072
            The maximum row count for scanned record batches. If scanned
            record batches are overflowing memory then this method can be
            called to reduce their size.
        batch_readahead : int, default 16
            The number of batches to read ahead in a file. This might not work
            for all file formats. Increasing this number will increase
            RAM usage but could also improve IO utilization.
        fragment_readahead : int, default 4
            The number of files to read ahead. Increasing this number will increase
            RAM usage but could also improve IO utilization.
        fragment_scan_options : FragmentScanOptions, default None
            Options specific to a particular scan and fragment type, which
            can change between different scans of the same dataset.
        use_threads : bool, default True
            If enabled, then maximum parallelism will be used determined by
            the number of available CPU cores.
        cache_metadata : bool, default True
            If enabled, metadata may be cached when scanning to speed up
            repeated scans.
        memory_pool : MemoryPool, default None
            For memory allocations, if required. If not specified, uses the
            default pool.
        """
    @staticmethod
    def from_fragment(
        fragment: Fragment,
        *,
        schema: lib.Schema | None = None,
        columns: list[str] | dict[str, Expression] | None = None,
        filter: Expression | None = None,
        batch_size: int = ...,
        batch_readahead: int = 16,
        fragment_readahead: int = 4,
        fragment_scan_options: FragmentScanOptions | None = None,
        use_threads: bool = True,
        cache_metadata: bool = True,
        memory_pool: lib.MemoryPool | None = None,
    ) -> Scanner:
        """
        Create Scanner from Fragment,

        Parameters
        ----------
        fragment : Fragment
            fragment to scan.
        schema : Schema, optional
            The schema of the fragment.
        columns : list[str] or dict[str, Expression], default None
            The columns to project. This can be a list of column names to
            include (order and duplicates will be preserved), or a dictionary
            with {new_column_name: expression} values for more advanced
            projections.

            The list of columns or expressions may use the special fields
            `__batch_index` (the index of the batch within the fragment),
            `__fragment_index` (the index of the fragment within the dataset),
            `__last_in_fragment` (whether the batch is last in fragment), and
            `__filename` (the name of the source file or a description of the
            source fragment).

            The columns will be passed down to Datasets and corresponding data
            fragments to avoid loading, copying, and deserializing columns
            that will not be required further down the compute chain.
            By default all of the available columns are projected. Raises
            an exception if any of the referenced column names does not exist
            in the dataset's Schema.
        filter : Expression, default None
            Scan will return only the rows matching the filter.
            If possible the predicate will be pushed down to exploit the
            partition information or internal metadata found in the data
            source, e.g. Parquet statistics. Otherwise filters the loaded
            RecordBatches before yielding them.
        batch_size : int, default 131_072
            The maximum row count for scanned record batches. If scanned
            record batches are overflowing memory then this method can be
            called to reduce their size.
        batch_readahead : int, default 16
            The number of batches to read ahead in a file. This might not work
            for all file formats. Increasing this number will increase
            RAM usage but could also improve IO utilization.
        fragment_readahead : int, default 4
            The number of files to read ahead. Increasing this number will increase
            RAM usage but could also improve IO utilization.
        fragment_scan_options : FragmentScanOptions, default None
            Options specific to a particular scan and fragment type, which
            can change between different scans of the same dataset.
        use_threads : bool, default True
            If enabled, then maximum parallelism will be used determined by
            the number of available CPU cores.
        cache_metadata : bool, default True
            If enabled, metadata may be cached when scanning to speed up
            repeated scans.
        memory_pool : MemoryPool, default None
            For memory allocations, if required. If not specified, uses the
            default pool.
        """
    @overload
    @staticmethod
    def from_batches(
        source: Iterator[lib.RecordBatch],
        *,
        schema: lib.Schema,
        columns: list[str] | dict[str, Expression] | None = None,
        filter: Expression | None = None,
        batch_size: int = ...,
        batch_readahead: int = 16,
        fragment_readahead: int = 4,
        fragment_scan_options: FragmentScanOptions | None = None,
        use_threads: bool = True,
        cache_metadata: bool = True,
        memory_pool: lib.MemoryPool | None = None,
    ) -> Scanner: ...
    @overload
    @staticmethod
    def from_batches(
        source: RecordBatchReader,
        *,
        columns: list[str] | dict[str, Expression] | None = None,
        filter: Expression | None = None,
        batch_size: int = ...,
        batch_readahead: int = 16,
        fragment_readahead: int = 4,
        fragment_scan_options: FragmentScanOptions | None = None,
        use_threads: bool = True,
        cache_metadata: bool = True,
        memory_pool: lib.MemoryPool | None = None,
    ) -> Scanner: ...
    @staticmethod
    def from_batches(*args, **kwargs):
        """
        Create a Scanner from an iterator of batches.

        This creates a scanner which can be used only once. It is
        intended to support writing a dataset (which takes a scanner)
        from a source which can be read only once (e.g. a
        RecordBatchReader or generator).

        Parameters
        ----------
        source : Iterator or Arrow-compatible stream object
            The iterator of Batches. This can be a pyarrow RecordBatchReader,
            any object that implements the Arrow PyCapsule Protocol for
            streams, or an actual Python iterator of RecordBatches.
        schema : Schema
            The schema of the batches (required when passing a Python
            iterator).
        columns : list[str] or dict[str, Expression], default None
            The columns to project. This can be a list of column names to
            include (order and duplicates will be preserved), or a dictionary
            with {new_column_name: expression} values for more advanced
            projections.

            The list of columns or expressions may use the special fields
            `__batch_index` (the index of the batch within the fragment),
            `__fragment_index` (the index of the fragment within the dataset),
            `__last_in_fragment` (whether the batch is last in fragment), and
            `__filename` (the name of the source file or a description of the
            source fragment).

            The columns will be passed down to Datasets and corresponding data
            fragments to avoid loading, copying, and deserializing columns
            that will not be required further down the compute chain.
            By default all of the available columns are projected. Raises
            an exception if any of the referenced column names does not exist
            in the dataset's Schema.
        filter : Expression, default None
            Scan will return only the rows matching the filter.
            If possible the predicate will be pushed down to exploit the
            partition information or internal metadata found in the data
            source, e.g. Parquet statistics. Otherwise filters the loaded
            RecordBatches before yielding them.
        batch_size : int, default 131_072
            The maximum row count for scanned record batches. If scanned
            record batches are overflowing memory then this method can be
            called to reduce their size.
        batch_readahead : int, default 16
            The number of batches to read ahead in a file. This might not work
            for all file formats. Increasing this number will increase
            RAM usage but could also improve IO utilization.
        fragment_readahead : int, default 4
            The number of files to read ahead. Increasing this number will increase
            RAM usage but could also improve IO utilization.
        fragment_scan_options : FragmentScanOptions, default None
            Options specific to a particular scan and fragment type, which
            can change between different scans of the same dataset.
        use_threads : bool, default True
            If enabled, then maximum parallelism will be used determined by
            the number of available CPU cores.
        cache_metadata : bool, default True
            If enabled, metadata may be cached when scanning to speed up
            repeated scans.
        memory_pool : MemoryPool, default None
            For memory allocations, if required. If not specified, uses the
            default pool.
        """
    @property
    def dataset_schema(self) -> lib.Schema:
        """The schema with which batches will be read from fragments."""
    @property
    def projected_schema(self) -> lib.Schema:
        """
        The materialized schema of the data, accounting for projections.

        This is the schema of any data returned from the scanner.
        """
    def to_batches(self) -> Iterator[lib.RecordBatch]:
        """
        Consume a Scanner in record batches.

        Returns
        -------
        record_batches : iterator of RecordBatch
        """
    def scan_batches(self) -> TaggedRecordBatchIterator:
        """
        Consume a Scanner in record batches with corresponding fragments.

        Returns
        -------
        record_batches : iterator of TaggedRecordBatch
        """
    def to_table(self) -> lib.Table:
        """
        Convert a Scanner into a Table.

        Use this convenience utility with care. This will serially materialize
        the Scan result in memory before creating the Table.

        Returns
        -------
        Table
        """
    def take(self, indices: Indices) -> lib.Table:
        """
        Select rows of data by index.

        Will only consume as many batches of the underlying dataset as
        needed. Otherwise, this is equivalent to
        ``to_table().take(indices)``.

        Parameters
        ----------
        indices : Array or array-like
            indices of rows to select in the dataset.

        Returns
        -------
        Table
        """
    def head(self, num_rows: int) -> lib.Table:
        """
        Load the first N rows of the dataset.

        Parameters
        ----------
        num_rows : int
            The number of rows to load.

        Returns
        -------
        Table
        """
    def count_rows(self) -> int:
        """
        Count rows matching the scanner filter.

        Returns
        -------
        count : int
        """
    def to_reader(self) -> RecordBatchReader:
        """Consume this scanner as a RecordBatchReader.

        Returns
        -------
        RecordBatchReader
        """

def get_partition_keys(partition_expression: Expression) -> dict[str, Any]:
    """
    Extract partition keys (equality constraints between a field and a scalar)
    from an expression as a dict mapping the field's name to its value.

    NB: All expressions yielded by a HivePartitioning or DirectoryPartitioning
    will be conjunctions of equality conditions and are accessible through this
    function. Other subexpressions will be ignored.

    Parameters
    ----------
    partition_expression : pyarrow.dataset.Expression

    Returns
    -------
    dict

    Examples
    --------

    For example, an expression of
    <pyarrow.dataset.Expression ((part == A:string) and (year == 2016:int32))>
    is converted to {'part': 'A', 'year': 2016}
    """

class WrittenFile(lib._Weakrefable):
    """
    Metadata information about files written as
    part of a dataset write operation

    Parameters
    ----------
    path : str
        Path to the file.
    metadata : pyarrow.parquet.FileMetaData, optional
        For Parquet files, the Parquet file metadata.
    size : int
        The size of the file in bytes.
    """
    def __init__(self, path: str, metadata: _parquet.FileMetaData | None, size: int) -> None: ...

def _filesystemdataset_write(
    data: Scanner,
    base_dir: StrPath,
    basename_template: str,
    filesystem: SupportedFileSystem,
    partitioning: Partitioning,
    file_options: FileWriteOptions,
    max_partitions: int,
    file_visitor: Callable[[str], None],
    existing_data_behavior: Literal["error", "overwrite_or_ignore", "delete_matching"],
    max_open_files: int,
    max_rows_per_file: int,
    min_rows_per_group: int,
    max_rows_per_group: int,
    create_dir: bool,
): ...

class _ScanNodeOptions(ExecNodeOptions):
    def _set_options(self, dataset: Dataset, scan_options: dict) -> None: ...

class ScanNodeOptions(_ScanNodeOptions):
    """
    A Source node which yields batches from a Dataset scan.

    This is the option class for the "scan" node factory.

    This node is capable of applying pushdown projections or filters
    to the file readers which reduce the amount of data that needs to
    be read (if supported by the file format). But note that this does not
    construct associated filter or project nodes to perform the final
    filtering or projection. Rather, you may supply the same filter
    expression or projection to the scan node that you also supply
    to the filter or project node.

    Yielded batches will be augmented with fragment/batch indices when
    implicit_ordering=True to enable stable ordering for simple ExecPlans.

    Parameters
    ----------
    dataset : pyarrow.dataset.Dataset
        The table which acts as the data source.
    **kwargs : dict, optional
        Scan options. See `Scanner.from_dataset` for possible arguments.
    require_sequenced_output : bool, default False
        Batches are yielded sequentially, like single-threaded
    implicit_ordering : bool, default False
        Preserve implicit ordering of data.
    """

    def __init__(
        self, dataset: Dataset, require_sequenced_output: bool = False, **kwargs
    ) -> None: ...
