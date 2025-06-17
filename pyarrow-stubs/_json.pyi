from typing import IO, Any, Literal

from _typeshed import StrPath

from .lib import MemoryPool, RecordBatchReader, Schema, Table, _Weakrefable

class ReadOptions(_Weakrefable):
    """
    Options for reading JSON files.

    Parameters
    ----------
    use_threads : bool, optional (default True)
        Whether to use multiple threads to accelerate reading
    block_size : int, optional
        How much bytes to process at a time from the input stream.
        This will determine multi-threading granularity as well as
        the size of individual chunks in the Table.
    """

    use_threads: bool
    """
    Whether to use multiple threads to accelerate reading.
    """
    block_size: int
    """
    How much bytes to process at a time from the input stream.

    This will determine multi-threading granularity as well as the size of
    individual chunks in the Table.
    """
    def __init__(self, use_threads: bool | None = None, block_size: int | None = None): ...
    def equals(self, other: ReadOptions) -> bool:
        """
        Parameters
        ----------
        other : pyarrow.json.ReadOptions

        Returns
        -------
        bool
        """

class ParseOptions(_Weakrefable):
    """
    Options for parsing JSON files.

    Parameters
    ----------
    explicit_schema : Schema, optional (default None)
        Optional explicit schema (no type inference, ignores other fields).
    newlines_in_values : bool, optional (default False)
        Whether objects may be printed across multiple lines (for example
        pretty printed). If false, input must end with an empty line.
    unexpected_field_behavior : str, default "infer"
        How JSON fields outside of explicit_schema (if given) are treated.

        Possible behaviors:

         - "ignore": unexpected JSON fields are ignored
         - "error": error out on unexpected JSON fields
         - "infer": unexpected JSON fields are type-inferred and included in
           the output
    """

    explicit_schema: Schema
    """
    Optional explicit schema (no type inference, ignores other fields)
    """
    newlines_in_values: bool
    """
    Whether newline characters are allowed in JSON values.
    Setting this to True reduces the performance of multi-threaded
    JSON reading.
    """
    unexpected_field_behavior: Literal["ignore", "error", "infer"]
    """
    How JSON fields outside of explicit_schema (if given) are treated.

    Possible behaviors:

        - "ignore": unexpected JSON fields are ignored
        - "error": error out on unexpected JSON fields
        - "infer": unexpected JSON fields are type-inferred and included in
        the output

    Set to "infer" by default.
    """
    def __init__(
        self,
        explicit_schema: Schema | None = None,
        newlines_in_values: bool | None = None,
        unexpected_field_behavior: Literal["ignore", "error", "infer"] = "infer",
    ): ...
    def equals(self, other: ParseOptions) -> bool:
        """
        Parameters
        ----------
        other : pyarrow.json.ParseOptions

        Returns
        -------
        bool
        """

class JSONStreamingReader(RecordBatchReader):
    """An object that reads record batches incrementally from a JSON file.

    Should not be instantiated directly by user code.
    """

def read_json(
    input_file: StrPath | IO[Any],
    read_options: ReadOptions | None = None,
    parse_options: ParseOptions | None = None,
    memory_pool: MemoryPool | None = None,
) -> Table:
    """
    Read a Table from a stream of JSON data.

    Parameters
    ----------
    input_file : str, path or file-like object
        The location of JSON data. Currently only the line-delimited JSON
        format is supported.
    read_options : pyarrow.json.ReadOptions, optional
        Options for the JSON reader (see ReadOptions constructor for defaults).
    parse_options : pyarrow.json.ParseOptions, optional
        Options for the JSON parser
        (see ParseOptions constructor for defaults).
    memory_pool : MemoryPool, optional
        Pool to allocate Table memory from.

    Returns
    -------
    :class:`pyarrow.Table`
        Contents of the JSON file as a in-memory table.
    """

def open_json(
    input_file: StrPath | IO[Any],
    read_options: ReadOptions | None = None,
    parse_options: ParseOptions | None = None,
    memory_pool: MemoryPool | None = None,
) -> JSONStreamingReader:
    """
    Open a streaming reader of JSON data.

    Reading using this function is always single-threaded.

    Parameters
    ----------
    input_file : string, path or file-like object
        The location of JSON data.  If a string or path, and if it ends
        with a recognized compressed file extension (e.g. ".gz" or ".bz2"),
        the data is automatically decompressed when reading.
    read_options : pyarrow.json.ReadOptions, optional
        Options for the JSON reader (see pyarrow.json.ReadOptions constructor
        for defaults)
    parse_options : pyarrow.json.ParseOptions, optional
        Options for the JSON parser
        (see pyarrow.json.ParseOptions constructor for defaults)
    memory_pool : MemoryPool, optional
        Pool to allocate RecordBatch memory from

    Returns
    -------
    :class:`pyarrow.json.JSONStreamingReader`
    """
