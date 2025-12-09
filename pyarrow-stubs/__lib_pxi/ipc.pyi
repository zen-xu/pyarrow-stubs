import enum
import sys

from io import IOBase

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self
from typing import Iterable, Iterator, Literal, Mapping, NamedTuple

import pandas as pd

from pyarrow._stubs_typing import SupportArrowStream, SupportPyBuffer
from pyarrow.lib import MemoryPool, RecordBatch, Schema, Table, Tensor, _Weakrefable

from .io import Buffer, Codec, NativeFile
from .types import DictionaryMemo, KeyValueMetadata

class MetadataVersion(enum.IntEnum):
    V1 = enum.auto()
    V2 = enum.auto()
    V3 = enum.auto()
    V4 = enum.auto()
    V5 = enum.auto()

class WriteStats(NamedTuple):
    """IPC write statistics

    Parameters
    ----------
    num_messages : int
        Number of messages.
    num_record_batches : int
        Number of record batches.
    num_dictionary_batches : int
        Number of dictionary batches.
    num_dictionary_deltas : int
        Delta of dictionaries.
    num_replaced_dictionaries : int
        Number of replaced dictionaries.
    """

    num_messages: int
    num_record_batches: int
    num_dictionary_batches: int
    num_dictionary_deltas: int
    num_replaced_dictionaries: int

class ReadStats(NamedTuple):
    """IPC read statistics

    Parameters
    ----------
    num_messages : int
        Number of messages.
    num_record_batches : int
        Number of record batches.
    num_dictionary_batches : int
        Number of dictionary batches.
    num_dictionary_deltas : int
        Delta of dictionaries.
    num_replaced_dictionaries : int
        Number of replaced dictionaries.
    """

    num_messages: int
    num_record_batches: int
    num_dictionary_batches: int
    num_dictionary_deltas: int
    num_replaced_dictionaries: int

class IpcReadOptions(_Weakrefable):
    """
    Serialization options for reading IPC format.

    Parameters
    ----------
    ensure_native_endian : bool, default True
        Whether to convert incoming data to platform-native endianness.
    use_threads : bool
        Whether to use the global CPU thread pool to parallelize any
        computational tasks like decompression
    included_fields : list
        If empty (the default), return all deserialized fields.
        If non-empty, the values are the indices of fields to read on
        the top-level schema
    """

    ensure_native_endian: bool
    use_threads: bool
    included_fields: list[int]
    def __init__(
        self,
        *,
        ensure_native_endian: bool = True,
        use_threads: bool = True,
        included_fields: list[int] | None = None,
    ) -> None: ...

class IpcWriteOptions(_Weakrefable):
    """
    Serialization options for the IPC format.

    Parameters
    ----------
    metadata_version : MetadataVersion, default MetadataVersion.V5
        The metadata version to write.  V5 is the current and latest,
        V4 is the pre-1.0 metadata version (with incompatible Union layout).
    allow_64bit : bool, default False
        If true, allow field lengths that don't fit in a signed 32-bit int.
    use_legacy_format : bool, default False
        Whether to use the pre-Arrow 0.15 IPC format.
    compression : str, Codec, or None
        compression codec to use for record batch buffers.
        If None then batch buffers will be uncompressed.
        Must be "lz4", "zstd" or None.
        To specify a compression_level use `pyarrow.Codec`
    use_threads : bool
        Whether to use the global CPU thread pool to parallelize any
        computational tasks like compression.
    emit_dictionary_deltas : bool
        Whether to emit dictionary deltas.  Default is false for maximum
        stream compatibility.
    unify_dictionaries : bool
        If true then calls to write_table will attempt to unify dictionaries
        across all batches in the table.  This can help avoid the need for
        replacement dictionaries (which the file format does not support)
        but requires computing the unified dictionary and then remapping
        the indices arrays.

        This parameter is ignored when writing to the IPC stream format as
        the IPC stream format can support replacement dictionaries.
    """

    metadata_version: MetadataVersion
    allow_64bit: bool
    use_legacy_format: bool
    compression: Codec | Literal["lz4", "zstd"] | None
    use_threads: bool
    emit_dictionary_deltas: bool
    unify_dictionaries: bool
    def __init__(
        self,
        *,
        metadata_version: MetadataVersion = MetadataVersion.V5,
        allow_64bit: bool = False,
        use_legacy_format: bool = False,
        compression: Codec | Literal["lz4", "zstd"] | None = None,
        use_threads: bool = True,
        emit_dictionary_deltas: bool = False,
        unify_dictionaries: bool = False,
    ) -> None: ...

class Message(_Weakrefable):
    """
    Container for an Arrow IPC message with metadata and optional body
    """

    @property
    def type(self) -> str: ...
    @property
    def metadata(self) -> Buffer: ...
    @property
    def metadata_version(self) -> MetadataVersion: ...
    @property
    def body(self) -> Buffer | None: ...
    def equals(self, other: Message) -> bool: ...
    def serialize_to(
        self, sink: NativeFile, alignment: int = 8, memory_pool: MemoryPool | None = None
    ):
        """
        Write message to generic OutputStream

        Parameters
        ----------
        sink : NativeFile
        alignment : int, default 8
            Byte alignment for metadata and body
        memory_pool : MemoryPool, default None
            Uses default memory pool if not specified
        """
    def serialize(self, alignment: int = 8, memory_pool: MemoryPool | None = None) -> Buffer:
        """
        Write message as encapsulated IPC message

        Parameters
        ----------
        alignment : int, default 8
            Byte alignment for metadata and body
        memory_pool : MemoryPool, default None
            Uses default memory pool if not specified

        Returns
        -------
        serialized : Buffer
        """

class MessageReader(_Weakrefable):
    """
    Interface for reading Message objects from some source (like an
    InputStream)
    """
    @classmethod
    def open_stream(cls, source: bytes | NativeFile | IOBase | SupportPyBuffer) -> Self:
        """
        Open stream from source, if you want to use memory map use
        MemoryMappedFile as source.

        Parameters
        ----------
        source : bytes/buffer-like, pyarrow.NativeFile, or file-like Python object
            A readable source, like an InputStream
        """
    def __iter__(self) -> Self: ...
    def read_next_message(self) -> Message:
        """
        Read next Message from the stream.

        Raises
        ------
        StopIteration
            At end of stream
        """
    __next__ = read_next_message

# ----------------------------------------------------------------------
# File and stream readers and writers

class _CRecordBatchWriter(_Weakrefable):
    """The base RecordBatchWriter wrapper.

    Provides common implementations of convenience methods. Should not
    be instantiated directly by user code.
    """
    def write(self, table_or_batch: Table | RecordBatch):
        """
        Write RecordBatch or Table to stream.

        Parameters
        ----------
        table_or_batch : {RecordBatch, Table}
        """
    def write_batch(
        self,
        batch: RecordBatch,
        custom_metadata: Mapping[bytes, bytes] | KeyValueMetadata | None = None,
    ):
        """
        Write RecordBatch to stream.

        Parameters
        ----------
        batch : RecordBatch
        custom_metadata : mapping or KeyValueMetadata
            Keys and values must be string-like / coercible to bytes
        """
    def write_table(self, table: Table, max_chunksize: int | None = None) -> None:
        """
        Write Table to stream in (contiguous) RecordBatch objects.

        Parameters
        ----------
        table : Table
        max_chunksize : int, default None
            Maximum number of rows for RecordBatch chunks. Individual chunks may
            be smaller depending on the chunk layout of individual columns.
        """
    def close(self) -> None:
        """
        Close stream and write end-of-stream 0 marker.
        """
    def __enter__(self) -> Self: ...
    def __exit__(self, exc_type, exc_val, exc_tb): ...
    @property
    def stats(self) -> WriteStats:
        """
        Current IPC write statistics.
        """

class _RecordBatchStreamWriter(_CRecordBatchWriter):
    def __dealloc__(self) -> None: ...
    def _open(self, sink, schema: Schema, options: IpcWriteOptions = IpcWriteOptions()): ...

class _ReadPandasMixin:
    def read_pandas(self, **options) -> pd.DataFrame:
        """
        Read contents of stream to a pandas.DataFrame.

        Read all record batches as a pyarrow.Table then convert it to a
        pandas.DataFrame using Table.to_pandas.

        Parameters
        ----------
        **options
            Arguments to forward to :meth:`Table.to_pandas`.

        Returns
        -------
        df : pandas.DataFrame
        """

class RecordBatchReader(_Weakrefable, _ReadPandasMixin):
    """Base class for reading stream of record batches.

    Record batch readers function as iterators of record batches that also
    provide the schema (without the need to get any batches).

    Warnings
    --------
    Do not call this class's constructor directly, use one of the
    ``RecordBatchReader.from_*`` functions instead.

    Notes
    -----
    To import and export using the Arrow C stream interface, use the
    ``_import_from_c`` and ``_export_to_c`` methods. However, keep in mind this
    interface is intended for expert users.

    Examples
    --------
    >>> import pyarrow as pa
    >>> schema = pa.schema([("x", pa.int64())])
    >>> def iter_record_batches():
    ...     for i in range(2):
    ...         yield pa.RecordBatch.from_arrays([pa.array([1, 2, 3])], schema=schema)
    >>> reader = pa.RecordBatchReader.from_batches(schema, iter_record_batches())
    >>> print(reader.schema)
    x: int64
    >>> for batch in reader:
    ...     print(batch)
    pyarrow.RecordBatch
    x: int64
    ----
    x: [1,2,3]
    pyarrow.RecordBatch
    x: int64
    ----
    x: [1,2,3]
    """

    def __iter__(self) -> Self: ...
    def read_next_batch(self) -> RecordBatch:
        """
        Read next RecordBatch from the stream.

        Raises
        ------
        StopIteration:
            At end of stream.

        Returns
        -------
        RecordBatch
        """
    __next__ = read_next_batch
    @property
    def schema(self) -> Schema:
        """
        Shared schema of the record batches in the stream.

        Returns
        -------
        Schema
        """
    def read_next_batch_with_custom_metadata(self) -> RecordBatchWithMetadata:
        """
        Read next RecordBatch from the stream along with its custom metadata.

        Raises
        ------
        StopIteration:
            At end of stream.

        Returns
        -------
        batch : RecordBatch
        custom_metadata : KeyValueMetadata
        """
    def iter_batches_with_custom_metadata(
        self,
    ) -> Iterator[RecordBatchWithMetadata]:
        """
        Iterate over record batches from the stream along with their custom
        metadata.

        Yields
        ------
        RecordBatchWithMetadata
        """
    def read_all(self) -> Table:
        """
        Read all record batches as a pyarrow.Table.

        Returns
        -------
        Table
        """
    def close(self) -> None:
        """
        Release any resources associated with the reader.
        """
    def __enter__(self) -> Self: ...
    def __exit__(self, exc_type, exc_val, exc_tb): ...
    def cast(self, target_schema: Schema) -> Self:
        """
        Wrap this reader with one that casts each batch lazily as it is pulled.
        Currently only a safe cast to target_schema is implemented.

        Parameters
        ----------
        target_schema : Schema
            Schema to cast to, the names and order of fields must match.

        Returns
        -------
        RecordBatchReader
        """
    def _export_to_c(self, out_ptr: int) -> None:
        """
        Export to a C ArrowArrayStream struct, given its pointer.

        Parameters
        ----------
        out_ptr: int
            The raw pointer to a C ArrowArrayStream struct.

        Be careful: if you don't pass the ArrowArrayStream struct to a
        consumer, array memory will leak.  This is a low-level function
        intended for expert users.
        """
    @classmethod
    def _import_from_c(cls, in_ptr: int) -> Self:
        """
        Import RecordBatchReader from a C ArrowArrayStream struct,
        given its pointer.

        Parameters
        ----------
        in_ptr: int
            The raw pointer to a C ArrowArrayStream struct.

        This is a low-level function intended for expert users.
        """
    def __arrow_c_stream__(self, requested_schema=None):
        """
        Export to a C ArrowArrayStream PyCapsule.

        Parameters
        ----------
        requested_schema : PyCapsule, default None
            The schema to which the stream should be casted, passed as a
            PyCapsule containing a C ArrowSchema representation of the
            requested schema.

        Returns
        -------
        PyCapsule
            A capsule containing a C ArrowArrayStream struct.
        """
    @classmethod
    def _import_from_c_capsule(cls, stream) -> Self:
        """
        Import RecordBatchReader from a C ArrowArrayStream PyCapsule.

        Parameters
        ----------
        stream: PyCapsule
            A capsule containing a C ArrowArrayStream PyCapsule.

        Returns
        -------
        RecordBatchReader
        """
    @classmethod
    def from_stream(cls, data: SupportArrowStream, schema: Schema | None = None) -> Self:
        """
        Create RecordBatchReader from a Arrow-compatible stream object.

        This accepts objects implementing the Arrow PyCapsule Protocol for
        streams, i.e. objects that have a ``__arrow_c_stream__`` method.

        Parameters
        ----------
        data : Arrow-compatible stream object
            Any object that implements the Arrow PyCapsule Protocol for
            streams.
        schema : Schema, default None
            The schema to which the stream should be casted, if supported
            by the stream object.

        Returns
        -------
        RecordBatchReader
        """
    @classmethod
    def from_batches(cls, schema: Schema, batches: Iterable[RecordBatch]) -> Self:
        """
        Create RecordBatchReader from an iterable of batches.

        Parameters
        ----------
        schema : Schema
            The shared schema of the record batches
        batches : Iterable[RecordBatch]
            The batches that this reader will return.

        Returns
        -------
        reader : RecordBatchReader
        """

class _RecordBatchStreamReader(RecordBatchReader):
    @property
    def stats(self) -> ReadStats:
        """
        Current IPC read statistics.
        """

class _RecordBatchFileWriter(_RecordBatchStreamWriter): ...

class RecordBatchWithMetadata(NamedTuple):
    """RecordBatch with its custom metadata

    Parameters
    ----------
    batch : RecordBatch
    custom_metadata : KeyValueMetadata
    """

    batch: RecordBatch
    custom_metadata: KeyValueMetadata

class _RecordBatchFileReader(_Weakrefable, _ReadPandasMixin):
    @property
    def num_record_batches(self) -> int:
        """
        The number of record batches in the IPC file.
        """
    def get_batch(self, i: int) -> RecordBatch:
        """
        Read the record batch with the given index.

        Parameters
        ----------
        i : int
            The index of the record batch in the IPC file.

        Returns
        -------
        batch : RecordBatch
        """
    get_record_batch = get_batch
    def get_batch_with_custom_metadata(self, i: int) -> RecordBatchWithMetadata:
        """
        Read the record batch with the given index along with
        its custom metadata

        Parameters
        ----------
        i : int
            The index of the record batch in the IPC file.

        Returns
        -------
        batch : RecordBatch
        custom_metadata : KeyValueMetadata
        """
    def read_all(self) -> Table:
        """
        Read all record batches as a pyarrow.Table
        """
    def __enter__(self) -> Self: ...
    def __exit__(self, exc_type, exc_val, exc_tb): ...
    @property
    def schema(self) -> Schema: ...
    @property
    def stats(self) -> ReadStats: ...

def get_tensor_size(tensor: Tensor) -> int:
    """
    Return total size of serialized Tensor including metadata and padding.

    Parameters
    ----------
    tensor : Tensor
        The tensor for which we want to known the size.
    """

def get_record_batch_size(batch: RecordBatch) -> int:
    """
    Return total size of serialized RecordBatch including metadata and padding.

    Parameters
    ----------
    batch : RecordBatch
        The recordbatch for which we want to know the size.
    """

def write_tensor(tensor: Tensor, dest: NativeFile) -> int:
    """
    Write pyarrow.Tensor to pyarrow.NativeFile object its current position.

    Parameters
    ----------
    tensor : pyarrow.Tensor
    dest : pyarrow.NativeFile

    Returns
    -------
    bytes_written : int
        Total number of bytes written to the file
    """

def read_tensor(source: NativeFile) -> Tensor:
    """Read pyarrow.Tensor from pyarrow.NativeFile object from current
    position. If the file source supports zero copy (e.g. a memory map), then
    this operation does not allocate any memory. This function not assume that
    the stream is aligned

    Parameters
    ----------
    source : pyarrow.NativeFile

    Returns
    -------
    tensor : Tensor

    """

def read_message(source: NativeFile | IOBase | SupportPyBuffer) -> Message:
    """
    Read length-prefixed message from file or buffer-like object

    Parameters
    ----------
    source : pyarrow.NativeFile, file-like object, or buffer-like object

    Returns
    -------
    message : Message
    """

def read_schema(obj: Buffer | Message, dictionary_memo: DictionaryMemo | None = None) -> Schema:
    """
    Read Schema from message or buffer

    Parameters
    ----------
    obj : buffer or Message
    dictionary_memo : DictionaryMemo, optional
        Needed to be able to reconstruct dictionary-encoded fields
        with read_record_batch

    Returns
    -------
    schema : Schema
    """

def read_record_batch(
    obj: Message | SupportPyBuffer, schema: Schema, dictionary_memo: DictionaryMemo | None = None
) -> RecordBatch:
    """
    Read RecordBatch from message, given a known schema. If reading data from a
    complete IPC stream, use ipc.open_stream instead

    Parameters
    ----------
    obj : Message or Buffer-like
    schema : Schema
    dictionary_memo : DictionaryMemo, optional
        If message contains dictionaries, must pass a populated
        DictionaryMemo

    Returns
    -------
    batch : RecordBatch
    """

__all__ = [
    "MetadataVersion",
    "WriteStats",
    "ReadStats",
    "IpcReadOptions",
    "IpcWriteOptions",
    "Message",
    "MessageReader",
    "_CRecordBatchWriter",
    "_RecordBatchStreamWriter",
    "_ReadPandasMixin",
    "RecordBatchReader",
    "_RecordBatchStreamReader",
    "_RecordBatchFileWriter",
    "RecordBatchWithMetadata",
    "_RecordBatchFileReader",
    "get_tensor_size",
    "get_record_batch_size",
    "write_tensor",
    "read_tensor",
    "read_message",
    "read_schema",
    "read_record_batch",
]
