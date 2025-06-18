import sys

from collections.abc import Callable
from io import IOBase

from _typeshed import StrPath

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self
if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

from typing import Any, Literal, SupportsIndex, overload

from pyarrow._stubs_typing import Compression, SupportPyBuffer
from pyarrow.lib import MemoryPool, _Weakrefable

from .device import Device, DeviceAllocationType, MemoryManager
from .types import KeyValueMetadata

def have_libhdfs() -> bool:
    """
    Return true if HDFS (HadoopFileSystem) library is set up correctly.
    """

def io_thread_count() -> int:
    """
    Return the number of threads to use for I/O operations.

    Many operations, such as scanning a dataset, will implicitly make
    use of this pool. The number of threads is set to a fixed value at
    startup. It can be modified at runtime by calling
    :func:`set_io_thread_count()`.

    See Also
    --------
    set_io_thread_count : Modify the size of this pool.
    cpu_count : The analogous function for the CPU thread pool.
    """

def set_io_thread_count(count: int) -> None:
    """
    Set the number of threads to use for I/O operations.

    Many operations, such as scanning a dataset, will implicitly make
    use of this pool.

    Parameters
    ----------
    count : int
        The max number of threads that may be used for I/O.
        Must be positive.

    See Also
    --------
    io_thread_count : Get the size of this pool.
    set_cpu_count : The analogous function for the CPU thread pool.
    """

Mode: TypeAlias = Literal["rb", "wb", "rb+", "ab"]

class NativeFile(_Weakrefable):
    """
    The base class for all Arrow streams.

    Streams are either readable, writable, or both.
    They optionally support seeking.

    While this class exposes methods to read or write data from Python, the
    primary intent of using a Arrow stream is to pass it to other Arrow
    facilities that will make use of it, such as Arrow IPC routines.

    Be aware that there are subtle differences with regular Python files,
    e.g. destroying a writable Arrow stream without closing it explicitly
    will not flush any pending data.
    """

    _default_chunk_size: int

    def __enter__(self) -> Self: ...
    def __exit__(self, *args) -> None: ...
    @property
    def mode(self) -> Mode:
        """
        The file mode. Currently instances of NativeFile may support:

        * rb: binary read
        * wb: binary write
        * rb+: binary read and write
        * ab: binary append
        """
    def readable(self) -> bool: ...
    def seekable(self) -> bool: ...
    def isatty(self) -> bool: ...
    def fileno(self) -> int: ...
    @property
    def closed(self) -> bool: ...
    def close(self) -> None: ...
    def size(self) -> int:
        """
        Return file size
        """
    def metadata(self) -> KeyValueMetadata:
        """
        Return file metadata
        """
    def tell(self) -> int:
        """
        Return current stream position
        """
    def seek(self, position: int, whence: int = 0) -> int:
        """
        Change current file stream position

        Parameters
        ----------
        position : int
            Byte offset, interpreted relative to value of whence argument
        whence : int, default 0
            Point of reference for seek offset

        Notes
        -----
        Values of whence:
        * 0 -- start of stream (the default); offset should be zero or positive
        * 1 -- current stream position; offset may be negative
        * 2 -- end of stream; offset is usually negative

        Returns
        -------
        int
            The new absolute stream position.
        """
    def flush(self) -> None:
        """
        Flush the stream, if applicable.

        An error is raised if stream is not writable.
        """
    def write(self, data: bytes | SupportPyBuffer) -> int:
        """
        Write data to the file.

        Parameters
        ----------
        data : bytes-like object or exporter of buffer protocol

        Returns
        -------
        int
            nbytes: number of bytes written
        """
    def read(self, nbytes: int | None = None) -> bytes:
        """
        Read and return up to n bytes.

        If *nbytes* is None, then the entire remaining file contents are read.

        Parameters
        ----------
        nbytes : int, default None

        Returns
        -------
        data : bytes
        """
    def get_stream(self, file_offset: int, nbytes: int) -> Self:
        """
        Return an input stream that reads a file segment independent of the
        state of the file.

        Allows reading portions of a random access file as an input stream
        without interfering with each other.

        Parameters
        ----------
        file_offset : int
        nbytes : int

        Returns
        -------
        stream : NativeFile
        """
    def read_at(self) -> bytes:
        """
        Read indicated number of bytes at offset from the file

        Parameters
        ----------
        nbytes : int
        offset : int

        Returns
        -------
        data : bytes
        """
    def read1(self) -> bytes:
        """Read and return up to n bytes.

        Unlike read(), if *nbytes* is None then a chunk is read, not the
        entire file.

        Parameters
        ----------
        nbytes : int, default None
            The maximum number of bytes to read.

        Returns
        -------
        data : bytes
        """
    def readall(self) -> bytes: ...
    def readinto(self, b: SupportPyBuffer) -> int:
        """
        Read into the supplied buffer

        Parameters
        ----------
        b : buffer-like object
            A writable buffer object (such as a bytearray).

        Returns
        -------
        written : int
            number of bytes written
        """

    def readline(self, size: int | None = None) -> bytes:
        """Read and return a line of bytes from the file.

        If size is specified, read at most size bytes.

        Line terminator is always b"\\n".

        Parameters
        ----------
        size : int
            maximum number of bytes read
        """
    def readlines(self, hint: int | None = None) -> list[bytes]:
        """Read lines of the file

        Parameters
        ----------
        hint : int
            maximum number of bytes read until we stop
        """
    def __iter__(self) -> Self: ...
    def __next__(self) -> bytes: ...
    def read_buffer(self, nbytes: int | None = None) -> Buffer:
        """
        Read from buffer.

        Parameters
        ----------
        nbytes : int, optional
            maximum number of bytes read
        """
    def truncate(self) -> None: ...
    def writelines(self, lines: list[bytes]):
        """
        Write lines to the file.

        Parameters
        ----------
        lines : iterable
            Iterable of bytes-like objects or exporters of buffer protocol
        """
    def download(self, stream_or_path: StrPath | IOBase, buffer_size: int | None = None) -> None:
        """
        Read this file completely to a local path or destination stream.

        This method first seeks to the beginning of the file.

        Parameters
        ----------
        stream_or_path : str or file-like object
            If a string, a local file path to write to; otherwise,
            should be a writable stream.
        buffer_size : int, optional
            The buffer size to use for data transfers.
        """
    def upload(self, stream: IOBase, buffer_size: int | None) -> None:
        """
        Write from a source stream to this file.

        Parameters
        ----------
        stream : file-like object
            Source stream to pipe to this file.
        buffer_size : int, optional
            The buffer size to use for data transfers.
        """

# ----------------------------------------------------------------------
# Python file-like objects

class PythonFile(NativeFile):
    """
    A stream backed by a Python file object.

    This class allows using Python file objects with arbitrary Arrow
    functions, including functions written in another language than Python.

    As a downside, there is a non-zero redirection cost in translating
    Arrow stream calls to Python method calls.  Furthermore, Python's
    Global Interpreter Lock may limit parallelism in some situations.

    Examples
    --------
    >>> import io
    >>> import pyarrow as pa
    >>> pa.PythonFile(io.BytesIO())
    <pyarrow.PythonFile closed=False own_file=False is_seekable=False is_writable=True is_readable=False>

    Create a stream for writing:

    >>> buf = io.BytesIO()
    >>> f = pa.PythonFile(buf, mode="w")
    >>> f.writable()
    True
    >>> f.write(b"PythonFile")
    10
    >>> buf.getvalue()
    b'PythonFile'
    >>> f.close()
    >>> f
    <pyarrow.PythonFile closed=True own_file=False is_seekable=False is_writable=True is_readable=False>

    Create a stream for reading:

    >>> buf = io.BytesIO(b"PythonFile")
    >>> f = pa.PythonFile(buf, mode="r")
    >>> f.mode
    'rb'
    >>> f.read()
    b'PythonFile'
    >>> f
    <pyarrow.PythonFile closed=False own_file=False is_seekable=True is_writable=False is_readable=True>
    >>> f.close()
    >>> f
    <pyarrow.PythonFile closed=True own_file=False is_seekable=True is_writable=False is_readable=True>
    """
    def __init__(self, handle: IOBase, mode: Literal["r", "w"] | None = None) -> None: ...
    def truncate(self, pos: int | None = None) -> None:
        """
        Parameters
        ----------
        pos : int, optional
        """

class MemoryMappedFile(NativeFile):
    """
    A stream that represents a memory-mapped file.

    Supports 'r', 'r+', 'w' modes.

    Examples
    --------
    Create a new file with memory map:

    >>> import pyarrow as pa
    >>> mmap = pa.create_memory_map("example_mmap.dat", 10)
    >>> mmap
    <pyarrow.MemoryMappedFile closed=False own_file=False is_seekable=True is_writable=True is_readable=True>
    >>> mmap.close()

    Open an existing file with memory map:

    >>> with pa.memory_map("example_mmap.dat") as mmap:
    ...     mmap
    <pyarrow.MemoryMappedFile closed=False own_file=False is_seekable=True is_writable=False is_readable=True>
    """
    @classmethod
    def create(cls, path: str, size: int) -> Self:
        """
        Create a MemoryMappedFile

        Parameters
        ----------
        path : str
            Where to create the file.
        size : int
            Size of the memory mapped file.
        """
    def _open(self, path: str, mode: Literal["r", "rb", "w", "wb", "r+", "r+b", "rb+"] = "r"): ...
    def resize(self, new_size: int) -> None:
        """
        Resize the map and underlying file.

        Parameters
        ----------
        new_size : new size in bytes
        """

def memory_map(
    path: str, mode: Literal["r", "rb", "w", "wb", "r+", "r+b", "rb+"] = "r"
) -> MemoryMappedFile:
    """
    Open memory map at file path. Size of the memory map cannot change.

    Parameters
    ----------
    path : str
    mode : {'r', 'r+', 'w'}, default 'r'
        Whether the file is opened for reading ('r'), writing ('w')
        or both ('r+').

    Returns
    -------
    mmap : MemoryMappedFile

    Examples
    --------
    Reading from a memory map without any memory allocation or copying:

    >>> import pyarrow as pa
    >>> with pa.output_stream("example_mmap.txt") as stream:
    ...     stream.write(b"Constructing a buffer referencing the mapped memory")
    51
    >>> with pa.memory_map("example_mmap.txt") as mmap:
    ...     mmap.read_at(6, 45)
    b'memory'
    """

create_memory_map = MemoryMappedFile.create

class OSFile(NativeFile):
    """
    A stream backed by a regular file descriptor.

    Examples
    --------
    Create a new file to write to:

    >>> import pyarrow as pa
    >>> with pa.OSFile("example_osfile.arrow", mode="w") as f:
    ...     f.writable()
    ...     f.write(b"OSFile")
    ...     f.seekable()
    True
    6
    False

    Open the file to read:

    >>> with pa.OSFile("example_osfile.arrow", mode="r") as f:
    ...     f.mode
    ...     f.read()
    'rb'
    b'OSFile'

    Open the file to append:

    >>> with pa.OSFile("example_osfile.arrow", mode="ab") as f:
    ...     f.mode
    ...     f.write(b" is super!")
    'ab'
    10
    >>> with pa.OSFile("example_osfile.arrow") as f:
    ...     f.read()
    b'OSFile is super!'

    Inspect created OSFile:

    >>> pa.OSFile("example_osfile.arrow")
    <pyarrow.OSFile closed=False own_file=False is_seekable=True is_writable=False is_readable=True>
    """
    def __init__(
        self,
        path: str,
        mode: Literal["r", "rb", "w", "wb", "a", "ab"],
        memory_pool: MemoryPool | None = None,
    ) -> None: ...

class FixedSizeBufferWriter(NativeFile):
    """
    A stream writing to a Arrow buffer.

    Examples
    --------
    Create a stream to write to ``pyarrow.Buffer``:

    >>> import pyarrow as pa
    >>> buf = pa.allocate_buffer(5)
    >>> with pa.output_stream(buf) as stream:
    ...     stream.write(b"abcde")
    ...     stream
    5
    <pyarrow.FixedSizeBufferWriter closed=False own_file=False is_seekable=False is_writable=True is_readable=False>

    Inspect the buffer:

    >>> buf.to_pybytes()
    b'abcde'
    >>> buf
    <pyarrow.Buffer address=... size=5 is_cpu=True is_mutable=True>
    """
    def __init__(self, buffer: Buffer) -> None: ...
    def set_memcopy_threads(self, num_threads: int) -> None: ...
    def set_memcopy_blocksize(self, blocksize: int) -> None: ...
    def set_memcopy_threshold(self, threshold: int) -> None: ...

# ----------------------------------------------------------------------
# Arrow buffers

class Buffer(_Weakrefable):
    """
    The base class for all Arrow buffers.

    A buffer represents a contiguous memory area.  Many buffers will own
    their memory, though not all of them do.
    """
    def __len__(self) -> int: ...
    def _assert_cpu(self) -> None: ...
    @property
    def size(self) -> int:
        """
        The buffer size in bytes.
        """
    @property
    def address(self) -> int:
        """
        The buffer's address, as an integer.

        The returned address may point to CPU or device memory.
        Use `is_cpu()` to disambiguate.
        """
    def hex(self) -> bytes:
        """
        Compute hexadecimal representation of the buffer.

        Returns
        -------
        : bytes
        """
    @property
    def is_mutable(self) -> bool:
        """
        Whether the buffer is mutable.
        """
    @property
    def is_cpu(self) -> bool:
        """
        Whether the buffer is CPU-accessible.
        """
    @property
    def device(self) -> Device:
        """
        The device where the buffer resides.

        Returns
        -------
        Device
        """
    @property
    def memory_manager(self) -> MemoryManager:
        """
        The memory manager associated with the buffer.

        Returns
        -------
        MemoryManager
        """
    @property
    def device_type(self) -> DeviceAllocationType:
        """
        The device type where the buffer resides.

        Returns
        -------
        DeviceAllocationType
        """
    @property
    def parent(self) -> Buffer | None: ...
    @overload
    def __getitem__(self, key: slice) -> Self: ...
    @overload
    def __getitem__(self, key: int) -> int: ...
    def slice(self, offset: int = 0, length: int | None = None) -> Self:
        """
        Slice this buffer.  Memory is not copied.

        You can also use the Python slice notation ``buffer[start:stop]``.

        Parameters
        ----------
        offset : int, default 0
            Offset from start of buffer to slice.
        length : int, default None
            Length of slice (default is until end of Buffer starting from
            offset).

        Returns
        -------
        sliced : Buffer
            A logical view over this buffer.
        """
    def equals(self, other: Self) -> bool:
        """
        Determine if two buffers contain exactly the same data.

        Parameters
        ----------
        other : Buffer

        Returns
        -------
        are_equal : bool
            True if buffer contents and size are equal
        """
    def __reduce_ex__(self, protocol: SupportsIndex) -> str | tuple[Any, ...]: ...
    def to_pybytes(self) -> bytes:
        """
        Return this buffer as a Python bytes object. Memory is copied.
        """
    def __buffer__(self, flags: int, /) -> memoryview: ...

class ResizableBuffer(Buffer):
    """
    A base class for buffers that can be resized.
    """

    def resize(self, new_size: int, shrink_to_fit: bool = False) -> None:
        """
        Resize buffer to indicated size.

        Parameters
        ----------
        new_size : int
            New size of buffer (padding may be added internally).
        shrink_to_fit : bool, default False
            If this is true, the buffer is shrunk when new_size is less
            than the current size.
            If this is false, the buffer is never shrunk.
        """

@overload
def allocate_buffer(size: int, memory_pool: MemoryPool | None = None) -> Buffer: ...
@overload
def allocate_buffer(
    size: int, memory_pool: MemoryPool | None, resizable: Literal[False]
) -> Buffer: ...
@overload
def allocate_buffer(
    size: int, memory_pool: MemoryPool | None, resizable: Literal[True]
) -> ResizableBuffer: ...
def allocate_buffer(*args, **kwargs):
    """
    Allocate a mutable buffer.

    Parameters
    ----------
    size : int
        Number of bytes to allocate (plus internal padding)
    memory_pool : MemoryPool, optional
        The pool to allocate memory from.
        If not given, the default memory pool is used.
    resizable : bool, default False
        If true, the returned buffer is resizable.

    Returns
    -------
    buffer : Buffer or ResizableBuffer
    """

# ----------------------------------------------------------------------
# Arrow Stream
class BufferOutputStream(NativeFile):
    """
    An output stream that writes to a resizable buffer.

    The buffer is produced as a result when ``getvalue()`` is called.

    Examples
    --------
    Create an output stream, write data to it and finalize it with
    ``getvalue()``:

    >>> import pyarrow as pa
    >>> f = pa.BufferOutputStream()
    >>> f.write(b"pyarrow.Buffer")
    14
    >>> f.closed
    False
    >>> f.getvalue()
    <pyarrow.Buffer address=... size=14 is_cpu=True is_mutable=True>
    >>> f.closed
    True
    """
    def __init__(self, memory_pool: MemoryPool | None = None) -> None: ...
    def getvalue(self) -> Buffer:
        """
        Finalize output stream and return result as pyarrow.Buffer.

        Returns
        -------
        value : Buffer
        """

class MockOutputStream(NativeFile): ...

class BufferReader(NativeFile):
    """
    Zero-copy reader from objects convertible to Arrow buffer.

    Parameters
    ----------
    obj : Python bytes or pyarrow.Buffer

    Examples
    --------
    Create an Arrow input stream and inspect it:

    >>> import pyarrow as pa
    >>> data = b"reader data"
    >>> buf = memoryview(data)
    >>> with pa.input_stream(buf) as stream:
    ...     stream.size()
    ...     stream.read(6)
    ...     stream.seek(7)
    ...     stream.read(15)
    11
    b'reader'
    7
    b'data'
    """
    def __init__(self, obj) -> None: ...

class CompressedInputStream(NativeFile):
    """
    An input stream wrapper which decompresses data on the fly.

    Parameters
    ----------
    stream : string, path, pyarrow.NativeFile, or file-like object
        Input stream object to wrap with the compression.
    compression : str
        The compression type ("bz2", "brotli", "gzip", "lz4" or "zstd").

    Examples
    --------
    Create an output stream which compresses the data:

    >>> import pyarrow as pa
    >>> data = b"Compressed stream"
    >>> raw = pa.BufferOutputStream()
    >>> with pa.CompressedOutputStream(raw, "gzip") as compressed:
    ...     compressed.write(data)
    17

    Create an input stream with decompression referencing the
    buffer with compressed data:

    >>> cdata = raw.getvalue()
    >>> with pa.input_stream(cdata, compression="gzip") as compressed:
    ...     compressed.read()
    b'Compressed stream'

    which actually translates to the use of ``BufferReader``and
    ``CompressedInputStream``:

    >>> raw = pa.BufferReader(cdata)
    >>> with pa.CompressedInputStream(raw, "gzip") as compressed:
    ...     compressed.read()
    b'Compressed stream'
    """

    def __init__(
        self,
        stream: StrPath | NativeFile | IOBase,
        compression: Literal["bz2", "brotli", "gzip", "lz4", "zstd"],
    ) -> None: ...

class CompressedOutputStream(NativeFile):
    """
    An output stream wrapper which compresses data on the fly.

    Parameters
    ----------
    stream : string, path, pyarrow.NativeFile, or file-like object
        Input stream object to wrap with the compression.
    compression : str
        The compression type ("bz2", "brotli", "gzip", "lz4" or "zstd").

    Examples
    --------
    Create an output stream which compresses the data:

    >>> import pyarrow as pa
    >>> data = b"Compressed stream"
    >>> raw = pa.BufferOutputStream()
    >>> with pa.CompressedOutputStream(raw, "gzip") as compressed:
    ...     compressed.write(data)
    17
    """
    def __init__(
        self,
        stream: StrPath | NativeFile | IOBase,
        compression: Literal["bz2", "brotli", "gzip", "lz4", "zstd"],
    ) -> None: ...

class BufferedInputStream(NativeFile):
    """
    An input stream that performs buffered reads from
    an unbuffered input stream, which can mitigate the overhead
    of many small reads in some cases.

    Parameters
    ----------
    stream : NativeFile
        The input stream to wrap with the buffer
    buffer_size : int
        Size of the temporary read buffer.
    memory_pool : MemoryPool
        The memory pool used to allocate the buffer.
    """
    def __init__(
        self, stream: NativeFile, buffer_size: int, memory_pool: MemoryPool | None = None
    ) -> None: ...
    def detach(self) -> NativeFile:
        """
        Release the raw InputStream.
        Further operations on this stream are invalid.

        Returns
        -------
        raw : NativeFile
            The underlying raw input stream
        """

class BufferedOutputStream(NativeFile):
    """
    An output stream that performs buffered reads from
    an unbuffered output stream, which can mitigate the overhead
    of many small writes in some cases.

    Parameters
    ----------
    stream : NativeFile
        The writable output stream to wrap with the buffer
    buffer_size : int
        Size of the buffer that should be added.
    memory_pool : MemoryPool
        The memory pool used to allocate the buffer.
    """
    def __init__(
        self, stream: NativeFile, buffer_size: int, memory_pool: MemoryPool | None = None
    ) -> None: ...
    def detach(self) -> NativeFile:
        """
        Flush any buffered writes and release the raw OutputStream.
        Further operations on this stream are invalid.

        Returns
        -------
        raw : NativeFile
            The underlying raw output stream.
        """

class TransformInputStream(NativeFile):
    """
    Transform an input stream.

    Parameters
    ----------
    stream : NativeFile
        The stream to transform.
    transform_func : callable
        The transformation to apply.
    """
    def __init__(self, stream: NativeFile, transform_func: Callable[[Buffer], Any]) -> None: ...

class Transcoder:
    def __init__(self, decoder, encoder) -> None: ...
    def __call__(self, buf: Buffer): ...

def transcoding_input_stream(
    stream: NativeFile, src_encoding: str, dest_encoding: str
) -> TransformInputStream:
    """
    Add a transcoding transformation to the stream.
    Incoming data will be decoded according to ``src_encoding`` and
    then re-encoded according to ``dest_encoding``.

    Parameters
    ----------
    stream : NativeFile
        The stream to which the transformation should be applied.
    src_encoding : str
        The codec to use when reading data.
    dest_encoding : str
        The codec to use for emitted data.
    """

def py_buffer(obj: SupportPyBuffer) -> Buffer:
    """
    Construct an Arrow buffer from a Python bytes-like or buffer-like object

    Parameters
    ----------
    obj : object
        the object from which the buffer should be constructed.
    """

def foreign_buffer(address: int, size: int, base: Any | None = None) -> Buffer:
    """
    Construct an Arrow buffer with the given *address* and *size*.

    The buffer will be optionally backed by the Python *base* object, if given.
    The *base* object will be kept alive as long as this buffer is alive,
    including across language boundaries (for example if the buffer is
    referenced by C++ code).

    Parameters
    ----------
    address : int
        The starting address of the buffer. The address can
        refer to both device or host memory but it must be
        accessible from device after mapping it with
        `get_device_address` method.
    size : int
        The size of device buffer in bytes.
    base : {None, object}
        Object that owns the referenced memory.
    """

def as_buffer(o: Buffer | SupportPyBuffer) -> Buffer: ...

# ---------------------------------------------------------------------

class CacheOptions(_Weakrefable):
    """
    Cache options for a pre-buffered fragment scan.

    Parameters
    ----------
    hole_size_limit : int, default 8KiB
        The maximum distance in bytes between two consecutive ranges; beyond
        this value, ranges are not combined.
    range_size_limit : int, default 32MiB
        The maximum size in bytes of a combined range; if combining two
        consecutive ranges would produce a range of a size greater than this,
        they are not combined
    lazy : bool, default True
        lazy = false: request all byte ranges when PreBuffer or WillNeed is called.
        lazy = True, prefetch_limit = 0: request merged byte ranges only after the reader
        needs them.
        lazy = True, prefetch_limit = k: prefetch up to k merged byte ranges ahead of the
        range that is currently being read.
    prefetch_limit : int, default 0
        The maximum number of ranges to be prefetched. This is only used for
        lazy cache to asynchronously read some ranges after reading the target
        range.
    """

    hole_size_limit: int
    range_size_limit: int
    lazy: bool
    prefetch_limit: int
    def __init__(
        self,
        *,
        hole_size_limit: int | None = None,
        range_size_limit: int | None = None,
        lazy: bool = True,
        prefetch_limit: int = 0,
    ) -> None: ...
    @classmethod
    def from_network_metrics(
        cls,
        time_to_first_byte_millis: int,
        transfer_bandwidth_mib_per_sec: int,
        ideal_bandwidth_utilization_frac: float = 0.9,
        max_ideal_request_size_mib: int = 64,
    ) -> Self:
        """
        Create suitable CacheOptions based on provided network metrics.

        Typically this will be used with object storage solutions like Amazon S3,
        Google Cloud Storage and Azure Blob Storage.

        Parameters
        ----------
        time_to_first_byte_millis : int
            Seek-time or Time-To-First-Byte (TTFB) in milliseconds, also called call
            setup latency of a new read request. The value is a positive integer.
        transfer_bandwidth_mib_per_sec : int
            Data transfer Bandwidth (BW) in MiB/sec (per connection). The value is a positive
            integer.
        ideal_bandwidth_utilization_frac : int, default 0.9
            Transfer bandwidth utilization fraction (per connection) to maximize the net
            data load. The value is a positive float less than 1.
        max_ideal_request_size_mib : int, default 64
            The maximum single data request size (in MiB) to maximize the net data load.

        Returns
        -------
        CacheOptions
        """

class Codec(_Weakrefable):
    """
    Compression codec.

    Parameters
    ----------
    compression : str
        Type of compression codec to initialize, valid values are: 'gzip',
        'bz2', 'brotli', 'lz4' (or 'lz4_frame'), 'lz4_raw', 'zstd' and
        'snappy'.
    compression_level : int, None
        Optional parameter specifying how aggressively to compress.  The
        possible ranges and effect of this parameter depend on the specific
        codec chosen.  Higher values compress more but typically use more
        resources (CPU/RAM).  Some codecs support negative values.

        gzip
            The compression_level maps to the memlevel parameter of
            deflateInit2.  Higher levels use more RAM but are faster
            and should have higher compression ratios.

        bz2
            The compression level maps to the blockSize100k parameter of
            the BZ2_bzCompressInit function.  Higher levels use more RAM
            but are faster and should have higher compression ratios.

        brotli
            The compression level maps to the BROTLI_PARAM_QUALITY
            parameter.  Higher values are slower and should have higher
            compression ratios.

        lz4/lz4_frame/lz4_raw
            The compression level parameter is not supported and must
            be None

        zstd
            The compression level maps to the compressionLevel parameter
            of ZSTD_initCStream.  Negative values are supported.  Higher
            values are slower and should have higher compression ratios.

        snappy
            The compression level parameter is not supported and must
            be None


    Raises
    ------
    ValueError
        If invalid compression value is passed.

    Examples
    --------
    >>> import pyarrow as pa
    >>> pa.Codec.is_available("gzip")
    True
    >>> codec = pa.Codec("gzip")
    >>> codec.name
    'gzip'
    >>> codec.compression_level
    9
    """
    def __init__(self, compression: Compression, compression_level: int | None = None) -> None: ...
    @classmethod
    def detect(cls, path: StrPath) -> Self:
        """
        Detect and instantiate compression codec based on file extension.

        Parameters
        ----------
        path : str, path-like
            File-path to detect compression from.

        Raises
        ------
        TypeError
            If the passed value is not path-like.
        ValueError
            If the compression can't be detected from the path.

        Returns
        -------
        Codec
        """
    @staticmethod
    def is_available(compression: Compression) -> bool:
        """
        Returns whether the compression support has been built and enabled.

        Parameters
        ----------
        compression : str
             Type of compression codec,
             refer to Codec docstring for a list of supported ones.

        Returns
        -------
        bool
        """
    @staticmethod
    def supports_compression_level(compression: Compression) -> int:
        """
        Returns true if the compression level parameter is supported
        for the given codec.

        Parameters
        ----------
        compression : str
            Type of compression codec,
            refer to Codec docstring for a list of supported ones.
        """
    @staticmethod
    def default_compression_level(compression: Compression) -> int:
        """
        Returns the compression level that Arrow will use for the codec if
        None is specified.

        Parameters
        ----------
        compression : str
            Type of compression codec,
            refer to Codec docstring for a list of supported ones.
        """
    @staticmethod
    def minimum_compression_level(compression: Compression) -> int:
        """
        Returns the smallest valid value for the compression level

        Parameters
        ----------
        compression : str
            Type of compression codec,
            refer to Codec docstring for a list of supported ones.
        """
    @staticmethod
    def maximum_compression_level(compression: Compression) -> int:
        """
        Returns the largest valid value for the compression level

        Parameters
        ----------
        compression : str
            Type of compression codec,
            refer to Codec docstring for a list of supported ones.
        """
    @property
    def name(self) -> Compression:
        """Returns the name of the codec"""
    @property
    def compression_level(self) -> int:
        """Returns the compression level parameter of the codec"""
    @overload
    def compress(
        self,
        buf: Buffer | bytes | SupportPyBuffer,
        *,
        memory_pool: MemoryPool | None = None,
    ) -> Buffer: ...
    @overload
    def compress(
        self,
        buf: Buffer | bytes | SupportPyBuffer,
        *,
        asbytes: Literal[False],
        memory_pool: MemoryPool | None = None,
    ) -> Buffer: ...
    @overload
    def compress(
        self,
        buf: Buffer | bytes | SupportPyBuffer,
        *,
        asbytes: Literal[True],
        memory_pool: MemoryPool | None = None,
    ) -> bytes: ...
    def compress(self, *args, **kwargs):
        """
        Compress data from buffer-like object.

        Parameters
        ----------
        buf : pyarrow.Buffer, bytes, or other object supporting buffer protocol
        asbytes : bool, default False
            Return result as Python bytes object, otherwise Buffer
        memory_pool : MemoryPool, default None
            Memory pool to use for buffer allocations, if any

        Returns
        -------
        compressed : pyarrow.Buffer or bytes (if asbytes=True)
        """
    @overload
    def decompress(
        self,
        buf: Buffer | bytes | SupportPyBuffer,
        decompressed_size: int | None = None,
        *,
        memory_pool: MemoryPool | None = None,
    ) -> Buffer: ...
    @overload
    def decompress(
        self,
        buf: Buffer | bytes | SupportPyBuffer,
        decompressed_size: int | None = None,
        *,
        asbytes: Literal[False],
        memory_pool: MemoryPool | None = None,
    ) -> Buffer: ...
    @overload
    def decompress(
        self,
        buf: Buffer | bytes | SupportPyBuffer,
        decompressed_size: int | None = None,
        *,
        asbytes: Literal[True],
        memory_pool: MemoryPool | None = None,
    ) -> bytes: ...
    def decompress(self, *args, **kwargs):
        """
        Decompress data from buffer-like object.

        Parameters
        ----------
        buf : pyarrow.Buffer, bytes, or memoryview-compatible object
        decompressed_size : int, default None
            Size of the decompressed result
        asbytes : boolean, default False
            Return result as Python bytes object, otherwise Buffer
        memory_pool : MemoryPool, default None
            Memory pool to use for buffer allocations, if any.

        Returns
        -------
        uncompressed : pyarrow.Buffer or bytes (if asbytes=True)
        """

@overload
def compress(
    buf: Buffer | bytes | SupportPyBuffer,
    codec: Compression = "lz4",
    *,
    memory_pool: MemoryPool | None = None,
) -> Buffer: ...
@overload
def compress(
    buf: Buffer | bytes | SupportPyBuffer,
    codec: Compression = "lz4",
    *,
    asbytes: Literal[False],
    memory_pool: MemoryPool | None = None,
) -> Buffer: ...
@overload
def compress(
    buf: Buffer | bytes | SupportPyBuffer,
    codec: Compression = "lz4",
    *,
    asbytes: Literal[True],
    memory_pool: MemoryPool | None = None,
) -> bytes: ...
def compress(*args, **kwargs):
    """
    Compress data from buffer-like object.

    Parameters
    ----------
    buf : pyarrow.Buffer, bytes, or other object supporting buffer protocol
    codec : str, default 'lz4'
        Compression codec.
        Supported types: {'brotli, 'gzip', 'lz4', 'lz4_raw', 'snappy', 'zstd'}
    asbytes : bool, default False
        Return result as Python bytes object, otherwise Buffer.
    memory_pool : MemoryPool, default None
        Memory pool to use for buffer allocations, if any.

    Returns
    -------
    compressed : pyarrow.Buffer or bytes (if asbytes=True)
    """

@overload
def decompress(
    buf: Buffer | bytes | SupportPyBuffer,
    decompressed_size: int | None = None,
    codec: Compression = "lz4",
    *,
    memory_pool: MemoryPool | None = None,
) -> Buffer: ...
@overload
def decompress(
    buf: Buffer | bytes | SupportPyBuffer,
    decompressed_size: int | None = None,
    codec: Compression = "lz4",
    *,
    asbytes: Literal[False],
    memory_pool: MemoryPool | None = None,
) -> Buffer: ...
@overload
def decompress(
    buf: Buffer | bytes | SupportPyBuffer,
    decompressed_size: int | None = None,
    codec: Compression = "lz4",
    *,
    asbytes: Literal[True],
    memory_pool: MemoryPool | None = None,
) -> bytes: ...
def decompress(*args, **kwargs):
    """
    Decompress data from buffer-like object.

    Parameters
    ----------
    buf : pyarrow.Buffer, bytes, or memoryview-compatible object
        Input object to decompress data from.
    decompressed_size : int, default None
        Size of the decompressed result
    codec : str, default 'lz4'
        Compression codec.
        Supported types: {'brotli, 'gzip', 'lz4', 'lz4_raw', 'snappy', 'zstd'}
    asbytes : bool, default False
        Return result as Python bytes object, otherwise Buffer.
    memory_pool : MemoryPool, default None
        Memory pool to use for buffer allocations, if any.

    Returns
    -------
    uncompressed : pyarrow.Buffer or bytes (if asbytes=True)
    """

def input_stream(
    source: StrPath | Buffer | IOBase,
    compression: Literal["detect", "bz2", "brotli", "gzip", "lz4", "zstd"] = "detect",
    buffer_size: int | None = None,
) -> BufferReader:
    """
    Create an Arrow input stream.

    Parameters
    ----------
    source : str, Path, buffer, or file-like object
        The source to open for reading.
    compression : str optional, default 'detect'
        The compression algorithm to use for on-the-fly decompression.
        If "detect" and source is a file path, then compression will be
        chosen based on the file extension.
        If None, no compression will be applied.
        Otherwise, a well-known algorithm name must be supplied (e.g. "gzip").
    buffer_size : int, default None
        If None or 0, no buffering will happen. Otherwise the size of the
        temporary read buffer.

    Examples
    --------
    Create a readable BufferReader (NativeFile) from a Buffer or a memoryview object:

    >>> import pyarrow as pa
    >>> buf = memoryview(b"some data")
    >>> with pa.input_stream(buf) as stream:
    ...     stream.read(4)
    b'some'

    Create a readable OSFile (NativeFile) from a string or file path:

    >>> import gzip
    >>> with gzip.open("example.gz", "wb") as f:
    ...     f.write(b"some data")
    9
    >>> with pa.input_stream("example.gz") as stream:
    ...     stream.read()
    b'some data'

    Create a readable PythonFile (NativeFile) from a a Python file object:

    >>> with open("example.txt", mode="w") as f:
    ...     f.write("some text")
    9
    >>> with pa.input_stream("example.txt") as stream:
    ...     stream.read(6)
    b'some t'
    """

def output_stream(
    source: StrPath | Buffer | IOBase,
    compression: Literal["detect", "bz2", "brotli", "gzip", "lz4", "zstd"] = "detect",
    buffer_size: int | None = None,
) -> NativeFile:
    """
    Create an Arrow output stream.

    Parameters
    ----------
    source : str, Path, buffer, file-like object
        The source to open for writing.
    compression : str optional, default 'detect'
        The compression algorithm to use for on-the-fly compression.
        If "detect" and source is a file path, then compression will be
        chosen based on the file extension.
        If None, no compression will be applied.
        Otherwise, a well-known algorithm name must be supplied (e.g. "gzip").
    buffer_size : int, default None
        If None or 0, no buffering will happen. Otherwise the size of the
        temporary write buffer.

    Examples
    --------
    Create a writable NativeFile from a pyarrow Buffer:

    >>> import pyarrow as pa
    >>> data = b"buffer data"
    >>> empty_obj = bytearray(11)
    >>> buf = pa.py_buffer(empty_obj)
    >>> with pa.output_stream(buf) as stream:
    ...     stream.write(data)
    11
    >>> with pa.input_stream(buf) as stream:
    ...     stream.read(6)
    b'buffer'

    or from a memoryview object:

    >>> buf = memoryview(empty_obj)
    >>> with pa.output_stream(buf) as stream:
    ...     stream.write(data)
    11
    >>> with pa.input_stream(buf) as stream:
    ...     stream.read()
    b'buffer data'

    Create a writable NativeFile from a string or file path:

    >>> with pa.output_stream("example_second.txt") as stream:
    ...     stream.write(b"Write some data")
    15
    >>> with pa.input_stream("example_second.txt") as stream:
    ...     stream.read()
    b'Write some data'
    """

__all__ = [
    "have_libhdfs",
    "io_thread_count",
    "set_io_thread_count",
    "NativeFile",
    "PythonFile",
    "MemoryMappedFile",
    "memory_map",
    "create_memory_map",
    "OSFile",
    "FixedSizeBufferWriter",
    "Buffer",
    "ResizableBuffer",
    "allocate_buffer",
    "BufferOutputStream",
    "MockOutputStream",
    "BufferReader",
    "CompressedInputStream",
    "CompressedOutputStream",
    "BufferedInputStream",
    "BufferedOutputStream",
    "TransformInputStream",
    "Transcoder",
    "transcoding_input_stream",
    "py_buffer",
    "foreign_buffer",
    "as_buffer",
    "CacheOptions",
    "Codec",
    "compress",
    "decompress",
    "input_stream",
    "output_stream",
]
