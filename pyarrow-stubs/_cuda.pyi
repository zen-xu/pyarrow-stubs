from typing import Any

import cuda  # type: ignore[import-not-found]

from numba.cuda.cudadrv import driver as _numba_driver  # type: ignore[import-not-found]

from . import lib
from ._stubs_typing import ArrayLike

class Context(lib._Weakrefable):
    """
    CUDA driver context.
    """

    def __init__(self, device_number: int = 0, handle: int | None = None) -> None:
        """
        Create a CUDA driver context for a particular device.

        If a CUDA context handle is passed, it is wrapped, otherwise
        a default CUDA context for the given device is requested.

        Parameters
        ----------
        device_number : int (default 0)
          Specify the GPU device for which the CUDA driver context is
          requested.
        handle : int, optional
          Specify CUDA handle for a shared context that has been created
          by another library.
        """
    @staticmethod
    def from_numba(context: _numba_driver.Context | None = None) -> Context:
        """
        Create a Context instance from a Numba CUDA context.

        Parameters
        ----------
        context : {numba.cuda.cudadrv.driver.Context, None}
          A Numba CUDA context instance.
          If None, the current Numba context is used.

        Returns
        -------
        shared_context : pyarrow.cuda.Context
          Context instance.
        """
    def to_numba(self) -> _numba_driver.Context:
        """
        Convert Context to a Numba CUDA context.

        Returns
        -------
        context : numba.cuda.cudadrv.driver.Context
          Numba CUDA context instance.
        """
    @staticmethod
    def get_num_devices() -> int:
        """Return the number of GPU devices."""
    @property
    def device_number(self) -> int:
        """Return context device number."""
    @property
    def handle(self) -> int:
        """Return pointer to context handle."""
    def synchronize(self) -> None:
        """Blocks until the device has completed all preceding requested
        tasks.
        """
    @property
    def bytes_allocated(self) -> int:
        """Return the number of allocated bytes."""
    def get_device_address(self, address: int) -> int:
        """Return the device address that is reachable from kernels running in
        the context

        Parameters
        ----------
        address : int
          Specify memory address value

        Returns
        -------
        device_address : int
          Device address accessible from device context

        Notes
        -----
        The device address is defined as a memory address accessible
        by device. While it is often a device memory address but it
        can be also a host memory address, for instance, when the
        memory is allocated as host memory (using cudaMallocHost or
        cudaHostAlloc) or as managed memory (using cudaMallocManaged)
        or the host memory is page-locked (using cudaHostRegister).
        """
    def new_buffer(self, nbytes: int) -> CudaBuffer:
        """Return new device buffer.

        Parameters
        ----------
        nbytes : int
          Specify the number of bytes to be allocated.

        Returns
        -------
        buf : CudaBuffer
          Allocated buffer.
        """
    @property
    def memory_manager(self) -> lib.MemoryManager:
        """
        The default memory manager tied to this context's device.

        Returns
        -------
        MemoryManager
        """
    @property
    def device(self) -> lib.Device:
        """
        The device instance associated with this context.

        Returns
        -------
        Device
        """
    def foreign_buffer(self, address: int, size: int, base: Any | None = None) -> CudaBuffer:
        """
        Create device buffer from address and size as a view.

        The caller is responsible for allocating and freeing the
        memory. When `address==size==0` then a new zero-sized buffer
        is returned.

        Parameters
        ----------
        address : int
          Specify the starting address of the buffer. The address can
          refer to both device or host memory but it must be
          accessible from device after mapping it with
          `get_device_address` method.
        size : int
          Specify the size of device buffer in bytes.
        base : {None, object}
          Specify object that owns the referenced memory.

        Returns
        -------
        cbuf : CudaBuffer
          Device buffer as a view of device reachable memory.

        """
    def open_ipc_buffer(self, ipc_handle: IpcMemHandle) -> CudaBuffer:
        """Open existing CUDA IPC memory handle

        Parameters
        ----------
        ipc_handle : IpcMemHandle
          Specify opaque pointer to CUipcMemHandle (driver API).

        Returns
        -------
        buf : CudaBuffer
          referencing device buffer
        """
    def buffer_from_data(
        self,
        data: CudaBuffer | HostBuffer | lib.Buffer | ArrayLike,
        offset: int = 0,
        size: int = -1,
    ) -> CudaBuffer:
        """Create device buffer and initialize with data.

        Parameters
        ----------
        data : {CudaBuffer, HostBuffer, Buffer, array-like}
          Specify data to be copied to device buffer.
        offset : int
          Specify the offset of input buffer for device data
          buffering. Default: 0.
        size : int
          Specify the size of device buffer in bytes. Default: all
          (starting from input offset)

        Returns
        -------
        cbuf : CudaBuffer
          Device buffer with copied data.
        """
    def buffer_from_object(self, obj: Any) -> CudaBuffer:
        """Create device buffer view of arbitrary object that references
        device accessible memory.

        When the object contains a non-contiguous view of device
        accessible memory then the returned device buffer will contain
        contiguous view of the memory, that is, including the
        intermediate data that is otherwise invisible to the input
        object.

        Parameters
        ----------
        obj : {object, Buffer, HostBuffer, CudaBuffer, ...}
          Specify an object that holds (device or host) address that
          can be accessed from device. This includes objects with
          types defined in pyarrow.cuda as well as arbitrary objects
          that implement the CUDA array interface as defined by numba.

        Returns
        -------
        cbuf : CudaBuffer
          Device buffer as a view of device accessible memory.

        """

class IpcMemHandle(lib._Weakrefable):
    """A serializable container for a CUDA IPC handle."""
    @staticmethod
    def from_buffer(opaque_handle: lib.Buffer) -> IpcMemHandle:
        """Create IpcMemHandle from opaque buffer (e.g. from another
        process)

        Parameters
        ----------
        opaque_handle :
          a CUipcMemHandle as a const void*

        Returns
        -------
        ipc_handle : IpcMemHandle
        """
    def serialize(self, pool: lib.MemoryPool | None = None) -> lib.Buffer:
        """Write IpcMemHandle to a Buffer

        Parameters
        ----------
        pool : {MemoryPool, None}
          Specify a pool to allocate memory from

        Returns
        -------
        buf : Buffer
          The serialized buffer.
        """

class CudaBuffer(lib.Buffer):
    """An Arrow buffer with data located in a GPU device.

    To create a CudaBuffer instance, use Context.device_buffer().

    The memory allocated in a CudaBuffer is freed when the buffer object
    is deleted.
    """

    @staticmethod
    def from_buffer(buf: lib.Buffer) -> CudaBuffer:
        """Convert back generic buffer into CudaBuffer

        Parameters
        ----------
        buf : Buffer
          Specify buffer containing CudaBuffer

        Returns
        -------
        dbuf : CudaBuffer
          Resulting device buffer.
        """
    @staticmethod
    def from_numba(mem: _numba_driver.MemoryPointer) -> CudaBuffer:
        """Create a CudaBuffer view from numba MemoryPointer instance.

        Parameters
        ----------
        mem :  numba.cuda.cudadrv.driver.MemoryPointer

        Returns
        -------
        cbuf : CudaBuffer
          Device buffer as a view of numba MemoryPointer.
        """
    def to_numba(self) -> _numba_driver.MemoryPointer:
        """Return numba memory pointer of CudaBuffer instance."""
    def copy_to_host(
        self,
        position: int = 0,
        nbytes: int = -1,
        buf: lib.Buffer | None = None,
        memory_pool: lib.MemoryPool | None = None,
        resizable: bool = False,
    ) -> lib.Buffer:
        """Copy memory from GPU device to CPU host

        Caller is responsible for ensuring that all tasks affecting
        the memory are finished. Use

          `<CudaBuffer instance>.context.synchronize()`

        when needed.

        Parameters
        ----------
        position : int
          Specify the starting position of the source data in GPU
          device buffer. Default: 0.
        nbytes : int
          Specify the number of bytes to copy. Default: -1 (all from
          the position until host buffer is full).
        buf : Buffer
          Specify a pre-allocated output buffer in host. Default: None
          (allocate new output buffer).
        memory_pool : MemoryPool
        resizable : bool
          Specify extra arguments to allocate_buffer. Used only when
          buf is None.

        Returns
        -------
        buf : Buffer
          Output buffer in host.

        """
    def copy_from_host(
        self, data: lib.Buffer | ArrayLike, position: int = 0, nbytes: int = -1
    ) -> int:
        """Copy data from host to device.

        The device buffer must be pre-allocated.

        Parameters
        ----------
        data : {Buffer, array-like}
          Specify data in host. It can be array-like that is valid
          argument to py_buffer
        position : int
          Specify the starting position of the copy in device buffer.
          Default: 0.
        nbytes : int
          Specify the number of bytes to copy. Default: -1 (all from
          source until device buffer, starting from position, is full)

        Returns
        -------
        nbytes : int
          Number of bytes copied.
        """
    def copy_from_device(self, buf: CudaBuffer, position: int = 0, nbytes: int = -1) -> int:
        """Copy data from device to device.

        Parameters
        ----------
        buf : CudaBuffer
          Specify source device buffer.
        position : int
          Specify the starting position of the copy in device buffer.
          Default: 0.
        nbytes : int
          Specify the number of bytes to copy. Default: -1 (all from
          source until device buffer, starting from position, is full)

        Returns
        -------
        nbytes : int
          Number of bytes copied.

        """
    def export_for_ipc(self) -> IpcMemHandle:
        """
        Expose this device buffer as IPC memory which can be used in other
        processes.

        After calling this function, this device memory will not be
        freed when the CudaBuffer is destructed.

        Returns
        -------
        ipc_handle : IpcMemHandle
          The exported IPC handle

        """
    @property
    def context(self) -> Context:
        """Returns the CUDA driver context of this buffer."""
    def slice(self, offset: int = 0, length: int | None = None) -> CudaBuffer:
        """Return slice of device buffer

        Parameters
        ----------
        offset : int, default 0
          Specify offset from the start of device buffer to slice
        length : int, default None
          Specify the length of slice (default is until end of device
          buffer starting from offset). If the length is larger than
          the data available, the returned slice will have a size of
          the available data starting from the offset.

        Returns
        -------
        sliced : CudaBuffer
          Zero-copy slice of device buffer.

        """
    def to_pybytes(self) -> bytes:
        """Return device buffer content as Python bytes."""

class HostBuffer(lib.Buffer):
    """Device-accessible CPU memory created using cudaHostAlloc.

    To create a HostBuffer instance, use

      cuda.new_host_buffer(<nbytes>)
    """
    @property
    def size(self) -> int: ...

class BufferReader(lib.NativeFile):
    """File interface for zero-copy read from CUDA buffers.

    Note: Read methods return pointers to device memory. This means
    you must be careful using this interface with any Arrow code which
    may expect to be able to do anything other than pointer arithmetic
    on the returned buffers.
    """
    def __init__(self, obj: CudaBuffer) -> None: ...
    def read_buffer(self, nbytes: int | None = None) -> CudaBuffer:
        """Return a slice view of the underlying device buffer.

        The slice will start at the current reader position and will
        have specified size in bytes.

        Parameters
        ----------
        nbytes : int, default None
          Specify the number of bytes to read. Default: None (read all
          remaining bytes).

        Returns
        -------
        cbuf : CudaBuffer
          New device buffer.

        """

class BufferWriter(lib.NativeFile):
    """File interface for writing to CUDA buffers.

    By default writes are unbuffered. Use set_buffer_size to enable
    buffering.
    """
    def __init__(self, obj: CudaBuffer) -> None: ...
    def writeat(self, position: int, data: ArrayLike) -> None:
        """Write data to buffer starting from position.

        Parameters
        ----------
        position : int
          Specify device buffer position where the data will be
          written.
        data : array-like
          Specify data, the data instance must implement buffer
          protocol.
        """
    @property
    def buffer_size(self) -> int:
        """Returns size of host (CPU) buffer, 0 for unbuffered"""
    @buffer_size.setter
    def buffer_size(self, buffer_size: int):
        """Set CPU buffer size to limit calls to cudaMemcpy

        Parameters
        ----------
        buffer_size : int
          Specify the size of CPU buffer to allocate in bytes.
        """
    @property
    def num_bytes_buffered(self) -> int:
        """Returns number of bytes buffered on host"""

def new_host_buffer(size: int, device: int = 0) -> HostBuffer:
    """Return buffer with CUDA-accessible memory on CPU host

    Parameters
    ----------
    size : int
      Specify the number of bytes to be allocated.
    device : int
      Specify GPU device number.

    Returns
    -------
    dbuf : HostBuffer
      Allocated host buffer
    """

def serialize_record_batch(batch: lib.RecordBatch, ctx: Context) -> CudaBuffer:
    """Write record batch message to GPU device memory

    Parameters
    ----------
    batch : RecordBatch
      Record batch to write
    ctx : Context
      CUDA Context to allocate device memory from

    Returns
    -------
    dbuf : CudaBuffer
      device buffer which contains the record batch message
    """

def read_message(
    source: CudaBuffer | cuda.BufferReader, pool: lib.MemoryManager | None = None
) -> lib.Message:
    """Read Arrow IPC message located on GPU device

    Parameters
    ----------
    source : {CudaBuffer, cuda.BufferReader}
      Device buffer or reader of device buffer.
    pool : MemoryPool (optional)
      Pool to allocate CPU memory for the metadata

    Returns
    -------
    message : Message
      The deserialized message, body still on device
    """

def read_record_batch(
    buffer: lib.Buffer,
    object: lib.Schema,
    *,
    dictionary_memo: lib.DictionaryMemo | None = None,
    pool: lib.MemoryPool | None = None,
) -> lib.RecordBatch:
    """Construct RecordBatch referencing IPC message located on CUDA device.

    While the metadata is copied to host memory for deserialization,
    the record batch data remains on the device.

    Parameters
    ----------
    buffer :
      Device buffer containing the complete IPC message
    schema : Schema
      The schema for the record batch
    dictionary_memo : DictionaryMemo, optional
        If message contains dictionaries, must pass a populated
        DictionaryMemo
    pool : MemoryPool (optional)
      Pool to allocate metadata from

    Returns
    -------
    batch : RecordBatch
      Reconstructed record batch, with device pointers

    """
