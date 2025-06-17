import enum

from pyarrow.lib import Buffer

class DlpackDeviceType(enum.IntEnum):
    """Integer enum for device type codes matching DLPack."""

    CPU = 1
    CUDA = 2
    CPU_PINNED = 3
    OPENCL = 4
    VULKAN = 7
    METAL = 8
    VPI = 9
    ROCM = 10

class _PyArrowBuffer:
    """
    Data in the buffer is guaranteed to be contiguous in memory.

    Note that there is no dtype attribute present, a buffer can be thought of
    as simply a block of memory. However, if the column that the buffer is
    attached to has a dtype that's supported by DLPack and ``__dlpack__`` is
    implemented, then that dtype information will be contained in the return
    value from ``__dlpack__``.

    This distinction is useful to support both data exchange via DLPack on a
    buffer and (b) dtypes like variable-length strings which do not have a
    fixed number of bytes per element.
    """
    def __init__(self, x: Buffer, allow_copy: bool = True) -> None: ...
    @property
    def bufsize(self) -> int:
        """
        Buffer size in bytes.
        """
    @property
    def ptr(self) -> int:
        """
        Pointer to start of the buffer as an integer.
        """
    def __dlpack__(self):
        """
        Produce DLPack capsule (see array API standard).

        Raises:
            - TypeError : if the buffer contains unsupported dtypes.
            - NotImplementedError : if DLPack support is not implemented

        Useful to have to connect to array libraries. Support optional because
        it's not completely trivial to implement for a Python-only library.
        """
    def __dlpack_device__(self) -> tuple[DlpackDeviceType, int | None]:
        """
        Device type and device ID for where the data in the buffer resides.
        Uses device type codes matching DLPack.
        Note: must be implemented even if ``__dlpack__`` is not.
        """
