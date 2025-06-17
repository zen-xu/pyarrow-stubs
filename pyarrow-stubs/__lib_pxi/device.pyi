import enum

from pyarrow.lib import _Weakrefable

class DeviceAllocationType(enum.Flag):
    CPU = enum.auto()
    CUDA = enum.auto()
    CUDA_HOST = enum.auto()
    OPENCL = enum.auto()
    VULKAN = enum.auto()
    METAL = enum.auto()
    VPI = enum.auto()
    ROCM = enum.auto()
    ROCM_HOST = enum.auto()
    EXT_DEV = enum.auto()
    CUDA_MANAGED = enum.auto()
    ONEAPI = enum.auto()
    WEBGPU = enum.auto()
    HEXAGON = enum.auto()

class Device(_Weakrefable):
    """
    Abstract interface for hardware devices

    This object represents a device with access to some memory spaces.
    When handling a Buffer or raw memory address, it allows deciding in which
    context the raw memory address should be interpreted
    (e.g. CPU-accessible memory, or embedded memory on some particular GPU).
    """

    @property
    def type_name(self) -> str:
        """
        A shorthand for this device's type.
        """
    @property
    def device_id(self) -> int:
        """
        A device ID to identify this device if there are multiple of this type.

        If there is no "device_id" equivalent (such as for the main CPU device on
        non-numa systems) returns -1.
        """
    @property
    def is_cpu(self) -> bool:
        """
        Whether this device is the main CPU device.

        This shorthand method is very useful when deciding whether a memory address
        is CPU-accessible.
        """
    @property
    def device_type(self) -> DeviceAllocationType:
        """
        Return the DeviceAllocationType of this device.
        """

class MemoryManager(_Weakrefable):
    """
    An object that provides memory management primitives.

    A MemoryManager is always tied to a particular Device instance.
    It can also have additional parameters (such as a MemoryPool to
    allocate CPU memory).

    """
    @property
    def device(self) -> Device:
        """
        The device this MemoryManager is tied to.
        """
    @property
    def is_cpu(self) -> bool:
        """
        Whether this MemoryManager is tied to the main CPU device.

        This shorthand method is very useful when deciding whether a memory
        address is CPU-accessible.
        """

def default_cpu_memory_manager() -> MemoryManager:
    """
    Return the default CPU MemoryManager instance.

    The returned singleton instance uses the default MemoryPool.
    """

__all__ = ["DeviceAllocationType", "Device", "MemoryManager", "default_cpu_memory_manager"]
