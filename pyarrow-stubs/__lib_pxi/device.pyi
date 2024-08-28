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
    @property
    def type_name(self) -> str: ...
    @property
    def device_id(self) -> int: ...
    @property
    def is_cpu(self) -> bool: ...
    @property
    def device_type(self) -> DeviceAllocationType: ...

class MemoryManager(_Weakrefable):
    @property
    def device(self) -> Device: ...
    @property
    def is_cpu(self) -> bool: ...

def default_cpu_memory_manager() -> MemoryManager: ...

__all__ = ["DeviceAllocationType", "Device", "MemoryManager", "default_cpu_memory_manager"]
