from pyarrow.lib import _Weakrefable

class MemoryPool(_Weakrefable):
    """
    Base class for memory allocation.

    Besides tracking its number of allocated bytes, a memory pool also
    takes care of the required 64-byte alignment for Arrow data.
    """

    def release_unused(self) -> None:
        """
        Attempt to return to the OS any memory being held onto by the pool.

        This function should not be called except potentially for
        benchmarking or debugging as it could be expensive and detrimental to
        performance.

        This is best effort and may not have any effect on some memory pools
        or in some situations (e.g. fragmentation).
        """
    def bytes_allocated(self) -> int:
        """
        Return the number of bytes that are currently allocated from this
        memory pool.
        """
    def total_bytes_allocated(self) -> int:
        """
        Return the total number of bytes that have been allocated from this
        memory pool.
        """
    def max_memory(self) -> int | None:
        """
        Return the peak memory allocation in this memory pool.
        This can be an approximate number in multi-threaded applications.

        None is returned if the pool implementation doesn't know how to
        compute this number.
        """
    def num_allocations(self) -> int:
        """
        Return the number of allocations or reallocations that were made
        using this memory pool.
        """
    def print_stats(self) -> None:
        """
        Print statistics about this memory pool.

        The output format is implementation-specific. Not all memory pools
        implement this method.
        """
    @property
    def backend_name(self) -> str:
        """
        The name of the backend used by this MemoryPool (e.g. "jemalloc").
        """

class LoggingMemoryPool(MemoryPool): ...
class ProxyMemoryPool(MemoryPool): ...

def default_memory_pool() -> MemoryPool:
    """
    Return the process-global memory pool.

    Examples
    --------
    >>> default_memory_pool()
    <pyarrow.MemoryPool backend_name=... bytes_allocated=0 max_memory=...>
    """

def proxy_memory_pool(parent: MemoryPool) -> ProxyMemoryPool:
    """
    Create and return a MemoryPool instance that redirects to the
    *parent*, but with separate allocation statistics.

    Parameters
    ----------
    parent : MemoryPool
        The real memory pool that should be used for allocations.
    """

def logging_memory_pool(parent: MemoryPool) -> LoggingMemoryPool:
    """
    Create and return a MemoryPool instance that redirects to the
    *parent*, but also dumps allocation logs on stderr.

    Parameters
    ----------
    parent : MemoryPool
        The real memory pool that should be used for allocations.
    """

def system_memory_pool() -> MemoryPool:
    """
    Return a memory pool based on the C malloc heap.
    """

def jemalloc_memory_pool() -> MemoryPool:
    """
    Return a memory pool based on the jemalloc heap.

    NotImplementedError is raised if jemalloc support is not enabled.
    """

def mimalloc_memory_pool() -> MemoryPool:
    """
    Return a memory pool based on the mimalloc heap.

    NotImplementedError is raised if mimalloc support is not enabled.
    """

def set_memory_pool(pool: MemoryPool) -> None:
    """
    Set the default memory pool.

    Parameters
    ----------
    pool : MemoryPool
        The memory pool that should be used by default.
    """

def log_memory_allocations(enable: bool = True) -> None:
    """
    Enable or disable memory allocator logging for debugging purposes

    Parameters
    ----------
    enable : bool, default True
        Pass False to disable logging
    """

def total_allocated_bytes() -> int:
    """
    Return the currently allocated bytes from the default memory pool.
    Other memory pools may not be accounted for.
    """

def jemalloc_set_decay_ms(decay_ms: int) -> None:
    """
    Set arenas.dirty_decay_ms and arenas.muzzy_decay_ms to indicated number of
    milliseconds. A value of 0 (the default) results in dirty / muzzy memory
    pages being released right away to the OS, while a higher value will result
    in a time-based decay. See the jemalloc docs for more information

    It's best to set this at the start of your application.

    Parameters
    ----------
    decay_ms : int
        Number of milliseconds to set for jemalloc decay conf parameters. Note
        that this change will only affect future memory arenas
    """

def supported_memory_backends() -> list[str]:
    """
    Return a list of available memory pool backends
    """

__all__ = [
    "MemoryPool",
    "LoggingMemoryPool",
    "ProxyMemoryPool",
    "default_memory_pool",
    "proxy_memory_pool",
    "logging_memory_pool",
    "system_memory_pool",
    "jemalloc_memory_pool",
    "mimalloc_memory_pool",
    "set_memory_pool",
    "log_memory_allocations",
    "total_allocated_bytes",
    "jemalloc_set_decay_ms",
    "supported_memory_backends",
]
