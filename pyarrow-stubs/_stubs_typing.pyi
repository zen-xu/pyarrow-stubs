from typing import Any, Collection, Literal, Protocol, TypeAlias

ArrayLike: TypeAlias = Any
Order: TypeAlias = Literal["ascending", "descending"]
JoinType: TypeAlias = Literal[
    "left semi",
    "right semi",
    "left anti",
    "right anti",
    "inner",
    "left outer",
    "right outer",
    "full outer",
]
Compression: TypeAlias = Literal[
    "gzip", "bz2", "brotli", "lz4", "lz4_frame", "lz4_raw", "zstd", "snappy"
]

class SupportEq(Protocol):
    def __eq__(self, other) -> bool: ...

class SupportLt(Protocol):
    def __lt__(self, other) -> bool: ...

class SupportGt(Protocol):
    def __gt__(self, other) -> bool: ...

class SupportLe(Protocol):
    def __le__(self, other) -> bool: ...

class SupportGe(Protocol):
    def __ge__(self, other) -> bool: ...

FilterTuple: TypeAlias = (
    tuple[str, Literal["=", "==", "!="], SupportEq]
    | tuple[str, Literal["<"], SupportLt]
    | tuple[str, Literal[">"], SupportGt]
    | tuple[str, Literal["<="], SupportLe]
    | tuple[str, Literal[">="], SupportGe]
    | tuple[str, Literal["in", "not in"], Collection]
)

class Buffer(Protocol):
    def __buffer__(self, flags: int, /) -> memoryview: ...

SupportPyBuffer: TypeAlias = Any
