from collections.abc import Generator

from _typeshed import Incomplete
import pyarrow._hdfsio as _hdfsio
from pyarrow.filesystem import FileSystem as FileSystem
from pyarrow.util import implements as implements

class HadoopFileSystem(_hdfsio.HadoopFileSystem, FileSystem):  # type: ignore
    def __init__(
        self,
        host: str = ...,
        port: int = ...,
        user: str | None = ...,
        kerb_ticket: Incomplete | None = ...,
        driver: str = ...,
        extra_conf: Incomplete | None = ...,
    ) -> None: ...
    def __reduce__(self) -> tuple: ...
    def walk(
        self, top_path: str
    ) -> Generator[tuple[str, list[str], list[str]], None, None]: ...

def connect(
    host: str = ...,
    port: int = ...,
    user: Incomplete | None = ...,
    kerb_ticket: Incomplete | None = ...,
    extra_conf: Incomplete | None = ...,
): ...
