from _typeshed import StrPath

from ._fs import FileSystem

class HadoopFileSystem(FileSystem):
    def __init__(
        self,
        host: str,
        port: int = 8020,
        *,
        user: str | None = None,
        replication: int = 3,
        buffer_size: int = 0,
        default_block_size: int | None = None,
        kerb_ticket: StrPath | None = None,
        extra_conf: dict | None = None,
    ): ...
    @staticmethod
    def from_uri(uri: str) -> HadoopFileSystem: ...  # type: ignore[override]
