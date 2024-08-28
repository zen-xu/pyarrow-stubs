from typing import Literal

from ._fs import FileSystem

class AzureFileSystem(FileSystem):
    def __init__(
        self,
        account_name: str,
        account_key: str | None = None,
        blob_storage_authority: str | None = None,
        dfs_storage_authority: str | None = None,
        blob_storage_schema: Literal["http", "https"] = "https",
        dfs_storage_schema: Literal["http", "https"] = "https",
    ) -> None: ...
