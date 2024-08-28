import datetime as dt

from ._fs import FileSystem
from .lib import KeyValueMetadata

class GcsFileSystem(FileSystem):
    def __init__(
        self,
        *,
        anonymous: bool = False,
        access_token: str | None = None,
        target_service_account: str | None = None,
        credential_token_expiration: dt.datetime | None = None,
        default_bucket_location: str = "US",
        scheme: str = "https",
        endpoint_override: str | None = None,
        default_metadata: dict | KeyValueMetadata | None = None,
        retry_time_limit: dt.timedelta | None = None,
        project_id: str | None = None,
    ): ...
    @property
    def default_bucket_location(self) -> str: ...
    @property
    def project_id(self) -> str: ...
