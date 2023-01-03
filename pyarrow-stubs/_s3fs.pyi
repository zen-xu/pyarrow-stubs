import enum
import importlib._bootstrap  # type: ignore
from typing import (
    Any,
    ClassVar,
)

import pyarrow._fs
import pyarrow.lib

Debug: importlib._bootstrap.S3LogLevel
Error: importlib._bootstrap.S3LogLevel
Fatal: importlib._bootstrap.S3LogLevel
Info: importlib._bootstrap.S3LogLevel
Off: importlib._bootstrap.S3LogLevel
Trace: importlib._bootstrap.S3LogLevel
Warn: importlib._bootstrap.S3LogLevel

class AwsDefaultS3RetryStrategy(S3RetryStrategy): ...
class AwsStandardS3RetryStrategy(S3RetryStrategy): ...

class S3FileSystem(pyarrow._fs.FileSystem):
    region: str
    def __init__(
        self,
        *,
        access_key: str | None = ...,
        secret_key: str | None = ...,
        session_token: str | None = ...,
        anonymous: bool = ...,
        role_arn: str | None = ...,
        session_name: str | None = ...,
        external_id: str | None = ...,
        load_frequency: int = ...,
        region: str = ...,
        request_timeout: float | None = ...,
        connect_timeout: float | None = ...,
        schema: str = ...,
        endpoint_override: str | None = ...,
        background_writes: bool = ...,
        default_metadata: dict | pyarrow.lib.KeyValueMetadata = ...,
        proxy_options: dict | str | None = ...,
        allow_bucket_creation: bool = ...,
        allow_bucket_deletion: bool = ...,
        retry_strategy: S3RetryStrategy = ...,
    ) -> None: ...
    @classmethod
    def _reconstruct(cls, kwargs: Any) -> S3FileSystem: ...

class S3LogLevel(enum.IntEnum):
    Debug: ClassVar[importlib._bootstrap.S3LogLevel] = ...
    Error: ClassVar[importlib._bootstrap.S3LogLevel] = ...
    Fatal: ClassVar[importlib._bootstrap.S3LogLevel] = ...
    Info: ClassVar[importlib._bootstrap.S3LogLevel] = ...
    Off: ClassVar[importlib._bootstrap.S3LogLevel] = ...
    Trace: ClassVar[importlib._bootstrap.S3LogLevel] = ...
    Warn: ClassVar[importlib._bootstrap.S3LogLevel] = ...

class S3RetryStrategy:
    def __init__(self, max_attempts: int = ...) -> None: ...

def finalize_s3() -> None: ...
def initialize_s3(log_level: S3LogLevel = ...) -> Any: ...
def resolve_s3_region(bucket: str) -> str: ...
