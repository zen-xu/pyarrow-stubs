from pyarrow._fs import (  # noqa
    FileSelector,
    FileType,
    FileInfo,
    FileSystem,
    LocalFileSystem,
    SubTreeFileSystem,
    _MockFileSystem,
    FileSystemHandler,
    PyFileSystem,
    SupportedFileSystem,
)
from pyarrow._azurefs import AzureFileSystem
from pyarrow._hdfs import HadoopFileSystem
from pyarrow._gcsfs import GcsFileSystem
from pyarrow._s3fs import (  # noqa
    AwsDefaultS3RetryStrategy,
    AwsStandardS3RetryStrategy,
    S3FileSystem,
    S3LogLevel,
    S3RetryStrategy,
    ensure_s3_initialized,
    finalize_s3,
    ensure_s3_finalized,
    initialize_s3,
    resolve_s3_region,
)

FileStats = FileInfo

def copy_files(
    source: str,
    destination: str,
    source_filesystem: SupportedFileSystem | None = None,
    destination_filesystem: SupportedFileSystem | None = None,
    *,
    chunk_size: int = 1024 * 1024,
    use_threads: bool = True,
) -> None: ...

class FSSpecHandler(FileSystemHandler):  # type: ignore[misc]
    fs: SupportedFileSystem
    def __init__(self, fs: SupportedFileSystem) -> None: ...

__all__ = [
    # _fs
    "FileSelector",
    "FileType",
    "FileInfo",
    "FileSystem",
    "LocalFileSystem",
    "SubTreeFileSystem",
    "_MockFileSystem",
    "FileSystemHandler",
    "PyFileSystem",
    # _azurefs
    "AzureFileSystem",
    # _hdfs
    "HadoopFileSystem",
    # _gcsfs
    "GcsFileSystem",
    # _s3fs
    "AwsDefaultS3RetryStrategy",
    "AwsStandardS3RetryStrategy",
    "S3FileSystem",
    "S3LogLevel",
    "S3RetryStrategy",
    "ensure_s3_initialized",
    "finalize_s3",
    "ensure_s3_finalized",
    "initialize_s3",
    "resolve_s3_region",
    # fs
    "FileStats",
    "copy_files",
    "FSSpecHandler",
]
