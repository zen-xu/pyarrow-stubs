from io import IOBase

import pandas as pd
import pyarrow.lib as lib

from pyarrow import ipc
from pyarrow.lib import Buffer
from pyarrow.lib import IpcReadOptions as IpcReadOptions
from pyarrow.lib import IpcWriteOptions as IpcWriteOptions
from pyarrow.lib import MemoryPool
from pyarrow.lib import Message as Message
from pyarrow.lib import MessageReader as MessageReader
from pyarrow.lib import MetadataVersion as MetadataVersion
from pyarrow.lib import NativeFile
from pyarrow.lib import ReadStats as ReadStats
from pyarrow.lib import RecordBatchReader as RecordBatchReader
from pyarrow.lib import Schema
from pyarrow.lib import WriteStats as WriteStats
from pyarrow.lib import get_record_batch_size as get_record_batch_size
from pyarrow.lib import get_tensor_size as get_tensor_size
from pyarrow.lib import read_message as read_message
from pyarrow.lib import read_record_batch as read_record_batch
from pyarrow.lib import read_schema as read_schema
from pyarrow.lib import read_tensor as read_tensor
from pyarrow.lib import write_tensor as write_tensor

class RecordBatchStreamReader(lib._RecordBatchStreamReader):
    def __init__(
        self,
        source: bytes | memoryview | Buffer | NativeFile | IOBase,
        *,
        options: ipc.IpcReadOptions | None = ...,
        memory_pool: MemoryPool | None = ...,
    ) -> None: ...

class RecordBatchStreamWriter(lib._RecordBatchStreamWriter):
    def __init__(
        self,
        sink: str | Buffer | NativeFile | IOBase,
        schema: Schema,
        *,
        use_legacy_format: bool | None = ...,
        options: ipc.IpcWriteOptions | None = ...,
    ) -> None: ...

class RecordBatchFileReader(lib._RecordBatchFileReader):
    def __init__(
        self,
        source: bytes | memoryview | Buffer | NativeFile | IOBase,
        footer_offset: int | None = ...,
        *,
        options: ipc.IpcReadOptions | None = ...,
        memory_pool: MemoryPool | None = ...,
    ) -> None: ...

class RecordBatchFileWriter(lib._RecordBatchFileWriter):
    def __init__(
        self,
        sink: str | Buffer | NativeFile | IOBase,
        schema: Schema,
        *,
        use_legacy_format: bool | None = ...,
        options: ipc.IpcWriteOptions | None = ...,
    ) -> None: ...

def new_stream(
    sink: str | Buffer | NativeFile | IOBase,
    schema: Schema,
    *,
    use_legacy_format: bool | None = ...,
    options: ipc.IpcWriteOptions | None = ...,
) -> RecordBatchStreamWriter: ...
def open_stream(
    source: bytes | memoryview | Buffer | NativeFile | IOBase,
    *,
    options: ipc.IpcReadOptions | None = ...,
    memory_pool: MemoryPool | None = ...,
) -> RecordBatchStreamReader: ...
def new_file(
    sink: str | NativeFile | IOBase,
    schema: Schema,
    *,
    use_legacy_format: bool | None = ...,
    options: ipc.IpcWriteOptions | None = ...,
) -> RecordBatchFileWriter: ...
def open_file(
    source: bytes | memoryview | Buffer | NativeFile | IOBase,
    footer_offset: int | None = ...,
    *,
    options: ipc.IpcReadOptions | None = ...,
    memory_pool: MemoryPool | None = ...,
) -> RecordBatchFileReader: ...
def serialize_pandas(
    df: pd.DataFrame,
    *,
    nthreads: int | None = ...,
    preserve_index: bool | None = ...,
) -> Buffer: ...
def deserialize_pandas(buf: memoryview | Buffer, *, use_threads: bool = ...) -> pd.DataFrame: ...
