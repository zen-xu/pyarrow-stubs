from io import IOBase

import pandas as pd
from pyarrow import ipc
import pyarrow.lib as lib
from pyarrow.lib import (
    Buffer,
    IpcReadOptions as IpcReadOptions,
    IpcWriteOptions as IpcWriteOptions,
    MemoryPool,
    Message as Message,
    MessageReader as MessageReader,
    MetadataVersion as MetadataVersion,
    NativeFile,
    ReadStats as ReadStats,
    RecordBatchReader as RecordBatchReader,
    Schema,
    WriteStats as WriteStats,
    get_record_batch_size as get_record_batch_size,
    get_tensor_size as get_tensor_size,
    read_message as read_message,
    read_record_batch as read_record_batch,
    read_schema as read_schema,
    read_tensor as read_tensor,
    write_tensor as write_tensor,
)

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
def deserialize_pandas(
    buf: memoryview | Buffer, *, use_threads: bool = ...
) -> pd.DataFrame: ...
