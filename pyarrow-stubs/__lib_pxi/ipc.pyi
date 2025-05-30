import enum
import sys

from io import IOBase

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self
from typing import Iterable, Iterator, Literal, Mapping, NamedTuple

import pandas as pd

from pyarrow._stubs_typing import SupportArrowStream, SupportPyBuffer
from pyarrow.lib import MemoryPool, RecordBatch, Schema, Table, Tensor, _Weakrefable

from .io import Buffer, Codec, NativeFile
from .types import DictionaryMemo, KeyValueMetadata

class MetadataVersion(enum.IntEnum):
    V1 = enum.auto()
    V2 = enum.auto()
    V3 = enum.auto()
    V4 = enum.auto()
    V5 = enum.auto()

class WriteStats(NamedTuple):
    num_messages: int
    num_record_batches: int
    num_dictionary_batches: int
    num_dictionary_deltas: int
    num_replaced_dictionaries: int

class ReadStats(NamedTuple):
    num_messages: int
    num_record_batches: int
    num_dictionary_batches: int
    num_dictionary_deltas: int
    num_replaced_dictionaries: int

class IpcReadOptions(_Weakrefable):
    ensure_native_endian: bool
    use_threads: bool
    included_fields: list[int]
    def __init__(
        self,
        *,
        ensure_native_endian: bool = True,
        use_threads: bool = True,
        included_fields: list[int] | None = None,
    ) -> None: ...

class IpcWriteOptions(_Weakrefable):
    metadata_version: MetadataVersion
    allow_64bit: bool
    use_legacy_format: bool
    compression: Codec | Literal["lz4", "zstd"] | None
    use_threads: bool
    emit_dictionary_deltas: bool
    unify_dictionaries: bool
    def __init__(
        self,
        *,
        metadata_version: MetadataVersion = MetadataVersion.V5,
        allow_64bit: bool = False,
        use_legacy_format: bool = False,
        compression: Codec | Literal["lz4", "zstd"] | None = None,
        use_threads: bool = True,
        emit_dictionary_deltas: bool = False,
        unify_dictionaries: bool = False,
    ) -> None: ...

class Message(_Weakrefable):
    @property
    def type(self) -> str: ...
    @property
    def metadata(self) -> Buffer: ...
    @property
    def metadata_version(self) -> MetadataVersion: ...
    @property
    def body(self) -> Buffer | None: ...
    def equals(self, other: Message) -> bool: ...
    def serialize_to(
        self, sink: NativeFile, alignment: int = 8, memory_pool: MemoryPool | None = None
    ): ...
    def serialize(self, alignment: int = 8, memory_pool: MemoryPool | None = None) -> Buffer: ...

class MessageReader(_Weakrefable):
    @classmethod
    def open_stream(cls, source: bytes | NativeFile | IOBase | SupportPyBuffer) -> Self: ...
    def __iter__(self) -> Self: ...
    def read_next_message(self) -> Message: ...
    __next__ = read_next_message

# ----------------------------------------------------------------------
# File and stream readers and writers

class _CRecordBatchWriter(_Weakrefable):
    def write(self, table_or_batch: Table | RecordBatch): ...
    def write_batch(
        self,
        batch: RecordBatch,
        custom_metadata: Mapping[bytes, bytes] | KeyValueMetadata | None = None,
    ): ...
    def write_table(self, table: Table, max_chunksize: int | None = None) -> None: ...
    def close(self) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(self, exc_type, exc_val, exc_tb): ...
    @property
    def stats(self) -> WriteStats: ...

class _RecordBatchStreamWriter(_CRecordBatchWriter):
    def __dealloc__(self) -> None: ...
    def _open(self, sink, schema: Schema, options: IpcWriteOptions = IpcWriteOptions()): ...

class _ReadPandasMixin:
    def read_pandas(self, **options) -> pd.DataFrame: ...

class RecordBatchReader(_Weakrefable):
    def __iter__(self) -> Self: ...
    def read_next_batch(self) -> RecordBatch: ...
    __next__ = read_next_batch
    @property
    def schema(self) -> Schema: ...
    def read_next_batch_with_custom_metadata(self) -> RecordBatchWithMetadata: ...
    def iter_batches_with_custom_metadata(
        self,
    ) -> Iterator[RecordBatchWithMetadata]: ...
    def read_all(self) -> Table: ...
    read_pandas = _ReadPandasMixin.read_pandas  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
    def close(self) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(self, exc_type, exc_val, exc_tb): ...
    def cast(self, target_schema: Schema) -> Self: ...
    def _export_to_c(self, out_ptr: int) -> None: ...
    @classmethod
    def _import_from_c(cls, in_ptr: int) -> Self: ...
    def __arrow_c_stream__(self, requested_schema=None): ...
    @classmethod
    def _import_from_c_capsule(cls, stream) -> Self: ...
    @classmethod
    def from_stream(cls, data: SupportArrowStream, schema: Schema | None = None) -> Self: ...
    @classmethod
    def from_batches(cls, schema: Schema, batches: Iterable[RecordBatch]) -> Self: ...

class _RecordBatchStreamReader(RecordBatchReader):
    @property
    def stats(self) -> ReadStats: ...

class _RecordBatchFileWriter(_RecordBatchStreamWriter): ...

class RecordBatchWithMetadata(NamedTuple):
    batch: RecordBatch
    custom_metadata: KeyValueMetadata

class _RecordBatchFileReader(_Weakrefable):
    @property
    def num_record_batches(self) -> int: ...
    def get_batch(self, i: int) -> RecordBatch: ...
    get_record_batch = get_batch
    def get_batch_with_custom_metadata(self, i: int) -> RecordBatchWithMetadata: ...
    def read_all(self) -> Table: ...
    read_pandas = _ReadPandasMixin.read_pandas  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
    def __enter__(self) -> Self: ...
    def __exit__(self, exc_type, exc_val, exc_tb): ...
    @property
    def schema(self) -> Schema: ...
    @property
    def stats(self) -> ReadStats: ...

def get_tensor_size(tensor: Tensor) -> int: ...
def get_record_batch_size(batch: RecordBatch) -> int: ...
def write_tensor(tensor: Tensor, dest: NativeFile) -> int: ...
def read_tensor(source: NativeFile) -> Tensor: ...
def read_message(source: NativeFile | IOBase | SupportPyBuffer) -> Message: ...
def read_schema(
    obj: Buffer | Message, dictionary_memo: DictionaryMemo | None = None
) -> Schema: ...
def read_record_batch(
    obj: Message | SupportPyBuffer, schema: Schema, dictionary_memo: DictionaryMemo | None = None
) -> RecordBatch: ...

__all__ = [
    "MetadataVersion",
    "WriteStats",
    "ReadStats",
    "IpcReadOptions",
    "IpcWriteOptions",
    "Message",
    "MessageReader",
    "_CRecordBatchWriter",
    "_RecordBatchStreamWriter",
    "_ReadPandasMixin",
    "RecordBatchReader",
    "_RecordBatchStreamReader",
    "_RecordBatchFileWriter",
    "RecordBatchWithMetadata",
    "_RecordBatchFileReader",
    "get_tensor_size",
    "get_record_batch_size",
    "write_tensor",
    "read_tensor",
    "read_message",
    "read_schema",
    "read_record_batch",
]
