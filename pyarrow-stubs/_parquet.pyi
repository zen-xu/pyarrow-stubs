from typing import Any, Literal, TypeAlias, TypedDict

from .lib import _Weakrefable

_PhysicalTypeName: TypeAlias = Literal[
    "BOOLEAN",
    "INT32",
    "INT64",
    "INT96",
    "FLOAT",
    "DOUBLE",
    "BYTE_ARRAY",
    "FIXED_LEN_BYTE_ARRAY",
    "UNKNOWN",
]
_LogicTypeName: TypeAlias = Literal[
    "UNDEFINED",
    "STRING",
    "MAP",
    "LIST",
    "ENUM",
    "DECIMAL",
    "DATE",
    "TIME",
    "TIMESTAMP",
    "INT",
    "FLOAT16",
    "JSON",
    "BSON",
    "UUID",
    "NONE",
    "UNKNOWN",
]
_ConvertedTypeName: TypeAlias = Literal[
    "NONE",
    "UTF8",
    "MAP",
    "MAP_KEY_VALUE",
    "LIST",
    "ENUM",
    "DECIMAL",
    "DATE",
    "TIME_MILLIS",
    "TIME_MICROS",
    "TIMESTAMP_MILLIS",
    "TIMESTAMP_MICROS",
    "UINT_8",
    "UINT_16",
    "UINT_32",
    "UINT_64",
    "INT_8",
    "INT_16",
    "INT_32",
    "INT_64",
    "JSON",
    "BSON",
    "INTERVAL",
    "UNKNOWN",
]
_EncodingName: TypeAlias = Literal[
    "PLAIN",
    "PLAIN_DICTIONARY",
    "RLE",
    "BIT_PACKED",
    "DELTA_BINARY_PACKED",
    "DELTA_LENGTH_BYTE_ARRAY",
    "DELTA_BYTE_ARRAY",
    "RLE_DICTIONARY",
    "BYTE_STREAM_SPLIT",
    "UNKNOWN",
]

class _Statistics(TypedDict):
    has_min_max: bool
    min: Any | None
    max: Any | None
    null_count: int | None
    distinct_count: int | None
    num_values: int
    physical_type: _PhysicalTypeName

class Statistics(_Weakrefable):
    def to_dict(self) -> _Statistics: ...
    def equals(self, other: Statistics) -> bool: ...
    @property
    def has_min_max(self) -> bool: ...
    @property
    def hash_null_count(self) -> bool: ...
    @property
    def has_distinct_count(self) -> bool: ...
    @property
    def min_raw(self) -> Any | None: ...
    @property
    def max_raw(self) -> Any | None: ...
    @property
    def min(self) -> Any | None: ...
    @property
    def max(self) -> Any | None: ...
    @property
    def null_count(self) -> int | None: ...
    @property
    def distinct_count(self) -> int | None: ...
    @property
    def num_values(self) -> int: ...
    @property
    def physical_type(self) -> _PhysicalTypeName: ...
    @property
    def logical_type(self) -> ParquetLogicalType: ...
    @property
    def converted_type(self) -> str | None: ...

class ParquetLogicalType(_Weakrefable):
    def to_json(self) -> str: ...
    @property
    def type(self) -> _LogicTypeName: ...

class _ColumnChunkMetaData(TypedDict):
    file_offset: int
    file_path: str | None
    physical_type: _PhysicalTypeName
    num_values: Any
    path_in_schema: Any
    is_stats_set: Any
    statistics: Any
    compression: Any
    encodings: Any
    has_dictionary_page: Any
    dictionary_page_offset: Any
    data_page_offset: Any
    total_compressed_size: Any
    total_uncompressed_size: Any

class ColumnChunkMetaData(_Weakrefable):
    def to_dict(self) -> _ColumnChunkMetaData: ...

class FileMetaData: ...
class FileDecryptionProperties: ...
