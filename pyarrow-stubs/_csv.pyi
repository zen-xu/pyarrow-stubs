from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Callable, Literal

from . import lib

@dataclass(kw_only=True)
class ReadOptions(lib._Weakrefable):
    use_threads: bool = field(default=True, kw_only=False)
    block_size: int | None = None
    skip_rows: int = 0
    skip_rows_after_names: int = 0
    column_names: list[str] | None = None
    autogenerate_column_names: bool = False
    encoding: str = "utf8"

    def validate(self) -> None: ...

@dataclass(kw_only=True)
class ParseOptions(lib._Weakrefable):
    delimiter: str = field(default=",", kw_only=False)
    quote_char: str | Literal[False] = '"'
    double_quote: bool = True
    escape_char: str | Literal[False] = False
    newlines_in_values: bool = False
    ignore_empty_lines: bool = True
    invalid_row_handler: Callable[[InvalidRow], Literal["skip", "error"]] | None = None

    def validate(self) -> None: ...

@dataclass(kw_only=True)
class ConvertOptions(lib._Weakrefable):
    check_utf8: bool = field(default=True, kw_only=False)
    check_types: lib.Schema | dict | None = None
    null_values: list[str] | None = None
    true_values: list[str] | None = None
    false_values: list[str] | None = None
    decimal_point: str = "."
    strings_can_be_null: bool = False
    quoted_strings_can_be_null: bool = True
    include_missing_columns: bool = False
    auto_dict_encode: bool = False
    auto_dict_max_cardinality: int | None = None
    timestamp_parsers: list[str] | None = None

    def validate(self) -> None: ...

@dataclass(kw_only=True)
class WriteOptions(lib._Weakrefable):
    include_header: bool = field(default=True, kw_only=False)
    batch_size: int = 1024
    delimiter: str = ","
    quoting_style: Literal["needed", "all_valid", "none"] = "needed"

    def validate(self) -> None: ...

@dataclass
class InvalidRow(lib._Weakrefable):
    expected_columns: int
    actual_columns: int
    number: int | None
    text: str

class CSVWriter(lib._CRecordBatchWriter):
    def __init__(
        self,
        # TODO: OutputStream
        sink: str | Path | IO,
        schema: lib.Schema,
        write_options: WriteOptions | None = None,
        *,
        memory_pool: lib.MemoryPool | None = None,
    ) -> None: ...

class CSVStreamingReader(lib.RecordBatchReader): ...

ISO8601: lib._Weakrefable

def open_csv(
    input_file: str | Path | IO,
    read_options: ReadOptions | None = None,
    parse_options: ParseOptions | None = None,
    convert_options: ConvertOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> CSVStreamingReader: ...
def read_csv(
    input_file: str | Path | IO,
    read_options: ReadOptions | None = None,
    parse_options: ParseOptions | None = None,
    convert_options: ConvertOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.Table: ...
def write_csv(
    data: lib.RecordBatch | lib.Table,
    output_file: str | Path | lib.NativeFile | IO,
    write_options: WriteOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> None: ...
