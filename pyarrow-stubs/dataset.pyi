from os import PathLike
from typing import (
    Callable,
    Iterable,
)

from pyarrow._dataset import (
    CsvFileFormat as CsvFileFormat,
    CsvFragmentScanOptions as CsvFragmentScanOptions,
    Dataset as Dataset,
    DatasetFactory as DatasetFactory,
    DirectoryPartitioning as DirectoryPartitioning,
    FeatherFileFormat as FeatherFileFormat,
    FileFormat as FileFormat,
    FileFragment as FileFragment,
    FilenamePartitioning as FilenamePartitioning,
    FileSystemDataset as FileSystemDataset,
    FileSystemDatasetFactory as FileSystemDatasetFactory,
    FileSystemFactoryOptions as FileSystemFactoryOptions,
    FileWriteOptions as FileWriteOptions,
    Fragment as Fragment,
    FragmentScanOptions as FragmentScanOptions,
    HivePartitioning as HivePartitioning,
    InMemoryDataset as InMemoryDataset,
    IpcFileFormat as IpcFileFormat,
    IpcFileWriteOptions as IpcFileWriteOptions,
    Partitioning as Partitioning,
    PartitioningFactory as PartitioningFactory,
    Scanner as Scanner,
    TaggedRecordBatch as TaggedRecordBatch,
    UnionDataset as UnionDataset,
    UnionDatasetFactory as UnionDatasetFactory,
    WrittenFile as WrittenFile,
)
from pyarrow._dataset_orc import OrcFileFormat as OrcFileFormat
from pyarrow._dataset_parquet import (
    ParquetDatasetFactory as ParquetDatasetFactory,
    ParquetFactoryOptions as ParquetFactoryOptions,
    ParquetFileFormat as ParquetFileFormat,
    ParquetFileFragment as ParquetFileFragment,
    ParquetFileWriteOptions as ParquetFileWriteOptions,
    ParquetFragmentScanOptions as ParquetFragmentScanOptions,
    ParquetReadOptions as ParquetReadOptions,
    RowGroupInfo as RowGroupInfo,
)
from pyarrow.compute import (
    Expression as Expression,
    field as field,
    scalar as scalar,
)
from pyarrow.dataset import Dataset
from pyarrow.filesystem import FileSystem
from pyarrow.lib import (
    Array,
    RecordBatch,
    RecordBatchReader,
    Schema,
    Table,
)
from typing_extensions import Literal

def __getattr__(name: str) -> None: ...
def partitioning(
    schema: Schema | None = ...,
    field_names: list[str] | None = ...,
    flavor: str | None = ...,
    dictionaries: dict[str, Array] | None = ...,
) -> Partitioning | PartitioningFactory: ...
def parquet_dataset(
    metadata_path: str | PathLike,
    schema: Schema | None = ...,
    filesystem: FileSystem | str | None = ...,
    format: ParquetFileFormat | str | None = ...,
    partitioning: Partitioning | PartitioningFactory | str | list[str] | None = ...,
    partition_base_dir: str | None = ...,
) -> FileSystemDataset: ...
def dataset(
    source: str | Dataset | Iterable[str | Dataset | RecordBatch | RecordBatchReader],
    schema: Schema | None = ...,
    format: FileFormat | str | None = ...,
    filesystem: FileSystem | str | None = ...,
    partitioning: Partitioning | PartitioningFactory | str | list[str] | None = ...,
    partition_base_dir: str | None = ...,
    exclude_invalid_files: bool | None = ...,
    ignore_prefixes: list[str] | None = ...,
) -> Dataset: ...
def write_dataset(
    data: Dataset
    | Table
    | RecordBatch
    | RecordBatchReader
    | Iterable[Table | RecordBatch],
    base_dir: str,
    *,
    basename_template: str | None = ...,
    format: FileFormat | str | None = ...,
    partitioning: Partitioning | list[str] | None = ...,
    partitioning_flavor: str | None = ...,
    schema: Schema | None = ...,
    filesystem: FileSystem | None = ...,
    file_options: FileWriteOptions | None = ...,
    use_threads: bool = ...,
    max_partitions: int | None = ...,
    max_open_files: int | None = ...,
    max_rows_per_file: int | None = ...,
    min_rows_per_group: int | None = ...,
    max_rows_per_group: int | None = ...,
    file_visitor: Callable[[WrittenFile], None] | None = ...,
    existing_data_behavior: Literal[
        "error", "overwrite_or_ignore", "delete_matching"
    ] = ...,
    create_dir: bool = ...,
) -> None: ...
