from os import PathLike
from typing import Callable
from typing import Iterable

from pyarrow._dataset import CsvFileFormat as CsvFileFormat
from pyarrow._dataset import CsvFragmentScanOptions as CsvFragmentScanOptions
from pyarrow._dataset import Dataset as Dataset
from pyarrow._dataset import DatasetFactory as DatasetFactory
from pyarrow._dataset import DirectoryPartitioning as DirectoryPartitioning
from pyarrow._dataset import FeatherFileFormat as FeatherFileFormat
from pyarrow._dataset import FileFormat as FileFormat
from pyarrow._dataset import FileFragment as FileFragment
from pyarrow._dataset import FilenamePartitioning as FilenamePartitioning
from pyarrow._dataset import FileSystemDataset as FileSystemDataset
from pyarrow._dataset import FileSystemDatasetFactory as FileSystemDatasetFactory
from pyarrow._dataset import FileSystemFactoryOptions as FileSystemFactoryOptions
from pyarrow._dataset import FileWriteOptions as FileWriteOptions
from pyarrow._dataset import Fragment as Fragment
from pyarrow._dataset import FragmentScanOptions as FragmentScanOptions
from pyarrow._dataset import HivePartitioning as HivePartitioning
from pyarrow._dataset import InMemoryDataset as InMemoryDataset
from pyarrow._dataset import IpcFileFormat as IpcFileFormat
from pyarrow._dataset import IpcFileWriteOptions as IpcFileWriteOptions
from pyarrow._dataset import Partitioning as Partitioning
from pyarrow._dataset import PartitioningFactory as PartitioningFactory
from pyarrow._dataset import Scanner as Scanner
from pyarrow._dataset import TaggedRecordBatch as TaggedRecordBatch
from pyarrow._dataset import UnionDataset as UnionDataset
from pyarrow._dataset import UnionDatasetFactory as UnionDatasetFactory
from pyarrow._dataset import WrittenFile as WrittenFile
from pyarrow._dataset_orc import OrcFileFormat as OrcFileFormat
from pyarrow._dataset_parquet import ParquetDatasetFactory as ParquetDatasetFactory
from pyarrow._dataset_parquet import ParquetFactoryOptions as ParquetFactoryOptions
from pyarrow._dataset_parquet import ParquetFileFormat as ParquetFileFormat
from pyarrow._dataset_parquet import ParquetFileFragment as ParquetFileFragment
from pyarrow._dataset_parquet import ParquetFileWriteOptions as ParquetFileWriteOptions
from pyarrow._dataset_parquet import ParquetFragmentScanOptions as ParquetFragmentScanOptions
from pyarrow._dataset_parquet import ParquetReadOptions as ParquetReadOptions
from pyarrow._dataset_parquet import RowGroupInfo as RowGroupInfo
from pyarrow.compute import Expression as Expression
from pyarrow.compute import field as field
from pyarrow.compute import scalar as scalar
from pyarrow.dataset import Dataset
from pyarrow.filesystem import FileSystem
from pyarrow.lib import Array
from pyarrow.lib import RecordBatch
from pyarrow.lib import RecordBatchReader
from pyarrow.lib import Schema
from pyarrow.lib import Table
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
    data: Dataset | Table | RecordBatch | RecordBatchReader | Iterable[Table | RecordBatch],
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
    existing_data_behavior: Literal["error", "overwrite_or_ignore", "delete_matching"] = ...,
    create_dir: bool = ...,
) -> None: ...
