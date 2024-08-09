from typing import TypeVar

from numpy.typing import ArrayLike
from pyarrow._compute import ArraySortOptions as ArraySortOptions
from pyarrow._compute import AssumeTimezoneOptions as AssumeTimezoneOptions
from pyarrow._compute import CastOptions as CastOptions
from pyarrow._compute import CountOptions as CountOptions
from pyarrow._compute import CumulativeSumOptions as CumulativeSumOptions
from pyarrow._compute import DayOfWeekOptions as DayOfWeekOptions
from pyarrow._compute import DictionaryEncodeOptions as DictionaryEncodeOptions
from pyarrow._compute import ElementWiseAggregateOptions as ElementWiseAggregateOptions
from pyarrow._compute import Expression as Expression
from pyarrow._compute import ExtractRegexOptions as ExtractRegexOptions
from pyarrow._compute import FilterOptions as FilterOptions
from pyarrow._compute import Function as Function
from pyarrow._compute import FunctionOptions as FunctionOptions
from pyarrow._compute import FunctionRegistry as FunctionRegistry
from pyarrow._compute import HashAggregateFunction as HashAggregateFunction
from pyarrow._compute import HashAggregateKernel as HashAggregateKernel
from pyarrow._compute import IndexOptions as IndexOptions
from pyarrow._compute import JoinOptions as JoinOptions
from pyarrow._compute import Kernel as Kernel
from pyarrow._compute import MakeStructOptions as MakeStructOptions
from pyarrow._compute import MapLookupOptions as MapLookupOptions
from pyarrow._compute import MatchSubstringOptions as MatchSubstringOptions
from pyarrow._compute import ModeOptions as ModeOptions
from pyarrow._compute import NullOptions as NullOptions
from pyarrow._compute import PadOptions as PadOptions
from pyarrow._compute import PartitionNthOptions as PartitionNthOptions
from pyarrow._compute import QuantileOptions as QuantileOptions
from pyarrow._compute import RandomOptions as RandomOptions
from pyarrow._compute import RankOptions as RankOptions
from pyarrow._compute import ReplaceSliceOptions as ReplaceSliceOptions
from pyarrow._compute import ReplaceSubstringOptions as ReplaceSubstringOptions
from pyarrow._compute import RoundOptions as RoundOptions
from pyarrow._compute import RoundTemporalOptions as RoundTemporalOptions
from pyarrow._compute import RoundToMultipleOptions as RoundToMultipleOptions
from pyarrow._compute import ScalarAggregateFunction as ScalarAggregateFunction
from pyarrow._compute import ScalarAggregateKernel as ScalarAggregateKernel
from pyarrow._compute import ScalarAggregateOptions as ScalarAggregateOptions
from pyarrow._compute import ScalarFunction as ScalarFunction
from pyarrow._compute import ScalarKernel as ScalarKernel
from pyarrow._compute import ScalarUdfContext as ScalarUdfContext
from pyarrow._compute import SelectKOptions as SelectKOptions
from pyarrow._compute import SetLookupOptions as SetLookupOptions
from pyarrow._compute import SliceOptions as SliceOptions
from pyarrow._compute import SortOptions as SortOptions
from pyarrow._compute import SplitOptions as SplitOptions
from pyarrow._compute import SplitPatternOptions as SplitPatternOptions
from pyarrow._compute import StrftimeOptions as StrftimeOptions
from pyarrow._compute import StrptimeOptions as StrptimeOptions
from pyarrow._compute import StructFieldOptions as StructFieldOptions
from pyarrow._compute import TakeOptions as TakeOptions
from pyarrow._compute import TDigestOptions as TDigestOptions
from pyarrow._compute import TrimOptions as TrimOptions
from pyarrow._compute import Utf8NormalizeOptions as Utf8NormalizeOptions
from pyarrow._compute import VarianceOptions as VarianceOptions
from pyarrow._compute import VectorFunction as VectorFunction
from pyarrow._compute import VectorKernel as VectorKernel
from pyarrow._compute import WeekOptions as WeekOptions
from pyarrow._compute import call_function as call_function
from pyarrow._compute import function_registry as function_registry
from pyarrow._compute import get_function as get_function
from pyarrow._compute import list_functions as list_functions
from pyarrow._compute import register_scalar_function as register_scalar_function
from pyarrow.lib import Array
from pyarrow.lib import ChunkedArray
from pyarrow.lib import DataType
from pyarrow.lib import MemoryPool
from pyarrow.lib import RecordBatch
from pyarrow.lib import Scalar
from pyarrow.lib import Table
from pyarrow.vendored import docscrape as docscrape

def cast(
    arr: ArrayLike,
    target_type: DataType | str | None = ...,
    safe: bool | None = ...,
    options: CastOptions | None = ...,
) -> Array: ...
def index(
    data: ArrayLike,
    value: Scalar,
    start: int | None = ...,
    end: int | None = ...,
    *,
    memory_pool: MemoryPool | None = ...,
) -> int: ...

_TakeData = TypeVar("_TakeData", Array, ChunkedArray, RecordBatch, Table)

def take(
    data: _TakeData,
    indices: Array | ChunkedArray,
    *,
    boundscheck: bool = ...,
    memory_pool: MemoryPool | None = ...,
) -> _TakeData: ...

_FillValues = TypeVar("_FillValues", bound=Array | ChunkedArray | Scalar)
_FillValue = TypeVar("_FillValue", bound=Array | ChunkedArray | Scalar)

def fill_null(values: _FillValues, fill_value: _FillValue) -> _FillValues: ...
def top_k_unstable(
    values: Array | ChunkedArray | RecordBatch | Table,
    k: int,
    sort_keys: list[str] | None = ...,
    *,
    memory_pool: MemoryPool | None = ...,
) -> Array: ...
def bottom_k_unstable(
    values: Array | ChunkedArray | RecordBatch | Table,
    k: int,
    sort_keys: list[str] | None = ...,
    *,
    memory_pool: MemoryPool | None = ...,
) -> Array: ...
def random(
    n: int,
    *,
    initializer: int | str = ...,
    options: RandomOptions | None = ...,
    memory_pool: MemoryPool | None = ...,
) -> Array: ...
def field(*name_or_index: int | str | tuple[int | str]): ...
def scalar(value: bool | int | float | str) -> Expression: ...
