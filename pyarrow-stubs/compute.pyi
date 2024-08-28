from typing import Literal, Sequence, TypeVar, overload

# Option classes
from pyarrow._compute import ArraySortOptions as ArraySortOptions
from pyarrow._compute import AssumeTimezoneOptions as AssumeTimezoneOptions
from pyarrow._compute import CastOptions as CastOptions
from pyarrow._compute import CountOptions as CountOptions
from pyarrow._compute import CumulativeOptions as CumulativeOptions
from pyarrow._compute import CumulativeSumOptions as CumulativeSumOptions
from pyarrow._compute import DayOfWeekOptions as DayOfWeekOptions
from pyarrow._compute import DictionaryEncodeOptions as DictionaryEncodeOptions
from pyarrow._compute import ElementWiseAggregateOptions as ElementWiseAggregateOptions

# Expressions
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
from pyarrow._compute import ListFlattenOptions as ListFlattenOptions
from pyarrow._compute import ListSliceOptions as ListSliceOptions
from pyarrow._compute import MakeStructOptions as MakeStructOptions
from pyarrow._compute import MapLookupOptions as MapLookupOptions
from pyarrow._compute import MatchSubstringOptions as MatchSubstringOptions
from pyarrow._compute import ModeOptions as ModeOptions
from pyarrow._compute import NullOptions as NullOptions
from pyarrow._compute import PadOptions as PadOptions
from pyarrow._compute import PairwiseOptions as PairwiseOptions
from pyarrow._compute import PartitionNthOptions as PartitionNthOptions
from pyarrow._compute import QuantileOptions as QuantileOptions
from pyarrow._compute import RandomOptions as RandomOptions
from pyarrow._compute import RankOptions as RankOptions
from pyarrow._compute import ReplaceSliceOptions as ReplaceSliceOptions
from pyarrow._compute import ReplaceSubstringOptions as ReplaceSubstringOptions
from pyarrow._compute import RoundBinaryOptions as RoundBinaryOptions
from pyarrow._compute import RoundOptions as RoundOptions
from pyarrow._compute import RoundTemporalOptions as RoundTemporalOptions
from pyarrow._compute import RoundToMultipleOptions as RoundToMultipleOptions
from pyarrow._compute import RunEndEncodeOptions as RunEndEncodeOptions
from pyarrow._compute import ScalarAggregateFunction as ScalarAggregateFunction
from pyarrow._compute import ScalarAggregateKernel as ScalarAggregateKernel
from pyarrow._compute import ScalarAggregateOptions as ScalarAggregateOptions
from pyarrow._compute import ScalarFunction as ScalarFunction
from pyarrow._compute import ScalarKernel as ScalarKernel
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
from pyarrow._compute import UdfContext as UdfContext
from pyarrow._compute import Utf8NormalizeOptions as Utf8NormalizeOptions
from pyarrow._compute import VarianceOptions as VarianceOptions
from pyarrow._compute import VectorFunction as VectorFunction
from pyarrow._compute import VectorKernel as VectorKernel
from pyarrow._compute import WeekOptions as WeekOptions

# Functions
from pyarrow._compute import call_function as call_function

# Udf
from pyarrow._compute import call_tabular_function as call_tabular_function
from pyarrow._compute import function_registry as function_registry
from pyarrow._compute import get_function as get_function
from pyarrow._compute import list_functions as list_functions
from pyarrow._compute import register_aggregate_function as register_aggregate_function
from pyarrow._compute import register_scalar_function as register_scalar_function
from pyarrow._compute import register_tabular_function as register_tabular_function
from pyarrow._compute import register_vector_function as register_vector_function
from pyarrow._stubs_typing import Indices

from . import lib

def cast(
    arr: lib.Array,
    target_type: str | lib.DataType,
    safe: bool = True,
    options: CastOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.Array: ...
def index(
    data: lib.Array,
    value: lib.Scalar,
    start: int | None = None,
    end: int | None = None,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> int: ...

_DataT = TypeVar("_DataT", bound=lib.Array | lib.ChunkedArray | lib.RecordBatch | lib.Table)

def take(
    data: _DataT,
    indices: Indices,
    *,
    boundscheck: bool = True,
    memory_pool: lib.MemoryPool | None = None,
) -> _DataT: ...
def fill_null(values: _DataT, fill_value: lib.Array | lib.ChunkedArray | lib.Scalar) -> _DataT: ...
@overload
def top_k_unstable(
    values: lib.Array | lib.ChunkedArray | lib.RecordBatch,
    k: int,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.Array: ...
@overload
def top_k_unstable(
    values: lib.Table,
    k: int,
    sort_keys: Sequence[str],
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.Array: ...
@overload
def bottom_k_unstable(
    values: lib.Array | lib.ChunkedArray | lib.RecordBatch,
    k: int,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.Array: ...
@overload
def bottom_k_unstable(
    values: lib.Table,
    k: int,
    sort_keys: Sequence[str],
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.Array: ...
def random(
    n: int,
    *,
    initializer: Literal["system"] | int = "system",
    options: RandomOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.DoubleArray: ...
def field(*name_or_index: str | tuple[str, ...] | int) -> Expression: ...
def scalar(value: bool | float | str) -> Expression: ...
