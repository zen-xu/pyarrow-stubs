from typing import TypeVar

from numpy.typing import ArrayLike
from pyarrow._compute import (
    ArraySortOptions as ArraySortOptions,
    AssumeTimezoneOptions as AssumeTimezoneOptions,
    CastOptions as CastOptions,
    CountOptions as CountOptions,
    CumulativeSumOptions as CumulativeSumOptions,
    DayOfWeekOptions as DayOfWeekOptions,
    DictionaryEncodeOptions as DictionaryEncodeOptions,
    ElementWiseAggregateOptions as ElementWiseAggregateOptions,
    Expression as Expression,
    ExtractRegexOptions as ExtractRegexOptions,
    FilterOptions as FilterOptions,
    Function as Function,
    FunctionOptions as FunctionOptions,
    FunctionRegistry as FunctionRegistry,
    HashAggregateFunction as HashAggregateFunction,
    HashAggregateKernel as HashAggregateKernel,
    IndexOptions as IndexOptions,
    JoinOptions as JoinOptions,
    Kernel as Kernel,
    MakeStructOptions as MakeStructOptions,
    MapLookupOptions as MapLookupOptions,
    MatchSubstringOptions as MatchSubstringOptions,
    ModeOptions as ModeOptions,
    NullOptions as NullOptions,
    PadOptions as PadOptions,
    PartitionNthOptions as PartitionNthOptions,
    QuantileOptions as QuantileOptions,
    RandomOptions as RandomOptions,
    RankOptions as RankOptions,
    ReplaceSliceOptions as ReplaceSliceOptions,
    ReplaceSubstringOptions as ReplaceSubstringOptions,
    RoundOptions as RoundOptions,
    RoundTemporalOptions as RoundTemporalOptions,
    RoundToMultipleOptions as RoundToMultipleOptions,
    ScalarAggregateFunction as ScalarAggregateFunction,
    ScalarAggregateKernel as ScalarAggregateKernel,
    ScalarAggregateOptions as ScalarAggregateOptions,
    ScalarFunction as ScalarFunction,
    ScalarKernel as ScalarKernel,
    ScalarUdfContext as ScalarUdfContext,
    SelectKOptions as SelectKOptions,
    SetLookupOptions as SetLookupOptions,
    SliceOptions as SliceOptions,
    SortOptions as SortOptions,
    SplitOptions as SplitOptions,
    SplitPatternOptions as SplitPatternOptions,
    StrftimeOptions as StrftimeOptions,
    StrptimeOptions as StrptimeOptions,
    StructFieldOptions as StructFieldOptions,
    TakeOptions as TakeOptions,
    TDigestOptions as TDigestOptions,
    TrimOptions as TrimOptions,
    Utf8NormalizeOptions as Utf8NormalizeOptions,
    VarianceOptions as VarianceOptions,
    VectorFunction as VectorFunction,
    VectorKernel as VectorKernel,
    WeekOptions as WeekOptions,
    call_function as call_function,
    function_registry as function_registry,
    get_function as get_function,
    list_functions as list_functions,
    register_scalar_function as register_scalar_function,
)
from pyarrow.lib import (
    Array,
    ChunkedArray,
    DataType,
    MemoryPool,
    RecordBatch,
    Scalar,
    Table,
)
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
