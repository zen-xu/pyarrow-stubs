# mypy: disable-error-code="misc"
# ruff: noqa: I001
from decimal import Decimal
from typing import Literal, Sequence, TypeAlias, TypeVar, overload

import numpy as np

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
from pyarrow._stubs_typing import Indices, NullSelectionBehavior, Order

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

# ============= compute functions =============
NumericScalar: TypeAlias = (
    lib.Scalar[lib.Int8Type]
    | lib.Scalar[lib.Int16Type]
    | lib.Scalar[lib.Int32Type]
    | lib.Scalar[lib.Int64Type]
    | lib.Scalar[lib.Uint8Type]
    | lib.Scalar[lib.Uint16Type]
    | lib.Scalar[lib.Uint32Type]
    | lib.Scalar[lib.Uint64Type]
    | lib.Scalar[lib.Float16Type]
    | lib.Scalar[lib.Float32Type]
    | lib.Scalar[lib.Float64Type]
    | lib.Scalar[lib.Decimal128Type]
    | lib.Scalar[lib.Decimal256Type]
)
_NumericScalarT = TypeVar("_NumericScalarT", bound=NumericScalar)
_NumericArrayT = TypeVar("_NumericArrayT", bound=lib.NumericArray)
Number: TypeAlias = int | float | Decimal
_T = TypeVar("_T")
NullableList: TypeAlias = list[_T] | list[_T | None]
NullableNumbers: TypeAlias = NullableList[int] | NullableList[float] | NullableList[Decimal]
_ArrayT = TypeVar("_ArrayT", bound=lib.Array)

@overload
def abs(x: int, /, *, memory_pool: lib.MemoryPool | None = None) -> lib.Int64Scalar: ...
@overload
def abs(x: float, /, *, memory_pool: lib.MemoryPool | None = None) -> lib.DoubleScalar: ...
@overload
def abs(x: Decimal, /, *, memory_pool: lib.MemoryPool | None = None) -> lib.Decimal128Scalar: ...
@overload
def abs(
    x: NullableList[int], /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.Int64Array: ...
@overload
def abs(
    x: NullableList[float], /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.DoubleArray: ...
@overload
def abs(
    x: NullableList[Decimal], /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.Decimal128Array: ...
@overload
def abs(
    x: NullableList[_NumericScalarT], /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.NumericArray[_NumericScalarT]: ...
@overload
def abs(
    x: _NumericScalarT, /, *, memory_pool: lib.MemoryPool | None = None
) -> _NumericScalarT: ...
@overload
def abs(x: _NumericArrayT, /, *, memory_pool: lib.MemoryPool | None = None) -> _NumericArrayT: ...
@overload
def abs(x: np.ndarray, /, *, memory_pool: lib.MemoryPool | None = None) -> lib.NumericArray: ...

abs_checked = abs

@overload
def acos(
    x: NumericScalar | Number, /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.DoubleScalar: ...
@overload
def acos(
    x: NullableList[Number] | lib.Array[NumericScalar],
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.DoubleArray: ...

acos_checked = acos
asin = acos
asin_checked = acos
atan = acos

@overload
def atan2(y: int, x: int, /, *, memory_pool: lib.MemoryPool | None = None) -> lib.Int64Scalar: ...
@overload
def atan2(
    y: Decimal, x: int, /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.Decimal128Scalar: ...
@overload
def atan2(
    y: int, x: Decimal, /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.Decimal128Scalar: ...
@overload
def atan2(
    y: Decimal, x: Decimal, /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.Decimal128Scalar: ...
@overload
def atan2(
    y: float, x: float, /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.DoubleScalar: ...
@overload
def atan2(
    y: Decimal, x: float, /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.DoubleScalar: ...
@overload
def atan2(
    y: float, x: Decimal, /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.DoubleScalar: ...
@overload
def atan2(
    y: NumericScalar, x: NumericScalar, /, *, memory_pool: lib.MemoryPool | None = None
) -> NumericScalar: ...
@overload
def atan2(
    y: NullableList[int], x: NullableList[int], /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.Int16Array: ...
@overload
def atan2(
    y: NullableList[Decimal], x: NullableList[int], /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.Decimal128Array: ...
@overload
def atan2(
    y: NullableList[int], x: NullableList[Decimal], /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.Decimal128Array: ...
@overload
def atan2(
    y: NullableList[Decimal],
    x: NullableList[Decimal],
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.Decimal128Array: ...
@overload
def atan2(
    y: NullableList[float], x: NullableList[float], /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.DoubleArray: ...
@overload
def atan2(
    y: NullableList[Decimal],
    x: NullableList[float],
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.DoubleArray: ...
@overload
def atan2(
    y: NullableList[float],
    x: NullableList[Decimal],
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.DoubleArray: ...
@overload
def atan2(
    y: lib.Array[NumericScalar],
    x: lib.Array[NumericScalar],
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.Array[NumericScalar]: ...
@overload
def atan2(
    y: Number,
    x: lib.Array[NumericScalar] | NullableNumbers,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.Array[NumericScalar]: ...
@overload
def atan2(
    y: lib.Array[NumericScalar] | NullableNumbers,
    x: Number,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.Array[NumericScalar]: ...
@overload
def add(x: int, y: int, /, *, memory_pool: lib.MemoryPool | None = None) -> lib.Int64Scalar: ...
@overload
def add(
    x: Decimal, y: int, /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.Decimal128Scalar: ...
@overload
def add(
    x: int, y: Decimal, /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.Decimal128Scalar: ...
@overload
def add(
    x: Decimal, y: Decimal, /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.Decimal128Scalar: ...
@overload
def add(
    x: float, y: float, /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.DoubleScalar: ...
@overload
def add(
    x: Decimal, y: float, /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.DoubleScalar: ...
@overload
def add(
    x: float, y: Decimal, /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.DoubleScalar: ...
@overload
def add(
    x: NumericScalar, y: NumericScalar, /, *, memory_pool: lib.MemoryPool | None = None
) -> NumericScalar: ...
@overload
def add(
    x: NullableList[int], y: NullableList[int], /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.Int16Array: ...
@overload
def add(
    x: NullableList[Decimal], y: NullableList[int], /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.Decimal128Array: ...
@overload
def add(
    x: NullableList[int], y: NullableList[Decimal], /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.Decimal128Array: ...
@overload
def add(
    x: NullableList[Decimal],
    y: NullableList[Decimal],
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.Decimal128Array: ...
@overload
def add(
    x: NullableList[float], y: NullableList[float], /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.DoubleArray: ...
@overload
def add(
    x: NullableList[Decimal],
    y: NullableList[float],
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.DoubleArray: ...
@overload
def add(
    x: NullableList[float],
    y: NullableList[Decimal],
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.DoubleArray: ...
@overload
def add(
    x: lib.Array[NumericScalar],
    y: lib.Array[NumericScalar],
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.Array[NumericScalar]: ...
@overload
def add(
    x: Number,
    y: lib.Array[NumericScalar] | NullableNumbers,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.Array[NumericScalar]: ...
@overload
def add(
    x: lib.Array[NumericScalar] | NullableNumbers,
    y: Number,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.Array[NumericScalar]: ...

add_checked = add

def all(
    array: NullableList[bool] | lib.BooleanArray,
    /,
    *,
    skip_nulls: bool = True,
    min_count: int = 1,
    options: ScalarAggregateOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.BooleanScalar: ...
def and_(
    x: NullableList[bool] | lib.BooleanArray,
    y: NullableList[bool] | lib.BooleanArray,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.BooleanArray: ...

and_kleene = and_
and_not = and_
and_not_kleene = and_
any = all

def approximate_median(
    array: lib.NumericArray | NullableNumbers,
    /,
    *,
    skip_nulls: bool = True,
    min_count: int = 1,
    options: ScalarAggregateOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.DoubleScalar: ...
@overload
def array_filter(
    array: NullableList[int],
    selection_filter: NullableList[bool] | lib.BooleanArray,
    /,
    null_selection_behavior: NullSelectionBehavior = "drop",
    *,
    options: FilterOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.Int64Array: ...
@overload
def array_filter(
    array: NullableList[float],
    selection_filter: NullableList[bool] | lib.BooleanArray,
    /,
    null_selection_behavior: NullSelectionBehavior = "drop",
    *,
    options: FilterOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.DoubleArray: ...
@overload
def array_filter(
    array: NullableList[Decimal],
    selection_filter: NullableList[bool] | lib.BooleanArray,
    /,
    null_selection_behavior: NullSelectionBehavior = "drop",
    *,
    options: FilterOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.Decimal128Array: ...
@overload
def array_filter(
    array: _ArrayT,
    selection_filter: NullableList[bool] | lib.BooleanArray,
    /,
    null_selection_behavior: NullSelectionBehavior = "drop",
    *,
    options: FilterOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> _ArrayT: ...
def array_sort_indices(
    array: list | lib.Array,
    /,
    order: Order = "ascending",
    *,
    null_placement: Literal["at_start", "at_end"] = "at_end",
    options: ArraySortOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.UInt64Array: ...
@overload
def array_take(
    array: NullableList[int],
    indices: Indices,
    /,
    *,
    boundscheck: bool = True,
    options: TakeOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.Int64Array: ...
@overload
def array_take(
    array: NullableList[float],
    indices: Indices,
    /,
    *,
    boundscheck: bool = True,
    options: TakeOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.DoubleArray: ...
@overload
def array_take(
    array: NullableList[Decimal],
    indices: Indices,
    /,
    *,
    boundscheck: bool = True,
    options: TakeOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.Decimal128Array: ...
@overload
def array_take(
    array: _ArrayT,
    indices: Indices,
    /,
    *,
    boundscheck: bool = True,
    options: TakeOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> _ArrayT: ...
@overload
def ascii_capitalize(
    strings: str | lib.StringScalar, /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.StringScalar: ...
@overload
def ascii_capitalize(
    strings: NullableList[str] | lib.StringArray, /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.StringArray: ...

ascii_lower = ascii_capitalize
ascii_upper = ascii_capitalize
ascii_reverse = ascii_capitalize
ascii_swapcase = ascii_capitalize
ascii_title = ascii_capitalize

@overload
def ascii_center(
    strings: str | lib.StringScalar,
    /,
    width: int,
    padding: str = " ",
    lean_left_on_odd_padding: bool = True,
    *,
    options: PadOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.StringScalar: ...
@overload
def ascii_center(
    strings: NullableList[str] | lib.StringArray,
    /,
    width: int,
    padding: str = " ",
    lean_left_on_odd_padding: bool = True,
    *,
    options: PadOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.StringArray: ...
@overload
def ascii_is_alnum(
    strings: str | lib.StringScalar, /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.BooleanScalar: ...
@overload
def ascii_is_alnum(
    strings: NullableList[str] | lib.StringArray, /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.BooleanArray: ...

ascii_is_alpha = ascii_is_alnum
ascii_is_decimal = ascii_is_alnum
ascii_is_lower = ascii_is_alnum
ascii_is_printable = ascii_is_alnum
ascii_is_space = ascii_is_alnum
ascii_is_title = ascii_is_alnum
ascii_is_upper = ascii_is_alnum

ascii_lpad = ascii_center
ascii_rpad = ascii_center

@overload
def ascii_trim(
    strings: str | lib.StringScalar,
    /,
    characters: str,
    *,
    options: TrimOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.StringScalar: ...
@overload
def ascii_trim(
    strings: NullableList[str] | lib.StringArray,
    /,
    characters: str,
    *,
    options: TrimOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.StringArray: ...

ascii_ltrim = ascii_trim
ascii_rtrim = ascii_trim
ascii_trim_whitespace = ascii_capitalize
ascii_rtrim_whitespace = ascii_capitalize
ascii_ltrim_whitespace = ascii_capitalize

@overload
def ascii_split_whitespace(
    strings: str | lib.StringScalar,
    /,
    *,
    max_splits: int | None = None,
    reverse: bool = False,
    options: SplitOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.ListArray[lib.StringScalar]: ...
@overload
def ascii_split_whitespace(
    strings: NullableList[str] | lib.StringArray,
    /,
    *,
    max_splits: int | None = None,
    reverse: bool = False,
    options: SplitOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.ListArray[lib.ListScalar[lib.StringType]]: ...
def binary_join(
    strings: lib.BinaryArray, separator: lib.BinaryArray, /, *, memory_pool=None
) -> lib.BinaryArray: ...
@overload
def binary_join_element_wise(
    *strings: bytes | lib.BinaryScalar,
    null_handling: Literal["emit_null", "skip", "replace"] = "emit_null",
    null_replacement: str = "",
    options: JoinOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.BinaryScalar: ...
@overload
def binary_join_element_wise(
    *strings: NullableList[bytes] | lib.BinaryArray,
    null_handling: Literal["emit_null", "skip", "replace"] = "emit_null",
    null_replacement: str = "",
    options: JoinOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.BinaryArray: ...
@overload
def binary_length(
    strings: bytes | lib.BinaryScalar, /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.Int32Scalar: ...
@overload
def binary_length(
    strings: NullableList[bytes] | lib.BinaryArray, /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.Int32Array: ...
@overload
def binary_repeat(
    strings: bytes | lib.BinaryScalar,
    num_repeats: int,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.BinaryScalar: ...
@overload
def binary_repeat(
    strings: bytes | lib.BinaryScalar,
    num_repeats: NullableList[int],
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.BinaryArray: ...
@overload
def binary_repeat(
    strings: NullableList[bytes] | lib.BinaryScalar,
    num_repeats: int | NullableList[int],
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.BinaryArray: ...
@overload
def binary_replace_slice(
    strings: bytes | lib.BinaryScalar,
    /,
    start: int,
    stop: int,
    replacement: str,
    *,
    options: ReplaceSliceOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.BinaryScalar: ...
@overload
def binary_replace_slice(
    strings: NullableList[bytes] | lib.BinaryArray,
    /,
    start: int,
    stop: int,
    replacement: str,
    *,
    options: ReplaceSliceOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.BinaryArray: ...
@overload
def binary_slice(
    strings: bytes | lib.BinaryScalar,
    /,
    start: int,
    stop: int | None = None,
    step: int = 1,
    *,
    options: ReplaceSliceOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.BinaryScalar: ...
@overload
def binary_slice(
    strings: NullableList[bytes] | lib.BinaryArray,
    /,
    start: int,
    stop: int | None = None,
    step: int = 1,
    *,
    options: ReplaceSliceOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.BinaryArray: ...
@overload
def binary_reverse(
    strings: bytes | lib.BinaryScalar,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.BinaryScalar: ...
@overload
def binary_reverse(
    strings: NullableList[bytes] | lib.BinaryArray,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.BinaryArray: ...
@overload
def assume_timezone(
    timestamps: lib.TimestampScalar,
    /,
    timezone: str,
    *,
    ambiguous: Literal["raise", "earliest", "latest"] = "raise",
    nonexistent: Literal["raise", "earliest", "latest"] = "raise",
    options: AssumeTimezoneOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.TimestampScalar: ...
@overload
def assume_timezone(
    timestamps: lib.TimestampArray,
    /,
    timezone: str,
    *,
    ambiguous: Literal["raise", "earliest", "latest"] = "raise",
    nonexistent: Literal["raise", "earliest", "latest"] = "raise",
    options: AssumeTimezoneOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.TimestampArray: ...
