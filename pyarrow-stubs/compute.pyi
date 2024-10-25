# mypy: disable-error-code="misc,type-var,var-annotated"
# ruff: noqa: I001
from typing import Literal, TypeAlias, TypeVar, overload, Any, Iterable, ParamSpec, Sequence
from collections.abc import Callable

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

from . import lib

_P = ParamSpec("_P")
_R = TypeVar("_R")

def field(*name_or_index: str | tuple[str, ...] | int) -> Expression: ...
def scalar(value: bool | float | str) -> Expression: ...
def _clone_signature(f: Callable[_P, _R]) -> Callable[_P, _R]: ...

# ============= compute functions =============
_DataTypeT = TypeVar("_DataTypeT", bound=lib.DataType)
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
BinaryScalar: TypeAlias = (
    lib.Scalar[lib.BinaryType]
    | lib.Scalar[lib.LargeBinaryType]
    | lib.Scalar[lib.FixedSizeBinaryType]
)
StringScalar: TypeAlias = lib.Scalar[lib.StringType] | lib.Scalar[lib.LargeStringType]
ListScalar: TypeAlias = (
    lib.ListScalar[_DataTypeT]
    | lib.LargeListScalar[_DataTypeT]
    | lib.ListViewScalar[_DataTypeT]
    | lib.LargeListViewScalar[_DataTypeT]
    | lib.FixedSizeListScalar[_DataTypeT, Any]
)
TemporalScalar: TypeAlias = (
    lib.Date32Scalar
    | lib.Date64Scalar
    | lib.Time32Scalar
    | lib.Time64Scalar
    | lib.TimestampScalar
    | lib.DurationScalar
    | lib.MonthDayNanoIntervalScalar
)
_NumericScalarT = TypeVar("_NumericScalarT", bound=NumericScalar)
NumericOrDurationScalar: TypeAlias = NumericScalar | lib.DurationScalar
_NumericOrDurationT = TypeVar("_NumericOrDurationT", bound=NumericOrDurationScalar)
NumericOrTemporalScalar: TypeAlias = NumericScalar | TemporalScalar
_NumericOrTemporalT = TypeVar("_NumericOrTemporalT", bound=NumericOrTemporalScalar)
NumericArray: TypeAlias = lib.NumericArray[_ScalarT] | lib.ChunkedArray[_ScalarT]
_NumericArrayT = TypeVar("_NumericArrayT", bound=lib.NumericArray)
NumericOrDurationArray: TypeAlias = (
    lib.NumericArray | lib.Array[lib.DurationScalar] | lib.ChunkedArray
)
_NumericOrDurationArrayT = TypeVar("_NumericOrDurationArrayT", bound=NumericOrDurationArray)
NumericOrTemporalArray: TypeAlias = (
    lib.NumericArray | lib.Array[TemporalScalar] | lib.ChunkedArray[TemporalScalar]
)
_NumericOrTemporalArrayT = TypeVar("_NumericOrTemporalArrayT", bound=NumericOrTemporalArray)
BooleanArray: TypeAlias = lib.BooleanArray | lib.ChunkedArray[lib.BooleanScalar]
FloatScalar: TypeAlias = lib.Scalar[lib.Float32Type] | lib.Scalar[lib.Float64Type]
DecimalScalar: TypeAlias = lib.Scalar[lib.Decimal128Type] | lib.Scalar[lib.Decimal256Type]
_FloatScalarT = TypeVar("_FloatScalarT", bound=FloatScalar)
FloatArray: TypeAlias = (
    lib.NumericArray[lib.FloatScalar]
    | lib.NumericArray[lib.DoubleScalar]
    | lib.ChunkedArray[lib.FloatScalar]
    | lib.ChunkedArray[lib.DoubleScalar]
)

_FloatArrayT = TypeVar("_FloatArrayT", bound=FloatArray)
_StringScalarT = TypeVar("_StringScalarT", bound=StringScalar)
StringArray: TypeAlias = (
    lib.StringArray
    | lib.LargeStringArray
    | lib.ChunkedArray[lib.StringScalar]
    | lib.ChunkedArray[lib.LargeStringScalar]
)
_StringArrayT = TypeVar("_StringArrayT", bound=StringArray)
_BinaryScalarT = TypeVar("_BinaryScalarT", bound=BinaryScalar)
BinaryArray: TypeAlias = (
    lib.BinaryArray
    | lib.LargeBinaryArray
    | lib.ChunkedArray[lib.BinaryScalar]
    | lib.ChunkedArray[lib.LargeBinaryScalar]
)
_BinaryArrayT = TypeVar("_BinaryArrayT", bound=BinaryArray)
StringOrBinaryScalar: TypeAlias = StringScalar | BinaryScalar
_StringOrBinaryScalarT = TypeVar("_StringOrBinaryScalarT", bound=StringOrBinaryScalar)
StringOrBinaryArray: TypeAlias = StringArray | BinaryArray
_StringOrBinaryArrayT = TypeVar("_StringOrBinaryArrayT", bound=StringOrBinaryArray)
_TemporalScalarT = TypeVar("_TemporalScalarT", bound=TemporalScalar)
TemporalArray: TypeAlias = (
    lib.Date32Array
    | lib.Date64Array
    | lib.Time32Array
    | lib.Time64Array
    | lib.TimestampArray
    | lib.DurationArray
    | lib.MonthDayNanoIntervalArray
    | lib.ChunkedArray[lib.Date32Scalar]
    | lib.ChunkedArray[lib.Date64Scalar]
    | lib.ChunkedArray[lib.Time32Scalar]
    | lib.ChunkedArray[lib.Time64Scalar]
    | lib.ChunkedArray[lib.DurationScalar]
    | lib.ChunkedArray[lib.MonthDayNanoIntervalScalar]
)
_TemporalArrayT = TypeVar("_TemporalArrayT", bound=TemporalArray)
_ScalarT = TypeVar("_ScalarT", bound=lib.Scalar)
_ArrayT = TypeVar("_ArrayT", bound=lib.Array)
_ScalarOrArrayT = TypeVar("_ScalarOrArrayT", bound=lib.Array | lib.Scalar)
# =============================== 1. Aggregation ===============================

# ========================= 1.1 functions =========================

def all(
    array: lib.BooleanScalar | BooleanArray,
    /,
    *,
    skip_nulls: bool = True,
    min_count: int = 1,
    options: ScalarAggregateOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.BooleanScalar: ...

any = _clone_signature(all)

def approximate_median(
    array: NumericScalar | NumericArray,
    /,
    *,
    skip_nulls: bool = True,
    min_count: int = 1,
    options: ScalarAggregateOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.DoubleScalar: ...
def count(
    array: lib.Array | lib.ChunkedArray,
    /,
    mode: Literal["only_valid", "only_null", "all"] = "only_valid",
    *,
    options: CountOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.Int64Scalar: ...
def count_distinct(
    array: lib.Array | lib.ChunkedArray,
    /,
    mode: Literal["only_valid", "only_null", "all"] = "only_valid",
    *,
    options: CountOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.Int64Scalar: ...
def first(
    array: lib.Array[_ScalarT] | lib.ChunkedArray[_ScalarT],
    /,
    *,
    skip_nulls: bool = True,
    min_count: int = 1,
    options: ScalarAggregateOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> _ScalarT: ...
def first_last(
    array: lib.Array | lib.ChunkedArray,
    /,
    *,
    skip_nulls: bool = True,
    min_count: int = 1,
    options: ScalarAggregateOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.StructScalar: ...
def index(
    data: lib.Array | lib.ChunkedArray,
    value,
    start: int | None = None,
    end: int | None = None,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.Int64Scalar: ...

last = _clone_signature(first)
max = _clone_signature(first)
min = _clone_signature(first)
min_max = _clone_signature(first_last)

@overload
def mean(
    array: FloatScalar | FloatArray,
    /,
    *,
    skip_nulls: bool = True,
    min_count: int = 1,
    options: ScalarAggregateOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.DoubleScalar: ...
@overload
def mean(
    array: lib.NumericArray[lib.Decimal128Scalar]
    | lib.ChunkedArray[lib.Decimal128Scalar]
    | lib.Decimal128Scalar,
    /,
    *,
    skip_nulls: bool = True,
    min_count: int = 1,
    options: ScalarAggregateOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.Decimal128Scalar: ...
@overload
def mean(
    array: lib.NumericArray[lib.Decimal256Scalar]
    | lib.ChunkedArray[lib.Decimal256Scalar]
    | lib.Decimal256Scalar,
    /,
    *,
    skip_nulls: bool = True,
    min_count: int = 1,
    options: ScalarAggregateOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.Decimal256Scalar: ...
def mode(
    array: NumericScalar | NumericArray,
    /,
    n: int = 1,
    *,
    skip_nulls: bool = True,
    min_count: int = 0,
    options: ModeOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.StructArray: ...
def product(
    array: _ScalarT | lib.NumericArray[_ScalarT],
    /,
    *,
    skip_nulls=True,
    min_count=1,
    options=None,
    memory_pool: lib.MemoryPool | None = None,
) -> _ScalarT: ...
def quantile(
    array: NumericScalar | NumericArray,
    /,
    q: float = 0.5,
    *,
    interpolation: Literal["linear", "lower", "higher", "nearest", "midpoint"] = "linear",
    skip_nulls: bool = True,
    min_count: int = 0,
    options: QuantileOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.DoubleArray: ...
def stddev(
    array: NumericScalar | NumericArray,
    /,
    *,
    ddof: float = 0,
    skip_nulls: bool = True,
    min_count: int = 0,
    options: VarianceOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.DoubleScalar: ...
def sum(
    array: _NumericScalarT | NumericArray[_NumericScalarT],
    /,
    *,
    skip_nulls: bool = True,
    min_count: int = 1,
    options: ScalarAggregateOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> _NumericScalarT: ...
def tdigest(
    array: NumericScalar | NumericArray,
    /,
    q: float = 0.5,
    *,
    delta: int = 100,
    buffer_size: int = 500,
    skip_nulls: bool = True,
    min_count: int = 0,
    options: TDigestOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.DoubleArray: ...
def variance(
    array: NumericScalar | NumericArray,
    /,
    *,
    ddof: int = 0,
    skip_nulls: bool = True,
    min_count: int = 0,
    options: VarianceOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.DoubleScalar: ...

# ========================= 2. Element-wise (“scalar”) functions =========================

# ========================= 2.1 Arithmetic =========================
@overload
def abs(
    x: _NumericOrDurationT, /, *, memory_pool: lib.MemoryPool | None = None
) -> _NumericOrDurationT: ...
@overload
def abs(
    x: _NumericOrDurationArrayT, /, *, memory_pool: lib.MemoryPool | None = None
) -> _NumericOrDurationArrayT: ...
@overload
def abs(x: Expression, /, *, memory_pool: lib.MemoryPool | None = None) -> Expression: ...

abs_checked = _clone_signature(abs)

@overload
def add(
    x: _NumericOrTemporalT, y: _NumericOrTemporalT, /, *, memory_pool: lib.MemoryPool | None = None
) -> _NumericOrTemporalT: ...
@overload
def add(
    x: _NumericOrTemporalArrayT,
    y: _NumericOrTemporalArrayT,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> _NumericOrTemporalArrayT: ...
@overload
def add(
    x: NumericScalar, y: NumericScalar, /, *, memory_pool: lib.MemoryPool | None = None
) -> NumericScalar: ...
@overload
def add(
    x: TemporalScalar, y: TemporalScalar, /, *, memory_pool: lib.MemoryPool | None = None
) -> TemporalScalar: ...
@overload
def add(
    x: NumericOrTemporalArray | NumericOrTemporalScalar,
    y: NumericOrTemporalArray | NumericOrTemporalScalar,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> NumericOrTemporalArray: ...
@overload
def add(
    x: Expression | Any, y: Expression | Any, /, *, memory_pool: lib.MemoryPool | None = None
) -> Expression: ...

add_checked = _clone_signature(add)

@overload
def divide(
    dividend: NumericScalar,
    divisor: NumericScalar,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> NumericScalar: ...
@overload
def divide(
    dividend: TemporalScalar,
    divisor: TemporalScalar,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> TemporalScalar: ...
@overload
def divide(
    dividend: NumericOrTemporalArray | NumericOrTemporalScalar,
    divisor: NumericOrTemporalArray | NumericOrTemporalScalar,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> NumericArray: ...
@overload
def divide(
    dividend: Expression | Any,
    divisor: Expression | Any,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> Expression: ...

divide_checked = _clone_signature(divide)

@overload
def exp(
    exponent: lib.FloatArray, /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.FloatArray: ...
@overload
def exp(
    exponent: NumericArray, /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.DoubleArray: ...
@overload
def exp(
    exponent: lib.FloatScalar, /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.FloatScalar: ...
@overload
def exp(
    exponent: NumericScalar, /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.DoubleScalar: ...
@overload
def exp(exponent: Expression, /, *, memory_pool: lib.MemoryPool | None = None) -> Expression: ...

multiply = _clone_signature(add)
multiply_checked = _clone_signature(multiply)

@overload
def negate(
    x: _NumericOrDurationT, /, *, memory_pool: lib.MemoryPool | None = None
) -> _NumericOrDurationT: ...
@overload
def negate(
    x: _NumericOrDurationArrayT, /, *, memory_pool: lib.MemoryPool | None = None
) -> _NumericOrDurationArrayT: ...
@overload
def negate(x: Expression, /, *, memory_pool: lib.MemoryPool | None = None) -> Expression: ...

negate_checked = _clone_signature(negate)

@overload
def power(
    base: _NumericScalarT,
    exponent: _NumericScalarT,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> _NumericScalarT: ...
@overload
def power(
    base: NumericScalar, exponent: NumericScalar, /, *, memory_pool: lib.MemoryPool | None = None
) -> NumericScalar: ...
@overload
def power(
    base: _NumericArrayT,
    exponent: _NumericArrayT,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> _NumericArrayT: ...
@overload
def power(
    base: NumericScalar | NumericArray,
    exponent: NumericScalar | NumericArray,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> NumericArray: ...
@overload
def power(
    base: Expression | Any,
    exponent: Expression | Any,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> Expression: ...

power_checked = _clone_signature(power)

@overload
def sign(
    x: NumericOrDurationArray, /, *, memory_pool: lib.MemoryPool | None = None
) -> (
    lib.NumericArray[lib.Int8Scalar]
    | lib.NumericArray[lib.FloatScalar]
    | lib.NumericArray[lib.DoubleScalar]
): ...
@overload
def sign(
    x: NumericOrDurationScalar, /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.Int8Scalar | lib.FloatScalar | lib.DoubleScalar: ...
@overload
def sign(x: Expression, /, *, memory_pool: lib.MemoryPool | None = None) -> Expression: ...
@overload
def sqrt(x: NumericArray, /, *, memory_pool: lib.MemoryPool | None = None) -> FloatArray: ...
@overload
def sqrt(x: NumericScalar, /, *, memory_pool: lib.MemoryPool | None = None) -> FloatScalar: ...
@overload
def sqrt(x: Expression, /, *, memory_pool: lib.MemoryPool | None = None) -> Expression: ...

sqrt_checked = _clone_signature(sqrt)

subtract = _clone_signature(add)
subtract_checked = _clone_signature(subtract)

# ========================= 2.1 Bit-wise functions =========================
@overload
def bit_wise_and(
    x: _NumericScalarT, y: _NumericScalarT, /, *, memory_pool: lib.MemoryPool | None = None
) -> _NumericScalarT: ...
@overload
def bit_wise_and(
    x: _NumericArrayT,
    y: _NumericArrayT,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> _NumericArrayT: ...
@overload
def bit_wise_and(
    x: NumericScalar, y: NumericScalar, /, *, memory_pool: lib.MemoryPool | None = None
) -> NumericScalar: ...
@overload
def bit_wise_and(
    x: NumericArray | NumericScalar,
    y: NumericArray | NumericScalar,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> NumericArray: ...
@overload
def bit_wise_and(
    x: Expression | Any, y: Expression | Any, /, *, memory_pool: lib.MemoryPool | None = None
) -> Expression: ...
@overload
def bit_wise_not(
    x: _NumericScalarT, /, *, memory_pool: lib.MemoryPool | None = None
) -> _NumericScalarT: ...
@overload
def bit_wise_not(
    x: _NumericArrayT, /, *, memory_pool: lib.MemoryPool | None = None
) -> _NumericArrayT: ...
@overload
def bit_wise_not(x: Expression, /, *, memory_pool: lib.MemoryPool | None = None) -> Expression: ...

bit_wise_or = _clone_signature(bit_wise_and)
bit_wise_xor = _clone_signature(bit_wise_and)
shift_left = _clone_signature(bit_wise_and)
shift_left_checked = _clone_signature(bit_wise_and)
shift_right = _clone_signature(bit_wise_and)
shift_right_checked = _clone_signature(bit_wise_and)

# ========================= 2.2 Rounding functions =========================
@overload
def ceil(x: _FloatScalarT, /, *, memory_pool: lib.MemoryPool | None = None) -> _FloatScalarT: ...
@overload
def ceil(x: _FloatArrayT, /, *, memory_pool: lib.MemoryPool | None = None) -> _FloatArrayT: ...
@overload
def ceil(x: Expression, /, *, memory_pool: lib.MemoryPool | None = None) -> Expression: ...

floor = _clone_signature(ceil)

@overload
def round(
    x: _NumericScalarT,
    /,
    ndigits: int = 0,
    round_mode: Literal[
        "down",
        "up",
        "towards_zero",
        "towards_infinity",
        "half_down",
        "half_up",
        "half_towards_zero",
        "half_towards_infinity",
        "half_to_even",
        "half_to_odd",
    ] = "half_to_even",
    *,
    options: RoundOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> _NumericScalarT: ...
@overload
def round(
    x: _NumericArrayT,
    /,
    ndigits: int = 0,
    round_mode: Literal[
        "down",
        "up",
        "towards_zero",
        "towards_infinity",
        "half_down",
        "half_up",
        "half_towards_zero",
        "half_towards_infinity",
        "half_to_even",
        "half_to_odd",
    ] = "half_to_even",
    *,
    options: RoundOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> _NumericArrayT: ...
@overload
def round(
    x: Expression,
    /,
    ndigits: int = 0,
    round_mode: Literal[
        "down",
        "up",
        "towards_zero",
        "towards_infinity",
        "half_down",
        "half_up",
        "half_towards_zero",
        "half_towards_infinity",
        "half_to_even",
        "half_to_odd",
    ] = "half_to_even",
    *,
    options: RoundOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> Expression: ...
@overload
def round_to_multiple(
    x: _NumericScalarT,
    /,
    multiple: int = 0,
    round_mode: Literal[
        "down",
        "up",
        "towards_zero",
        "towards_infinity",
        "half_down",
        "half_up",
        "half_towards_zero",
        "half_towards_infinity",
        "half_to_even",
        "half_to_odd",
    ] = "half_to_even",
    *,
    options: RoundToMultipleOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> _NumericScalarT: ...
@overload
def round_to_multiple(
    x: _NumericArrayT,
    /,
    multiple: int = 0,
    round_mode: Literal[
        "down",
        "up",
        "towards_zero",
        "towards_infinity",
        "half_down",
        "half_up",
        "half_towards_zero",
        "half_towards_infinity",
        "half_to_even",
        "half_to_odd",
    ] = "half_to_even",
    *,
    options: RoundToMultipleOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> _NumericArrayT: ...
@overload
def round_to_multiple(
    x: Expression,
    /,
    multiple: int = 0,
    round_mode: Literal[
        "down",
        "up",
        "towards_zero",
        "towards_infinity",
        "half_down",
        "half_up",
        "half_towards_zero",
        "half_towards_infinity",
        "half_to_even",
        "half_to_odd",
    ] = "half_to_even",
    *,
    options: RoundToMultipleOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> Expression: ...
@overload
def round_binary(
    x: _NumericScalarT,
    s: int | lib.Int8Scalar | lib.Int16Scalar | lib.Int32Scalar | lib.Int64Scalar,
    /,
    round_mode: Literal[
        "down",
        "up",
        "towards_zero",
        "towards_infinity",
        "half_down",
        "half_up",
        "half_towards_zero",
        "half_towards_infinity",
        "half_to_even",
        "half_to_odd",
    ] = "half_to_even",
    *,
    options: RoundBinaryOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> _NumericScalarT: ...
@overload
def round_binary(
    x: _NumericScalarT,
    s: Iterable,
    /,
    round_mode: Literal[
        "down",
        "up",
        "towards_zero",
        "towards_infinity",
        "half_down",
        "half_up",
        "half_towards_zero",
        "half_towards_infinity",
        "half_to_even",
        "half_to_odd",
    ] = "half_to_even",
    *,
    options: RoundBinaryOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.NumericArray[_NumericScalarT]: ...
@overload
def round_binary(
    x: _NumericArrayT,
    s: int | lib.Int8Scalar | lib.Int16Scalar | lib.Int32Scalar | lib.Int64Scalar | Iterable,
    /,
    round_mode: Literal[
        "down",
        "up",
        "towards_zero",
        "towards_infinity",
        "half_down",
        "half_up",
        "half_towards_zero",
        "half_towards_infinity",
        "half_to_even",
        "half_to_odd",
    ] = "half_to_even",
    *,
    options: RoundBinaryOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> _NumericArrayT: ...
@overload
def round_binary(
    x: Expression,
    s: Iterable,
    /,
    round_mode: Literal[
        "down",
        "up",
        "towards_zero",
        "towards_infinity",
        "half_down",
        "half_up",
        "half_towards_zero",
        "half_towards_infinity",
        "half_to_even",
        "half_to_odd",
    ] = "half_to_even",
    *,
    options: RoundBinaryOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> Expression: ...

trunc = _clone_signature(ceil)

# ========================= 2.3 Logarithmic functions =========================
@overload
def ln(
    x: FloatScalar, /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.FloatScalar | lib.DoubleScalar: ...
@overload
def ln(
    x: FloatArray, /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.NumericArray[lib.FloatScalar] | lib.NumericArray[lib.DoubleScalar]: ...
@overload
def ln(x: Expression, /, *, memory_pool: lib.MemoryPool | None = None) -> Expression: ...

ln_checked = _clone_signature(ln)
log10 = _clone_signature(ln)
log10_checked = _clone_signature(ln)
log1p = _clone_signature(ln)
log1p_checked = _clone_signature(ln)
log2 = _clone_signature(ln)
log2_checked = _clone_signature(ln)

@overload
def logb(
    x: FloatScalar, b: FloatScalar, /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.FloatScalar | lib.DoubleScalar: ...
@overload
def logb(
    x: FloatArray, b: FloatArray, /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.NumericArray[lib.FloatScalar] | lib.NumericArray[lib.DoubleScalar]: ...
@overload
def logb(
    x: FloatScalar | FloatArray,
    b: FloatScalar | FloatArray,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.NumericArray[lib.FloatScalar] | lib.NumericArray[lib.DoubleScalar]: ...
@overload
def logb(
    x: Expression | Any, b: Expression | Any, /, *, memory_pool: lib.MemoryPool | None = None
) -> Expression | Any: ...

logb_checked = _clone_signature(logb)

# ========================= 2.4 Trigonometric functions =========================
acos = _clone_signature(ln)
acos_checked = _clone_signature(ln)
asin = _clone_signature(ln)
asin_checked = _clone_signature(ln)
atan = _clone_signature(ln)
cos = _clone_signature(ln)
cos_checked = _clone_signature(ln)
sin = _clone_signature(ln)
sin_checked = _clone_signature(ln)
tan = _clone_signature(ln)
tan_checked = _clone_signature(ln)

@overload
def atan2(
    y: FloatScalar, x: FloatScalar, /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.FloatScalar | lib.DoubleScalar: ...
@overload
def atan2(
    y: FloatArray, x: FloatArray, /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.NumericArray[lib.FloatScalar] | lib.NumericArray[lib.DoubleScalar]: ...
@overload
def atan2(
    y: FloatScalar | FloatArray,
    x: FloatScalar | FloatArray,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.NumericArray[lib.FloatScalar] | lib.NumericArray[lib.DoubleScalar]: ...
@overload
def atan2(
    y: Expression | Any, x: Expression | Any, /, *, memory_pool: lib.MemoryPool | None = None
) -> Expression: ...

# ========================= 2.5 Comparisons functions =========================
@overload
def equal(
    x: lib.Scalar, y: lib.Scalar, /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.BooleanScalar: ...
@overload
def equal(
    x: lib.Scalar | lib.Array | lib.ChunkedArray,
    y: lib.Scalar | lib.Array | lib.ChunkedArray,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.BooleanArray: ...
@overload
def equal(
    x: Expression | Any,
    y: Expression | Any,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> Expression: ...

greater = _clone_signature(equal)
greater_equal = _clone_signature(equal)
less = _clone_signature(equal)
less_equal = _clone_signature(equal)
not_equal = _clone_signature(equal)

@overload
def max_element_wise(
    *args: _ScalarT,
    skip_nulls: bool = True,
    options: ElementWiseAggregateOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> _ScalarT: ...
@overload
def max_element_wise(
    *args: _ArrayT,
    skip_nulls: bool = True,
    options: ElementWiseAggregateOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> _ArrayT: ...
@overload
def max_element_wise(
    *args: Expression,
    skip_nulls: bool = True,
    options: ElementWiseAggregateOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> Expression: ...

min_element_wise = _clone_signature(equal)

# ========================= 2.6 Logical functions =========================
@overload
def and_(
    x: lib.BooleanScalar, y: lib.BooleanScalar, /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.BooleanScalar: ...
@overload
def and_(
    x: lib.BooleanScalar | BooleanArray,
    y: lib.BooleanScalar | BooleanArray,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.BooleanArray: ...
@overload
def and_(
    x: Expression | Any,
    y: Expression | Any,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> Expression: ...

and_kleene = _clone_signature(and_)
and_not = _clone_signature(and_)
and_not_kleene = _clone_signature(and_)
or_ = _clone_signature(and_)
or_kleene = _clone_signature(and_)
xor = _clone_signature(and_)

@overload
def invert(
    x: lib.BooleanScalar, /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.BooleanScalar: ...
@overload
def invert(
    x: lib.BooleanArray,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.BooleanArray: ...
@overload
def invert(
    x: Expression,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> Expression: ...

# ========================= 2.10 String predicates =========================
@overload
def ascii_is_alnum(
    strings: StringScalar, /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.BooleanScalar: ...
@overload
def ascii_is_alnum(
    strings: StringArray, /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.BooleanArray: ...
@overload
def ascii_is_alnum(
    strings: Expression, /, *, memory_pool: lib.MemoryPool | None = None
) -> Expression: ...

ascii_is_alpha = _clone_signature(ascii_is_alnum)
ascii_is_decimal = _clone_signature(ascii_is_alnum)
ascii_is_lower = _clone_signature(ascii_is_alnum)
ascii_is_printable = _clone_signature(ascii_is_alnum)
ascii_is_space = _clone_signature(ascii_is_alnum)
ascii_is_upper = _clone_signature(ascii_is_alnum)
utf8_is_alnum = _clone_signature(ascii_is_alnum)
utf8_is_alpha = _clone_signature(ascii_is_alnum)
utf8_is_decimal = _clone_signature(ascii_is_alnum)
utf8_is_digit = _clone_signature(ascii_is_alnum)
utf8_is_lower = _clone_signature(ascii_is_alnum)
utf8_is_numeric = _clone_signature(ascii_is_alnum)
utf8_is_printable = _clone_signature(ascii_is_alnum)
utf8_is_space = _clone_signature(ascii_is_alnum)
utf8_is_upper = _clone_signature(ascii_is_alnum)
ascii_is_title = _clone_signature(ascii_is_alnum)
utf8_is_title = _clone_signature(ascii_is_alnum)
string_is_ascii = _clone_signature(ascii_is_alnum)

# ========================= 2.11 String transforms =========================
@overload
def ascii_capitalize(
    strings: _StringScalarT, /, *, memory_pool: lib.MemoryPool | None = None
) -> _StringScalarT: ...
@overload
def ascii_capitalize(
    strings: _StringArrayT, /, *, memory_pool: lib.MemoryPool | None = None
) -> _StringArrayT: ...
@overload
def ascii_capitalize(
    strings: Expression, /, *, memory_pool: lib.MemoryPool | None = None
) -> Expression: ...

ascii_lower = _clone_signature(ascii_capitalize)
ascii_reverse = _clone_signature(ascii_capitalize)
ascii_swapcase = _clone_signature(ascii_capitalize)
ascii_title = _clone_signature(ascii_capitalize)
ascii_upper = _clone_signature(ascii_capitalize)

@overload
def binary_length(
    strings: lib.BinaryScalar | lib.StringScalar, /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.Int32Scalar: ...
@overload
def binary_length(
    strings: lib.LargeBinaryScalar | lib.LargeStringScalar,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.Int64Scalar: ...
@overload
def binary_length(
    strings: lib.BinaryArray
    | lib.StringArray
    | lib.ChunkedArray[lib.BinaryScalar]
    | lib.ChunkedArray[lib.StringScalar],
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.Int32Array: ...
@overload
def binary_length(
    strings: lib.LargeBinaryArray
    | lib.LargeStringArray
    | lib.ChunkedArray[lib.LargeBinaryScalar]
    | lib.ChunkedArray[lib.LargeStringScalar],
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.Int64Array: ...
@overload
def binary_length(
    strings: Expression,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> Expression: ...
@overload
def binary_repeat(
    strings: _StringOrBinaryScalarT,
    num_repeats: int,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> _StringOrBinaryScalarT: ...
@overload
def binary_repeat(
    strings: _StringOrBinaryScalarT,
    num_repeats: list[int] | list[int | None],
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.Array[_StringOrBinaryScalarT]: ...
@overload
def binary_repeat(
    strings: _StringOrBinaryArrayT,
    num_repeats: int | list[int] | list[int | None],
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> _StringOrBinaryArrayT: ...
@overload
def binary_repeat(
    strings: Expression,
    num_repeats: int | list[int] | list[int | None],
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> Expression: ...
@overload
def binary_replace_slice(
    strings: _StringOrBinaryScalarT,
    /,
    start: int,
    stop: int,
    replacement: str | bytes,
    *,
    options: ReplaceSliceOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> _StringOrBinaryScalarT: ...
@overload
def binary_replace_slice(
    strings: _StringOrBinaryArrayT,
    /,
    start: int,
    stop: int,
    replacement: str | bytes,
    *,
    options: ReplaceSliceOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> _StringOrBinaryArrayT: ...
@overload
def binary_replace_slice(
    strings: Expression,
    /,
    start: int,
    stop: int,
    replacement: str | bytes,
    *,
    options: ReplaceSliceOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> Expression: ...
@overload
def binary_reverse(
    strings: _BinaryScalarT, /, *, memory_pool: lib.MemoryPool | None = None
) -> _BinaryScalarT: ...
@overload
def binary_reverse(
    strings: _BinaryArrayT, /, *, memory_pool: lib.MemoryPool | None = None
) -> _BinaryArrayT: ...
@overload
def binary_reverse(
    strings: Expression, /, *, memory_pool: lib.MemoryPool | None = None
) -> Expression: ...
@overload
def replace_substring(
    strings: _StringScalarT,
    /,
    pattern: str | bytes,
    replacement: str | bytes,
    *,
    max_replacements: int | None = None,
    options: ReplaceSubstringOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> _StringScalarT: ...
@overload
def replace_substring(
    strings: _StringArrayT,
    /,
    pattern: str | bytes,
    replacement: str | bytes,
    *,
    max_replacements: int | None = None,
    options: ReplaceSubstringOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> _StringArrayT: ...
@overload
def replace_substring(
    strings: Expression,
    /,
    pattern: str | bytes,
    replacement: str | bytes,
    *,
    max_replacements: int | None = None,
    options: ReplaceSubstringOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> Expression: ...

replace_substring_regex = _clone_signature(replace_substring)

@overload
def utf8_capitalize(
    strings: _StringScalarT, /, *, memory_pool: lib.MemoryPool | None = None
) -> _StringScalarT: ...
@overload
def utf8_capitalize(
    strings: _StringArrayT, /, *, memory_pool: lib.MemoryPool | None = None
) -> _StringArrayT: ...
@overload
def utf8_capitalize(
    strings: Expression, /, *, memory_pool: lib.MemoryPool | None = None
) -> Expression: ...
@overload
def utf8_length(
    strings: lib.StringScalar, /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.Int32Scalar: ...
@overload
def utf8_length(
    strings: lib.LargeStringScalar,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.Int64Scalar: ...
@overload
def utf8_length(
    strings: lib.StringArray | lib.ChunkedArray[lib.StringScalar],
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.Int32Array: ...
@overload
def utf8_length(
    strings: lib.LargeStringArray | lib.ChunkedArray[lib.LargeStringScalar],
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.Int64Array: ...
@overload
def utf8_length(
    strings: Expression,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> Expression: ...

utf8_lower = _clone_signature(utf8_capitalize)

@overload
def utf8_replace_slice(
    strings: _StringScalarT,
    /,
    start: int,
    stop: int,
    replacement: str | bytes,
    *,
    options: ReplaceSliceOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> _StringScalarT: ...
@overload
def utf8_replace_slice(
    strings: _StringArrayT,
    /,
    start: int,
    stop: int,
    replacement: str | bytes,
    *,
    options: ReplaceSliceOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> _StringArrayT: ...
@overload
def utf8_replace_slice(
    strings: Expression,
    /,
    start: int,
    stop: int,
    replacement: str | bytes,
    *,
    options: ReplaceSliceOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> Expression: ...

utf8_reverse = _clone_signature(utf8_capitalize)
utf8_swapcase = _clone_signature(utf8_capitalize)
utf8_title = _clone_signature(utf8_capitalize)
utf8_upper = _clone_signature(utf8_capitalize)

# ========================= 2.12 String padding =========================
@overload
def ascii_center(
    strings: _StringScalarT,
    /,
    width: int,
    padding: str = " ",
    lean_left_on_odd_padding: bool = True,
    *,
    options: PadOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> _StringScalarT: ...
@overload
def ascii_center(
    strings: _StringArrayT,
    /,
    width: int,
    padding: str = " ",
    lean_left_on_odd_padding: bool = True,
    *,
    options: PadOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> _StringArrayT: ...
@overload
def ascii_center(
    strings: Expression,
    /,
    width: int,
    padding: str = " ",
    lean_left_on_odd_padding: bool = True,
    *,
    options: PadOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> Expression: ...

ascii_lpad = _clone_signature(ascii_center)
ascii_rpad = _clone_signature(ascii_center)
utf8_center = _clone_signature(ascii_center)
utf8_lpad = _clone_signature(ascii_center)
utf8_rpad = _clone_signature(ascii_center)

# ========================= 2.13 String trimming =========================
@overload
def ascii_ltrim(
    strings: _StringScalarT,
    /,
    characters: str,
    *,
    options: TrimOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> _StringScalarT: ...
@overload
def ascii_ltrim(
    strings: _StringArrayT,
    /,
    characters: str,
    *,
    options: TrimOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> _StringArrayT: ...
@overload
def ascii_ltrim(
    strings: Expression,
    /,
    characters: str,
    *,
    options: TrimOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> Expression: ...

ascii_rtrim = _clone_signature(ascii_ltrim)
ascii_trim = _clone_signature(ascii_ltrim)
utf8_ltrim = _clone_signature(ascii_ltrim)
utf8_rtrim = _clone_signature(ascii_ltrim)
utf8_trim = _clone_signature(ascii_ltrim)

@overload
def ascii_ltrim_whitespace(
    strings: _StringScalarT,
    /,
    *,
    options: TrimOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> _StringScalarT: ...
@overload
def ascii_ltrim_whitespace(
    strings: _StringArrayT,
    /,
    *,
    options: TrimOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> _StringArrayT: ...
@overload
def ascii_ltrim_whitespace(
    strings: Expression,
    /,
    *,
    options: TrimOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> Expression: ...

ascii_rtrim_whitespace = _clone_signature(ascii_ltrim_whitespace)
ascii_trim_whitespace = _clone_signature(ascii_ltrim_whitespace)
utf8_ltrim_whitespace = _clone_signature(ascii_ltrim_whitespace)
utf8_rtrim_whitespace = _clone_signature(ascii_ltrim_whitespace)
utf8_trim_whitespace = _clone_signature(ascii_ltrim_whitespace)

# ========================= 2.14 String splitting =========================
@overload
def ascii_split_whitespace(
    strings: _StringScalarT,
    /,
    *,
    max_splits: int | None = None,
    reverse: bool = False,
    options: SplitOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.ListArray[_StringScalarT]: ...
@overload
def ascii_split_whitespace(
    strings: lib.Array[lib.Scalar[_DataTypeT]],
    /,
    *,
    max_splits: int | None = None,
    reverse: bool = False,
    options: SplitOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.ListArray[lib.ListScalar[_DataTypeT]]: ...
@overload
def ascii_split_whitespace(
    strings: Expression,
    /,
    *,
    max_splits: int | None = None,
    reverse: bool = False,
    options: SplitOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> Expression: ...
@overload
def split_pattern(
    strings: _StringOrBinaryScalarT,
    /,
    pattern: str,
    *,
    max_splits: int | None = None,
    reverse: bool = False,
    options: SplitOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.ListArray[_StringOrBinaryScalarT]: ...
@overload
def split_pattern(
    strings: lib.Array[lib.Scalar[_DataTypeT]],
    /,
    pattern: str,
    *,
    max_splits: int | None = None,
    reverse: bool = False,
    options: SplitPatternOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.ListArray[lib.ListScalar[_DataTypeT]]: ...
@overload
def split_pattern(
    strings: Expression,
    /,
    pattern: str,
    *,
    max_splits: int | None = None,
    reverse: bool = False,
    options: SplitPatternOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> Expression: ...

split_pattern_regex = _clone_signature(split_pattern)
utf8_split_whitespace = _clone_signature(ascii_split_whitespace)

# ========================= 2.15 String component extraction =========================
@overload
def extract_regex(
    strings: StringOrBinaryScalar,
    /,
    pattern: str,
    *,
    options: ExtractRegexOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.StructScalar: ...
@overload
def extract_regex(
    strings: StringOrBinaryArray,
    /,
    pattern: str,
    *,
    options: ExtractRegexOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.StructArray: ...
@overload
def extract_regex(
    strings: Expression,
    /,
    pattern: str,
    *,
    options: ExtractRegexOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> Expression: ...

# ========================= 2.16 String join =========================
def binary_join(
    strings, separator, /, *, memory_pool: lib.MemoryPool | None = None
) -> StringScalar | StringArray: ...
@overload
def binary_join_element_wise(
    *strings: _StringOrBinaryScalarT,
    null_handling: Literal["emit_null", "skip", "replace"] = "emit_null",
    null_replacement: str = "",
    options: JoinOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> _StringOrBinaryScalarT: ...
@overload
def binary_join_element_wise(
    *strings: _StringOrBinaryArrayT,
    null_handling: Literal["emit_null", "skip", "replace"] = "emit_null",
    null_replacement: str = "",
    options: JoinOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> _StringOrBinaryArrayT: ...
@overload
def binary_join_element_wise(
    *strings: Expression,
    null_handling: Literal["emit_null", "skip", "replace"] = "emit_null",
    null_replacement: str = "",
    options: JoinOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> Expression: ...

# ========================= 2.17 String Slicing =========================
@overload
def binary_slice(
    strings: _BinaryScalarT,
    /,
    start: int,
    stop: int | None = None,
    step: int = 1,
    *,
    options: SliceOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> _BinaryScalarT: ...
@overload
def binary_slice(
    strings: _BinaryArrayT,
    /,
    start: int,
    stop: int | None = None,
    step: int = 1,
    *,
    options: SliceOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> _BinaryArrayT: ...
@overload
def binary_slice(
    strings: Expression,
    /,
    start: int,
    stop: int | None = None,
    step: int = 1,
    *,
    options: SliceOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> Expression: ...
@overload
def utf8_slice_codeunits(
    strings: _StringScalarT,
    /,
    start: int,
    stop: int | None = None,
    step: int = 1,
    *,
    options: SliceOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> _StringScalarT: ...
@overload
def utf8_slice_codeunits(
    strings: _StringArrayT,
    /,
    start: int,
    stop: int | None = None,
    step: int = 1,
    *,
    options: SliceOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> _StringArrayT: ...
@overload
def utf8_slice_codeunits(
    strings: Expression,
    /,
    start: int,
    stop: int | None = None,
    step: int = 1,
    *,
    options: SliceOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> Expression: ...

# ========================= 2.18 Containment tests =========================
@overload
def count_substring(
    strings: lib.StringScalar | lib.BinaryScalar,
    /,
    pattern: str,
    *,
    ignore_case: bool = False,
    options: MatchSubstringOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.Int32Scalar: ...
@overload
def count_substring(
    strings: lib.LargeStringScalar | lib.LargeBinaryScalar,
    /,
    pattern: str,
    *,
    ignore_case: bool = False,
    options: MatchSubstringOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.Int64Scalar: ...
@overload
def count_substring(
    strings: lib.StringArray
    | lib.BinaryArray
    | lib.ChunkedArray[lib.StringScalar]
    | lib.ChunkedArray[lib.BinaryScalar],
    /,
    pattern: str,
    *,
    ignore_case: bool = False,
    options: MatchSubstringOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.Int32Array: ...
@overload
def count_substring(
    strings: lib.LargeStringArray
    | lib.LargeBinaryArray
    | lib.ChunkedArray[lib.LargeStringScalar]
    | lib.ChunkedArray[lib.LargeBinaryScalar],
    /,
    pattern: str,
    *,
    ignore_case: bool = False,
    options: MatchSubstringOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.Int64Array: ...
@overload
def count_substring(
    strings: Expression,
    /,
    pattern: str,
    *,
    ignore_case: bool = False,
    options: MatchSubstringOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> Expression: ...

count_substring_regex = _clone_signature(count_substring)

@overload
def ends_with(
    strings: StringScalar | BinaryScalar,
    /,
    pattern: str,
    *,
    ignore_case: bool = False,
    options: MatchSubstringOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.BooleanScalar: ...
@overload
def ends_with(
    strings: StringArray | BinaryArray,
    /,
    pattern: str,
    *,
    ignore_case: bool = False,
    options: MatchSubstringOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.BooleanArray: ...
@overload
def ends_with(
    strings: Expression,
    /,
    pattern: str,
    *,
    ignore_case: bool = False,
    options: MatchSubstringOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> Expression: ...

find_substring = _clone_signature(count_substring)
find_substring_regex = _clone_signature(count_substring)

@overload
def index_in(
    values: lib.Scalar,
    /,
    value_set: lib.Array | lib.ChunkedArray,
    *,
    skip_nulls: bool = False,
    options: SetLookupOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.Int32Scalar: ...
@overload
def index_in(
    values: lib.Array | lib.ChunkedArray,
    /,
    value_set: lib.Array | lib.ChunkedArray,
    *,
    skip_nulls: bool = False,
    options: SetLookupOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.Int32Array: ...
@overload
def index_in(
    values: Expression,
    /,
    value_set: lib.Array | lib.ChunkedArray,
    *,
    skip_nulls: bool = False,
    options: SetLookupOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> Expression: ...
@overload
def is_in(
    values: lib.Scalar,
    /,
    value_set: lib.Array | lib.ChunkedArray,
    *,
    skip_nulls: bool = False,
    options: SetLookupOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.BooleanScalar: ...
@overload
def is_in(
    values: lib.Array | lib.ChunkedArray,
    /,
    value_set: lib.Array | lib.ChunkedArray,
    *,
    skip_nulls: bool = False,
    options: SetLookupOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.BooleanArray: ...
@overload
def is_in(
    values: Expression,
    /,
    value_set: lib.Array | lib.ChunkedArray,
    *,
    skip_nulls: bool = False,
    options: SetLookupOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> Expression: ...

match_like = _clone_signature(ends_with)
match_substring = _clone_signature(ends_with)
match_substring_regex = _clone_signature(ends_with)
starts_with = _clone_signature(ends_with)

# ========================= 2.19 Categorizations =========================
@overload
def is_finite(
    values: NumericScalar | lib.NullScalar, /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.BooleanScalar: ...
@overload
def is_finite(
    values: NumericArray | lib.NullArray, /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.BooleanArray: ...
@overload
def is_finite(
    values: Expression, /, *, memory_pool: lib.MemoryPool | None = None
) -> Expression: ...

is_inf = _clone_signature(is_finite)
is_nan = _clone_signature(is_finite)

@overload
def is_null(
    values: lib.Scalar,
    /,
    *,
    nan_is_null: bool = False,
    options: NullOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.BooleanScalar: ...
@overload
def is_null(
    values: lib.Array | lib.ChunkedArray,
    /,
    *,
    nan_is_null: bool = False,
    options: NullOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.BooleanArray: ...
@overload
def is_null(
    values: Expression,
    /,
    *,
    nan_is_null: bool = False,
    options: NullOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> Expression: ...
@overload
def is_valid(
    values: lib.Scalar, /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.BooleanScalar: ...
@overload
def is_valid(
    values: lib.Array | lib.ChunkedArray, /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.BooleanArray: ...
@overload
def is_valid(
    values: Expression, /, *, memory_pool: lib.MemoryPool | None = None
) -> Expression: ...

true_unless_null = _clone_signature(is_valid)

# ========================= 2.20 Selecting / multiplexing =========================
def case_when(cond, /, *cases, memory_pool: lib.MemoryPool | None = None): ...
def choose(indices, /, *values, memory_pool: lib.MemoryPool | None = None): ...
def coalesce(
    *values: _ScalarOrArrayT, memory_pool: lib.MemoryPool | None = None
) -> _ScalarOrArrayT: ...
def if_else(cond, left, right, /, *, memory_pool: lib.MemoryPool | None = None): ...

# ========================= 2.21 Structural transforms =========================

@overload
def list_value_length(
    lists: lib.ListArray | lib.ListViewArray | lib.FixedSizeListArray | lib.ChunkedArray,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.Int32Array: ...
@overload
def list_value_length(
    lists: lib.LargeListArray | lib.LargeListViewArray | lib.ChunkedArray,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.Int64Array: ...
@overload
def list_value_length(
    lists: Expression,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> Expression: ...
@overload
def make_struct(
    *args: lib.Scalar,
    field_names: list[str] | tuple[str, ...] = (),
    field_nullability: bool | None = None,
    field_metadata: list[lib.KeyValueMetadata] | None = None,
    options: MakeStructOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.StructScalar: ...
@overload
def make_struct(
    *args: lib.Array | lib.ChunkedArray,
    field_names: list[str] | tuple[str, ...] = (),
    field_nullability: bool | None = None,
    field_metadata: list[lib.KeyValueMetadata] | None = None,
    options: MakeStructOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.StructArray: ...
@overload
def make_struct(
    *args: Expression,
    field_names: list[str] | tuple[str, ...] = (),
    field_nullability: bool | None = None,
    field_metadata: list[lib.KeyValueMetadata] | None = None,
    options: MakeStructOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> Expression: ...

# ========================= 2.22 Conversions =========================
@overload
def ceil_temporal(
    timestamps: _TemporalScalarT,
    /,
    multiple: int = 1,
    unit: Literal[
        "year",
        "quarter",
        "month",
        "week",
        "day",
        "hour",
        "minute",
        "second",
        "millisecond",
        "microsecond",
        "nanosecond",
    ] = "day",
    *,
    week_starts_monday: bool = True,
    ceil_is_strictly_greater: bool = False,
    calendar_based_origin: bool = False,
    options: RoundTemporalOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> _TemporalScalarT: ...
@overload
def ceil_temporal(
    timestamps: _TemporalArrayT,
    /,
    multiple: int = 1,
    unit: Literal[
        "year",
        "quarter",
        "month",
        "week",
        "day",
        "hour",
        "minute",
        "second",
        "millisecond",
        "microsecond",
        "nanosecond",
    ] = "day",
    *,
    week_starts_monday: bool = True,
    ceil_is_strictly_greater: bool = False,
    calendar_based_origin: bool = False,
    options: RoundTemporalOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> _TemporalArrayT: ...
@overload
def ceil_temporal(
    timestamps: Expression,
    /,
    multiple: int = 1,
    unit: Literal[
        "year",
        "quarter",
        "month",
        "week",
        "day",
        "hour",
        "minute",
        "second",
        "millisecond",
        "microsecond",
        "nanosecond",
    ] = "day",
    *,
    week_starts_monday: bool = True,
    ceil_is_strictly_greater: bool = False,
    calendar_based_origin: bool = False,
    options: RoundTemporalOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> Expression: ...

floor_temporal = _clone_signature(ceil_temporal)
round_temporal = _clone_signature(ceil_temporal)

@overload
def cast(
    arr: lib.Scalar,
    target_type: _DataTypeT,
    safe: bool | None = None,
    options: CastOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.Scalar[_DataTypeT]: ...
@overload
def cast(
    arr: lib.Array,
    target_type: _DataTypeT,
    safe: bool | None = None,
    options: CastOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.Array[lib.Scalar[_DataTypeT]]: ...
@overload
def cast(
    arr: lib.ChunkedArray,
    target_type: _DataTypeT,
    safe: bool | None = None,
    options: CastOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.ChunkedArray[lib.Scalar[_DataTypeT]]: ...
@overload
def strftime(
    timestamps: TemporalScalar,
    /,
    format: str = "%Y-%m-%dT%H:%M:%S",
    locale: str = "C",
    *,
    options: StrftimeOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.StringScalar: ...
@overload
def strftime(
    timestamps: TemporalArray,
    /,
    format: str = "%Y-%m-%dT%H:%M:%S",
    locale: str = "C",
    *,
    options: StrftimeOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.StringArray: ...
@overload
def strftime(
    timestamps: Expression,
    /,
    format: str = "%Y-%m-%dT%H:%M:%S",
    locale: str = "C",
    *,
    options: StrftimeOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> Expression: ...
@overload
def strptime(
    strings: StringScalar,
    /,
    format: str,
    unit: Literal["s", "ms", "us", "ns"],
    error_is_null: bool = False,
    *,
    options: StrptimeOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.TimestampScalar: ...
@overload
def strptime(
    strings: StringArray,
    /,
    format: str,
    unit: Literal["s", "ms", "us", "ns"],
    error_is_null: bool = False,
    *,
    options: StrptimeOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.TimestampArray: ...
@overload
def strptime(
    strings: Expression,
    /,
    format: str,
    unit: Literal["s", "ms", "us", "ns"],
    error_is_null: bool = False,
    *,
    options: StrptimeOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> Expression: ...

# ========================= 2.23 Temporal component extraction =========================
@overload
def day(
    values: TemporalScalar, /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.Int64Scalar: ...
@overload
def day(
    values: TemporalArray, /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.Int64Array: ...
@overload
def day(values: Expression, /, *, memory_pool: lib.MemoryPool | None = None) -> Expression: ...
@overload
def day_of_week(
    values: TemporalScalar,
    /,
    *,
    count_from_zero: bool = True,
    week_start: int = 1,
    options: DayOfWeekOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.Int64Scalar: ...
@overload
def day_of_week(
    values: TemporalArray,
    /,
    *,
    count_from_zero: bool = True,
    week_start: int = 1,
    options: DayOfWeekOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.Int64Array: ...
@overload
def day_of_week(
    values: Expression,
    /,
    *,
    count_from_zero: bool = True,
    week_start: int = 1,
    options: DayOfWeekOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> Expression: ...

day_of_year = _clone_signature(day)

@overload
def hour(
    values: lib.TimestampScalar | lib.Time32Scalar | lib.Time64Scalar,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.Int64Scalar: ...
@overload
def hour(
    values: lib.TimestampArray
    | lib.Time32Array
    | lib.Time64Array
    | lib.ChunkedArray[lib.TimestampScalar]
    | lib.ChunkedArray[lib.Time32Scalar]
    | lib.ChunkedArray[lib.Time64Scalar],
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.Int64Array: ...
@overload
def hour(
    values: Expression,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> Expression: ...
@overload
def is_dst(
    values: lib.TimestampScalar, /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.BooleanScalar: ...
@overload
def is_dst(
    values: lib.TimestampArray | lib.ChunkedArray[lib.TimestampScalar],
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.BooleanArray: ...
@overload
def is_dst(values: Expression, /, *, memory_pool: lib.MemoryPool | None = None) -> Expression: ...
@overload
def iso_week(
    values: lib.TimestampScalar, /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.Int64Scalar: ...
@overload
def iso_week(
    values: lib.TimestampArray | lib.ChunkedArray[lib.TimestampScalar],
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.Int64Array: ...
@overload
def iso_week(
    values: Expression, /, *, memory_pool: lib.MemoryPool | None = None
) -> Expression: ...

iso_year = _clone_signature(iso_week)

@overload
def is_leap_year(
    values: lib.TimestampScalar | lib.Date32Scalar | lib.Date64Scalar,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.BooleanScalar: ...
@overload
def is_leap_year(
    values: lib.TimestampArray
    | lib.Date32Array
    | lib.Date64Array
    | lib.ChunkedArray[lib.TimestampScalar]
    | lib.ChunkedArray[lib.Date32Scalar]
    | lib.ChunkedArray[lib.Date64Scalar],
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.BooleanArray: ...
@overload
def is_leap_year(
    values: Expression,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> Expression: ...

microsecond = _clone_signature(iso_week)
millisecond = _clone_signature(iso_week)
minute = _clone_signature(iso_week)
month = _clone_signature(day_of_week)
nanosecond = _clone_signature(hour)
quarter = _clone_signature(day_of_week)
second = _clone_signature(hour)
subsecond = _clone_signature(hour)
us_week = _clone_signature(iso_week)
us_year = _clone_signature(iso_week)
year = _clone_signature(iso_week)

@overload
def week(
    values: lib.TimestampScalar,
    /,
    *,
    week_starts_monday: bool = True,
    count_from_zero: bool = False,
    first_week_is_fully_in_year: bool = False,
    options: WeekOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.Int64Scalar: ...
@overload
def week(
    values: lib.TimestampArray | lib.ChunkedArray[lib.TimestampScalar],
    /,
    *,
    week_starts_monday: bool = True,
    count_from_zero: bool = False,
    first_week_is_fully_in_year: bool = False,
    options: WeekOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.Int64Array: ...
@overload
def week(
    values: Expression,
    /,
    *,
    week_starts_monday: bool = True,
    count_from_zero: bool = False,
    first_week_is_fully_in_year: bool = False,
    options: WeekOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> Expression: ...
@overload
def year_month_day(
    values: TemporalScalar, /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.StructScalar: ...
@overload
def year_month_day(
    values: TemporalArray, /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.StructArray: ...
@overload
def year_month_day(
    values: Expression, /, *, memory_pool: lib.MemoryPool | None = None
) -> Expression: ...

# ========================= 2.24 Temporal difference =========================
def day_time_interval_between(start, end, /, *, memory_pool: lib.MemoryPool | None = None): ...
def days_between(
    start, end, /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.Int64Scalar | lib.Int64Array: ...

hours_between = _clone_signature(days_between)
microseconds_between = _clone_signature(days_between)
milliseconds_between = _clone_signature(days_between)
minutes_between = _clone_signature(days_between)

def month_day_nano_interval_between(
    start, end, /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.MonthDayNanoIntervalScalar | lib.MonthDayNanoIntervalArray: ...
def month_interval_between(start, end, /, *, memory_pool: lib.MemoryPool | None = None): ...

nanoseconds_between = _clone_signature(days_between)
quarters_between = _clone_signature(days_between)
seconds_between = _clone_signature(days_between)

def weeks_between(
    start,
    end,
    /,
    *,
    count_from_zero: bool = True,
    week_start: int = 1,
    options: DayOfWeekOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.Int64Scalar | lib.Int64Array: ...

years_between = _clone_signature(days_between)

# ========================= 2.25 Timezone handling =========================
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
    timestamps: lib.TimestampArray | lib.ChunkedArray[lib.TimestampScalar],
    /,
    timezone: str,
    *,
    ambiguous: Literal["raise", "earliest", "latest"] = "raise",
    nonexistent: Literal["raise", "earliest", "latest"] = "raise",
    options: AssumeTimezoneOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.TimestampArray: ...
@overload
def assume_timezone(
    timestamps: Expression,
    /,
    timezone: str,
    *,
    ambiguous: Literal["raise", "earliest", "latest"] = "raise",
    nonexistent: Literal["raise", "earliest", "latest"] = "raise",
    options: AssumeTimezoneOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> Expression: ...
@overload
def local_timestamp(
    timestamps: lib.TimestampScalar, /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.TimestampScalar: ...
@overload
def local_timestamp(
    timestamps: lib.TimestampArray | lib.ChunkedArray[lib.TimestampScalar],
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.TimestampArray: ...
@overload
def local_timestamp(
    timestamps: Expression, /, *, memory_pool: lib.MemoryPool | None = None
) -> Expression: ...

# ========================= 2.26 Random number generation =========================
def random(
    n: int,
    *,
    initializer: Literal["system"] | int = "system",
    options: RandomOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.DoubleArray: ...

# ========================= 3. Array-wise (“vector”) functions =========================

# ========================= 3.1 Cumulative Functions =========================
@overload
def cumulative_sum(
    values: _NumericArrayT,
    /,
    start: lib.Scalar | None = None,
    *,
    skip_nulls: bool = False,
    options: CumulativeSumOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> _NumericArrayT: ...
@overload
def cumulative_sum(
    values: Expression,
    /,
    start: lib.Scalar | None = None,
    *,
    skip_nulls: bool = False,
    options: CumulativeSumOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> Expression: ...

cumulative_sum_checked = _clone_signature(cumulative_sum)
cumulative_prod = _clone_signature(cumulative_sum)
cumulative_prod_checked = _clone_signature(cumulative_sum)
cumulative_max = _clone_signature(cumulative_sum)
cumulative_min = _clone_signature(cumulative_sum)
cumulative_mean = _clone_signature(cumulative_sum)

# ========================= 3.2 Associative transforms =========================

@overload
def dictionary_encode(
    array: _ScalarOrArrayT,
    /,
    null_encoding: Literal["mask", "encode"] = "mask",
    *,
    options=None,
    memory_pool: lib.MemoryPool | None = None,
) -> _ScalarOrArrayT: ...
@overload
def dictionary_encode(
    array: Expression,
    /,
    null_encoding: Literal["mask", "encode"] = "mask",
    *,
    options=None,
    memory_pool: lib.MemoryPool | None = None,
) -> Expression: ...
@overload
def unique(array: _ArrayT, /, *, memory_pool: lib.MemoryPool | None = None) -> _ArrayT: ...
@overload
def unique(array: Expression, /, *, memory_pool: lib.MemoryPool | None = None) -> Expression: ...
@overload
def value_counts(
    array: lib.Array | lib.ChunkedArray, /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.StructArray: ...
@overload
def value_counts(
    array: Expression, /, *, memory_pool: lib.MemoryPool | None = None
) -> Expression: ...

# ========================= 3.3 Selections =========================
@overload
def array_filter(
    array: _ArrayT,
    selection_filter: list[bool] | list[bool | None] | BooleanArray,
    /,
    null_selection_behavior: Literal["drop", "emit_null"] = "drop",
    *,
    options: FilterOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> _ArrayT: ...
@overload
def array_filter(
    array: Expression,
    selection_filter: list[bool] | list[bool | None] | BooleanArray,
    /,
    null_selection_behavior: Literal["drop", "emit_null"] = "drop",
    *,
    options: FilterOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> Expression: ...
@overload
def array_take(
    array: _ArrayT,
    indices: list[int]
    | list[int | None]
    | lib.Int16Array
    | lib.Int32Array
    | lib.Int64Array
    | lib.ChunkedArray[lib.Int16Scalar]
    | lib.ChunkedArray[lib.Int32Scalar]
    | lib.ChunkedArray[lib.Int64Scalar],
    /,
    *,
    boundscheck: bool = True,
    options: TakeOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> _ArrayT: ...
@overload
def array_take(
    array: Expression,
    indices: list[int]
    | list[int | None]
    | lib.Int16Array
    | lib.Int32Array
    | lib.Int64Array
    | lib.ChunkedArray[lib.Int16Scalar]
    | lib.ChunkedArray[lib.Int32Scalar]
    | lib.ChunkedArray[lib.Int64Scalar],
    /,
    *,
    boundscheck: bool = True,
    options: TakeOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> Expression: ...
@overload
def drop_null(input: _ArrayT, /, *, memory_pool: lib.MemoryPool | None = None) -> _ArrayT: ...
@overload
def drop_null(
    input: Expression, /, *, memory_pool: lib.MemoryPool | None = None
) -> Expression: ...

filter = array_filter
take = array_take

# ========================= 3.4 Containment tests  =========================
@overload
def indices_nonzero(
    values: lib.BooleanArray
    | lib.NullArray
    | NumericArray
    | lib.Decimal128Array
    | lib.Decimal256Array,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.UInt64Array: ...
@overload
def indices_nonzero(
    values: Expression,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> Expression: ...

# ========================= 3.5 Sorts and partitions  =========================
@overload
def array_sort_indices(
    array: lib.Array | lib.ChunkedArray,
    /,
    order: Literal["ascending", "descending"] = "ascending",
    *,
    null_placement: Literal["at_start", "at_end"] = "at_end",
    options: ArraySortOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.UInt64Array: ...
@overload
def array_sort_indices(
    array: Expression,
    /,
    order: Literal["ascending", "descending"] = "ascending",
    *,
    null_placement: Literal["at_start", "at_end"] = "at_end",
    options: ArraySortOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> Expression: ...
@overload
def partition_nth_indices(
    array: lib.Array | lib.ChunkedArray,
    /,
    pivot: int,
    *,
    null_placement: Literal["at_start", "at_end"] = "at_end",
    options: PartitionNthOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.UInt64Array: ...
@overload
def partition_nth_indices(
    array: Expression,
    /,
    pivot: int,
    *,
    null_placement: Literal["at_start", "at_end"] = "at_end",
    options: PartitionNthOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> Expression: ...
def rank(
    input: lib.Array | lib.ChunkedArray,
    /,
    sort_keys: Literal["ascending", "descending"] = "ascending",
    *,
    null_placement: Literal["at_start", "at_end"] = "at_end",
    tiebreaker: Literal["min", "max", "first", "dense"] = "first",
    options: RankOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.UInt64Array: ...
@overload
def select_k_unstable(
    input: lib.Array | lib.ChunkedArray,
    /,
    k: int,
    sort_keys: list[tuple[str, Literal["ascending", "descending"]]],
    *,
    options: SelectKOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.UInt64Array: ...
@overload
def select_k_unstable(
    input: Expression,
    /,
    k: int,
    sort_keys: list[tuple[str, Literal["ascending", "descending"]]],
    *,
    options: SelectKOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> Expression: ...
@overload
def sort_indices(
    input: lib.Array | lib.ChunkedArray | lib.RecordBatch | lib.Table,
    /,
    sort_keys: Sequence[tuple[str, Literal["ascending", "descending"]]],
    *,
    null_placement: Literal["at_start", "at_end"] = "at_end",
    options: SortOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.UInt64Array: ...
@overload
def sort_indices(
    input: Expression,
    /,
    sort_keys: Sequence[tuple[str, Literal["ascending", "descending"]]],
    *,
    null_placement: Literal["at_start", "at_end"] = "at_end",
    options: SortOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> Expression: ...

# ========================= 3.6 Structural transforms =========================
@overload
def list_element(
    lists: Expression, index, /, *, memory_pool: lib.MemoryPool | None = None
) -> Expression: ...
@overload
def list_element(
    lists, index, /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.ListArray: ...
@overload
def list_flatten(
    lists: Expression,
    /,
    recursive: bool = False,
    *,
    options: ListFlattenOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> Expression: ...
@overload
def list_flatten(
    lists,
    /,
    recursive: bool = False,
    *,
    options: ListFlattenOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.ListArray: ...
@overload
def list_parent_indices(
    lists: Expression, /, *, memory_pool: lib.MemoryPool | None = None
) -> Expression: ...
@overload
def list_parent_indices(
    lists, /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.Int64Array: ...
@overload
def list_slice(
    lists: Expression,
    /,
    start: int,
    stop: int | None = None,
    step: int = 1,
    return_fixed_size_list: bool | None = None,
    *,
    options: ListSliceOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> Expression: ...
@overload
def list_slice(
    lists,
    /,
    start: int,
    stop: int | None = None,
    step: int = 1,
    return_fixed_size_list: bool | None = None,
    *,
    options: ListSliceOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.ListArray: ...
def map_lookup(
    container,
    /,
    query_key,
    occurrence: str,
    *,
    options: MapLookupOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
): ...
def struct_field(
    values,
    /,
    indices,
    *,
    options: StructFieldOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
): ...
def fill_null_backward(values, /, *, memory_pool: lib.MemoryPool | None = None): ...
def fill_null_forward(values, /, *, memory_pool: lib.MemoryPool | None = None): ...
def replace_with_mask(
    values,
    mask: list[bool] | list[bool | None] | BooleanArray,
    replacements,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
): ...

# ========================= 3.7 Pairwise functions =========================
@overload
def pairwise_diff(
    input: _NumericOrTemporalArrayT,
    /,
    period: int = 1,
    *,
    options: PairwiseOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> _NumericOrTemporalArrayT: ...
@overload
def pairwise_diff(
    input: Expression,
    /,
    period: int = 1,
    *,
    options: PairwiseOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> Expression: ...

pairwise_diff_checked = _clone_signature(pairwise_diff)
