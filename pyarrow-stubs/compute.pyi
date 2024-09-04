# mypy: disable-error-code="misc,type-var,var-annotated"
# ruff: noqa: I001
from typing import Literal, TypeAlias, TypeVar, overload, Any, Iterable, ParamSpec
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
import typing_extensions

_P = ParamSpec("_P")
_R = TypeVar("_R")

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
NumericArray: TypeAlias = lib.NumericArray
_NumericArrayT = TypeVar("_NumericArrayT", bound=lib.NumericArray)
NumericOrDurationArray: TypeAlias = lib.NumericArray | lib.Array[lib.DurationScalar]
_NumericOrDurationArrayT = TypeVar("_NumericOrDurationArrayT", bound=NumericOrDurationArray)
NumericOrTemporalArray: TypeAlias = lib.NumericArray | lib.Array[TemporalScalar]
_NumericOrTemporalArrayT = TypeVar("_NumericOrTemporalArrayT", bound=NumericOrTemporalArray)
FloatScalar: typing_extensions.TypeAlias = (
    lib.Scalar[lib.Float32Type]
    | lib.Scalar[lib.Float64Type]
    | lib.Scalar[lib.Decimal128Type]
    | lib.Scalar[lib.Decimal256Type]
)
_FloatScalarT = TypeVar("_FloatScalarT", bound=FloatScalar)
FloatArray: typing_extensions.TypeAlias = (
    lib.NumericArray[lib.FloatScalar]
    | lib.NumericArray[lib.DoubleScalar]
    | lib.NumericArray[lib.Decimal128Scalar]
    | lib.NumericArray[lib.Decimal256Scalar]
)
_FloatArrayT = TypeVar("_FloatArrayT", bound=FloatArray)
_StringScalarT = TypeVar("_StringScalarT", bound=StringScalar)
StringArray: TypeAlias = lib.StringArray | lib.LargeStringArray
_StringArrayT = TypeVar("_StringArrayT", bound=StringArray)
_BinaryScalarT = TypeVar("_BinaryScalarT", bound=BinaryScalar)
BinaryArray: TypeAlias = lib.BinaryArray | lib.LargeBinaryArray
_BinaryArrayT = TypeVar("_BinaryArrayT", bound=BinaryArray)
StringOrBinaryScalar: TypeAlias = StringScalar | BinaryScalar
_StringOrBinaryScalarT = TypeVar("_StringOrBinaryScalarT", bound=StringOrBinaryScalar)
StringOrBinaryArray: TypeAlias = StringArray | BinaryArray
_StringOrBinaryArrayT = TypeVar("_StringOrBinaryArrayT", bound=StringOrBinaryArray)
_ScalarT = TypeVar("_ScalarT", bound=lib.Scalar)
_ArrayT = TypeVar("_ArrayT", bound=lib.Array)
# =============================== 1. Aggregation ===============================

# ========================= 1.1 functions =========================

def all(
    array: lib.BooleanScalar | lib.BooleanArray,
    /,
    *,
    skip_nulls: bool = True,
    min_count: int = 1,
    options: ScalarAggregateOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.BooleanScalar: ...

any = _clone_signature(all)

def approximate_median(
    array: NumericScalar | lib.NumericArray,
    /,
    *,
    skip_nulls: bool = True,
    min_count: int = 1,
    options: ScalarAggregateOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.DoubleScalar: ...
def count(
    array: lib.Array,
    /,
    mode: Literal["only_valid", "only_null", "all"] = "only_valid",
    *,
    options: CountOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.Int64Scalar: ...
def count_distinct(
    array: lib.Array,
    /,
    mode: Literal["only_valid", "only_null", "all"] = "only_valid",
    *,
    options: CountOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.Int64Scalar: ...
def first(
    array: lib.Array[_ScalarT],
    /,
    *,
    skip_nulls: bool = True,
    min_count: int = 1,
    options: ScalarAggregateOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> _ScalarT: ...
def first_last(
    array: lib.Array,
    /,
    *,
    skip_nulls: bool = True,
    min_count: int = 1,
    options: ScalarAggregateOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.StructScalar: ...
def index(
    data: lib.Array,
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

def mean(
    array: NumericScalar | lib.NumericArray,
    /,
    *,
    skip_nulls: bool = True,
    min_count: int = 1,
    options: ScalarAggregateOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.DoubleScalar | lib.Decimal128Scalar: ...
def mode(
    array: NumericScalar | lib.NumericArray,
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
    array: NumericScalar | lib.NumericArray,
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
    array: NumericScalar | lib.NumericArray,
    /,
    *,
    ddof: float = 0,
    skip_nulls: bool = True,
    min_count: int = 0,
    options: VarianceOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.DoubleScalar: ...
def sum(
    array: _NumericScalarT | lib.NumericArray[_NumericScalarT],
    /,
    *,
    skip_nulls: bool = True,
    min_count: int = 1,
    options: ScalarAggregateOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> _NumericScalarT: ...
def tdigest(
    array: NumericScalar | lib.NumericArray,
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
    array: NumericScalar | lib.NumericArray,
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

divide_checked = _clone_signature(divide)

@overload
def exp(
    exponent: NumericArray, /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.FloatArray | lib.DoubleArray: ...
@overload
def exp(
    exponent: NumericScalar, /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.FloatScalar | lib.DoubleScalar: ...

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
def sqrt(x: NumericArray, /, *, memory_pool: lib.MemoryPool | None = None) -> FloatArray: ...
@overload
def sqrt(x: NumericScalar, /, *, memory_pool: lib.MemoryPool | None = None) -> FloatScalar: ...

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
def bit_wise_not(
    x: _NumericScalarT, /, *, memory_pool: lib.MemoryPool | None = None
) -> _NumericScalarT: ...
@overload
def bit_wise_not(
    x: _NumericArrayT, /, *, memory_pool: lib.MemoryPool | None = None
) -> _NumericArrayT: ...

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
def logb(
    x: FloatScalar, b: FloatScalar, /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.FloatScalar | lib.DoubleScalar: ...

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
def atan2(
    y: FloatScalar, x: FloatScalar, /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.FloatScalar | lib.DoubleScalar: ...

# ========================= 2.5 Comparisons functions =========================
@overload
def equal(
    x: lib.Scalar, y: lib.Scalar, /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.BooleanScalar: ...
@overload
def equal(
    x: lib.Scalar | lib.Array,
    y: lib.Scalar | lib.Array,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.BooleanArray: ...

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

min_element_wise = _clone_signature(equal)

# ========================= 2.6 Logical functions =========================
@overload
def and_(
    x: lib.BooleanScalar, y: lib.BooleanScalar, /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.BooleanScalar: ...
@overload
def and_(
    x: lib.BooleanScalar | lib.BooleanArray,
    y: lib.BooleanScalar | lib.BooleanArray,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.BooleanArray: ...

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

# ========================= 2.7 String predicates =========================
@overload
def ascii_is_alnum(
    strings: StringScalar, /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.BooleanScalar: ...
@overload
def ascii_is_alnum(
    strings: StringArray, /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.BooleanArray: ...

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

# ========================= 2.7 String transforms =========================
@overload
def ascii_capitalize(
    strings: _StringScalarT, /, *, memory_pool: lib.MemoryPool | None = None
) -> _StringScalarT: ...
@overload
def ascii_capitalize(
    strings: _StringArrayT, /, *, memory_pool: lib.MemoryPool | None = None
) -> _StringArrayT: ...

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
    strings: lib.BinaryArray | lib.StringArray, /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.Int32Array: ...
@overload
def binary_length(
    strings: lib.LargeBinaryArray | lib.LargeStringArray,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.Int64Array: ...
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
def binary_reverse(
    strings: _BinaryScalarT, /, *, memory_pool: lib.MemoryPool | None = None
) -> _BinaryScalarT: ...
@overload
def binary_reverse(
    strings: _BinaryArrayT, /, *, memory_pool: lib.MemoryPool | None = None
) -> _BinaryArrayT: ...
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
    strings: lib.StringArray, /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.Int32Array: ...
@overload
def utf8_length(
    strings: lib.LargeStringArray,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.Int64Array: ...

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

utf8_reverse = _clone_signature(utf8_capitalize)
utf8_swapcase = _clone_signature(utf8_capitalize)
utf8_title = _clone_signature(utf8_capitalize)
utf8_upper = _clone_signature(utf8_capitalize)
