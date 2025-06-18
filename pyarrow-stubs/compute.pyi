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
from pyarrow._compute import ExtractRegexSpanOptions as ExtractRegexSpanOptions
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
from pyarrow._compute import PivotWiderOptions as PivotWiderOptions
from pyarrow._compute import QuantileOptions as QuantileOptions
from pyarrow._compute import RandomOptions as RandomOptions
from pyarrow._compute import RankOptions as RankOptions
from pyarrow._compute import RankQuantileOptions as RankQuantileOptions
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
from pyarrow._compute import SkewOptions as SkewOptions
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
from pyarrow._compute import WinsorizeOptions as WinsorizeOptions

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

from pyarrow._compute import _Order, _Placement
from pyarrow._stubs_typing import ArrayLike, ScalarLike
from . import lib

_P = ParamSpec("_P")
_R = TypeVar("_R")

def field(*name_or_index: str | tuple[str, ...] | int) -> Expression:
    """Reference a column of the dataset.

    Stores only the field's name. Type and other information is known only when
    the expression is bound to a dataset having an explicit scheme.

    Nested references are allowed by passing multiple names or a tuple of
    names. For example ``('foo', 'bar')`` references the field named "bar"
    inside the field named "foo".

    Parameters
    ----------
    *name_or_index : string, multiple strings, tuple or int
        The name or index of the (possibly nested) field the expression
        references to.

    Returns
    -------
    field_expr : Expression
        Reference to the given field

    Examples
    --------
    >>> import pyarrow.compute as pc
    >>> pc.field("a")
    <pyarrow.compute.Expression a>
    >>> pc.field(1)
    <pyarrow.compute.Expression FieldPath(1)>
    >>> pc.field(("a", "b"))
    <pyarrow.compute.Expression FieldRef.Nested(FieldRef.Name(a) ...
    >>> pc.field("a", "b")
    <pyarrow.compute.Expression FieldRef.Nested(FieldRef.Name(a) ...
    """

def scalar(value: bool | float | str) -> Expression:
    """Expression representing a scalar value.

    Creates an Expression object representing a scalar value that can be used
    in compute expressions and predicates.

    Parameters
    ----------
    value : bool, int, float or string
        Python value of the scalar. This function accepts any value that can be
        converted to a ``pyarrow.Scalar`` using ``pa.scalar()``.

    Notes
    -----
    This function differs from ``pyarrow.scalar()`` in the following way:

    * ``pyarrow.scalar()`` creates a ``pyarrow.Scalar`` object that represents
      a single value in Arrow's memory model.
    * ``pyarrow.compute.scalar()`` creates an ``Expression`` object representing
      a scalar value that can be used in compute expressions, predicates, and
      dataset filtering operations.

    Returns
    -------
    scalar_expr : Expression
        An Expression representing the scalar value
    """

def _clone_signature(f: Callable[_P, _R]) -> Callable[_P, _R]: ...

# ============= compute functions =============
_DataTypeT = TypeVar("_DataTypeT", bound=lib.DataType)
_Scalar_CoT = TypeVar("_Scalar_CoT", bound=lib.Scalar, covariant=True)
_ScalarT = TypeVar("_ScalarT", bound=lib.Scalar)
_ArrayT = TypeVar("_ArrayT", bound=lib.Array | lib.ChunkedArray)
_ScalarOrArrayT = TypeVar("_ScalarOrArrayT", bound=lib.Array | lib.Scalar | lib.ChunkedArray)
ArrayOrChunkedArray: TypeAlias = lib.Array[_Scalar_CoT] | lib.ChunkedArray[_Scalar_CoT]
ScalarOrArray: TypeAlias = ArrayOrChunkedArray[_Scalar_CoT] | _Scalar_CoT

SignedIntegerScalar: TypeAlias = (
    lib.Scalar[lib.Int8Type]
    | lib.Scalar[lib.Int16Type]
    | lib.Scalar[lib.Int32Type]
    | lib.Scalar[lib.Int64Type]
)
UnsignedIntegerScalar: TypeAlias = (
    lib.Scalar[lib.UInt8Type]
    | lib.Scalar[lib.UInt16Type]
    | lib.Scalar[lib.Uint32Type]
    | lib.Scalar[lib.UInt64Type]
)
IntegerScalar: TypeAlias = SignedIntegerScalar | UnsignedIntegerScalar
FloatScalar: TypeAlias = (
    lib.Scalar[lib.Float16Type] | lib.Scalar[lib.Float32Type] | lib.Scalar[lib.Float64Type]
)
DecimalScalar: TypeAlias = (
    lib.Scalar[lib.Decimal32Type]
    | lib.Scalar[lib.Decimal64Type]
    | lib.Scalar[lib.Decimal128Type]
    | lib.Scalar[lib.Decimal256Type]
)
NonFloatNumericScalar: TypeAlias = IntegerScalar | DecimalScalar
NumericScalar: TypeAlias = IntegerScalar | FloatScalar | DecimalScalar
BinaryScalar: TypeAlias = (
    lib.Scalar[lib.BinaryType]
    | lib.Scalar[lib.LargeBinaryType]
    | lib.Scalar[lib.FixedSizeBinaryType]
)
StringScalar: TypeAlias = lib.Scalar[lib.StringType] | lib.Scalar[lib.LargeStringType]
StringOrBinaryScalar: TypeAlias = StringScalar | BinaryScalar
_ListScalar: TypeAlias = lib.ListViewScalar[_DataTypeT] | lib.FixedSizeListScalar[_DataTypeT, Any]
_LargeListScalar: TypeAlias = lib.LargeListScalar[_DataTypeT] | lib.LargeListViewScalar[_DataTypeT]
ListScalar: TypeAlias = (
    lib.ListScalar[_DataTypeT] | _ListScalar[_DataTypeT] | _LargeListScalar[_DataTypeT]
)
TemporalScalar: TypeAlias = (
    lib.Date32Scalar
    | lib.Date64Scalar
    | lib.Time32Scalar[Any]
    | lib.Time64Scalar[Any]
    | lib.TimestampScalar[Any]
    | lib.DurationScalar[Any]
    | lib.MonthDayNanoIntervalScalar
)
NumericOrDurationScalar: TypeAlias = NumericScalar | lib.DurationScalar
NumericOrTemporalScalar: TypeAlias = NumericScalar | TemporalScalar

_NumericOrTemporalScalarT = TypeVar("_NumericOrTemporalScalarT", bound=NumericOrTemporalScalar)
NumericArray: TypeAlias = ArrayOrChunkedArray[_NumericScalarT]
_NumericArrayT = TypeVar("_NumericArrayT", bound=NumericArray)
_NumericScalarT = TypeVar("_NumericScalarT", bound=NumericScalar)
_NumericOrDurationT = TypeVar("_NumericOrDurationT", bound=NumericOrDurationScalar)
NumericOrDurationArray: TypeAlias = ArrayOrChunkedArray[NumericOrDurationScalar]
_NumericOrDurationArrayT = TypeVar("_NumericOrDurationArrayT", bound=NumericOrDurationArray)
NumericOrTemporalArray: TypeAlias = ArrayOrChunkedArray[_NumericOrTemporalScalarT]
_NumericOrTemporalArrayT = TypeVar("_NumericOrTemporalArrayT", bound=NumericOrTemporalArray)
BooleanArray: TypeAlias = ArrayOrChunkedArray[lib.BooleanScalar]
_BooleanArrayT = TypeVar("_BooleanArrayT", bound=BooleanArray)
IntegerArray: TypeAlias = ArrayOrChunkedArray[IntegerScalar]
_FloatScalarT = TypeVar("_FloatScalarT", bound=FloatScalar)
FloatArray: TypeAlias = ArrayOrChunkedArray[FloatScalar]
_FloatArrayT = TypeVar("_FloatArrayT", bound=FloatArray)
_StringScalarT = TypeVar("_StringScalarT", bound=StringScalar)
StringArray: TypeAlias = ArrayOrChunkedArray[StringScalar]
_StringArrayT = TypeVar("_StringArrayT", bound=StringArray)
_BinaryScalarT = TypeVar("_BinaryScalarT", bound=BinaryScalar)
BinaryArray: TypeAlias = ArrayOrChunkedArray[BinaryScalar]
_BinaryArrayT = TypeVar("_BinaryArrayT", bound=BinaryArray)
_StringOrBinaryScalarT = TypeVar("_StringOrBinaryScalarT", bound=StringOrBinaryScalar)
StringOrBinaryArray: TypeAlias = StringArray | BinaryArray
_StringOrBinaryArrayT = TypeVar("_StringOrBinaryArrayT", bound=StringOrBinaryArray)
_TemporalScalarT = TypeVar("_TemporalScalarT", bound=TemporalScalar)
TemporalArray: TypeAlias = ArrayOrChunkedArray[TemporalScalar]
_TemporalArrayT = TypeVar("_TemporalArrayT", bound=TemporalArray)
_ListArray: TypeAlias = ArrayOrChunkedArray[_ListScalar[_DataTypeT]]
_LargeListArray: TypeAlias = ArrayOrChunkedArray[_LargeListScalar[_DataTypeT]]
ListArray: TypeAlias = ArrayOrChunkedArray[ListScalar[_DataTypeT]]
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
) -> lib.BooleanScalar:
    """
    Test whether all elements in a boolean array evaluate to true.

    Null values are ignored by default.
    If the `skip_nulls` option is set to false, then Kleene logic is used.
    See "kleene_and" for more details on Kleene logic.

    Parameters
    ----------
    array : Array-like
        Argument to compute function.
    skip_nulls : bool, default True
        Whether to skip (ignore) nulls in the input.
        If False, any null in the input forces the output to null.
    min_count : int, default 1
        Minimum number of non-null values in the input.  If the number
        of non-null values is below `min_count`, the output is null.
    options : pyarrow.compute.ScalarAggregateOptions, optional
        Alternative way of passing options.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

any = _clone_signature(all)
"""
Test whether any element in a boolean array evaluates to true.

Null values are ignored by default.
If the `skip_nulls` option is set to false, then Kleene logic is used.
See "kleene_or" for more details on Kleene logic.

Parameters
----------
array : Array-like
    Argument to compute function.
skip_nulls : bool, default True
    Whether to skip (ignore) nulls in the input.
    If False, any null in the input forces the output to null.
min_count : int, default 1
    Minimum number of non-null values in the input.  If the number
    of non-null values is below `min_count`, the output is null.
options : pyarrow.compute.ScalarAggregateOptions, optional
    Alternative way of passing options.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""

def approximate_median(
    array: NumericScalar | NumericArray,
    /,
    *,
    skip_nulls: bool = True,
    min_count: int = 1,
    options: ScalarAggregateOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.DoubleScalar:
    """
    Approximate median of a numeric array with T-Digest algorithm.

    Nulls and NaNs are ignored.
    A null scalar is returned if there is no valid data point.

    Parameters
    ----------
    array : Array-like
        Argument to compute function.
    skip_nulls : bool, default True
        Whether to skip (ignore) nulls in the input.
        If False, any null in the input forces the output to null.
    min_count : int, default 1
        Minimum number of non-null values in the input.  If the number
        of non-null values is below `min_count`, the output is null.
    options : pyarrow.compute.ScalarAggregateOptions, optional
        Alternative way of passing options.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

def count(
    array: lib.Array | lib.ChunkedArray,
    /,
    mode: Literal["only_valid", "only_null", "all"] = "only_valid",
    *,
    options: CountOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.Int64Scalar:
    """
    Count the number of null / non-null values.

    By default, only non-null values are counted.
    This can be changed through CountOptions.

    Parameters
    ----------
    array : Array-like
        Argument to compute function.
    mode : str, default "only_valid"
        Which values to count in the input.
        Accepted values are "only_valid", "only_null", "all".
    options : pyarrow.compute.CountOptions, optional
        Alternative way of passing options.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

def count_distinct(
    array: lib.Array | lib.ChunkedArray,
    /,
    mode: Literal["only_valid", "only_null", "all"] = "only_valid",
    *,
    options: CountOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.Int64Scalar:
    """
    Count the number of unique values.

    By default, only non-null values are counted.
    This can be changed through CountOptions.

    Parameters
    ----------
    array : Array-like
        Argument to compute function.
    mode : str, default "only_valid"
        Which values to count in the input.
        Accepted values are "only_valid", "only_null", "all".
    options : pyarrow.compute.CountOptions, optional
        Alternative way of passing options.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

def first(
    array: lib.Array[_ScalarT] | lib.ChunkedArray[_ScalarT],
    /,
    *,
    skip_nulls: bool = True,
    min_count: int = 1,
    options: ScalarAggregateOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> _ScalarT:
    """
    Compute the first value in each group.

    Null values are ignored by default.
    If skip_nulls = false, then this will return the first and last values
    regardless if it is null

    Parameters
    ----------
    array : Array-like
        Argument to compute function.
    skip_nulls : bool, default True
        Whether to skip (ignore) nulls in the input.
        If False, any null in the input forces the output to null.
    min_count : int, default 1
        Minimum number of non-null values in the input.  If the number
        of non-null values is below `min_count`, the output is null.
    options : pyarrow.compute.ScalarAggregateOptions, optional
        Alternative way of passing options.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

def first_last(
    array: lib.Array[Any] | lib.ChunkedArray[Any],
    /,
    *,
    skip_nulls: bool = True,
    min_count: int = 1,
    options: ScalarAggregateOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.StructScalar:
    """
    Compute the first and last values of an array.

    Null values are ignored by default.
    If skip_nulls = false, then this will return the first and last values
    regardless if it is null

    Parameters
    ----------
    array : Array-like
        Argument to compute function.
    skip_nulls : bool, default True
        Whether to skip (ignore) nulls in the input.
        If False, any null in the input forces the output to null.
    min_count : int, default 1
        Minimum number of non-null values in the input.  If the number
        of non-null values is below `min_count`, the output is null.
    options : pyarrow.compute.ScalarAggregateOptions, optional
        Alternative way of passing options.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

def index(
    data: lib.Array[Any] | lib.ChunkedArray[Any],
    value,
    start: int | None = None,
    end: int | None = None,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.Int64Scalar:
    """
    Find the index of the first occurrence of a given value.

    Parameters
    ----------
    data : Array-like
    value : Scalar-like object
        The value to search for.
    start : int, optional
    end : int, optional
    memory_pool : MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.

    Returns
    -------
    index : int
        the index, or -1 if not found

    Examples
    --------
    >>> import pyarrow as pa
    >>> import pyarrow.compute as pc
    >>> arr = pa.array(["Lorem", "ipsum", "dolor", "sit", "Lorem", "ipsum"])
    >>> pc.index(arr, "ipsum")
    <pyarrow.Int64Scalar: 1>
    >>> pc.index(arr, "ipsum", start=2)
    <pyarrow.Int64Scalar: 5>
    >>> pc.index(arr, "amet")
    <pyarrow.Int64Scalar: -1>
    """

last = _clone_signature(first)
"""
Compute the first and last values of an array.

Null values are ignored by default.
If skip_nulls = false, then this will return the first and last values
regardless if it is null

Parameters
----------
array : Array-like
    Argument to compute function.
skip_nulls : bool, default True
In [15]: print(pc.last.__doc__)
Compute the first value in each group.

Null values are ignored by default.
If skip_nulls = false, then this will return the first and last values
regardless if it is null

Parameters
----------
array : Array-like
    Argument to compute function.
skip_nulls : bool, default True
    Whether to skip (ignore) nulls in the input.
    If False, any null in the input forces the output to null.
min_count : int, default 1
    Minimum number of non-null values in the input.  If the number
    of non-null values is below `min_count`, the output is null.
options : pyarrow.compute.ScalarAggregateOptions, optional
    Alternative way of passing options.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
max = _clone_signature(first)
"""
Compute the minimum or maximum values of a numeric array.

Null values are ignored by default.
This can be changed through ScalarAggregateOptions.

Parameters
----------
array : Array-like
    Argument to compute function.
skip_nulls : bool, default True
    Whether to skip (ignore) nulls in the input.
    If False, any null in the input forces the output to null.
min_count : int, default 1
    Minimum number of non-null values in the input.  If the number
    of non-null values is below `min_count`, the output is null.
options : pyarrow.compute.ScalarAggregateOptions, optional
    Alternative way of passing options.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
min = _clone_signature(first)
"""
Compute the minimum or maximum values of a numeric array.

Null values are ignored by default.
This can be changed through ScalarAggregateOptions.

Parameters
----------
array : Array-like
    Argument to compute function.
skip_nulls : bool, default True
    Whether to skip (ignore) nulls in the input.
    If False, any null in the input forces the output to null.
min_count : int, default 1
    Minimum number of non-null values in the input.  If the number
    of non-null values is below `min_count`, the output is null.
options : pyarrow.compute.ScalarAggregateOptions, optional
    Alternative way of passing options.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
min_max = _clone_signature(first_last)
"""
Compute the minimum and maximum values of a numeric array.

Null values are ignored by default.
This can be changed through ScalarAggregateOptions.

Parameters
----------
array : Array-like
    Argument to compute function.
skip_nulls : bool, default True
    Whether to skip (ignore) nulls in the input.
    If False, any null in the input forces the output to null.
min_count : int, default 1
    Minimum number of non-null values in the input.  If the number
    of non-null values is below `min_count`, the output is null.
options : pyarrow.compute.ScalarAggregateOptions, optional
    Alternative way of passing options.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""

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
def mean(*args, **kwargs):
    """
    Compute the mean of a numeric array.

    Null values are ignored by default. Minimum count of non-null
    values can be set and null is returned if too few are present.
    This can be changed through ScalarAggregateOptions.
    The result is a double for integer and floating point arguments,
    and a decimal with the same bit-width/precision/scale for decimal arguments.
    For integers and floats, NaN is returned if min_count = 0 and
    there are no values. For decimals, null is returned instead.

    Parameters
    ----------
    array : Array-like
        Argument to compute function.
    skip_nulls : bool, default True
        Whether to skip (ignore) nulls in the input.
        If False, any null in the input forces the output to null.
    min_count : int, default 1
        Minimum number of non-null values in the input.  If the number
        of non-null values is below `min_count`, the output is null.
    options : pyarrow.compute.ScalarAggregateOptions, optional
        Alternative way of passing options.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

def mode(
    array: NumericScalar | NumericArray,
    /,
    n: int = 1,
    *,
    skip_nulls: bool = True,
    min_count: int = 0,
    options: ModeOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.StructArray:
    """
    Compute the modal (most common) values of a numeric array.

    Compute the n most common values and their respective occurrence counts.
    The output has type `struct<mode: T, count: int64>`, where T is the
    input type.
    The results are ordered by descending `count` first, and ascending `mode`
    when breaking ties.
    Nulls are ignored.  If there are no non-null values in the array,
    an empty array is returned.

    Parameters
    ----------
    array : Array-like
        Argument to compute function.
    n : int, default 1
        Number of distinct most-common values to return.
    skip_nulls : bool, default True
        Whether to skip (ignore) nulls in the input.
        If False, any null in the input forces the output to null.
    min_count : int, default 0
        Minimum number of non-null values in the input.  If the number
        of non-null values is below `min_count`, the output is null.
    options : pyarrow.compute.ModeOptions, optional
        Alternative way of passing options.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.

    Examples
    --------
    >>> import pyarrow as pa
    >>> import pyarrow.compute as pc
    >>> arr = pa.array([1, 1, 2, 2, 3, 2, 2, 2])
    >>> modes = pc.mode(arr, 2)
    >>> modes[0]
    <pyarrow.StructScalar: [('mode', 2), ('count', 5)]>
    >>> modes[1]
    <pyarrow.StructScalar: [('mode', 1), ('count', 2)]>
    """

def product(
    array: _ScalarT | lib.NumericArray[_ScalarT],
    /,
    *,
    skip_nulls: bool = True,
    min_count: int = 1,
    options: ScalarAggregateOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> _ScalarT:
    """
    Compute the product of values in a numeric array.

    Null values are ignored by default. Minimum count of non-null
    values can be set and null is returned if too few are present.
    This can be changed through ScalarAggregateOptions.

    Parameters
    ----------
    array : Array-like
        Argument to compute function.
    skip_nulls : bool, default True
        Whether to skip (ignore) nulls in the input.
        If False, any null in the input forces the output to null.
    min_count : int, default 1
        Minimum number of non-null values in the input.  If the number
        of non-null values is below `min_count`, the output is null.
    options : pyarrow.compute.ScalarAggregateOptions, optional
        Alternative way of passing options.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

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
) -> lib.DoubleArray:
    """
    Compute an array of quantiles of a numeric array or chunked array.

    By default, 0.5 quantile (median) is returned.
    If quantile lies between two data points, an interpolated value is
    returned based on selected interpolation method.
    Nulls and NaNs are ignored.
    An array of nulls is returned if there is no valid data point.

    Parameters
    ----------
    array : Array-like
        Argument to compute function.
    q : double or sequence of double, default 0.5
        Probability levels of the quantiles to compute. All values must be in
        [0, 1].
    interpolation : str, default "linear"
        How to break ties between competing data points for a given quantile.
        Accepted values are:

        - "linear": compute an interpolation
        - "lower": always use the smallest of the two data points
        - "higher": always use the largest of the two data points
        - "nearest": select the data point that is closest to the quantile
        - "midpoint": compute the (unweighted) mean of the two data points
    skip_nulls : bool, default True
        Whether to skip (ignore) nulls in the input.
        If False, any null in the input forces the output to null.
    min_count : int, default 0
        Minimum number of non-null values in the input.  If the number
        of non-null values is below `min_count`, the output is null.
    options : pyarrow.compute.QuantileOptions, optional
        Alternative way of passing options.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

def stddev(
    array: NumericScalar | NumericArray,
    /,
    *,
    ddof: float = 0,
    skip_nulls: bool = True,
    min_count: int = 0,
    options: VarianceOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.DoubleScalar:
    """
    Calculate the standard deviation of a numeric array.

    The number of degrees of freedom can be controlled using VarianceOptions.
    By default (`ddof` = 0), the population standard deviation is calculated.
    Nulls are ignored.  If there are not enough non-null values in the array
    to satisfy `ddof`, null is returned.

    Parameters
    ----------
    array : Array-like
        Argument to compute function.
    ddof : int, default 0
        Number of degrees of freedom.
    skip_nulls : bool, default True
        Whether to skip (ignore) nulls in the input.
        If False, any null in the input forces the output to null.
    min_count : int, default 0
        Minimum number of non-null values in the input.  If the number
        of non-null values is below `min_count`, the output is null.
    options : pyarrow.compute.VarianceOptions, optional
        Alternative way of passing options.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

def sum(
    array: _NumericScalarT | NumericArray[_NumericScalarT],
    /,
    *,
    skip_nulls: bool = True,
    min_count: int = 1,
    options: ScalarAggregateOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> _NumericScalarT:
    """
    Compute the sum of a numeric array.

    Null values are ignored by default. Minimum count of non-null
    values can be set and null is returned if too few are present.
    This can be changed through ScalarAggregateOptions.

    Parameters
    ----------
    array : Array-like
        Argument to compute function.
    skip_nulls : bool, default True
        Whether to skip (ignore) nulls in the input.
        If False, any null in the input forces the output to null.
    min_count : int, default 1
        Minimum number of non-null values in the input.  If the number
        of non-null values is below `min_count`, the output is null.
    options : pyarrow.compute.ScalarAggregateOptions, optional
        Alternative way of passing options.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

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
) -> lib.DoubleArray:
    """
    Approximate quantiles of a numeric array with T-Digest algorithm.

    By default, 0.5 quantile (median) is returned.
    Nulls and NaNs are ignored.
    An array of nulls is returned if there is no valid data point.

    Parameters
    ----------
    array : Array-like
        Argument to compute function.
    q : double or sequence of double, default 0.5
        Probability levels of the quantiles to approximate. All values must be
        in [0, 1].
    delta : int, default 100
        Compression parameter for the T-digest algorithm.
    buffer_size : int, default 500
        Buffer size for the T-digest algorithm.
    skip_nulls : bool, default True
        Whether to skip (ignore) nulls in the input.
        If False, any null in the input forces the output to null.
    min_count : int, default 0
        Minimum number of non-null values in the input.  If the number
        of non-null values is below `min_count`, the output is null.
    options : pyarrow.compute.TDigestOptions, optional
        Alternative way of passing options.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.

    """

def variance(
    array: NumericScalar | NumericArray,
    /,
    *,
    ddof: int = 0,
    skip_nulls: bool = True,
    min_count: int = 0,
    options: VarianceOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.DoubleScalar:
    """
    Calculate the variance of a numeric array.

    The number of degrees of freedom can be controlled using VarianceOptions.
    By default (`ddof` = 0), the population variance is calculated.
    Nulls are ignored.  If there are not enough non-null values in the array
    to satisfy `ddof`, null is returned.

    Parameters
    ----------
    array : Array-like
        Argument to compute function.
    ddof : int, default 0
        Number of degrees of freedom.
    skip_nulls : bool, default True
        Whether to skip (ignore) nulls in the input.
        If False, any null in the input forces the output to null.
    min_count : int, default 0
        Minimum number of non-null values in the input.  If the number
        of non-null values is below `min_count`, the output is null.
    options : pyarrow.compute.VarianceOptions, optional
        Alternative way of passing options.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

def top_k_unstable(
    values: lib.Array | lib.ChunkedArray | lib.RecordBatch | lib.Table,
    k: int,
    sort_keys: list | None = None,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.Array:
    """
    Select the indices of the top-k ordered elements from array- or table-like
    data.

    This is a specialization for :func:`select_k_unstable`. Output is not
    guaranteed to be stable.

    Parameters
    ----------
    values : Array, ChunkedArray, RecordBatch, or Table
        Data to sort and get top indices from.
    k : int
        The number of `k` elements to keep.
    sort_keys : List-like
        Column key names to order by when input is table-like data.
    memory_pool : MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.

    Returns
    -------
    result : Array
        Indices of the top-k ordered elements

    Examples
    --------
    >>> import pyarrow as pa
    >>> import pyarrow.compute as pc
    >>> arr = pa.array(["a", "b", "c", None, "e", "f"])
    >>> pc.top_k_unstable(arr, k=3)
    <pyarrow.lib.UInt64Array object at ...>
    [
      5,
      4,
      2
    ]
    """

def bottom_k_unstable(
    values: lib.Array | lib.ChunkedArray | lib.RecordBatch | lib.Table,
    k: int,
    sort_keys: list | None = None,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.Array:
    """
    Select the indices of the bottom-k ordered elements from
    array- or table-like data.

    This is a specialization for :func:`select_k_unstable`. Output is not
    guaranteed to be stable.

    Parameters
    ----------
    values : Array, ChunkedArray, RecordBatch, or Table
        Data to sort and get bottom indices from.
    k : int
        The number of `k` elements to keep.
    sort_keys : List-like
        Column key names to order by when input is table-like data.
    memory_pool : MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.

    Returns
    -------
    result : Array of indices
        Indices of the bottom-k ordered elements

    Examples
    --------
    >>> import pyarrow as pa
    >>> import pyarrow.compute as pc
    >>> arr = pa.array(["a", "b", "c", None, "e", "f"])
    >>> pc.bottom_k_unstable(arr, k=3)
    <pyarrow.lib.UInt64Array object at ...>
    [
      0,
      1,
      2
    ]
    """

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
def abs(*args, **kwargs):
    """
    Calculate the absolute value of the argument element-wise.

    Results will wrap around on integer overflow.
    Use function "abs_checked" if you want overflow
    to return an error.

    Parameters
    ----------
    x : Array-like or scalar-like
        Argument to compute function.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

abs_checked = _clone_signature(abs)
"""
Calculate the absolute value of the argument element-wise.

This function returns an error on overflow.  For a variant that
doesn't fail on overflow, use function "abs".

Parameters
----------
x : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""

@overload
def add(
    x: _NumericOrTemporalScalarT,
    y: _NumericOrTemporalScalarT,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> _NumericOrTemporalScalarT: ...
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
    x: Expression, y: Expression, /, *, memory_pool: lib.MemoryPool | None = None
) -> Expression: ...
@overload
def add(
    x: NumericOrTemporalScalar,
    y: _NumericOrTemporalArrayT,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> _NumericOrTemporalArrayT: ...
@overload
def add(
    x: _NumericOrTemporalArrayT,
    y: NumericOrTemporalScalar,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> _NumericOrTemporalArrayT: ...
@overload
def add(
    x: NumericOrTemporalScalar, y: Expression, /, *, memory_pool: lib.MemoryPool | None = None
) -> Expression: ...
@overload
def add(
    x: Expression, y: NumericOrTemporalScalar, /, *, memory_pool: lib.MemoryPool | None = None
) -> Expression: ...
def add(*args, **kwargs):
    """
    Add the arguments element-wise.

    Results will wrap around on integer overflow.
    Use function "add_checked" if you want overflow
    to return an error.

    Parameters
    ----------
    x : Array-like or scalar-like
        Argument to compute function.
    y : Array-like or scalar-like
        Argument to compute function.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

add_checked = _clone_signature(add)
"""
Add the arguments element-wise.

This function returns an error on overflow.  For a variant that
doesn't fail on overflow, use function "add".

Parameters
----------
x : Array-like or scalar-like
    Argument to compute function.
y : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.

"""

@overload
def divide(
    dividend: _NumericOrTemporalScalarT,
    divisor: _NumericOrTemporalScalarT,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> _NumericOrTemporalScalarT: ...
@overload
def divide(
    dividend: _NumericOrTemporalArrayT,
    divisor: _NumericOrTemporalArrayT,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> _NumericOrTemporalArrayT: ...
@overload
def divide(
    dividend: Expression,
    divisor: Expression,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> Expression: ...
@overload
def divide(
    dividend: NumericOrTemporalScalar,
    divisor: _NumericOrTemporalArrayT,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> _NumericOrTemporalArrayT: ...
@overload
def divide(
    dividend: _NumericOrTemporalArrayT,
    divisor: NumericOrTemporalScalar,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> _NumericOrTemporalArrayT: ...
@overload
def divide(
    dividend: NumericOrTemporalScalar,
    divisor: Expression,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> Expression: ...
@overload
def divide(
    dividend: Expression,
    divisor: NumericOrTemporalScalar,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> Expression: ...
def divide(*args, **kwargs):
    """
    Divide the arguments element-wise.

    Integer division by zero returns an error. However, integer overflow
    wraps around, and floating-point division by zero returns an infinite.
    Use function "divide_checked" if you want to get an error
    in all the aforementioned cases.

    Parameters
    ----------
    dividend : Array-like or scalar-like
        Argument to compute function.
    divisor : Array-like or scalar-like
        Argument to compute function.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.

    """

divide_checked = _clone_signature(divide)
"""
Divide the arguments element-wise.

An error is returned when trying to divide by zero, or when
integer overflow is encountered.

Parameters
----------
dividend : Array-like or scalar-like
    Argument to compute function.
divisor : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""

@overload
def exp(
    exponent: _FloatArrayT, /, *, memory_pool: lib.MemoryPool | None = None
) -> _FloatArrayT: ...
@overload
def exp(
    exponent: ArrayOrChunkedArray[NonFloatNumericScalar],
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.DoubleArray: ...
@overload
def exp(
    exponent: _FloatScalarT, /, *, memory_pool: lib.MemoryPool | None = None
) -> _FloatScalarT: ...
@overload
def exp(
    exponent: NonFloatNumericScalar, /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.DoubleScalar: ...
@overload
def exp(exponent: Expression, /, *, memory_pool: lib.MemoryPool | None = None) -> Expression: ...
def exp(*args, **kwargs):
    """
    Compute Euler's number raised to the power of specified exponent, element-wise.

    If exponent is null the result will be null.

    Parameters
    ----------
    exponent : Array-like or scalar-like
        Argument to compute function.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

multiply = _clone_signature(add)
"""
Multiply the arguments element-wise.

Results will wrap around on integer overflow.
Use function "multiply_checked" if you want overflow
to return an error.

Parameters
----------
x : Array-like or scalar-like
    Argument to compute function.
y : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
multiply_checked = _clone_signature(add)
"""
Multiply the arguments element-wise.

This function returns an error on overflow.  For a variant that
doesn't fail on overflow, use function "multiply".

Parameters
----------
x : Array-like or scalar-like
    Argument to compute function.
y : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""

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
def negate(*args, **kwargs):
    """
    Negate the argument element-wise.

    Results will wrap around on integer overflow.
    Use function "negate_checked" if you want overflow
    to return an error.

    Parameters
    ----------
    x : Array-like or scalar-like
        Argument to compute function.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

negate_checked = _clone_signature(negate)
"""
Negate the arguments element-wise.

This function returns an error on overflow.  For a variant that
doesn't fail on overflow, use function "negate".

Parameters
----------
x : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""

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
    base: _NumericArrayT,
    exponent: _NumericArrayT,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> _NumericArrayT: ...
@overload
def power(
    base: Expression,
    exponent: Expression,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> Expression: ...
@overload
def power(
    base: _NumericArrayT,
    exponent: NumericScalar,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> _NumericArrayT: ...
@overload
def power(
    base: NumericScalar,
    exponent: _NumericArrayT,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> _NumericArrayT: ...
@overload
def power(
    base: NumericScalar,
    exponent: Expression,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> Expression: ...
@overload
def power(
    base: Expression,
    exponent: NumericScalar,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> Expression: ...
def power(*args, **kwargs):
    """
    Raise arguments to power element-wise.

    Integer to negative integer power returns an error. However, integer overflow
    wraps around. If either base or exponent is null the result will be null.

    Parameters
    ----------
    base : Array-like or scalar-like
        Argument to compute function.
    exponent : Array-like or scalar-like
        Argument to compute function.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

power_checked = _clone_signature(power)
"""
Raise arguments to power element-wise.

An error is returned when integer to negative integer power is encountered,
or integer overflow is encountered.

Parameters
----------
base : Array-like or scalar-like
    Argument to compute function.
exponent : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""

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
def sign(*args, **kwargs):
    """
    Get the signedness of the arguments element-wise.

    Output is any of (-1,1) for nonzero inputs and 0 for zero input.
    NaN values return NaN.  Integral values return signedness as Int8 and
    floating-point values return it with the same type as the input values.

    Parameters
    ----------
    x : Array-like or scalar-like
        Argument to compute function.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.

    """

@overload
def sqrt(x: NumericArray, /, *, memory_pool: lib.MemoryPool | None = None) -> FloatArray: ...
@overload
def sqrt(x: NumericScalar, /, *, memory_pool: lib.MemoryPool | None = None) -> FloatScalar: ...
@overload
def sqrt(x: Expression, /, *, memory_pool: lib.MemoryPool | None = None) -> Expression: ...
def sqrt(*args, **kwargs):
    """
    Takes the square root of arguments element-wise.

    A negative argument returns a NaN.  For a variant that returns an
    error, use function "sqrt_checked".

    Parameters
    ----------
    x : Array-like or scalar-like
        Argument to compute function.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.

    """

sqrt_checked = _clone_signature(sqrt)
"""
Takes the square root of arguments element-wise.

A negative argument returns an error.  For a variant that returns a
NaN, use function "sqrt".

Parameters
----------
x : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""

subtract = _clone_signature(add)
"""
Subtract the arguments element-wise.

Results will wrap around on integer overflow.
Use function "subtract_checked" if you want overflow
to return an error.

Parameters
----------
x : Array-like or scalar-like
    Argument to compute function.
y : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
subtract_checked = _clone_signature(add)
"""
Subtract the arguments element-wise.

This function returns an error on overflow.  For a variant that
doesn't fail on overflow, use function "subtract".

Parameters
----------
x : Array-like or scalar-like
    Argument to compute function.
y : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""

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
    x: NumericScalar, y: _NumericArrayT, /, *, memory_pool: lib.MemoryPool | None = None
) -> _NumericArrayT: ...
@overload
def bit_wise_and(
    x: _NumericArrayT, y: NumericScalar, /, *, memory_pool: lib.MemoryPool | None = None
) -> _NumericArrayT: ...
@overload
def bit_wise_and(
    x: Expression,
    y: Expression,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> Expression: ...
@overload
def bit_wise_and(
    x: Expression,
    y: NumericScalar | ArrayOrChunkedArray[NumericScalar],
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> Expression: ...
@overload
def bit_wise_and(
    x: NumericScalar | ArrayOrChunkedArray[NumericScalar],
    y: Expression,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> Expression: ...
def bit_wise_and(*args, **kwargs):
    """
    Bit-wise AND the arguments element-wise.

    Null values return null.

    Parameters
    ----------
    x : Array-like or scalar-like
        Argument to compute function.
    y : Array-like or scalar-like
        Argument to compute function.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

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
def bit_wise_not(*args, **kwargs):
    """
    Bit-wise negate the arguments element-wise.

    Null values return null.

    Parameters
    ----------
    x : Array-like or scalar-like
        Argument to compute function.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

bit_wise_or = _clone_signature(bit_wise_and)
"""
Bit-wise OR the arguments element-wise.

Null values return null.

Parameters
----------
x : Array-like or scalar-like
    Argument to compute function.
y : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
bit_wise_xor = _clone_signature(bit_wise_and)
"""
Bit-wise XOR the arguments element-wise.

Null values return null.

Parameters
----------
x : Array-like or scalar-like
    Argument to compute function.
y : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
shift_left = _clone_signature(bit_wise_and)
"""
Left shift `x` by `y`.

The shift operates as if on the two's complement representation of the number.
In other words, this is equivalent to multiplying `x` by 2 to the power `y`,
even if overflow occurs.
`x` is returned if `y` (the amount to shift by) is (1) negative or
(2) greater than or equal to the precision of `x`.
Use function "shift_left_checked" if you want an invalid shift amount
to return an error.

Parameters
----------
x : Array-like or scalar-like
    Argument to compute function.
y : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
shift_left_checked = _clone_signature(bit_wise_and)
"""
Left shift `x` by `y`.

The shift operates as if on the two's complement representation of the number.
In other words, this is equivalent to multiplying `x` by 2 to the power `y`,
even if overflow occurs.
An error is raised if `y` (the amount to shift by) is (1) negative or
(2) greater than or equal to the precision of `x`.
See "shift_left" for a variant that doesn't fail for an invalid shift amount.

Parameters
----------
x : Array-like or scalar-like
    Argument to compute function.
y : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
shift_right = _clone_signature(bit_wise_and)
"""
Right shift `x` by `y`.

This is equivalent to dividing `x` by 2 to the power `y`.
`x` is returned if `y` (the amount to shift by) is: (1) negative or
(2) greater than or equal to the precision of `x`.
Use function "shift_right_checked" if you want an invalid shift amount
to return an error.

Parameters
----------
x : Array-like or scalar-like
    Argument to compute function.
y : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
shift_right_checked = _clone_signature(bit_wise_and)
"""
Right shift `x` by `y`.

This is equivalent to dividing `x` by 2 to the power `y`.
An error is raised if `y` (the amount to shift by) is (1) negative or
(2) greater than or equal to the precision of `x`.
See "shift_right" for a variant that doesn't fail for an invalid shift amount

Parameters
----------
x : Array-like or scalar-like
    Argument to compute function.
y : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""

# ========================= 2.2 Rounding functions =========================
@overload
def ceil(x: _FloatScalarT, /, *, memory_pool: lib.MemoryPool | None = None) -> _FloatScalarT: ...
@overload
def ceil(x: _FloatArrayT, /, *, memory_pool: lib.MemoryPool | None = None) -> _FloatArrayT: ...
@overload
def ceil(x: Expression, /, *, memory_pool: lib.MemoryPool | None = None) -> Expression: ...
def ceil(*args, **kwargs):
    """
    Round up to the nearest integer.

    Compute the smallest integer value not less in magnitude than `x`.

    Parameters
    ----------
    x : Array-like or scalar-like
        Argument to compute function.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

floor = _clone_signature(ceil)
"""
Round down to the nearest integer.

Compute the largest integer value not greater in magnitude than `x`.

Parameters
----------
x : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""

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
def round(*args, **kwargs):
    """
    Round to a given precision.

    Options are used to control the number of digits and rounding mode.
    Default behavior is to round to the nearest integer and
    use half-to-even rule to break ties.

    Parameters
    ----------
    x : Array-like or scalar-like
        Argument to compute function.
    ndigits : int, default 0
        Number of fractional digits to round to.
    round_mode : str, default "half_to_even"
        Rounding and tie-breaking mode.
        Accepted values are "down", "up", "towards_zero", "towards_infinity",
        "half_down", "half_up", "half_towards_zero", "half_towards_infinity",
        "half_to_even", "half_to_odd".
    options : pyarrow.compute.RoundOptions, optional
        Alternative way of passing options.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

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
def round_to_multiple(*args, **kwargs):
    """
    Round to a given multiple.

    Options are used to control the rounding multiple and rounding mode.
    Default behavior is to round to the nearest integer and
    use half-to-even rule to break ties.

    Parameters
    ----------
    x : Array-like or scalar-like
        Argument to compute function.
    multiple : numeric scalar, default 1.0
        Multiple to round to. Should be a scalar of a type compatible
        with the argument to be rounded.
    round_mode : str, default "half_to_even"
        Rounding and tie-breaking mode.
        Accepted values are "down", "up", "towards_zero", "towards_infinity",
        "half_down", "half_up", "half_towards_zero", "half_towards_infinity",
        "half_to_even", "half_to_odd".
    options : pyarrow.compute.RoundToMultipleOptions, optional
        Alternative way of passing options.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

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
def round_binary(*args, **kwargs):
    """
    Round to the given precision.

    Options are used to control the rounding mode.
    Default behavior is to use the half-to-even rule to break ties.

    Parameters
    ----------
    x : Array-like or scalar-like
        Argument to compute function.
    s : Array-like or scalar-like
        Argument to compute function.
    round_mode : str, default "half_to_even"
        Rounding and tie-breaking mode.
        Accepted values are "down", "up", "towards_zero", "towards_infinity",
        "half_down", "half_up", "half_towards_zero", "half_towards_infinity",
        "half_to_even", "half_to_odd".
    options : pyarrow.compute.RoundBinaryOptions, optional
        Alternative way of passing options.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

trunc = _clone_signature(ceil)
"""
Compute the integral part.

Compute the nearest integer not greater in magnitude than `x`.

Parameters
----------
x : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""

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
def ln(*args, **kwargs):
    """
    Compute natural logarithm.

    Non-positive values return -inf or NaN. Null values return null.
    Use function "ln_checked" if you want non-positive values to raise an error.

    Parameters
    ----------
    x : Array-like or scalar-like
        Argument to compute function.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

ln_checked = _clone_signature(ln)
"""
Compute natural logarithm.

Non-positive values raise an error. Null values return null.
Use function "ln" if you want non-positive values to return -inf or NaN.

Parameters
----------
x : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
log10 = _clone_signature(ln)
"""
Compute base 10 logarithm.

Non-positive values return -inf or NaN. Null values return null.
Use function "log10_checked" if you want non-positive values
to raise an error.

Parameters
----------
x : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
log10_checked = _clone_signature(ln)
"""
Compute base 10 logarithm.

Non-positive values raise an error. Null values return null.
Use function "log10" if you want non-positive values
to return -inf or NaN.

Parameters
----------
x : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
log1p = _clone_signature(ln)
"""
Compute natural log of (1+x).

Values <= -1 return -inf or NaN. Null values return null.
This function may be more precise than log(1 + x) for x close to zero.
Use function "log1p_checked" if you want invalid values to raise an error.

Parameters
----------
x : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
log1p_checked = _clone_signature(ln)
"""
Compute natural log of (1+x).

Values <= -1 return -inf or NaN. Null values return null.
This function may be more precise than log(1 + x) for x close to zero.
Use function "log1p" if you want invalid values to return -inf or NaN.

Parameters
----------
x : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
log2 = _clone_signature(ln)
"""
Compute base 2 logarithm.

Non-positive values return -inf or NaN. Null values return null.
Use function "log2_checked" if you want non-positive values
to raise an error.

Parameters
----------
x : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
log2_checked = _clone_signature(ln)
"""
Compute base 2 logarithm.

Non-positive values raise an error. Null values return null.
Use function "log2" if you want non-positive values
to return -inf or NaN.

Parameters
----------
x : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""

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
    x: FloatScalar,
    b: FloatArray,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.NumericArray[lib.FloatScalar] | lib.NumericArray[lib.DoubleScalar]: ...
@overload
def logb(
    x: FloatArray,
    b: FloatScalar,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.NumericArray[lib.FloatScalar] | lib.NumericArray[lib.DoubleScalar]: ...
@overload
def logb(
    x: Expression | Any, b: Expression | Any, /, *, memory_pool: lib.MemoryPool | None = None
) -> Expression | Any: ...
def logb(*args, **kwargs):
    """
    Compute base `b` logarithm.

    Values <= 0 return -inf or NaN. Null values return null.
    Use function "logb_checked" if you want non-positive values to raise an error.

    Parameters
    ----------
    x : Array-like or scalar-like
        Argument to compute function.
    b : Array-like or scalar-like
        Argument to compute function.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

logb_checked = _clone_signature(logb)
"""
Compute base `b` logarithm.

Values <= 0 return -inf or NaN. Null values return null.
Use function "logb" if you want non-positive values to return -inf or NaN.

Parameters
----------
x : Array-like or scalar-like
    Argument to compute function.
b : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""

# ========================= 2.4 Trigonometric functions =========================
acos = _clone_signature(ln)
"""
Compute the inverse cosine.

NaN is returned for invalid input values;
to raise an error instead, see "acos_checked".

Parameters
----------
x : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
acos_checked = _clone_signature(ln)
"""
Compute the inverse cosine.

Invalid input values raise an error;
to return NaN instead, see "acos".

Parameters
----------
x : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
asin = _clone_signature(ln)
"""
Compute the inverse sine.

NaN is returned for invalid input values;
to raise an error instead, see "asin_checked".

Parameters
----------
x : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
asin_checked = _clone_signature(ln)
"""
Compute the inverse sine.

Invalid input values raise an error;
to return NaN instead, see "asin".

Parameters
----------
x : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
atan = _clone_signature(ln)
"""
Compute the inverse tangent of x.

The return value is in the range [-pi/2, pi/2];
for a full return range [-pi, pi], see "atan2".

Parameters
----------
x : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
cos = _clone_signature(ln)
"""
Compute the cosine.

NaN is returned for invalid input values;
to raise an error instead, see "cos_checked".

Parameters
----------
x : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
cos_checked = _clone_signature(ln)
"""
Compute the cosine.

Infinite values raise an error;
to return NaN instead, see "cos".

Parameters
----------
x : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
sin = _clone_signature(ln)
"""
Compute the sine.

NaN is returned for invalid input values;
to raise an error instead, see "sin_checked".

Parameters
----------
x : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
sin_checked = _clone_signature(ln)
"""
Compute the sine.

Invalid input values raise an error;
to return NaN instead, see "sin".

Parameters
----------
x : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
tan = _clone_signature(ln)
"""
Compute the tangent.

NaN is returned for invalid input values;
to raise an error instead, see "tan_checked".

Parameters
----------
x : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
tan_checked = _clone_signature(ln)
"""
Compute the tangent.

Infinite values raise an error;
to return NaN instead, see "tan".

Parameters
----------
x : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""

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
    y: FloatArray,
    x: FloatScalar,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.NumericArray[lib.FloatScalar] | lib.NumericArray[lib.DoubleScalar]: ...
@overload
def atan2(
    y: FloatScalar,
    x: FloatArray,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.NumericArray[lib.FloatScalar] | lib.NumericArray[lib.DoubleScalar]: ...
@overload
def atan2(
    y: Expression, x: Any, /, *, memory_pool: lib.MemoryPool | None = None
) -> Expression: ...
@overload
def atan2(
    y: Any, x: Expression, /, *, memory_pool: lib.MemoryPool | None = None
) -> Expression: ...
def atan2(*args, **kwargs):
    """
    Compute the inverse tangent of y/x.

    The return value is in the range [-pi, pi].

    Parameters
    ----------
    y : Array-like or scalar-like
        Argument to compute function.
    x : Array-like or scalar-like
        Argument to compute function.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

# ========================= 2.5 Comparisons functions =========================
@overload
def equal(
    x: lib.Scalar, y: lib.Scalar, /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.BooleanScalar: ...
@overload
def equal(
    x: lib.Scalar,
    y: lib.Array | lib.ChunkedArray,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.BooleanArray: ...
@overload
def equal(
    x: lib.Array | lib.ChunkedArray,
    y: lib.Scalar,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.BooleanArray: ...
@overload
def equal(
    x: lib.Array | lib.ChunkedArray,
    y: lib.Array | lib.ChunkedArray,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.BooleanArray: ...
@overload
def equal(
    x: Expression,
    y: Expression,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> Expression: ...
@overload
def equal(
    x: lib.Scalar,
    y: Expression,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> Expression: ...
@overload
def equal(
    x: Expression,
    y: lib.Scalar,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> Expression: ...
def equal(*args, **kwargs):
    """
    Compare values for equality (x == y).

    A null on either side emits a null comparison result.

    Parameters
    ----------
    x : Array-like or scalar-like
        Argument to compute function.
    y : Array-like or scalar-like
        Argument to compute function.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

greater = _clone_signature(equal)
"""
Compare values for ordered inequality (x > y).

A null on either side emits a null comparison result.

Parameters
----------
x : Array-like or scalar-like
    Argument to compute function.
y : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
greater_equal = _clone_signature(equal)
"""
Compare values for ordered inequality (x >= y).

A null on either side emits a null comparison result.

Parameters
----------
x : Array-like or scalar-like
    Argument to compute function.
y : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
less = _clone_signature(equal)
"""
Compare values for ordered inequality (x < y).

A null on either side emits a null comparison result.

Parameters
----------
x : Array-like or scalar-like
    Argument to compute function.
y : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
less_equal = _clone_signature(equal)
"""
Compare values for ordered inequality (x <= y).

A null on either side emits a null comparison result.

Parameters
----------
x : Array-like or scalar-like
    Argument to compute function.
y : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
not_equal = _clone_signature(equal)
"""
Compare values for inequality (x != y).

A null on either side emits a null comparison result.

Parameters
----------
x : Array-like or scalar-like
    Argument to compute function.
y : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""

@overload
def max_element_wise(
    *args: ScalarOrArray[_Scalar_CoT],
    skip_nulls: bool = True,
    options: ElementWiseAggregateOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> _Scalar_CoT: ...
@overload
def max_element_wise(
    *args: Expression,
    skip_nulls: bool = True,
    options: ElementWiseAggregateOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> Expression: ...
def max_element_wise(*args, **kwargs):
    """
    Find the element-wise maximum value.

    Nulls are ignored (by default) or propagated.
    NaN is preferred over null, but not over any valid value.

    Parameters
    ----------
    *args : Array-like or scalar-like
        Argument to compute function.
    skip_nulls : bool, default True
        Whether to skip (ignore) nulls in the input.
        If False, any null in the input forces the output to null.
    options : pyarrow.compute.ElementWiseAggregateOptions, optional
        Alternative way of passing options.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

min_element_wise = _clone_signature(max_element_wise)
"""
Find the element-wise minimum value.

Nulls are ignored (by default) or propagated.
NaN is preferred over null, but not over any valid value.

Parameters
----------
*args : Array-like or scalar-like
    Argument to compute function.
skip_nulls : bool, default True
    Whether to skip (ignore) nulls in the input.
    If False, any null in the input forces the output to null.
options : pyarrow.compute.ElementWiseAggregateOptions, optional
    Alternative way of passing options.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""

# ========================= 2.6 Logical functions =========================
@overload
def and_(
    x: lib.BooleanScalar, y: lib.BooleanScalar, /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.BooleanScalar: ...
@overload
def and_(
    x: BooleanArray,
    y: BooleanArray,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.BooleanArray: ...
@overload
def and_(
    x: Expression,
    y: Expression,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> Expression: ...
@overload
def and_(
    x: lib.BooleanScalar,
    y: BooleanArray,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.BooleanArray: ...
@overload
def and_(
    x: BooleanArray,
    y: lib.BooleanScalar,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.BooleanArray: ...
@overload
def and_(
    x: lib.BooleanScalar,
    y: Expression,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> Expression: ...
@overload
def and_(
    x: Expression,
    y: lib.BooleanScalar,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> Expression: ...
@overload
def and_(
    x: ScalarOrArray[lib.BooleanScalar],
    y: ScalarOrArray[lib.BooleanScalar],
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> ScalarOrArray[lib.BooleanScalar]: ...
def and_(*args, **kwargs):
    """
    Logical 'and' boolean values.

    When a null is encountered in either input, a null is output.
    For a different null behavior, see function "and_kleene".

    Parameters
    ----------
    x : Array-like or scalar-like
        Argument to compute function.
    y : Array-like or scalar-like
        Argument to compute function.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

and_kleene = _clone_signature(and_)
"""
Logical 'and' boolean values (Kleene logic).

This function behaves as follows with nulls:

- true and null = null
- null and true = null
- false and null = false
- null and false = false
- null and null = null

In other words, in this context a null value really means "unknown",
and an unknown value 'and' false is always false.
For a different null behavior, see function "and".

Parameters
----------
x : Array-like or scalar-like
    Argument to compute function.
y : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
and_not = _clone_signature(and_)
"""
Logical 'and not' boolean values.

When a null is encountered in either input, a null is output.
For a different null behavior, see function "and_not_kleene".

Parameters
----------
x : Array-like or scalar-like
    Argument to compute function.
y : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
and_not_kleene = _clone_signature(and_)
"""
Logical 'and not' boolean values (Kleene logic).

This function behaves as follows with nulls:

- true and not null = null
- null and not false = null
- false and not null = false
- null and not true = false
- null and not null = null

In other words, in this context a null value really means "unknown",
and an unknown value 'and not' true is always false, as is false
'and not' an unknown value.
For a different null behavior, see function "and_not".

Parameters
----------
x : Array-like or scalar-like
    Argument to compute function.
y : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
or_ = _clone_signature(and_)
"""
Logical 'or' boolean values.

When a null is encountered in either input, a null is output.
For a different null behavior, see function "or_kleene".

Parameters
----------
x : Array-like or scalar-like
    Argument to compute function.
y : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
or_kleene = _clone_signature(and_)
"""
Logical 'or' boolean values (Kleene logic).

This function behaves as follows with nulls:

- true or null = true
- null or true = true
- false or null = null
- null or false = null
- null or null = null

In other words, in this context a null value really means "unknown",
and an unknown value 'or' true is always true.
For a different null behavior, see function "or".

Parameters
----------
x : Array-like or scalar-like
    Argument to compute function.
y : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
xor = _clone_signature(and_)
"""
Logical 'xor' boolean values.

When a null is encountered in either input, a null is output.

Parameters
----------
x : Array-like or scalar-like
    Argument to compute function.
y : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""

@overload
def invert(
    x: lib.BooleanScalar, /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.BooleanScalar: ...
@overload
def invert(
    x: _BooleanArrayT,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> _BooleanArrayT: ...
@overload
def invert(
    x: Expression,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> Expression: ...
def invert(*args, **kwargs):
    """
    Invert boolean values.

    Parameters
    ----------
    values : Array-like or scalar-like
        Argument to compute function.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

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
def ascii_is_alnum(*args, **kwargs):
    """
    Classify strings as ASCII alphanumeric.

    For each string in `strings`, emit true iff the string is non-empty
    and consists only of alphanumeric ASCII characters.  Null strings emit null.

    Parameters
    ----------
    strings : Array-like or scalar-like
        Argument to compute function.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

ascii_is_alpha = _clone_signature(ascii_is_alnum)
"""
Classify strings as ASCII alphabetic.

For each string in `strings`, emit true iff the string is non-empty
and consists only of alphabetic ASCII characters.  Null strings emit null.

Parameters
----------
strings : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
ascii_is_decimal = _clone_signature(ascii_is_alnum)
"""
Classify strings as ASCII decimal.

For each string in `strings`, emit true iff the string is non-empty
and consists only of decimal ASCII characters.  Null strings emit null.

Parameters
----------
strings : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
ascii_is_lower = _clone_signature(ascii_is_alnum)
"""
Classify strings as ASCII lowercase.

For each string in `strings`, emit true iff the string is non-empty
and consists only of lowercase ASCII characters.  Null strings emit null.

Parameters
----------
strings : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
ascii_is_printable = _clone_signature(ascii_is_alnum)
"""
Classify strings as ASCII printable.

For each string in `strings`, emit true iff the string is non-empty
and consists only of printable ASCII characters.  Null strings emit null.

Parameters
----------
strings : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
ascii_is_space = _clone_signature(ascii_is_alnum)
"""
Classify strings as ASCII whitespace.

For each string in `strings`, emit true iff the string is non-empty
and consists only of whitespace ASCII characters.  Null strings emit null.

Parameters
----------
strings : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
ascii_is_upper = _clone_signature(ascii_is_alnum)
"""
Classify strings as ASCII uppercase.

For each string in `strings`, emit true iff the string is non-empty
and consists only of uppercase ASCII characters.  Null strings emit null.

Parameters
----------
strings : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
utf8_is_alnum = _clone_signature(ascii_is_alnum)
"""
Classify strings as alphanumeric.

For each string in `strings`, emit true iff the string is non-empty
and consists only of alphanumeric Unicode characters.  Null strings emit null.

Parameters
----------
strings : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
utf8_is_alpha = _clone_signature(ascii_is_alnum)
"""
Classify strings as alphabetic.

For each string in `strings`, emit true iff the string is non-empty
and consists only of alphabetic Unicode characters.  Null strings emit null.

Parameters
----------
strings : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
utf8_is_decimal = _clone_signature(ascii_is_alnum)
"""
Classify strings as decimal.

For each string in `strings`, emit true iff the string is non-empty
and consists only of decimal Unicode characters.  Null strings emit null.

Parameters
----------
strings : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
utf8_is_digit = _clone_signature(ascii_is_alnum)
"""
Classify strings as digits.

For each string in `strings`, emit true iff the string is non-empty
and consists only of Unicode digits.  Null strings emit null.

Parameters
----------
strings : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
utf8_is_lower = _clone_signature(ascii_is_alnum)
"""
Classify strings as lowercase.

For each string in `strings`, emit true iff the string is non-empty
and consists only of lowercase Unicode characters.  Null strings emit null.

Parameters
----------
strings : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
utf8_is_numeric = _clone_signature(ascii_is_alnum)
"""
Classify strings as numeric.

For each string in `strings`, emit true iff the string is non-empty
and consists only of numeric Unicode characters.  Null strings emit null.

Parameters
----------
strings : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
utf8_is_printable = _clone_signature(ascii_is_alnum)
"""
Classify strings as printable.

For each string in `strings`, emit true iff the string is non-empty
and consists only of printable Unicode characters.  Null strings emit null.

Parameters
----------
strings : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
utf8_is_space = _clone_signature(ascii_is_alnum)
"""
Classify strings as whitespace.

For each string in `strings`, emit true iff the string is non-empty
and consists only of whitespace Unicode characters.  Null strings emit null.

Parameters
----------
strings : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
utf8_is_upper = _clone_signature(ascii_is_alnum)
"""
Classify strings as uppercase.

For each string in `strings`, emit true iff the string is non-empty
and consists only of uppercase Unicode characters.  Null strings emit null.

Parameters
----------
strings : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
ascii_is_title = _clone_signature(ascii_is_alnum)
"""
Classify strings as ASCII titlecase.

For each string in `strings`, emit true iff the string is title-cased,
i.e. it has at least one cased character, each uppercase character
follows an uncased character, and each lowercase character follows
an uppercase character.

Parameters
----------
strings : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
utf8_is_title = _clone_signature(ascii_is_alnum)
"""
Classify strings as titlecase.

For each string in `strings`, emit true iff the string is title-cased,
i.e. it has at least one cased character, each uppercase character
follows an uncased character, and each lowercase character follows
an uppercase character.

Parameters
----------
strings : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
string_is_ascii = _clone_signature(ascii_is_alnum)
"""
Classify strings as ASCII.

For each string in `strings`, emit true iff the string consists only
of ASCII characters.  Null strings emit null.

Parameters
----------
strings : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""

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
def ascii_capitalize(*args, **kwargs):
    """
    Capitalize the first character of ASCII input.

    For each string in `strings`, return a capitalized version.

    This function assumes the input is fully ASCII.  If it may contain
    non-ASCII characters, use "utf8_capitalize" instead.

    Parameters
    ----------
    strings : Array-like or scalar-like
        Argument to compute function.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

ascii_lower = _clone_signature(ascii_capitalize)
"""
Transform ASCII input to lowercase.

For each string in `strings`, return a lowercase version.

This function assumes the input is fully ASCII.  If it may contain
non-ASCII characters, use "utf8_lower" instead.

Parameters
----------
strings : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
ascii_reverse = _clone_signature(ascii_capitalize)
"""
Reverse ASCII input.

For each ASCII string in `strings`, return a reversed version.

This function assumes the input is fully ASCII.  If it may contain
non-ASCII characters, use "utf8_reverse" instead.

Parameters
----------
strings : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
ascii_swapcase = _clone_signature(ascii_capitalize)
"""
Transform ASCII input by inverting casing.

For each string in `strings`, return a string with opposite casing.

This function assumes the input is fully ASCII.  If it may contain
non-ASCII characters, use "utf8_swapcase" instead.

Parameters
----------
strings : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
ascii_title = _clone_signature(ascii_capitalize)
"""
Titlecase each word of ASCII input.

For each string in `strings`, return a titlecased version.
Each word in the output will start with an uppercase character and its
remaining characters will be lowercase.

This function assumes the input is fully ASCII.  If it may contain
non-ASCII characters, use "utf8_title" instead.

Parameters
----------
strings : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
ascii_upper = _clone_signature(ascii_capitalize)
"""
Transform ASCII input to uppercase.

For each string in `strings`, return an uppercase version.

This function assumes the input is fully ASCII.  It it may contain
non-ASCII characters, use "utf8_upper" instead.

Parameters
----------
strings : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""

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
def binary_length(*args, **kwargs):
    """
    Compute string lengths.

    For each string in `strings`, emit its length of bytes.
    Null values emit null.

    Parameters
    ----------
    strings : Array-like or scalar-like
        Argument to compute function.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

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
def binary_repeat(*args, **kwargs):
    """
    Repeat a binary string.

    For each binary string in `strings`, return a replicated version.

    Parameters
    ----------
    strings : Array-like or scalar-like
        Argument to compute function.
    num_repeats : Array-like or scalar-like
        Argument to compute function.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

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
def binary_replace_slice(*args, **kwargs):
    """
    Replace a slice of a binary string.

    For each string in `strings`, replace a slice of the string defined by `start`
    and `stop` indices with the given `replacement`. `start` is inclusive
    and `stop` is exclusive, and both are measured in bytes.
    Null values emit null.

    Parameters
    ----------
    strings : Array-like or scalar-like
        Argument to compute function.
    start : int
        Index to start slicing at (inclusive).
    stop : int
        Index to stop slicing at (exclusive).
    replacement : str
        What to replace the slice with.
    options : pyarrow.compute.ReplaceSliceOptions, optional
        Alternative way of passing options.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

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
def binary_reverse(*args, **kwargs):
    """
    Reverse binary input.

    For each binary string in `strings`, return a reversed version.

    This function reverses the binary data at a byte-level.

    Parameters
    ----------
    strings : Array-like or scalar-like
        Argument to compute function.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

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
def replace_substring(*args, **kwargs):
    """
    Replace matching non-overlapping substrings with replacement.

    For each string in `strings`, replace non-overlapping substrings that match
    the given literal `pattern` with the given `replacement`.
    If `max_replacements` is given and not equal to -1, it limits the
    maximum amount replacements per input, counted from the left.
    Null values emit null.

    Parameters
    ----------
    strings : Array-like or scalar-like
        Argument to compute function.
    pattern : str
        Substring pattern to look for inside input values.
    replacement : str
        What to replace the pattern with.
    max_replacements : int or None, default None
        The maximum number of strings to replace in each
        input value (unlimited if None).
    options : pyarrow.compute.ReplaceSubstringOptions, optional
        Alternative way of passing options.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

replace_substring_regex = _clone_signature(replace_substring)
"""
Replace matching non-overlapping substrings with replacement.

For each string in `strings`, replace non-overlapping substrings that match
the given regular expression `pattern` with the given `replacement`.
If `max_replacements` is given and not equal to -1, it limits the
maximum amount replacements per input, counted from the left.
Null values emit null.

Parameters
----------
strings : Array-like or scalar-like
    Argument to compute function.
pattern : str
    Substring pattern to look for inside input values.
replacement : str
    What to replace the pattern with.
max_replacements : int or None, default None
    The maximum number of strings to replace in each
    input value (unlimited if None).
options : pyarrow.compute.ReplaceSubstringOptions, optional
    Alternative way of passing options.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""

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
def utf8_capitalize(*args, **kwargs):
    """
    Capitalize the first character of input.

    For each string in `strings`, return a capitalized version,
    with the first character uppercased and the others lowercased.

    Parameters
    ----------
    strings : Array-like or scalar-like
        Argument to compute function.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

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
def utf8_length(*args, **kwargs):
    """
    Compute UTF8 string lengths.

    For each string in `strings`, emit its length in UTF8 characters.
    Null values emit null.

    Parameters
    ----------
    strings : Array-like or scalar-like
        Argument to compute function.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

utf8_lower = _clone_signature(utf8_capitalize)
"""
Transform input to lowercase.

For each string in `strings`, return a lowercase version.

Parameters
----------
strings : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""

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
def utf8_replace_slice(*args, **kwargs):
    """
    Replace a slice of a string.

    For each string in `strings`, replace a slice of the string defined by `start`
    and `stop` indices with the given `replacement`. `start` is inclusive
    and `stop` is exclusive, and both are measured in UTF8 characters.
    Null values emit null.

    Parameters
    ----------
    strings : Array-like or scalar-like
        Argument to compute function.
    start : int
        Index to start slicing at (inclusive).
    stop : int
        Index to stop slicing at (exclusive).
    replacement : str
        What to replace the slice with.
    options : pyarrow.compute.ReplaceSliceOptions, optional
        Alternative way of passing options.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

utf8_reverse = _clone_signature(utf8_capitalize)
"""
Reverse input.

For each string in `strings`, return a reversed version.

This function operates on Unicode codepoints, not grapheme
clusters. Hence, it will not correctly reverse grapheme clusters
composed of multiple codepoints.

Parameters
----------
strings : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
utf8_swapcase = _clone_signature(utf8_capitalize)
"""
Transform input lowercase characters to uppercase and uppercase characters to lowercase.

For each string in `strings`, return an opposite case version.

Parameters
----------
strings : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
utf8_title = _clone_signature(utf8_capitalize)
"""
Titlecase each word of input.

For each string in `strings`, return a titlecased version.
Each word in the output will start with an uppercase character and its
remaining characters will be lowercase.

Parameters
----------
strings : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
utf8_upper = _clone_signature(utf8_capitalize)
"""
Transform input to uppercase.

For each string in `strings`, return an uppercase version.

Parameters
----------
strings : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory poo
"""

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
def ascii_center(*args, **kwargs):
    """
    Center strings by padding with a given character.

    For each string in `strings`, emit a centered string by padding both sides
    with the given ASCII character.
    Null values emit null.

    Parameters
    ----------
    strings : Array-like or scalar-like
        Argument to compute function.
    width : int
        Desired string length.
    padding : str, default " "
        What to pad the string with. Should be one byte or codepoint.
    lean_left_on_odd_padding : bool, default True
        What to do if there is an odd number of padding characters (in case
        of centered padding). Defaults to aligning on the left (i.e. adding
        the extra padding character on the right).
    options : pyarrow.compute.PadOptions, optional
        Alternative way of passing options.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

ascii_lpad = _clone_signature(ascii_center)
"""
Right-align strings by padding with a given character.

For each string in `strings`, emit a right-aligned string by prepending
the given ASCII character.
Null values emit null.

Parameters
----------
strings : Array-like or scalar-like
    Argument to compute function.
width : int
    Desired string length.
padding : str, default " "
    What to pad the string with. Should be one byte or codepoint.
lean_left_on_odd_padding : bool, default True
    What to do if there is an odd number of padding characters (in case
    of centered padding). Defaults to aligning on the left (i.e. adding
    the extra padding character on the right).
options : pyarrow.compute.PadOptions, optional
    Alternative way of passing options.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
ascii_rpad = _clone_signature(ascii_center)
"""
Left-align strings by padding with a given character.

For each string in `strings`, emit a left-aligned string by appending
the given ASCII character.
Null values emit null.

Parameters
----------
strings : Array-like or scalar-like
    Argument to compute function.
width : int
    Desired string length.
padding : str, default " "
    What to pad the string with. Should be one byte or codepoint.
lean_left_on_odd_padding : bool, default True
    What to do if there is an odd number of padding characters (in case
    of centered padding). Defaults to aligning on the left (i.e. adding
    the extra padding character on the right).
options : pyarrow.compute.PadOptions, optional
    Alternative way of passing options.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
utf8_center = _clone_signature(ascii_center)
"""
Center strings by padding with a given character.

For each string in `strings`, emit a centered string by padding both sides
with the given UTF8 codeunit.
Null values emit null.

Parameters
----------
strings : Array-like or scalar-like
    Argument to compute function.
width : int
    Desired string length.
padding : str, default " "
    What to pad the string with. Should be one byte or codepoint.
lean_left_on_odd_padding : bool, default True
    What to do if there is an odd number of padding characters (in case
    of centered padding). Defaults to aligning on the left (i.e. adding
    the extra padding character on the right).
options : pyarrow.compute.PadOptions, optional
    Alternative way of passing options.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
utf8_lpad = _clone_signature(ascii_center)
"""
Right-align strings by padding with a given character.

For each string in `strings`, emit a right-aligned string by prepending
the given UTF8 codeunit.
Null values emit null.

Parameters
----------
strings : Array-like or scalar-like
    Argument to compute function.
width : int
    Desired string length.
padding : str, default " "
    What to pad the string with. Should be one byte or codepoint.
lean_left_on_odd_padding : bool, default True
    What to do if there is an odd number of padding characters (in case
    of centered padding). Defaults to aligning on the left (i.e. adding
    the extra padding character on the right).
options : pyarrow.compute.PadOptions, optional
    Alternative way of passing options.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
utf8_rpad = _clone_signature(ascii_center)
"""
Left-align strings by padding with a given character.

For each string in `strings`, emit a left-aligned string by appending
the given UTF8 codeunit.
Null values emit null.

Parameters
----------
strings : Array-like or scalar-like
    Argument to compute function.
width : int
    Desired string length.
padding : str, default " "
    What to pad the string with. Should be one byte or codepoint.
lean_left_on_odd_padding : bool, default True
    What to do if there is an odd number of padding characters (in case
    of centered padding). Defaults to aligning on the left (i.e. adding
    the extra padding character on the right).
options : pyarrow.compute.PadOptions, optional
    Alternative way of passing options.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""

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
def ascii_ltrim(*args, **kwargs):
    """
    Trim leading characters.

    For each string in `strings`, remove any leading characters
    from the `characters` option (as given in TrimOptions).
    Null values emit null.
    Both the `strings` and the `characters` are interpreted as
    ASCII; to trim non-ASCII characters, use `utf8_ltrim`.

    Parameters
    ----------
    strings : Array-like or scalar-like
        Argument to compute function.
    characters : str
        Individual characters to be trimmed from the string.
    options : pyarrow.compute.TrimOptions, optional
        Alternative way of passing options.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

ascii_rtrim = _clone_signature(ascii_ltrim)
"""
Trim trailing characters.

For each string in `strings`, remove any trailing characters
from the `characters` option (as given in TrimOptions).
Null values emit null.
Both the `strings` and the `characters` are interpreted as
ASCII; to trim non-ASCII characters, use `utf8_rtrim`.

Parameters
----------
strings : Array-like or scalar-like
    Argument to compute function.
characters : str
    Individual characters to be trimmed from the string.
options : pyarrow.compute.TrimOptions, optional
    Alternative way of passing options.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
ascii_trim = _clone_signature(ascii_ltrim)
"""
Trim leading and trailing characters.

For each string in `strings`, remove any leading or trailing characters
from the `characters` option (as given in TrimOptions).
Null values emit null.
Both the `strings` and the `characters` are interpreted as
ASCII; to trim non-ASCII characters, use `utf8_trim`.

Parameters
----------
strings : Array-like or scalar-like
    Argument to compute function.
characters : str
    Individual characters to be trimmed from the string.
options : pyarrow.compute.TrimOptions, optional
    Alternative way of passing options.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
utf8_ltrim = _clone_signature(ascii_ltrim)
"""
Trim leading characters.

For each string in `strings`, remove any leading characters
from the `characters` option (as given in TrimOptions).
Null values emit null.

Parameters
----------
strings : Array-like or scalar-like
    Argument to compute function.
characters : str
    Individual characters to be trimmed from the string.
options : pyarrow.compute.TrimOptions, optional
    Alternative way of passing options.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
utf8_rtrim = _clone_signature(ascii_ltrim)
"""
Trim trailing characters.

For each string in `strings`, remove any trailing characters
from the `characters` option (as given in TrimOptions).
Null values emit null.

Parameters
----------
strings : Array-like or scalar-like
    Argument to compute function.
characters : str
    Individual characters to be trimmed from the string.
options : pyarrow.compute.TrimOptions, optional
    Alternative way of passing options.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
utf8_trim = _clone_signature(ascii_ltrim)
"""
Trim leading and trailing characters.

For each string in `strings`, remove any leading or trailing characters
from the `characters` option (as given in TrimOptions).
Null values emit null.

Parameters
----------
strings : Array-like or scalar-like
    Argument to compute function.
characters : str
    Individual characters to be trimmed from the string.
options : pyarrow.compute.TrimOptions, optional
    Alternative way of passing options.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""

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
def ascii_ltrim_whitespace(*args, **kwargs):
    """
    Trim leading ASCII whitespace characters.

    For each string in `strings`, emit a string with leading ASCII whitespace
    characters removed.  Use `utf8_ltrim_whitespace` to trim leading Unicode
    whitespace characters. Null values emit null.

    Parameters
    ----------
    strings : Array-like or scalar-like
        Argument to compute function.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

ascii_rtrim_whitespace = _clone_signature(ascii_ltrim_whitespace)
"""
Trim trailing ASCII whitespace characters.

For each string in `strings`, emit a string with trailing ASCII whitespace
characters removed. Use `utf8_rtrim_whitespace` to trim trailing Unicode
whitespace characters. Null values emit null.

Parameters
----------
strings : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
ascii_trim_whitespace = _clone_signature(ascii_ltrim_whitespace)
"""
Trim leading and trailing ASCII whitespace characters.

For each string in `strings`, emit a string with leading and trailing ASCII
whitespace characters removed. Use `utf8_trim_whitespace` to trim Unicode
whitespace characters. Null values emit null.

Parameters
----------
strings : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
utf8_ltrim_whitespace = _clone_signature(ascii_ltrim_whitespace)
"""
Trim leading whitespace characters.

For each string in `strings`, emit a string with leading whitespace
characters removed, where whitespace characters are defined by the Unicode
standard.  Null values emit null.

Parameters
----------
strings : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
utf8_rtrim_whitespace = _clone_signature(ascii_ltrim_whitespace)
"""
Trim trailing whitespace characters.

For each string in `strings`, emit a string with trailing whitespace
characters removed, where whitespace characters are defined by the Unicode
standard.  Null values emit null.

Parameters
----------
strings : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
utf8_trim_whitespace = _clone_signature(ascii_ltrim_whitespace)
"""
Trim leading and trailing whitespace characters.

For each string in `strings`, emit a string with leading and trailing
whitespace characters removed, where whitespace characters are defined
by the Unicode standard.  Null values emit null.

Parameters
----------
strings : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""

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
def ascii_split_whitespace(*args, **kwargs):
    """
    Split string according to any ASCII whitespace.

    Split each string according any non-zero length sequence of ASCII
    whitespace characters.  The output for each string input is a list
    of strings.

    The maximum number of splits and direction of splitting
    (forward, reverse) can optionally be defined in SplitOptions.

    Parameters
    ----------
    strings : Array-like or scalar-like
        Argument to compute function.
    max_splits : int or None, default None
        Maximum number of splits for each input value (unlimited if None).
    reverse : bool, default False
        Whether to start splitting from the end of each input value.
        This only has an effect if `max_splits` is not None.
    options : pyarrow.compute.SplitOptions, optional
        Alternative way of passing options.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

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
def split_pattern(*args, **kwargs):
    """
    Split string according to separator.

    Split each string according to the exact `pattern` defined in
    SplitPatternOptions.  The output for each string input is a list
    of strings.

    The maximum number of splits and direction of splitting
    (forward, reverse) can optionally be defined in SplitPatternOptions.

    Parameters
    ----------
    strings : Array-like or scalar-like
        Argument to compute function.
    pattern : str
        String pattern to split on.
    max_splits : int or None, default None
        Maximum number of splits for each input value (unlimited if None).
    reverse : bool, default False
        Whether to start splitting from the end of each input value.
        This only has an effect if `max_splits` is not None.
    options : pyarrow.compute.SplitPatternOptions, optional
        Alternative way of passing options.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

split_pattern_regex = _clone_signature(split_pattern)
"""
Split string according to regex pattern.

Split each string according to the regex `pattern` defined in
SplitPatternOptions.  The output for each string input is a list
of strings.

The maximum number of splits and direction of splitting
(forward, reverse) can optionally be defined in SplitPatternOptions.

Parameters
----------
strings : Array-like or scalar-like
    Argument to compute function.
pattern : str
    String pattern to split on.
max_splits : int or None, default None
    Maximum number of splits for each input value (unlimited if None).
reverse : bool, default False
    Whether to start splitting from the end of each input value.
    This only has an effect if `max_splits` is not None.
options : pyarrow.compute.SplitPatternOptions, optional
    Alternative way of passing options.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
utf8_split_whitespace = _clone_signature(ascii_split_whitespace)
"""
Split string according to any Unicode whitespace.

Split each string according any non-zero length sequence of Unicode
whitespace characters.  The output for each string input is a list
of strings.

The maximum number of splits and direction of splitting
(forward, reverse) can optionally be defined in SplitOptions.

Parameters
----------
strings : Array-like or scalar-like
    Argument to compute function.
max_splits : int or None, default None
    Maximum number of splits for each input value (unlimited if None).
reverse : bool, default False
    Whether to start splitting from the end of each input value.
    This only has an effect if `max_splits` is not None.
options : pyarrow.compute.SplitOptions, optional
    Alternative way of passing options.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""

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
def extract_regex(*args, **kwargs):
    """
    Extract substrings captured by a regex pattern.

    For each string in `strings`, match the regular expression and, if
    successful, emit a struct with field names and values coming from the
    regular expression's named capture groups. If the input is null or the
    regular expression fails matching, a null output value is emitted.

    Regular expression matching is done using the Google RE2 library.

    Parameters
    ----------
    strings : Array-like or scalar-like
        Argument to compute function.
    pattern : str
        Regular expression with named capture fields.
    options : pyarrow.compute.ExtractRegexOptions, optional
        Alternative way of passing options.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

# ========================= 2.16 String join =========================
def binary_join(
    strings, separator, /, *, memory_pool: lib.MemoryPool | None = None
) -> StringScalar | StringArray:
    """
    Join a list of strings together with a separator.

    Concatenate the strings in `list`. The `separator` is inserted
    between each given string.
    Any null input and any null `list` element emits a null output.

    Parameters
    ----------
    strings : Array-like or scalar-like
        Argument to compute function.
    separator : Array-like or scalar-like
        Argument to compute function.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

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
def binary_join_element_wise(*args, **kwargs):
    """
    Join string arguments together, with the last argument as separator.

    Concatenate the `strings` except for the last one. The last argument
    in `strings` is inserted between each given string.
    Any null separator element emits a null output. Null elements either
    emit a null (the default), are skipped, or replaced with a given string.

    Parameters
    ----------
    *strings : Array-like or scalar-like
        Argument to compute function.
    null_handling : str, default "emit_null"
        How to handle null values in the inputs.
        Accepted values are "emit_null", "skip", "replace".
    null_replacement : str, default ""
        Replacement string to emit for null inputs if `null_handling`
        is "replace".
    options : pyarrow.compute.JoinOptions, optional
        Alternative way of passing options.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

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
def binary_slice(*args, **kwargs):
    """
    Slice binary string.

    For each binary string in `strings`, emit the substring defined by
    (`start`, `stop`, `step`) as given by `SliceOptions` where `start` is
    inclusive and `stop` is exclusive. All three values are measured in
    bytes.
    If `step` is negative, the string will be advanced in reversed order.
    An error is raised if `step` is zero.
    Null inputs emit null.

    Parameters
    ----------
    strings : Array-like or scalar-like
        Argument to compute function.
    start : int
        Index to start slicing at (inclusive).
    stop : int or None, default None
        If given, index to stop slicing at (exclusive).
        If not given, slicing will stop at the end.
    step : int, default 1
        Slice step.
    options : pyarrow.compute.SliceOptions, optional
        Alternative way of passing options.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

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
def utf8_slice_codeunits(*args, **kwargs):
    """
    Slice string.

    For each string in `strings`, emit the substring defined by
    (`start`, `stop`, `step`) as given by `SliceOptions` where `start` is
    inclusive and `stop` is exclusive. All three values are measured in
    UTF8 codeunits.
    If `step` is negative, the string will be advanced in reversed order.
    An error is raised if `step` is zero.
    Null inputs emit null.

    Parameters
    ----------
    strings : Array-like or scalar-like
        Argument to compute function.
    start : int
        Index to start slicing at (inclusive).
    stop : int or None, default None
        If given, index to stop slicing at (exclusive).
        If not given, slicing will stop at the end.
    step : int, default 1
        Slice step.
    options : pyarrow.compute.SliceOptions, optional
        Alternative way of passing options.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

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
def count_substring(*args, **kwargs):
    """
    Count occurrences of substring.

    For each string in `strings`, emit the number of occurrences of the given
    literal pattern.
    Null inputs emit null. The pattern must be given in MatchSubstringOptions.

    Parameters
    ----------
    strings : Array-like or scalar-like
        Argument to compute function.
    pattern : str
        Substring pattern to look for inside input values.
    ignore_case : bool, default False
        Whether to perform a case-insensitive match.
    options : pyarrow.compute.MatchSubstringOptions, optional
        Alternative way of passing options.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

count_substring_regex = _clone_signature(count_substring)
"""
Count occurrences of substring.

For each string in `strings`, emit the number of occurrences of the given
regular expression pattern.
Null inputs emit null. The pattern must be given in MatchSubstringOptions.

Parameters
----------
strings : Array-like or scalar-like
    Argument to compute function.
pattern : str
    Substring pattern to look for inside input values.
ignore_case : bool, default False
    Whether to perform a case-insensitive match.
options : pyarrow.compute.MatchSubstringOptions, optional
    Alternative way of passing options.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""

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
def ends_with(*args, **kwargs):
    """
    Check if strings end with a literal pattern.

    For each string in `strings`, emit true iff it ends with a given pattern.
    The pattern must be given in MatchSubstringOptions.
    If ignore_case is set, only simple case folding is performed.

    Null inputs emit null.

    Parameters
    ----------
    strings : Array-like or scalar-like
        Argument to compute function.
    pattern : str
        Substring pattern to look for inside input values.
    ignore_case : bool, default False
        Whether to perform a case-insensitive match.
    options : pyarrow.compute.MatchSubstringOptions, optional
        Alternative way of passing options.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

find_substring = _clone_signature(count_substring)
"""
Find first occurrence of substring.

For each string in `strings`, emit the index in bytes of the first occurrence
of the given literal pattern, or -1 if not found.
Null inputs emit null. The pattern must be given in MatchSubstringOptions.

Parameters
----------
strings : Array-like or scalar-like
    Argument to compute function.
pattern : str
    Substring pattern to look for inside input values.
ignore_case : bool, default False
    Whether to perform a case-insensitive match.
options : pyarrow.compute.MatchSubstringOptions, optional
    Alternative way of passing options.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
find_substring_regex = _clone_signature(count_substring)
"""
Find location of first match of regex pattern.

For each string in `strings`, emit the index in bytes of the first occurrence
of the given literal pattern, or -1 if not found.
Null inputs emit null. The pattern must be given in MatchSubstringOptions.

Parameters
----------
strings : Array-like or scalar-like
    Argument to compute function.
pattern : str
    Substring pattern to look for inside input values.
ignore_case : bool, default False
    Whether to perform a case-insensitive match.
options : pyarrow.compute.MatchSubstringOptions, optional
    Alternative way of passing options.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""

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
def index_in(*args, **kwargs):
    """
    Return index of each element in a set of values.

    For each element in `values`, return its index in a given set of
    values, or null if it is not found there.
    The set of values to look for must be given in SetLookupOptions.
    By default, nulls are matched against the value set, this can be
    changed in SetLookupOptions.

    Parameters
    ----------
    values : Array-like or scalar-like
        Argument to compute function.
    value_set : Array
        Set of values to look for in the input.
    skip_nulls : bool, default False
        If False, nulls in the input are matched in the value_set just
        like regular values.
        If True, nulls in the input always fail matching.
    options : pyarrow.compute.SetLookupOptions, optional
        Alternative way of passing options.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

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
def is_in(*args, **kwargs):
    """
    Find each element in a set of values.

    For each element in `values`, return true if it is found in a given
    set of values, false otherwise.
    The set of values to look for must be given in SetLookupOptions.
    By default, nulls are matched against the value set, this can be
    changed in SetLookupOptions.

    Parameters
    ----------
    values : Array-like or scalar-like
        Argument to compute function.
    value_set : Array
        Set of values to look for in the input.
    skip_nulls : bool, default False
        If False, nulls in the input are matched in the value_set just
        like regular values.
        If True, nulls in the input always fail matching.
    options : pyarrow.compute.SetLookupOptions, optional
        Alternative way of passing options.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

match_like = _clone_signature(ends_with)
"""
Match strings against SQL-style LIKE pattern.

For each string in `strings`, emit true iff it matches a given pattern
at any position. '%' will match any number of characters, '_' will
match exactly one character, and any other character matches itself.
To match a literal '%', '_', or '\', precede the character with a backslash.
Null inputs emit null.  The pattern must be given in MatchSubstringOptions.

Parameters
----------
strings : Array-like or scalar-like
    Argument to compute function.
pattern : str
    Substring pattern to look for inside input values.
ignore_case : bool, default False
    Whether to perform a case-insensitive match.
options : pyarrow.compute.MatchSubstringOptions, optional
    Alternative way of passing options.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
match_substring = _clone_signature(ends_with)
"""
Match strings against literal pattern.

For each string in `strings`, emit true iff it contains a given pattern.
Null inputs emit null.
The pattern must be given in MatchSubstringOptions.
If ignore_case is set, only simple case folding is performed.

Parameters
----------
strings : Array-like or scalar-like
    Argument to compute function.
pattern : str
    Substring pattern to look for inside input values.
ignore_case : bool, default False
    Whether to perform a case-insensitive match.
options : pyarrow.compute.MatchSubstringOptions, optional
    Alternative way of passing options.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
match_substring_regex = _clone_signature(ends_with)
"""
Match strings against regex pattern.

For each string in `strings`, emit true iff it matches a given pattern
at any position. The pattern must be given in MatchSubstringOptions.
If ignore_case is set, only simple case folding is performed.

Null inputs emit null.

Parameters
----------
strings : Array-like or scalar-like
    Argument to compute function.
pattern : str
    Substring pattern to look for inside input values.
ignore_case : bool, default False
    Whether to perform a case-insensitive match.
options : pyarrow.compute.MatchSubstringOptions, optional
    Alternative way of passing options.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
starts_with = _clone_signature(ends_with)
"""
Check if strings start with a literal pattern.

For each string in `strings`, emit true iff it starts with a given pattern.
The pattern must be given in MatchSubstringOptions.
If ignore_case is set, only simple case folding is performed.

Null inputs emit null.

Parameters
----------
strings : Array-like or scalar-like
    Argument to compute function.
pattern : str
    Substring pattern to look for inside input values.
ignore_case : bool, default False
    Whether to perform a case-insensitive match.
options : pyarrow.compute.MatchSubstringOptions, optional
    Alternative way of passing options.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""

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
def is_finite(*args, **kwargs):
    """
    Return true if value is finite.

    For each input value, emit true iff the value is finite
    (i.e. neither NaN, inf, nor -inf).

    Parameters
    ----------
    values : Array-like or scalar-like
        Argument to compute function.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

is_inf = _clone_signature(is_finite)
"""
Return true if infinity.

For each input value, emit true iff the value is infinite (inf or -inf).

Parameters
----------
values : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
is_nan = _clone_signature(is_finite)
"""
Return true if NaN.

For each input value, emit true iff the value is NaN.

Parameters
----------
values : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""

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
def is_null(*args, **kwargs):
    """
    Return true if null (and optionally NaN).

    For each input value, emit true iff the value is null.
    True may also be emitted for NaN values by setting the `nan_is_null` flag.

    Parameters
    ----------
    values : Array-like or scalar-like
        Argument to compute function.
    nan_is_null : bool, default False
        Whether floating-point NaN values are considered null.
    options : pyarrow.compute.NullOptions, optional
        Alternative way of passing options.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

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
def is_valid(*args, **kwargs):
    """
    Return true if non-null.

    For each input value, emit true iff the value is valid (i.e. non-null).

    Parameters
    ----------
    values : Array-like or scalar-like
        Argument to compute function.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

true_unless_null = _clone_signature(is_valid)
"""
Return true if non-null, else return null.

For each input value, emit true iff the value
is valid (non-null), otherwise emit null.

Parameters
----------
values : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""

# ========================= 2.20 Selecting / multiplexing =========================
def case_when(cond, /, *cases, memory_pool: lib.MemoryPool | None = None):
    """
    Choose values based on multiple conditions.

    `cond` must be a struct of Boolean values. `cases` can be a mix
    of scalar and array arguments (of any type, but all must be the
    same type or castable to a common type), with either exactly one
    datum per child of `cond`, or one more `cases` than children of
    `cond` (in which case we have an "else" value).

    Each row of the output will be the corresponding value of the
    first datum in `cases` for which the corresponding child of `cond`
    is true, or otherwise the "else" value (if given), or null.

    Essentially, this implements a switch-case or if-else, if-else... statement.

    Parameters
    ----------
    cond : Array-like or scalar-like
        Argument to compute function.
    *cases : Array-like or scalar-like
        Argument to compute function.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

def choose(indices, /, *values, memory_pool: lib.MemoryPool | None = None):
    """
    Choose values from several arrays.

    For each row, the value of the first argument is used as a 0-based index
    into the list of `values` arrays (i.e. index 0 selects the first of the
    `values` arrays). The output value is the corresponding value of the
    selected argument.

    If an index is null, the output will be null.

    Parameters
    ----------
    indices : Array-like or scalar-like
        Argument to compute function.
    *values : Array-like or scalar-like
        Argument to compute function.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

def coalesce(
    *values: _ScalarOrArrayT, memory_pool: lib.MemoryPool | None = None
) -> _ScalarOrArrayT:
    """
    Select the first non-null value.

    Each row of the output will be the value from the first corresponding input
    for which the value is not null. If all inputs are null in a row, the output
    will be null.

    Parameters
    ----------
    *values : Array-like or scalar-like
        Argument to compute function.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

fill_null = coalesce
"""Replace each null element in values with a corresponding
element from fill_value.

If fill_value is scalar-like, then every null element in values
will be replaced with fill_value. If fill_value is array-like,
then the i-th element in values will be replaced with the i-th
element in fill_value.

The fill_value's type must be the same as that of values, or it
must be able to be implicitly casted to the array's type.

This is an alias for :func:`coalesce`.

Parameters
----------
values : Array, ChunkedArray, or Scalar-like object
    Each null element is replaced with the corresponding value
    from fill_value.
fill_value : Array, ChunkedArray, or Scalar-like object
    If not same type as values, will attempt to cast.

Returns
-------
result : depends on inputs
    Values with all null elements replaced

Examples
--------
>>> import pyarrow as pa
>>> arr = pa.array([1, 2, None, 3], type=pa.int8())
>>> fill_value = pa.scalar(5, type=pa.int8())
>>> arr.fill_null(fill_value)
<pyarrow.lib.Int8Array object at ...>
[
    1,
    2,
    5,
    3
]
>>> arr = pa.array([1, 2, None, 4, None])
>>> arr.fill_null(pa.array([10, 20, 30, 40, 50]))
<pyarrow.lib.Int64Array object at ...>
[
    1,
    2,
    30,
    4,
    50
]
"""

def if_else(
    cond: ArrayLike | ScalarLike,
    left: ArrayLike | ScalarLike,
    right: ArrayLike | ScalarLike,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> ArrayLike | ScalarLike:
    """
    Choose values based on a condition.

    `cond` must be a Boolean scalar/ array.
    `left` or `right` must be of the same type scalar/ array.
    `null` values in `cond` will be promoted to the output.

    Parameters
    ----------
    cond : Array-like or scalar-like
        Argument to compute function.
    left : Array-like or scalar-like
        Argument to compute function.
    right : Array-like or scalar-like
        Argument to compute function.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

# ========================= 2.21 Structural transforms =========================

@overload
def list_value_length(
    lists: _ListArray[Any],
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.Int32Array: ...
@overload
def list_value_length(
    lists: _LargeListArray[Any],
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.Int64Array: ...
@overload
def list_value_length(
    lists: ListArray[Any],
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.Int32Array | lib.Int64Array: ...
@overload
def list_value_length(
    lists: Expression,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> Expression: ...
def list_value_length(*args, **kwargs):
    """
    Compute list lengths.

    `lists` must have a list-like type.
    For each non-null value in `lists`, its length is emitted.
    Null values emit a null in the output.

    Parameters
    ----------
    lists : Array-like or scalar-like
        Argument to compute function.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

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
def make_struct(*args, **kwargs):
    """
    Wrap Arrays into a StructArray.

    Names of the StructArray's fields are
    specified through MakeStructOptions.

    Parameters
    ----------
    *args : Array-like or scalar-like
        Argument to compute function.
    field_names : sequence of str
        Names of the struct fields to create.
    field_nullability : sequence of bool, optional
        Nullability information for each struct field.
        If omitted, all fields are nullable.
    field_metadata : sequence of KeyValueMetadata, optional
        Metadata for each struct field.
    options : pyarrow.compute.MakeStructOptions, optional
        Alternative way of passing options.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

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
def ceil_temporal(*args, **kwargs):
    """
    Round temporal values up to nearest multiple of specified time unit.

    Null values emit null.
    An error is returned if the values have a defined timezone but it
    cannot be found in the timezone database.

    Parameters
    ----------
    timestamps : Array-like or scalar-like
        Argument to compute function.
    multiple : int, default 1
        Number of units to round to.
    unit : str, default "day"
        The unit in which `multiple` is expressed.
        Accepted values are "year", "quarter", "month", "week", "day",
        "hour", "minute", "second", "millisecond", "microsecond",
        "nanosecond".
    week_starts_monday : bool, default True
        If True, weeks start on Monday; if False, on Sunday.
    ceil_is_strictly_greater : bool, default False
        If True, ceil returns a rounded value that is strictly greater than the
        input. For example: ceiling 1970-01-01T00:00:00 to 3 hours would
        yield 1970-01-01T03:00:00 if set to True and 1970-01-01T00:00:00
        if set to False.
        This applies to the ceil_temporal function only.
    calendar_based_origin : bool, default False
        By default, the origin is 1970-01-01T00:00:00. By setting this to True,
        rounding origin will be beginning of one less precise calendar unit.
        E.g.: rounding to hours will use beginning of day as origin.

        By default time is rounded to a multiple of units since
        1970-01-01T00:00:00. By setting calendar_based_origin to true,
        time will be rounded to number of units since the last greater
        calendar unit.
        For example: rounding to multiple of days since the beginning of the
        month or to hours since the beginning of the day.
        Exceptions: week and quarter are not used as greater units,
        therefore days will be rounded to the beginning of the month not
        week. Greater unit of week is a year.
        Note that ceiling and rounding might change sorting order of an array
        near greater unit change. For example rounding YYYY-mm-dd 23:00:00 to
        5 hours will ceil and round to YYYY-mm-dd+1 01:00:00 and floor to
        YYYY-mm-dd 20:00:00. On the other hand YYYY-mm-dd+1 00:00:00 will
        ceil, round and floor to YYYY-mm-dd+1 00:00:00. This can break the
        order of an already ordered array.
    options : pyarrow.compute.RoundTemporalOptions, optional
        Alternative way of passing options.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

floor_temporal = _clone_signature(ceil_temporal)
"""
Round temporal values down to nearest multiple of specified time unit.

Null values emit null.
An error is returned if the values have a defined timezone but it
cannot be found in the timezone database.

Parameters
----------
timestamps : Array-like or scalar-like
    Argument to compute function.
multiple : int, default 1
    Number of units to round to.
unit : str, default "day"
    The unit in which `multiple` is expressed.
    Accepted values are "year", "quarter", "month", "week", "day",
    "hour", "minute", "second", "millisecond", "microsecond",
    "nanosecond".
week_starts_monday : bool, default True
    If True, weeks start on Monday; if False, on Sunday.
ceil_is_strictly_greater : bool, default False
    If True, ceil returns a rounded value that is strictly greater than the
    input. For example: ceiling 1970-01-01T00:00:00 to 3 hours would
    yield 1970-01-01T03:00:00 if set to True and 1970-01-01T00:00:00
    if set to False.
    This applies to the ceil_temporal function only.
calendar_based_origin : bool, default False
    By default, the origin is 1970-01-01T00:00:00. By setting this to True,
    rounding origin will be beginning of one less precise calendar unit.
    E.g.: rounding to hours will use beginning of day as origin.

    By default time is rounded to a multiple of units since
    1970-01-01T00:00:00. By setting calendar_based_origin to true,
    time will be rounded to number of units since the last greater
    calendar unit.
    For example: rounding to multiple of days since the beginning of the
    month or to hours since the beginning of the day.
    Exceptions: week and quarter are not used as greater units,
    therefore days will be rounded to the beginning of the month not
    week. Greater unit of week is a year.
    Note that ceiling and rounding might change sorting order of an array
    near greater unit change. For example rounding YYYY-mm-dd 23:00:00 to
    5 hours will ceil and round to YYYY-mm-dd+1 01:00:00 and floor to
    YYYY-mm-dd 20:00:00. On the other hand YYYY-mm-dd+1 00:00:00 will
    ceil, round and floor to YYYY-mm-dd+1 00:00:00. This can break the
    order of an already ordered array.
options : pyarrow.compute.RoundTemporalOptions, optional
    Alternative way of passing options.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
round_temporal = _clone_signature(ceil_temporal)
"""
Round temporal values to the nearest multiple of specified time unit.

Null values emit null.
An error is returned if the values have a defined timezone but it
cannot be found in the timezone database.

Parameters
----------
timestamps : Array-like or scalar-like
    Argument to compute function.
multiple : int, default 1
    Number of units to round to.
unit : str, default "day"
    The unit in which `multiple` is expressed.
    Accepted values are "year", "quarter", "month", "week", "day",
    "hour", "minute", "second", "millisecond", "microsecond",
    "nanosecond".
week_starts_monday : bool, default True
    If True, weeks start on Monday; if False, on Sunday.
ceil_is_strictly_greater : bool, default False
    If True, ceil returns a rounded value that is strictly greater than the
    input. For example: ceiling 1970-01-01T00:00:00 to 3 hours would
    yield 1970-01-01T03:00:00 if set to True and 1970-01-01T00:00:00
    if set to False.
    This applies to the ceil_temporal function only.
calendar_based_origin : bool, default False
    By default, the origin is 1970-01-01T00:00:00. By setting this to True,
    rounding origin will be beginning of one less precise calendar unit.
    E.g.: rounding to hours will use beginning of day as origin.

    By default time is rounded to a multiple of units since
    1970-01-01T00:00:00. By setting calendar_based_origin to true,
    time will be rounded to number of units since the last greater
    calendar unit.
    For example: rounding to multiple of days since the beginning of the
    month or to hours since the beginning of the day.
    Exceptions: week and quarter are not used as greater units,
    therefore days will be rounded to the beginning of the month not
    week. Greater unit of week is a year.
    Note that ceiling and rounding might change sorting order of an array
    near greater unit change. For example rounding YYYY-mm-dd 23:00:00 to
    5 hours will ceil and round to YYYY-mm-dd+1 01:00:00 and floor to
    YYYY-mm-dd 20:00:00. On the other hand YYYY-mm-dd+1 00:00:00 will
    ceil, round and floor to YYYY-mm-dd+1 00:00:00. This can break the
    order of an already ordered array.
options : pyarrow.compute.RoundTemporalOptions, optional
    Alternative way of passing options.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""

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
def cast(*args, **kwargs):
    """
    Cast array values to another data type. Can also be invoked as an array
    instance method.

    Parameters
    ----------
    arr : Array-like
    target_type : DataType or str
        Type to cast to
    safe : bool, default True
        Check for overflows or other unsafe conversions
    options : CastOptions, default None
        Additional checks pass by CastOptions
    memory_pool : MemoryPool, optional
        memory pool to use for allocations during function execution.

    Examples
    --------
    >>> from datetime import datetime
    >>> import pyarrow as pa
    >>> arr = pa.array([datetime(2010, 1, 1), datetime(2015, 1, 1)])
    >>> arr.type
    TimestampType(timestamp[us])

    You can use ``pyarrow.DataType`` objects to specify the target type:

    >>> cast(arr, pa.timestamp("ms"))
    <pyarrow.lib.TimestampArray object at ...>
    [
      2010-01-01 00:00:00.000,
      2015-01-01 00:00:00.000
    ]

    >>> cast(arr, pa.timestamp("ms")).type
    TimestampType(timestamp[ms])

    Alternatively, it is also supported to use the string aliases for these
    types:

    >>> arr.cast("timestamp[ms]")
    <pyarrow.lib.TimestampArray object at ...>
    [
      2010-01-01 00:00:00.000,
      2015-01-01 00:00:00.000
    ]
    >>> arr.cast("timestamp[ms]").type
    TimestampType(timestamp[ms])

    Returns
    -------
    casted : Array
        The cast result as a new Array
    """

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
def strftime(*args, **kwargs):
    """
    Format temporal values according to a format string.

    For each input value, emit a formatted string.
    The time format string and locale can be set using StrftimeOptions.
    The output precision of the "%S" (seconds) format code depends on
    the input time precision: it is an integer for timestamps with
    second precision, a real number with the required number of fractional
    digits for higher precisions.
    Null values emit null.
    An error is returned if the values have a defined timezone but it
    cannot be found in the timezone database, or if the specified locale
    does not exist on this system.

    Parameters
    ----------
    timestamps : Array-like or scalar-like
        Argument to compute function.
    format : str, default "%Y-%m-%dT%H:%M:%S"
        Pattern for formatting input values.
    locale : str, default "C"
        Locale to use for locale-specific format specifiers.
    options : pyarrow.compute.StrftimeOptions, optional
        Alternative way of passing options.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

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
def strptime(*args, **kwargs):
    """
    Parse timestamps.

    For each string in `strings`, parse it as a timestamp.
    The timestamp unit and the expected string pattern must be given
    in StrptimeOptions. Null inputs emit null. If a non-null string
    fails parsing, an error is returned by default.

    Parameters
    ----------
    strings : Array-like or scalar-like
        Argument to compute function.
    format : str
        Pattern for parsing input strings as timestamps, such as "%Y/%m/%d".
        Note that the semantics of the format follow the C/C++ strptime, not the Python one.
        There are differences in behavior, for example how the "%y" placeholder
        handles years with less than four digits.
    unit : str
        Timestamp unit of the output.
        Accepted values are "s", "ms", "us", "ns".
    error_is_null : boolean, default False
        Return null on parsing errors if true or raise if false.
    options : pyarrow.compute.StrptimeOptions, optional
        Alternative way of passing options.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

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
def day(*args, **kwargs):
    """
    Extract day number.

    Null values emit null.
    An error is returned if the values have a defined timezone but it
    cannot be found in the timezone database.

    Parameters
    ----------
    values : Array-like or scalar-like
        Argument to compute function.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

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
def day_of_week(*args, **kwargs):
    """
    Extract day of the week number.

    By default, the week starts on Monday represented by 0 and ends on Sunday
    represented by 6.
    `DayOfWeekOptions.week_start` can be used to set another starting day using
    the ISO numbering convention (1=start week on Monday, 7=start week on Sunday).
    Day numbers can start at 0 or 1 based on `DayOfWeekOptions.count_from_zero`.
    Null values emit null.
    An error is returned if the values have a defined timezone but it
    cannot be found in the timezone database.

    Parameters
    ----------
    values : Array-like or scalar-like
        Argument to compute function.
    count_from_zero : bool, default True
        If True, number days from 0, otherwise from 1.
    week_start : int, default 1
        Which day does the week start with (Monday=1, Sunday=7).
        How this value is numbered is unaffected by `count_from_zero`.
    options : pyarrow.compute.DayOfWeekOptions, optional
        Alternative way of passing options.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

day_of_year = _clone_signature(day)
"""
Extract day of year number.

January 1st maps to day number 1, February 1st to 32, etc.
Null values emit null.
An error is returned if the values have a defined timezone but it
cannot be found in the timezone database.

Parameters
----------
values : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""

@overload
def hour(
    values: lib.TimestampScalar[Any] | lib.Time32Scalar[Any] | lib.Time64Scalar[Any],
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.Int64Scalar: ...
@overload
def hour(
    values: lib.TimestampArray[Any]
    | lib.Time32Array[Any]
    | lib.Time64Array[Any]
    | lib.ChunkedArray[lib.TimestampScalar[Any]]
    | lib.ChunkedArray[lib.Time32Scalar[Any]]
    | lib.ChunkedArray[lib.Time64Scalar[Any]],
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
def hour(*args, **kwargs):
    """
    Extract hour value.

    Null values emit null.
    An error is returned if the values have a defined timezone but it
    cannot be found in the timezone database.

    Parameters
    ----------
    values : Array-like or scalar-like
        Argument to compute function.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

@overload
def is_dst(
    values: lib.TimestampScalar[Any], /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.BooleanScalar: ...
@overload
def is_dst(
    values: lib.TimestampArray[Any] | lib.ChunkedArray[lib.TimestampScalar[Any]],
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.BooleanArray: ...
@overload
def is_dst(values: Expression, /, *, memory_pool: lib.MemoryPool | None = None) -> Expression: ...
def is_dst(*args, **kwargs):
    """
    Extracts if currently observing daylight savings.

    IsDaylightSavings returns true if a timestamp has a daylight saving
    offset in the given timezone.
    Null values emit null.
    An error is returned if the values do not have a defined timezone.

    Parameters
    ----------
    values : Array-like or scalar-like
        Argument to compute function.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

@overload
def iso_week(
    values: lib.TimestampScalar[Any], /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.Int64Scalar: ...
@overload
def iso_week(
    values: lib.TimestampArray[Any] | lib.ChunkedArray[lib.TimestampScalar[Any]],
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.Int64Array: ...
@overload
def iso_week(
    values: Expression, /, *, memory_pool: lib.MemoryPool | None = None
) -> Expression: ...
def iso_week(*args, **kwargs):
    """
    Extract ISO week of year number.

    First ISO week has the majority (4 or more) of its days in January.
    ISO week starts on Monday. The week number starts with 1 and can run
    up to 53.
    Null values emit null.
    An error is returned if the values have a defined timezone but it
    cannot be found in the timezone database.

    Parameters
    ----------
    values : Array-like or scalar-like
        Argument to compute function.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

iso_year = _clone_signature(iso_week)
"""
Extract ISO year number.

First week of an ISO year has the majority (4 or more) of its days in January.
Null values emit null.
An error is returned if the values have a defined timezone but it
cannot be found in the timezone database.

Parameters
----------
values : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""

@overload
def is_leap_year(
    values: lib.TimestampScalar[Any] | lib.Date32Scalar | lib.Date64Scalar,
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
def is_leap_year(*args, **kwargs):
    """
    Extract if year is a leap year.

    Null values emit null.
    An error is returned if the values have a defined timezone but it
    cannot be found in the timezone database.

    Parameters
    ----------
    values : Array-like or scalar-like
        Argument to compute function.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

microsecond = _clone_signature(iso_week)
"""
Extract microsecond values.

Microsecond returns number of microseconds since the last full millisecond.
Null values emit null.
An error is returned if the values have a defined timezone but it
cannot be found in the timezone database.

Parameters
----------
values : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
millisecond = _clone_signature(iso_week)
"""
Extract millisecond values.

Millisecond returns number of milliseconds since the last full second.
Null values emit null.
An error is returned if the values have a defined timezone but it
cannot be found in the timezone database.

Parameters
----------
values : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
minute = _clone_signature(iso_week)
"""
Extract minute values.

Null values emit null.
An error is returned if the values have a defined timezone but it
cannot be found in the timezone database.

Parameters
----------
values : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
month = _clone_signature(day_of_week)
"""
Extract month number.

Month is encoded as January=1, December=12.
Null values emit null.
An error is returned if the values have a defined timezone but it
cannot be found in the timezone database.

Parameters
----------
values : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
nanosecond = _clone_signature(hour)
"""
Extract nanosecond values.

Nanosecond returns number of nanoseconds since the last full microsecond.
Null values emit null.
An error is returned if the values have a defined timezone but it
cannot be found in the timezone database.

Parameters
----------
values : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
quarter = _clone_signature(day_of_week)
"""
Extract quarter of year number.

First quarter maps to 1 and forth quarter maps to 4.
Null values emit null.
An error is returned if the values have a defined timezone but it
cannot be found in the timezone database.

Parameters
----------
values : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
second = _clone_signature(hour)
"""
Extract second values.

Null values emit null.
An error is returned if the values have a defined timezone but it
cannot be found in the timezone database.

Parameters
----------
values : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
subsecond = _clone_signature(hour)
"""
Extract subsecond values.

Subsecond returns the fraction of a second since the last full second.
Null values emit null.
An error is returned if the values have a defined timezone but it
cannot be found in the timezone database.

Parameters
----------
values : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
us_week = _clone_signature(iso_week)
"""
Extract US week of year number.

First US week has the majority (4 or more) of its days in January.
US week starts on Monday. The week number starts with 1 and can run
up to 53.
Null values emit null.
An error is returned if the values have a defined timezone but it
cannot be found in the timezone database.

Parameters
----------
values : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
us_year = _clone_signature(iso_week)
"""
Extract US epidemiological year number.

First week of US epidemiological year has the majority (4 or more) of
it's days in January. Last week of US epidemiological year has the
year's last Wednesday in it. US epidemiological week starts on Sunday.
Null values emit null.
An error is returned if the values have a defined timezone but it
cannot be found in the timezone database.

Parameters
----------
values : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
year = _clone_signature(iso_week)
"""
Extract year number.

Null values emit null.
An error is returned if the values have a defined timezone but it
cannot be found in the timezone database.

Parameters
----------
values : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""

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
def week(*args, **kwargs):
    """
    Extract week of year number.

    First week has the majority (4 or more) of its days in January.
    Year can have 52 or 53 weeks. Week numbering can start with 0 or 1 using
    DayOfWeekOptions.count_from_zero.
    An error is returned if the values have a defined timezone but it
    cannot be found in the timezone database.

    Parameters
    ----------
    values : Array-like or scalar-like
        Argument to compute function.
    week_starts_monday : bool, default True
        If True, weeks start on Monday; if False, on Sunday.
    count_from_zero : bool, default False
        If True, dates at the start of a year that fall into the last week
        of the previous year emit 0.
        If False, they emit 52 or 53 (the week number of the last week
        of the previous year).
    first_week_is_fully_in_year : bool, default False
        If True, week number 0 is fully in January.
        If False, a week that begins on December 29, 30 or 31 is considered
        to be week number 0 of the following year.
    options : pyarrow.compute.WeekOptions, optional
        Alternative way of passing options.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

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
def year_month_day(*args, **kwargs):
    """
    Extract (year, month, day) struct.

    Null values emit null.
    An error is returned in the values have a defined timezone but it
    cannot be found in the timezone database.

    Parameters
    ----------
    values : Array-like or scalar-like
        Argument to compute function.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

# ========================= 2.24 Temporal difference =========================
def day_time_interval_between(start, end, /, *, memory_pool: lib.MemoryPool | None = None):
    """
    Compute the number of days and milliseconds between two timestamps.

    Returns the number of days and milliseconds from `start` to `end`.
    That is, first the difference in days is computed as if both
    timestamps were truncated to the day, then the difference between time times
    of the two timestamps is computed as if both times were truncated to the
    millisecond.
    Null values return null.

    Parameters
    ----------
    start : Array-like or scalar-like
        Argument to compute function.
    end : Array-like or scalar-like
        Argument to compute function.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

def days_between(
    start, end, /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.Int64Scalar | lib.Int64Array:
    """
    Compute the number of days between two timestamps.

    Returns the number of day boundaries crossed from `start` to `end`.
    That is, the difference is calculated as if the timestamps were
    truncated to the day.
    Null values emit null.

    Parameters
    ----------
    start : Array-like or scalar-like
        Argument to compute function.
    end : Array-like or scalar-like
        Argument to compute function.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

hours_between = _clone_signature(days_between)
"""
Compute the number of hours between two timestamps.

Returns the number of hour boundaries crossed from `start` to `end`.
That is, the difference is calculated as if the timestamps were
truncated to the hour.
Null values emit null.

Parameters
----------
start : Array-like or scalar-like
    Argument to compute function.
end : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
microseconds_between = _clone_signature(days_between)
"""
Compute the number of microseconds between two timestamps.

Returns the number of microsecond boundaries crossed from `start` to `end`.
That is, the difference is calculated as if the timestamps were
truncated to the microsecond.
Null values emit null.

Parameters
----------
start : Array-like or scalar-like
    Argument to compute function.
end : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
milliseconds_between = _clone_signature(days_between)
"""
Compute the number of millisecond boundaries between two timestamps.

Returns the number of millisecond boundaries crossed from `start` to `end`.
That is, the difference is calculated as if the timestamps were
truncated to the millisecond.
Null values emit null.

Parameters
----------
start : Array-like or scalar-like
    Argument to compute function.
end : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
minutes_between = _clone_signature(days_between)
"""
Compute the number of millisecond boundaries between two timestamps.

Returns the number of millisecond boundaries crossed from `start` to `end`.
That is, the difference is calculated as if the timestamps were
truncated to the millisecond.
Null values emit null.
In [152]: print(pc.minutes_between.__doc__)
Compute the number of minute boundaries between two timestamps.

Returns the number of minute boundaries crossed from `start` to `end`.
That is, the difference is calculated as if the timestamps were
truncated to the minute.
Null values emit null.

Parameters
----------
start : Array-like or scalar-like
    Argument to compute function.
end : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""

def month_day_nano_interval_between(
    start, end, /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.MonthDayNanoIntervalScalar | lib.MonthDayNanoIntervalArray:
    """
    Compute the number of months, days and nanoseconds between two timestamps.

    Returns the number of months, days, and nanoseconds from `start` to `end`.
    That is, first the difference in months is computed as if both timestamps
    were truncated to the months, then the difference between the days
    is computed, and finally the difference between the times of the two
    timestamps is computed as if both times were truncated to the nanosecond.
    Null values return null.

    Parameters
    ----------
    start : Array-like or scalar-like
        Argument to compute function.
    end : Array-like or scalar-like
        Argument to compute function.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

def month_interval_between(start, end, /, *, memory_pool: lib.MemoryPool | None = None):
    """
    Compute the number of months between two timestamps.

    Returns the number of month boundaries crossed from `start` to `end`.
    That is, the difference is calculated as if the timestamps were
    truncated to the month.
    Null values emit null.

    Parameters
    ----------
    start : Array-like or scalar-like
        Argument to compute function.
    end : Array-like or scalar-like
        Argument to compute function.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

nanoseconds_between = _clone_signature(days_between)
"""
Compute the number of nanoseconds between two timestamps.

Returns the number of nanosecond boundaries crossed from `start` to `end`.
That is, the difference is calculated as if the timestamps were
truncated to the nanosecond.
Null values emit null.

Parameters
----------
start : Array-like or scalar-like
    Argument to compute function.
end : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
quarters_between = _clone_signature(days_between)
"""
Compute the number of quarters between two timestamps.

Returns the number of quarter start boundaries crossed from `start` to `end`.
That is, the difference is calculated as if the timestamps were
truncated to the quarter.
Null values emit null.

Parameters
----------
start : Array-like or scalar-like
    Argument to compute function.
end : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
seconds_between = _clone_signature(days_between)
"""
Compute the number of seconds between two timestamps.

Returns the number of second boundaries crossed from `start` to `end`.
That is, the difference is calculated as if the timestamps were
truncated to the second.
Null values emit null.

Parameters
----------
start : Array-like or scalar-like
    Argument to compute function.
end : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""

def weeks_between(
    start,
    end,
    /,
    *,
    count_from_zero: bool = True,
    week_start: int = 1,
    options: DayOfWeekOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.Int64Scalar | lib.Int64Array:
    """
    Compute the number of weeks between two timestamps.

    Returns the number of week boundaries crossed from `start` to `end`.
    That is, the difference is calculated as if the timestamps were
    truncated to the week.
    Null values emit null.

    Parameters
    ----------
    start : Array-like or scalar-like
        Argument to compute function.
    end : Array-like or scalar-like
        Argument to compute function.
    count_from_zero : bool, default True
        If True, number days from 0, otherwise from 1.
    week_start : int, default 1
        Which day does the week start with (Monday=1, Sunday=7).
        How this value is numbered is unaffected by `count_from_zero`.
    options : pyarrow.compute.DayOfWeekOptions, optional
        Alternative way of passing options.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

years_between = _clone_signature(days_between)
"""
Compute the number of years between two timestamps.

Returns the number of year boundaries crossed from `start` to `end`.
That is, the difference is calculated as if the timestamps were
truncated to the year.
Null values emit null.

Parameters
----------
start : Array-like or scalar-like
    Argument to compute function.
end : Array-like or scalar-like
    Argument to compute function.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""

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
def assume_timezone(*args, **kwargs):
    """
    Convert naive timestamp to timezone-aware timestamp.

    Input timestamps are assumed to be relative to the timezone given in the
    `timezone` option. They are converted to UTC-relative timestamps and
    the output type has its timezone set to the value of the `timezone`
    option. Null values emit null.
    This function is meant to be used when an external system produces
    "timezone-naive" timestamps which need to be converted to
    "timezone-aware" timestamps. An error is returned if the timestamps
    already have a defined timezone.

    Parameters
    ----------
    timestamps : Array-like or scalar-like
        Argument to compute function.
    timezone : str
        Timezone to assume for the input.
    ambiguous : str, default "raise"
        How to handle timestamps that are ambiguous in the assumed timezone.
        Accepted values are "raise", "earliest", "latest".
    nonexistent : str, default "raise"
        How to handle timestamps that don't exist in the assumed timezone.
        Accepted values are "raise", "earliest", "latest".
    options : pyarrow.compute.AssumeTimezoneOptions, optional
        Alternative way of passing options.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

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
def local_timestamp(*args, **kwargs):
    """
    Convert timestamp to a timezone-naive local time timestamp.

    LocalTimestamp converts timezone-aware timestamp to local timestamp
    of the given timestamp's timezone and removes timezone metadata.
    Alternative name for this timestamp is also wall clock time.
    If input is in UTC or without timezone, then unchanged input values
    without timezone metadata are returned.
    Null values emit null.

    Parameters
    ----------
    values : Array-like or scalar-like
        Argument to compute function.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

# ========================= 2.26 Random number generation =========================
def random(
    n: int,
    *,
    initializer: Literal["system"] | int = "system",
    options: RandomOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.DoubleArray:
    """
    Generate numbers in the range [0, 1).

    Generated values are uniformly-distributed, double-precision
    in range [0, 1). Algorithm and seed can be changed via RandomOptions.

    Parameters
    ----------
    n : int
        Number of values to generate, must be greater than or equal to 0
    initializer : int or str
        How to initialize the underlying random generator.
        If an integer is given, it is used as a seed.
        If "system" is given, the random generator is initialized with
        a system-specific source of (hopefully true) randomness.
        Other values are invalid.
    options : pyarrow.compute.RandomOptions, optional
        Alternative way of passing options.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

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
def cumulative_sum(*args, **kwargs):
    """
    Compute the cumulative sum over a numeric input.

    `values` must be numeric. Return an array/chunked array which is the
    cumulative sum computed over `values`. Results will wrap around on
    integer overflow. Use function "cumulative_sum_checked" if you want
    overflow to return an error. The default start is 0.

    Parameters
    ----------
    values : Array-like
        Argument to compute function.
    start : Scalar, default None
        Starting value for the cumulative operation. If none is given,
        a default value depending on the operation and input type is used.
    skip_nulls : bool, default False
        When false, the first encountered null is propagated.
    options : pyarrow.compute.CumulativeOptions, optional
        Alternative way of passing options.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

cumulative_sum_checked = _clone_signature(cumulative_sum)
"""
Compute the cumulative sum over a numeric input.

`values` must be numeric. Return an array/chunked array which is the
cumulative sum computed over `values`. This function returns an error
on overflow. For a variant that doesn't fail on overflow, use
function "cumulative_sum". The default start is 0.

Parameters
----------
values : Array-like
    Argument to compute function.
start : Scalar, default None
    Starting value for the cumulative operation. If none is given,
    a default value depending on the operation and input type is used.
skip_nulls : bool, default False
    When false, the first encountered null is propagated.
options : pyarrow.compute.CumulativeOptions, optional
    Alternative way of passing options.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
cumulative_prod = _clone_signature(cumulative_sum)
"""
Compute the cumulative product over a numeric input.

`values` must be numeric. Return an array/chunked array which is the
cumulative product computed over `values`. Results will wrap around on
integer overflow. Use function "cumulative_prod_checked" if you want
overflow to return an error. The default start is 1.

Parameters
----------
values : Array-like
    Argument to compute function.
start : Scalar, default None
    Starting value for the cumulative operation. If none is given,
    a default value depending on the operation and input type is used.
skip_nulls : bool, default False
    When false, the first encountered null is propagated.
options : pyarrow.compute.CumulativeOptions, optional
    Alternative way of passing options.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
cumulative_prod_checked = _clone_signature(cumulative_sum)
"""
Compute the cumulative product over a numeric input.

`values` must be numeric. Return an array/chunked array which is the
cumulative product computed over `values`. This function returns an error
on overflow. For a variant that doesn't fail on overflow, use
function "cumulative_prod". The default start is 1.

Parameters
----------
values : Array-like
    Argument to compute function.
start : Scalar, default None
    Starting value for the cumulative operation. If none is given,
    a default value depending on the operation and input type is used.
skip_nulls : bool, default False
    When false, the first encountered null is propagated.
options : pyarrow.compute.CumulativeOptions, optional
    Alternative way of passing options.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
cumulative_max = _clone_signature(cumulative_sum)
"""
Compute the cumulative max over a numeric input.

`values` must be numeric. Return an array/chunked array which is the
cumulative max computed over `values`. The default start is the minimum
value of input type (so that any other value will replace the
start as the new maximum).

Parameters
----------
values : Array-like
    Argument to compute function.
start : Scalar, default None
    Starting value for the cumulative operation. If none is given,
    a default value depending on the operation and input type is used.
skip_nulls : bool, default False
    When false, the first encountered null is propagated.
options : pyarrow.compute.CumulativeOptions, optional
    Alternative way of passing options.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
cumulative_min = _clone_signature(cumulative_sum)
"""
Compute the cumulative min over a numeric input.

`values` must be numeric. Return an array/chunked array which is the
cumulative min computed over `values`. The default start is the maximum
value of input type (so that any other value will replace the
start as the new minimum).

Parameters
----------
values : Array-like
    Argument to compute function.
start : Scalar, default None
    Starting value for the cumulative operation. If none is given,
    a default value depending on the operation and input type is used.
skip_nulls : bool, default False
    When false, the first encountered null is propagated.
options : pyarrow.compute.CumulativeOptions, optional
    Alternative way of passing options.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
cumulative_mean = _clone_signature(cumulative_sum)
"""
Compute the cumulative max over a numeric input.

`values` must be numeric. Return an array/chunked array which is the
cumulative max computed over `values`. The default start is the minimum
value of input type (so that any other value will replace the
start as the new maximum).

Parameters
----------
values : Array-like
    Argument to compute function.
start : Scalar, default None
    Starting value for the cumulative operation. If none is given,
    a default value depending on the operation and input type is used.
skip_nulls : bool, default False
    When false, the first encountered null is propagated.
options : pyarrow.compute.CumulativeOptions, optional
    Alternative way of passing options.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
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
"""
Select values (or records) from array- or table-like data given integer
selection indices.

The result will be of the same type(s) as the input, with elements taken
from the input array (or record batch / table fields) at the given
indices. If an index is null then the corresponding value in the output
will be null.

Parameters
----------
data : Array, ChunkedArray, RecordBatch, or Table
indices : Array, ChunkedArray
    Must be of integer type
boundscheck : boolean, default True
    Whether to boundscheck the indices. If False and there is an out of
    bounds index, will likely cause the process to crash.
memory_pool : MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.

Returns
-------
result : depends on inputs
    Selected values for the given indices

Examples
--------
>>> import pyarrow as pa
>>> arr = pa.array(["a", "b", "c", None, "e", "f"])
>>> indices = pa.array([0, None, 4, 3])
>>> arr.take(indices)
<pyarrow.lib.StringArray object at ...>
[
    "a",
    null,
    "e",
    null
]
"""

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
def indices_nonzero(*args, **kwargs):
    """
    Return the indices of the values in the array that are non-zero.

    For each input value, check if it's zero, false or null. Emit the index
    of the value in the array if it's none of the those.

    Parameters
    ----------
    values : Array-like
        Argument to compute function.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

# ========================= 3.5 Sorts and partitions  =========================
@overload
def array_sort_indices(
    array: lib.Array | lib.ChunkedArray,
    /,
    order: _Order = "ascending",
    *,
    null_placement: _Placement = "at_end",
    options: ArraySortOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.UInt64Array: ...
@overload
def array_sort_indices(
    array: Expression,
    /,
    order: _Order = "ascending",
    *,
    null_placement: _Placement = "at_end",
    options: ArraySortOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> Expression: ...
def array_sort_indices(*args, **kwargs):
    """
    Return the indices that would sort an array.

    This function computes an array of indices that define a stable sort
    of the input array.  By default, Null values are considered greater
    than any other value and are therefore sorted at the end of the array.
    For floating-point types, NaNs are considered greater than any
    other non-null value, but smaller than null values.

    The handling of nulls and NaNs can be changed in ArraySortOptions.

    Parameters
    ----------
    array : Array-like
        Argument to compute function.
    order : str, default "ascending"
        Which order to sort values in.
        Accepted values are "ascending", "descending".
    null_placement : str, default "at_end"
        Where nulls in the input should be sorted.
        Accepted values are "at_start", "at_end".
    options : pyarrow.compute.ArraySortOptions, optional
        Alternative way of passing options.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

@overload
def partition_nth_indices(
    array: lib.Array | lib.ChunkedArray,
    /,
    pivot: int,
    *,
    null_placement: _Placement = "at_end",
    options: PartitionNthOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.UInt64Array: ...
@overload
def partition_nth_indices(
    array: Expression,
    /,
    pivot: int,
    *,
    null_placement: _Placement = "at_end",
    options: PartitionNthOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> Expression: ...
def partition_nth_indices(*args, **kwargs):
    """
    Return the indices that would partition an array around a pivot.

    This functions computes an array of indices that define a non-stable
    partial sort of the input array.

    The output is such that the `N`'th index points to the `N`'th element
    of the input in sorted order, and all indices before the `N`'th point
    to elements in the input less or equal to elements at or after the `N`'th.

    By default, null values are considered greater than any other value
    and are therefore partitioned towards the end of the array.
    For floating-point types, NaNs are considered greater than any
    other non-null value, but smaller than null values.

    The pivot index `N` must be given in PartitionNthOptions.
    The handling of nulls and NaNs can also be changed in PartitionNthOptions.

    Parameters
    ----------
    array : Array-like
        Argument to compute function.
    pivot : int
        Index into the equivalent sorted array of the pivot element.
    null_placement : str, default "at_end"
        Where nulls in the input should be partitioned.
        Accepted values are "at_start", "at_end".
    options : pyarrow.compute.PartitionNthOptions, optional
        Alternative way of passing options.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

def rank(
    input: lib.Array | lib.ChunkedArray,
    /,
    sort_keys: _Order = "ascending",
    *,
    null_placement: _Placement = "at_end",
    tiebreaker: Literal["min", "max", "first", "dense"] = "first",
    options: RankOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.UInt64Array:
    """
    Compute ordinal ranks of an array (1-based).

    This function computes a rank of the input array.
    By default, null values are considered greater than any other value and
    are therefore sorted at the end of the input. For floating-point types,
    NaNs are considered greater than any other non-null value, but smaller
    than null values. The default tiebreaker is to assign ranks in order of
    when ties appear in the input.

    The handling of nulls, NaNs and tiebreakers can be changed in RankOptions.

    Parameters
    ----------
    input : Array-like or scalar-like
        Argument to compute function.
    sort_keys : sequence of (name, order) tuples or str, default "ascending"
        Names of field/column keys to sort the input on,
        along with the order each field/column is sorted in.
        Accepted values for `order` are "ascending", "descending".
        The field name can be a string column name or expression.
        Alternatively, one can simply pass "ascending" or "descending" as a string
        if the input is array-like.
    null_placement : str, default "at_end"
        Where nulls in input should be sorted.
        Accepted values are "at_start", "at_end".
    tiebreaker : str, default "first"
        Configure how ties between equal values are handled.
        Accepted values are:

        - "min": Ties get the smallest possible rank in sorted order.
        - "max": Ties get the largest possible rank in sorted order.
        - "first": Ranks are assigned in order of when ties appear in the
                   input. This ensures the ranks are a stable permutation
                   of the input.
        - "dense": The ranks span a dense [1, M] interval where M is the
                   number of distinct values in the input.
    options : pyarrow.compute.RankOptions, optional
        Alternative way of passing options.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

@overload
def select_k_unstable(
    input: lib.Array | lib.ChunkedArray,
    /,
    k: int,
    sort_keys: list[tuple[str, _Order]],
    *,
    options: SelectKOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.UInt64Array: ...
@overload
def select_k_unstable(
    input: Expression,
    /,
    k: int,
    sort_keys: list[tuple[str, _Order]],
    *,
    options: SelectKOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> Expression: ...
def select_k_unstable(*args, **kwargs):
    """
    Select the indices of the first `k` ordered elements from the input.

    This function selects an array of indices of the first `k` ordered elements
    from the `input` array, record batch or table specified in the column keys
    (`options.sort_keys`). Output is not guaranteed to be stable.
    Null values are considered greater than any other value and are
    therefore ordered at the end. For floating-point types, NaNs are considered
    greater than any other non-null value, but smaller than null values.

    Parameters
    ----------
    input : Array-like or scalar-like
        Argument to compute function.
    k : int
        Number of leading values to select in sorted order
        (i.e. the largest values if sort order is "descending",
        the smallest otherwise).
    sort_keys : sequence of (name, order) tuples
        Names of field/column keys to sort the input on,
        along with the order each field/column is sorted in.
        Accepted values for `order` are "ascending", "descending".
        The field name can be a string column name or expression.
    options : pyarrow.compute.SelectKOptions, optional
        Alternative way of passing options.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

@overload
def sort_indices(
    input: lib.Array | lib.ChunkedArray | lib.RecordBatch | lib.Table,
    /,
    sort_keys: Sequence[tuple[str, _Order]] = (),
    *,
    null_placement: _Placement = "at_end",
    options: SortOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.UInt64Array: ...
@overload
def sort_indices(
    input: Expression,
    /,
    sort_keys: Sequence[tuple[str, _Order]] = (),
    *,
    null_placement: _Placement = "at_end",
    options: SortOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> Expression: ...
def sort_indices(*args, **kwargs):
    """
    Return the indices that would sort an array, record batch or table.

    This function computes an array of indices that define a stable sort
    of the input array, record batch or table.  By default, null values are
    considered greater than any other value and are therefore sorted at the
    end of the input. For floating-point types, NaNs are considered greater
    than any other non-null value, but smaller than null values.

    The handling of nulls and NaNs can be changed in SortOptions.

    Parameters
    ----------
    input : Array-like or scalar-like
        Argument to compute function.
    sort_keys : sequence of (name, order) tuples
        Names of field/column keys to sort the input on,
        along with the order each field/column is sorted in.
        Accepted values for `order` are "ascending", "descending".
        The field name can be a string column name or expression.
    null_placement : str, default "at_end"
        Where nulls in input should be sorted, only applying to
        columns/fields mentioned in `sort_keys`.
        Accepted values are "at_start", "at_end".
    options : pyarrow.compute.SortOptions, optional
        Alternative way of passing options.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

# ========================= 3.6 Structural transforms =========================
@overload
def list_element(
    lists: Expression, index: ScalarLike, /, *, memory_pool: lib.MemoryPool | None = None
) -> Expression: ...
@overload
def list_element(
    lists: lib.Array[ListScalar[_DataTypeT]],
    index: ScalarLike,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.Array[lib.Scalar[_DataTypeT]]: ...
@overload
def list_element(
    lists: lib.ChunkedArray[ListScalar[_DataTypeT]],
    index: ScalarLike,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.ChunkedArray[lib.Scalar[_DataTypeT]]: ...
@overload
def list_element(
    lists: ListScalar[_DataTypeT],
    index: ScalarLike,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
) -> _DataTypeT: ...
def list_element(*args, **kwargs):
    """
    Compute elements using of nested list values using an index.

    `lists` must have a list-like type.
    For each value in each list of `lists`, the element at `index`
    is emitted. Null values emit a null in the output.

    Parameters
    ----------
    lists : Array-like or scalar-like
        Argument to compute function.
    index : Array-like or scalar-like
        Argument to compute function.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

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
    lists: ArrayOrChunkedArray[ListScalar[Any]],
    /,
    recursive: bool = False,
    *,
    options: ListFlattenOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.ListArray[Any]: ...
def list_flatten(*args, **kwargs):
    """
    Flatten list values.

    `lists` must have a list-like type (lists, list-views, and
    fixed-size lists).
    Return an array with the top list level flattened unless
    `recursive` is set to true in ListFlattenOptions. When that
    is that case, flattening happens recursively until a non-list
    array is formed.

    Null list values do not emit anything to the output.

    Parameters
    ----------
    lists : Array-like
        Argument to compute function.
    recursive : bool, default False
        When True, the list array is flattened recursively until an array
        of non-list values is formed.
    options : pyarrow.compute.ListFlattenOptions, optional
        Alternative way of passing options.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

@overload
def list_parent_indices(
    lists: Expression, /, *, memory_pool: lib.MemoryPool | None = None
) -> Expression: ...
@overload
def list_parent_indices(
    lists: ArrayOrChunkedArray[Any], /, *, memory_pool: lib.MemoryPool | None = None
) -> lib.Int64Array: ...
def list_parent_indices(*args, **kwargs):
    """
    Compute parent indices of nested list values.

    `lists` must have a list-like or list-view type.
    For each value in each list of `lists`, the top-level list index
    is emitted.

    Parameters
    ----------
    lists : Array-like or scalar-like
        Argument to compute function.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

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
    lists: ArrayOrChunkedArray[Any],
    /,
    start: int,
    stop: int | None = None,
    step: int = 1,
    return_fixed_size_list: bool | None = None,
    *,
    options: ListSliceOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
) -> lib.ListArray[Any]: ...
def list_slice(*args, **kwargs):
    """
    Compute slice of list-like array.

    `lists` must have a list-like type.
    For each list element, compute a slice, returning a new list array.
    A variable or fixed size list array is returned, depending on options.

    Parameters
    ----------
    lists : Array-like or scalar-like
        Argument to compute function.
    start : int
        Index to start slicing inner list elements (inclusive).
    stop : Optional[int], default None
        If given, index to stop slicing at (exclusive).
        If not given, slicing will stop at the end. (NotImplemented)
    step : int, default 1
        Slice step.
    return_fixed_size_list : Optional[bool], default None
        Whether to return a FixedSizeListArray. If true _and_ stop is after
        a list element's length, nulls will be appended to create the
        requested slice size. The default of `None` will return the same
        type which was passed in.
    options : pyarrow.compute.ListSliceOptions, optional
        Alternative way of passing options.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

def map_lookup(
    container,
    /,
    query_key,
    occurrence: str,
    *,
    options: MapLookupOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
):
    """
    Find the items corresponding to a given key in a Map.

    For a given query key (passed via MapLookupOptions), extract
    either the FIRST, LAST or ALL items from a Map that have
    matching keys.

    Parameters
    ----------
    container : Array-like or scalar-like
        Argument to compute function.
    query_key : Scalar or Object can be converted to Scalar
        The key to search for.
    occurrence : str
        The occurrence(s) to return from the Map
        Accepted values are "first", "last", or "all".
    options : pyarrow.compute.MapLookupOptions, optional
        Alternative way of passing options.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

def struct_field(
    values,
    /,
    indices,
    *,
    options: StructFieldOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
):
    """
    Extract children of a struct or union by index.

    Given a list of indices (passed via StructFieldOptions), extract
    the child array or scalar with the given child index, recursively.

    For union inputs, nulls are emitted for union values that reference
    a different child than specified. Also, the indices are always
    in physical order, not logical type codes - for example, the first
    child is always index 0.

    An empty list of indices returns the argument unchanged.

    Parameters
    ----------
    values : Array-like or scalar-like
        Argument to compute function.
    indices : List[str], List[bytes], List[int], Expression, bytes, str, or int
        List of indices for chained field lookup, for example `[4, 1]`
        will look up the second nested field in the fifth outer field.
    options : pyarrow.compute.StructFieldOptions, optional
        Alternative way of passing options.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

def fill_null_backward(values, /, *, memory_pool: lib.MemoryPool | None = None):
    """
    Carry non-null values backward to fill null slots.

    Given an array, propagate next valid observation backward to previous valid
    or nothing if all next values are null.

    Parameters
    ----------
    values : Array-like
        Argument to compute function.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

def fill_null_forward(values, /, *, memory_pool: lib.MemoryPool | None = None):
    """
    Carry non-null values forward to fill null slots.

    Given an array, propagate last valid observation forward to next valid
    or nothing if all previous values are null.

    Parameters
    ----------
    values : Array-like
        Argument to compute function.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

def replace_with_mask(
    values,
    mask: list[bool] | list[bool | None] | BooleanArray,
    replacements,
    /,
    *,
    memory_pool: lib.MemoryPool | None = None,
):
    """
    Replace items selected with a mask.

    Given an array and a boolean mask (either scalar or of equal length),
    along with replacement values (either scalar or array),
    each element of the array for which the corresponding mask element is
    true will be replaced by the next value from the replacements,
    or with null if the mask is null.
    Hence, for replacement arrays, len(replacements) == sum(mask == true).

    Parameters
    ----------
    values : Array-like
        Argument to compute function.
    mask : Array-like
        Argument to compute function.
    replacements : Array-like
        Argument to compute function.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

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
def pairwise_diff(*args, **kwargs):
    """
    Compute first order difference of an array.

    Computes the first order difference of an array, It internally calls
    the scalar function "subtract" to compute
     differences, so its
    behavior and supported types are the same as
    "subtract". The period can be specified in :struct:`PairwiseOptions`.

    Results will wrap around on integer overflow. Use function
    "pairwise_diff_checked" if you want overflow to return an error.

    Parameters
    ----------
    input : Array-like
        Argument to compute function.
    period : int, default 1
        Period for applying the period function.
    options : pyarrow.compute.PairwiseOptions, optional
        Alternative way of passing options.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the default memory pool.
    """

pairwise_diff_checked = _clone_signature(pairwise_diff)
"""
Compute first order difference of an array.

Computes the first order difference of an array, It internally calls
the scalar function "subtract_checked" (or the checked variant) to compute
differences, so its behavior and supported types are the same as
"subtract_checked". The period can be specified in :struct:`PairwiseOptions`.

This function returns an error on overflow. For a variant that doesn't
fail on overflow, use function "pairwise_diff".

Parameters
----------
input : Array-like
    Argument to compute function.
period : int, default 1
    Period for applying the period function.
options : pyarrow.compute.PairwiseOptions, optional
    Alternative way of passing options.
memory_pool : pyarrow.MemoryPool, optional
    If not passed, will allocate memory from the default memory pool.
"""
