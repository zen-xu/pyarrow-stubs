from typing import (
    Any,
    Callable,
    Iterable,
    Literal,
    Sequence,
    TypeAlias,
    TypedDict,
    overload,
)

from . import lib

_Order: TypeAlias = Literal["ascending", "descending"]
_Placement: TypeAlias = Literal["at_start", "at_end"]

class Kernel(lib._Weakrefable):
    """
    A kernel object.

    Kernels handle the execution of a Function for a certain signature.
    """

class Function(lib._Weakrefable):
    """
    A compute function.

    A function implements a certain logical computation over a range of
    possible input signatures.  Each signature accepts a range of input
    types and is implemented by a given Kernel.

    Functions can be of different kinds:

    * "scalar" functions apply an item-wise computation over all items
      of their inputs.  Each item in the output only depends on the values
      of the inputs at the same position.  Examples: addition, comparisons,
      string predicates...

    * "vector" functions apply a collection-wise computation, such that
      each item in the output may depend on the values of several items
      in each input.  Examples: dictionary encoding, sorting, extracting
      unique values...

    * "scalar_aggregate" functions reduce the dimensionality of the inputs by
      applying a reduction function.  Examples: sum, min_max, mode...

    * "hash_aggregate" functions apply a reduction function to an input
      subdivided by grouping criteria.  They may not be directly called.
      Examples: hash_sum, hash_min_max...

    * "meta" functions dispatch to other functions.
    """
    @property
    def arity(self) -> int:
        """
        The function arity.

        If Ellipsis (i.e. `...`) is returned, the function takes a variable
        number of arguments.
        """
    @property
    def kind(
        self,
    ) -> Literal["scalar", "vector", "scalar_aggregate", "hash_aggregate", "meta"]:
        """
        The function kind.
        """
    @property
    def name(self) -> str:
        """
        The function name.
        """
    @property
    def num_kernels(self) -> int:
        """
        The number of kernels implementing this function.
        """
    def call(
        self,
        args: Iterable,
        options: FunctionOptions | None = None,
        memory_pool: lib.MemoryPool | None = None,
        length: int | None = None,
    ) -> Any:
        """
        Call the function on the given arguments.

        Parameters
        ----------
        args : iterable
            The arguments to pass to the function.  Accepted types depend
            on the specific function.
        options : FunctionOptions, optional
            Options instance for executing this function.  This should have
            the right concrete options type.
        memory_pool : pyarrow.MemoryPool, optional
            If not passed, will allocate memory from the default memory pool.
        length : int, optional
            Batch size for execution, for nullary (no argument) functions. If
            not passed, will be inferred from passed data.
        """

class FunctionOptions(lib._Weakrefable):
    def serialize(self) -> lib.Buffer: ...
    @classmethod
    def deserialize(cls, buf: lib.Buffer) -> FunctionOptions: ...

class FunctionRegistry(lib._Weakrefable):
    def get_function(self, name: str) -> Function:
        """
        Look up a function by name in the registry.

        Parameters
        ----------
        name : str
            The name of the function to lookup
        """

    def list_functions(self) -> list[str]:
        """
        Return all function names in the registry.
        """

class HashAggregateFunction(Function): ...
class HashAggregateKernel(Kernel): ...
class ScalarAggregateFunction(Function): ...
class ScalarAggregateKernel(Kernel): ...
class ScalarFunction(Function): ...
class ScalarKernel(Kernel): ...
class VectorFunction(Function): ...
class VectorKernel(Kernel): ...

# ==================== _compute.pyx Option classes ====================
class ArraySortOptions(FunctionOptions):
    """
    Options for the `array_sort_indices` function.

    Parameters
    ----------
    order : str, default "ascending"
        Which order to sort values in.
        Accepted values are "ascending", "descending".
    null_placement : str, default "at_end"
        Where nulls in the input should be sorted.
        Accepted values are "at_start", "at_end".
    """
    def __init__(
        self,
        order: _Order = "ascending",
        null_placement: _Placement = "at_end",
    ) -> None: ...

class AssumeTimezoneOptions(FunctionOptions):
    """
    Options for the `assume_timezone` function.

    Parameters
    ----------
    timezone : str
        Timezone to assume for the input.
    ambiguous : str, default "raise"
        How to handle timestamps that are ambiguous in the assumed timezone.
        Accepted values are "raise", "earliest", "latest".
    nonexistent : str, default "raise"
        How to handle timestamps that don't exist in the assumed timezone.
        Accepted values are "raise", "earliest", "latest".
    """

    def __init__(
        self,
        timezone: str,
        *,
        ambiguous: Literal["raise", "earliest", "latest"] = "raise",
        nonexistent: Literal["raise", "earliest", "latest"] = "raise",
    ) -> None: ...

class CastOptions(FunctionOptions):
    """
    Options for the `cast` function.

    Parameters
    ----------
    target_type : DataType, optional
        The PyArrow type to cast to.
    allow_int_overflow : bool, default False
        Whether integer overflow is allowed when casting.
    allow_time_truncate : bool, default False
        Whether time precision truncation is allowed when casting.
    allow_time_overflow : bool, default False
        Whether date/time range overflow is allowed when casting.
    allow_decimal_truncate : bool, default False
        Whether decimal precision truncation is allowed when casting.
    allow_float_truncate : bool, default False
        Whether floating-point precision truncation is allowed when casting.
    allow_invalid_utf8 : bool, default False
        Whether producing invalid utf8 data is allowed when casting.
    """

    allow_int_overflow: bool
    allow_time_truncate: bool
    allow_time_overflow: bool
    allow_decimal_truncate: bool
    allow_float_truncate: bool
    allow_invalid_utf8: bool

    def __init__(
        self,
        target_type: lib.DataType | None = None,
        *,
        allow_int_overflow: bool | None = None,
        allow_time_truncate: bool | None = None,
        allow_time_overflow: bool | None = None,
        allow_decimal_truncate: bool | None = None,
        allow_float_truncate: bool | None = None,
        allow_invalid_utf8: bool | None = None,
    ) -> None: ...
    @staticmethod
    def safe(target_type: lib.DataType | None = None) -> CastOptions: ...
    @staticmethod
    def unsafe(target_type: lib.DataType | None = None) -> CastOptions: ...
    def is_safe(self) -> bool: ...

class CountOptions(FunctionOptions):
    """
    Options for the `count` function.

    Parameters
    ----------
    mode : str, default "only_valid"
        Which values to count in the input.
        Accepted values are "only_valid", "only_null", "all".
    """
    def __init__(self, mode: Literal["only_valid", "only_null", "all"] = "only_valid") -> None: ...

class CumulativeOptions(FunctionOptions):
    """
    Options for `cumulative_*` functions.

    - cumulative_sum
    - cumulative_sum_checked
    - cumulative_prod
    - cumulative_prod_checked
    - cumulative_max
    - cumulative_min

    Parameters
    ----------
    start : Scalar, default None
        Starting value for the cumulative operation. If none is given,
        a default value depending on the operation and input type is used.
    skip_nulls : bool, default False
        When false, the first encountered null is propagated.
    """
    def __init__(self, start: lib.Scalar | None = None, *, skip_nulls: bool = False) -> None: ...

class CumulativeSumOptions(FunctionOptions):
    """
    Options for `cumulative_sum` function.

    Parameters
    ----------
    start : Scalar, default None
        Starting value for sum computation
    skip_nulls : bool, default False
        When false, the first encountered null is propagated.
    """
    def __init__(self, start: lib.Scalar | None = None, *, skip_nulls: bool = False) -> None: ...

class DayOfWeekOptions(FunctionOptions):
    """
    Options for the `day_of_week` function.

    Parameters
    ----------
    count_from_zero : bool, default True
        If True, number days from 0, otherwise from 1.
    week_start : int, default 1
        Which day does the week start with (Monday=1, Sunday=7).
        How this value is numbered is unaffected by `count_from_zero`.
    """

    def __init__(self, *, count_from_zero: bool = True, week_start: int = 1) -> None: ...

class DictionaryEncodeOptions(FunctionOptions):
    """
    Options for dictionary encoding.

    Parameters
    ----------
    null_encoding : str, default "mask"
        How to encode nulls in the input.
        Accepted values are "mask" (null inputs emit a null in the indices
        array), "encode" (null inputs emit a non-null index pointing to
        a null value in the dictionary array).
    """
    def __init__(self, null_encoding: Literal["mask", "encode"] = "mask") -> None: ...

class RunEndEncodeOptions(FunctionOptions):
    """
    Options for run-end encoding.

    Parameters
    ----------
    run_end_type : DataType, default pyarrow.int32()
        The data type of the run_ends array.

        Accepted values are pyarrow.{int16(), int32(), int64()}.
    """
    # TODO: default is DataType(int32)
    def __init__(self, run_end_type: lib.DataType = ...) -> None: ...

class ElementWiseAggregateOptions(FunctionOptions):
    """
    Options for element-wise aggregate functions.

    Parameters
    ----------
    skip_nulls : bool, default True
        Whether to skip (ignore) nulls in the input.
        If False, any null in the input forces the output to null.
    """
    def __init__(self, *, skip_nulls: bool = True) -> None: ...

class ExtractRegexOptions(FunctionOptions):
    """
    Options for the `extract_regex` function.

    Parameters
    ----------
    pattern : str
        Regular expression with named capture fields.
    """
    def __init__(self, pattern: str) -> None: ...

class ExtractRegexSpanOptions(FunctionOptions):
    """
    Options for the `extract_regex_span` function.

    Parameters
    ----------
    pattern : str
        Regular expression with named capture fields.
    """
    def __init__(self, pattern: str) -> None: ...

class FilterOptions(FunctionOptions):
    """
    Options for selecting with a boolean filter.

    Parameters
    ----------
    null_selection_behavior : str, default "drop"
        How to handle nulls in the selection filter.
        Accepted values are "drop", "emit_null".
    """

    def __init__(self, null_selection_behavior: Literal["drop", "emit_null"] = "drop") -> None: ...

class IndexOptions(FunctionOptions):
    """
    Options for the `index` function.

    Parameters
    ----------
    value : Scalar
        The value to search for.
    """
    def __init__(self, value: lib.Scalar) -> None: ...

class JoinOptions(FunctionOptions):
    """
    Options for the `binary_join_element_wise` function.

    Parameters
    ----------
    null_handling : str, default "emit_null"
        How to handle null values in the inputs.
        Accepted values are "emit_null", "skip", "replace".
    null_replacement : str, default ""
        Replacement string to emit for null inputs if `null_handling`
        is "replace".
    """
    @overload
    def __init__(self, null_handling: Literal["emit_null", "skip"] = "emit_null") -> None: ...
    @overload
    def __init__(self, null_handling: Literal["replace"], null_replacement: str = "") -> None: ...

class ListSliceOptions(FunctionOptions):
    """
    Options for list array slicing.

    Parameters
    ----------
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
    """
    def __init__(
        self,
        start: int,
        stop: int | None = None,
        step: int = 1,
        return_fixed_size_list: bool | None = None,
    ) -> None: ...

class ListFlattenOptions(FunctionOptions):
    """
    Options for `list_flatten` function

    Parameters
    ----------
    recursive : bool, default False
        When True, the list array is flattened recursively until an array
        of non-list values is formed.
    """
    def __init__(self, recursive: bool = False) -> None: ...

class MakeStructOptions(FunctionOptions):
    """
    Options for the `make_struct` function.

    Parameters
    ----------
    field_names : sequence of str
        Names of the struct fields to create.
    field_nullability : sequence of bool, optional
        Nullability information for each struct field.
        If omitted, all fields are nullable.
    field_metadata : sequence of KeyValueMetadata, optional
        Metadata for each struct field.
    """
    def __init__(
        self,
        field_names: Sequence[str] = (),
        *,
        field_nullability: Sequence[bool] | None = None,
        field_metadata: Sequence[lib.KeyValueMetadata] | None = None,
    ) -> None: ...

class MapLookupOptions(FunctionOptions):
    """
    Options for the `map_lookup` function.

    Parameters
    ----------
    query_key : Scalar or Object can be converted to Scalar
        The key to search for.
    occurrence : str
        The occurrence(s) to return from the Map
        Accepted values are "first", "last", or "all".
    """
    # TODO: query_key: Scalar or Object can be converted to Scalar
    def __init__(
        self, query_key: lib.Scalar, occurrence: Literal["first", "last", "all"]
    ) -> None: ...

class MatchSubstringOptions(FunctionOptions):
    """
    Options for looking for a substring.

    Parameters
    ----------
    pattern : str
        Substring pattern to look for inside input values.
    ignore_case : bool, default False
        Whether to perform a case-insensitive match.
    """

    def __init__(self, pattern: str, *, ignore_case: bool = False) -> None: ...

class ModeOptions(FunctionOptions):
    """
    Options for the `mode` function.

    Parameters
    ----------
    n : int, default 1
        Number of distinct most-common values to return.
    skip_nulls : bool, default True
        Whether to skip (ignore) nulls in the input.
        If False, any null in the input forces the output to null.
    min_count : int, default 0
        Minimum number of non-null values in the input.  If the number
        of non-null values is below `min_count`, the output is null.
    """
    def __init__(self, n: int = 1, *, skip_nulls: bool = True, min_count: int = 0) -> None: ...

class NullOptions(FunctionOptions):
    """
    Options for the `is_null` function.

    Parameters
    ----------
    nan_is_null : bool, default False
        Whether floating-point NaN values are considered null.
    """
    def __init__(self, *, nan_is_null: bool = False) -> None: ...

class PadOptions(FunctionOptions):
    """
    Options for padding strings.

    Parameters
    ----------
    width : int
        Desired string length.
    padding : str, default " "
        What to pad the string with. Should be one byte or codepoint.
    lean_left_on_odd_padding : bool, default True
        What to do if there is an odd number of padding characters (in case
        of centered padding). Defaults to aligning on the left (i.e. adding
        the extra padding character on the right).
    """
    def __init__(
        self, width: int, padding: str = " ", lean_left_on_odd_padding: bool = True
    ) -> None: ...

class PairwiseOptions(FunctionOptions):
    """
    Options for `pairwise` functions.

    Parameters
    ----------
    period : int, default 1
        Period for applying the period function.
    """
    def __init__(self, period: int = 1) -> None: ...

class PartitionNthOptions(FunctionOptions):
    """
    Options for the `partition_nth_indices` function.

    Parameters
    ----------
    pivot : int
        Index into the equivalent sorted array of the pivot element.
    null_placement : str, default "at_end"
        Where nulls in the input should be partitioned.
        Accepted values are "at_start", "at_end".
    """
    def __init__(self, pivot: int, *, null_placement: _Placement = "at_end") -> None: ...

class WinsorizeOptions(FunctionOptions):
    """
    Options for the `winsorize` function.

    Parameters
    ----------
    lower_limit : float, between 0 and 1
        The quantile below which all values are replaced with the quantile's value.
    upper_limit : float, between 0 and 1
        The quantile above which all values are replaced with the quantile's value.
    """
    def __init__(self, lower_limit: float, upper_limit: float) -> None: ...

class QuantileOptions(FunctionOptions):
    """
    Options for the `quantile` function.

    Parameters
    ----------
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
    """
    def __init__(
        self,
        q: float | Sequence[float],
        *,
        interpolation: Literal["linear", "lower", "higher", "nearest", "midpoint"] = "linear",
        skip_nulls: bool = True,
        min_count: int = 0,
    ) -> None: ...

class RandomOptions(FunctionOptions):
    """
    Options for random generation.

    Parameters
    ----------
    initializer : int or str
        How to initialize the underlying random generator.
        If an integer is given, it is used as a seed.
        If "system" is given, the random generator is initialized with
        a system-specific source of (hopefully true) randomness.
        Other values are invalid.
    """
    def __init__(self, *, initializer: int | Literal["system"] = "system") -> None: ...

class RankOptions(FunctionOptions):
    """
    Options for the `rank` function.

    Parameters
    ----------
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
    """
    def __init__(
        self,
        sort_keys: _Order | Sequence[tuple[str, _Order]] = "ascending",
        *,
        null_placement: _Placement = "at_end",
        tiebreaker: Literal["min", "max", "first", "dense"] = "first",
    ) -> None: ...

class RankQuantileOptions(FunctionOptions):
    """
    Options for the `rank_quantile` function.

    Parameters
    ----------
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
    """

    def __init__(
        self,
        sort_keys: _Order | Sequence[tuple[str, _Order]] = "ascending",
        *,
        null_placement: _Placement = "at_end",
    ) -> None: ...

class PivotWiderOptions(FunctionOptions):
    """
    Options for the `pivot_wider` function.

    Parameters
    ----------
    key_names : sequence of str
        The pivot key names expected in the pivot key column.
        For each entry in `key_names`, a column with the same name is emitted
        in the struct output.
    unexpected_key_behavior : str, default "ignore"
        The behavior when pivot keys not in `key_names` are encountered.
        Accepted values are "ignore", "raise".
        If "ignore", unexpected keys are silently ignored.
        If "raise", unexpected keys raise a KeyError.
    """
    def __init__(
        self,
        key_names: Sequence[str],
        *,
        unexpected_key_behavior: Literal["ignore", "raise"] = "ignore",
    ) -> None: ...

class ReplaceSliceOptions(FunctionOptions):
    """
    Options for replacing slices.

    Parameters
    ----------
    start : int
        Index to start slicing at (inclusive).
    stop : int
        Index to stop slicing at (exclusive).
    replacement : str
        What to replace the slice with.
    """
    def __init__(self, start: int, stop: int, replacement: str) -> None: ...

class ReplaceSubstringOptions(FunctionOptions):
    """
    Options for replacing matched substrings.

    Parameters
    ----------
    pattern : str
        Substring pattern to look for inside input values.
    replacement : str
        What to replace the pattern with.
    max_replacements : int or None, default None
        The maximum number of strings to replace in each
        input value (unlimited if None).
    """
    def __init__(
        self, pattern: str, replacement: str, *, max_replacements: int | None = None
    ) -> None: ...

_RoundMode: TypeAlias = Literal[
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
]

class RoundBinaryOptions(FunctionOptions):
    """
    Options for rounding numbers when ndigits is provided by a second array

    Parameters
    ----------
    round_mode : str, default "half_to_even"
        Rounding and tie-breaking mode.
        Accepted values are "down", "up", "towards_zero", "towards_infinity",
        "half_down", "half_up", "half_towards_zero", "half_towards_infinity",
        "half_to_even", "half_to_odd".
    """
    def __init__(
        self,
        round_mode: _RoundMode = "half_to_even",
    ) -> None: ...

class RoundOptions(FunctionOptions):
    """
    Options for rounding numbers.

    Parameters
    ----------
    ndigits : int, default 0
        Number of fractional digits to round to.
    round_mode : str, default "half_to_even"
        Rounding and tie-breaking mode.
        Accepted values are "down", "up", "towards_zero", "towards_infinity",
        "half_down", "half_up", "half_towards_zero", "half_towards_infinity",
        "half_to_even", "half_to_odd".
    """
    def __init__(
        self,
        ndigits: int = 0,
        round_mode: _RoundMode = "half_to_even",
    ) -> None: ...

_DateTimeUint: TypeAlias = Literal[
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
]

class RoundTemporalOptions(FunctionOptions):
    """
    Options for rounding temporal values.

    Parameters
    ----------
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
    """
    def __init__(
        self,
        multiple: int = 1,
        unit: _DateTimeUint = "day",
        *,
        week_starts_monday: bool = True,
        ceil_is_strictly_greater: bool = False,
        calendar_based_origin: bool = False,
    ) -> None: ...

class RoundToMultipleOptions(FunctionOptions):
    """
    Options for rounding numbers to a multiple.

    Parameters
    ----------
    multiple : numeric scalar, default 1.0
        Multiple to round to. Should be a scalar of a type compatible
        with the argument to be rounded.
    round_mode : str, default "half_to_even"
        Rounding and tie-breaking mode.
        Accepted values are "down", "up", "towards_zero", "towards_infinity",
        "half_down", "half_up", "half_towards_zero", "half_towards_infinity",
        "half_to_even", "half_to_odd".
    """
    def __init__(self, multiple: float = 1.0, round_mode: _RoundMode = "half_to_even") -> None: ...

class ScalarAggregateOptions(FunctionOptions):
    """
    Options for scalar aggregations.

    Parameters
    ----------
    skip_nulls : bool, default True
        Whether to skip (ignore) nulls in the input.
        If False, any null in the input forces the output to null.
    min_count : int, default 1
        Minimum number of non-null values in the input.  If the number
        of non-null values is below `min_count`, the output is null.
    """
    def __init__(self, *, skip_nulls: bool = True, min_count: int = 1) -> None: ...

class SelectKOptions(FunctionOptions):
    """
    Options for top/bottom k-selection.

    Parameters
    ----------
    k : int
        Number of leading values to select in sorted order
        (i.e. the largest values if sort order is "descending",
        the smallest otherwise).
    sort_keys : sequence of (name, order) tuples
        Names of field/column keys to sort the input on,
        along with the order each field/column is sorted in.
        Accepted values for `order` are "ascending", "descending".
        The field name can be a string column name or expression.
    """

    def __init__(self, k: int, sort_keys: Sequence[tuple[str, _Order]]) -> None: ...

class SetLookupOptions(FunctionOptions):
    """
    Options for the `is_in` and `index_in` functions.

    Parameters
    ----------
    value_set : Array
        Set of values to look for in the input.
    skip_nulls : bool, default False
        If False, nulls in the input are matched in the value_set just
        like regular values.
        If True, nulls in the input always fail matching.
    """
    def __init__(self, value_set: lib.Array, *, skip_nulls: bool = True) -> None: ...

class SliceOptions(FunctionOptions):
    """
    Options for slicing.

    Parameters
    ----------
    start : int
        Index to start slicing at (inclusive).
    stop : int or None, default None
        If given, index to stop slicing at (exclusive).
        If not given, slicing will stop at the end.
    step : int, default 1
        Slice step.
    """

    def __init__(self, start: int, stop: int | None = None, step: int = 1) -> None: ...

class SortOptions(FunctionOptions):
    """
    Options for the `sort_indices` function.

    Parameters
    ----------
    sort_keys : sequence of (name, order) tuples
        Names of field/column keys to sort the input on,
        along with the order each field/column is sorted in.
        Accepted values for `order` are "ascending", "descending".
        The field name can be a string column name or expression.
    null_placement : str, default "at_end"
        Where nulls in input should be sorted, only applying to
        columns/fields mentioned in `sort_keys`.
        Accepted values are "at_start", "at_end".
    """
    def __init__(
        self, sort_keys: Sequence[tuple[str, _Order]], *, null_placement: _Placement = "at_end"
    ) -> None: ...

class SplitOptions(FunctionOptions):
    """
    Options for splitting on whitespace.

    Parameters
    ----------
    max_splits : int or None, default None
        Maximum number of splits for each input value (unlimited if None).
    reverse : bool, default False
        Whether to start splitting from the end of each input value.
        This only has an effect if `max_splits` is not None.
    """

    def __init__(self, *, max_splits: int | None = None, reverse: bool = False) -> None: ...

class SplitPatternOptions(FunctionOptions):
    """
    Options for splitting on a string pattern.

    Parameters
    ----------
    pattern : str
        String pattern to split on.
    max_splits : int or None, default None
        Maximum number of splits for each input value (unlimited if None).
    reverse : bool, default False
        Whether to start splitting from the end of each input value.
        This only has an effect if `max_splits` is not None.
    """
    def __init__(
        self, pattern: str, *, max_splits: int | None = None, reverse: bool = False
    ) -> None: ...

class StrftimeOptions(FunctionOptions):
    """
    Options for the `strftime` function.

    Parameters
    ----------
    format : str, default "%Y-%m-%dT%H:%M:%S"
        Pattern for formatting input values.
    locale : str, default "C"
        Locale to use for locale-specific format specifiers.
    """
    def __init__(self, format: str = "%Y-%m-%dT%H:%M:%S", locale: str = "C") -> None: ...

class StrptimeOptions(FunctionOptions):
    """
    Options for the `strptime` function.

    Parameters
    ----------
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
    """
    def __init__(
        self, format: str, unit: Literal["s", "ms", "us", "ns"], error_is_null: bool = False
    ) -> None: ...

class StructFieldOptions(FunctionOptions):
    """
    Options for the `struct_field` function.

    Parameters
    ----------
    indices : List[str], List[bytes], List[int], Expression, bytes, str, or int
        List of indices for chained field lookup, for example `[4, 1]`
        will look up the second nested field in the fifth outer field.
    """
    def __init__(
        self, indices: list[str] | list[bytes] | list[int] | Expression | bytes | str | int
    ) -> None: ...

class TakeOptions(FunctionOptions):
    """
    Options for the `take` and `array_take` functions.

    Parameters
    ----------
    boundscheck : boolean, default True
        Whether to check indices are within bounds. If False and an
        index is out of bounds, behavior is undefined (the process
        may crash).
    """
    def __init__(self, boundscheck: bool = True) -> None: ...

class TDigestOptions(FunctionOptions):
    """
    Options for the `tdigest` function.

    Parameters
    ----------
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
    """
    def __init__(
        self,
        q: float | Sequence[float] = 0.5,
        *,
        delta: int = 100,
        buffer_size: int = 500,
        skip_nulls: bool = True,
        min_count: int = 0,
    ) -> None: ...

class TrimOptions(FunctionOptions):
    """
    Options for trimming characters from strings.

    Parameters
    ----------
    characters : str
        Individual characters to be trimmed from the string.
    """
    def __init__(self, characters: str) -> None: ...

class Utf8NormalizeOptions(FunctionOptions):
    """
    Options for the `utf8_normalize` function.

    Parameters
    ----------
    form : str
        Unicode normalization form.
        Accepted values are "NFC", "NFKC", "NFD", NFKD".
    """

    def __init__(self, form: Literal["NFC", "NFKC", "NFD", "NFKD"]) -> None: ...

class VarianceOptions(FunctionOptions):
    """
    Options for the `variance` and `stddev` functions.

    Parameters
    ----------
    ddof : int, default 0
        Number of degrees of freedom.
    skip_nulls : bool, default True
        Whether to skip (ignore) nulls in the input.
        If False, any null in the input forces the output to null.
    min_count : int, default 0
        Minimum number of non-null values in the input.  If the number
        of non-null values is below `min_count`, the output is null.
    """
    def __init__(self, *, ddof: int = 0, skip_nulls: bool = True, min_count: int = 0) -> None: ...

class SkewOptions(FunctionOptions):
    """
    Options for the `skew` and `kurtosis` functions.

    Parameters
    ----------
    skip_nulls : bool, default True
        Whether to skip (ignore) nulls in the input.
        If False, any null in the input forces the output to null.
    biased : bool, default True
        Whether the calculated value is biased.
        If False, the value computed includes a correction factor to reduce bias.
    min_count : int, default 0
        Minimum number of non-null values in the input.  If the number
        of non-null values is below `min_count`, the output is null.
    """
    def __init__(
        self, *, skip_nulls: bool = True, biased: bool = True, min_count: int = 0
    ) -> None: ...

class WeekOptions(FunctionOptions):
    """
    Options for the `week` function.

    Parameters
    ----------
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
    """
    def __init__(
        self,
        *,
        week_starts_monday: bool = True,
        count_from_zero: bool = False,
        first_week_is_fully_in_year: bool = False,
    ) -> None: ...

# ==================== _compute.pyx Functions ====================

def call_function(
    name: str,
    args: list,
    options: FunctionOptions | None = None,
    memory_pool: lib.MemoryPool | None = None,
    length: int | None = None,
) -> Any:
    """
    Call a named function.

    The function is looked up in the global registry
    (as returned by `function_registry()`).

    Parameters
    ----------
    name : str
        The name of the function to call.
    args : list
        The arguments to the function.
    options : optional
        options provided to the function.
    memory_pool : MemoryPool, optional
        memory pool to use for allocations during function execution.
    length : int, optional
        Batch size for execution, for nullary (no argument) functions. If not
        passed, inferred from data.
    """

def function_registry() -> FunctionRegistry: ...
def get_function(name: str) -> Function:
    """
    Get a function by name.

    The function is looked up in the global registry
    (as returned by `function_registry()`).

    Parameters
    ----------
    name : str
        The name of the function to lookup
    """

def list_functions() -> list[str]:
    """
    Return all function names in the global registry.
    """

# ==================== _compute.pyx Udf ====================

def call_tabular_function(
    function_name: str, args: Iterable | None = None, func_registry: FunctionRegistry | None = None
) -> lib.RecordBatchReader:
    """
    Get a record batch iterator from a tabular function.

    Parameters
    ----------
    function_name : str
        Name of the function.
    args : iterable
        The arguments to pass to the function.  Accepted types depend
        on the specific function.  Currently, only an empty args is supported.
    func_registry : FunctionRegistry
        Optional function registry to use instead of the default global one.
    """

class _FunctionDoc(TypedDict):
    summary: str
    description: str

def register_scalar_function(
    func: Callable,
    function_name: str,
    function_doc: _FunctionDoc,
    in_types: dict[str, lib.DataType],
    out_type: lib.DataType,
    func_registry: FunctionRegistry | None = None,
) -> None:
    """
    Register a user-defined scalar function.

    This API is EXPERIMENTAL.

    A scalar function is a function that executes elementwise
    operations on arrays or scalars, i.e. a scalar function must
    be computed row-by-row with no state where each output row
    is computed only from its corresponding input row.
    In other words, all argument arrays have the same length,
    and the output array is of the same length as the arguments.
    Scalar functions are the only functions allowed in query engine
    expressions.

    Parameters
    ----------
    func : callable
        A callable implementing the user-defined function.
        The first argument is the context argument of type
        UdfContext.
        Then, it must take arguments equal to the number of
        in_types defined. It must return an Array or Scalar
        matching the out_type. It must return a Scalar if
        all arguments are scalar, else it must return an Array.

        To define a varargs function, pass a callable that takes
        *args. The last in_type will be the type of all varargs
        arguments.
    function_name : str
        Name of the function. There should only be one function
        registered with this name in the function registry.
    function_doc : dict
        A dictionary object with keys "summary" (str),
        and "description" (str).
    in_types : Dict[str, DataType]
        A dictionary mapping function argument names to
        their respective DataType.
        The argument names will be used to generate
        documentation for the function. The number of
        arguments specified here determines the function
        arity.
    out_type : DataType
        Output type of the function.
    func_registry : FunctionRegistry
        Optional function registry to use instead of the default global one.

    Examples
    --------
    >>> import pyarrow as pa
    >>> import pyarrow.compute as pc
    >>>
    >>> func_doc = {}
    >>> func_doc["summary"] = "simple udf"
    >>> func_doc["description"] = "add a constant to a scalar"
    >>>
    >>> def add_constant(ctx, array):
    ...     return pc.add(array, 1, memory_pool=ctx.memory_pool)
    >>>
    >>> func_name = "py_add_func"
    >>> in_types = {"array": pa.int64()}
    >>> out_type = pa.int64()
    >>> pc.register_scalar_function(add_constant, func_name, func_doc, in_types, out_type)
    >>>
    >>> func = pc.get_function(func_name)
    >>> func.name
    'py_add_func'
    >>> answer = pc.call_function(func_name, [pa.array([20])])
    >>> answer
    <pyarrow.lib.Int64Array object at ...>
    [
      21
    ]
    """

def register_tabular_function(
    func: Callable,
    function_name: str,
    function_doc: _FunctionDoc,
    in_types: dict[str, lib.DataType],
    out_type: lib.DataType,
    func_registry: FunctionRegistry | None = None,
) -> None:
    """
    Register a user-defined tabular function.

    This API is EXPERIMENTAL.

    A tabular function is one accepting a context argument of type
    UdfContext and returning a generator of struct arrays.
    The in_types argument must be empty and the out_type argument
    specifies a schema. Each struct array must have field types
    corresponding to the schema.

    Parameters
    ----------
    func : callable
        A callable implementing the user-defined function.
        The only argument is the context argument of type
        UdfContext. It must return a callable that
        returns on each invocation a StructArray matching
        the out_type, where an empty array indicates end.
    function_name : str
        Name of the function. There should only be one function
        registered with this name in the function registry.
    function_doc : dict
        A dictionary object with keys "summary" (str),
        and "description" (str).
    in_types : Dict[str, DataType]
        Must be an empty dictionary (reserved for future use).
    out_type : Union[Schema, DataType]
        Schema of the function's output, or a corresponding flat struct type.
    func_registry : FunctionRegistry
        Optional function registry to use instead of the default global one.
    """

def register_aggregate_function(
    func: Callable,
    function_name: str,
    function_doc: _FunctionDoc,
    in_types: dict[str, lib.DataType],
    out_type: lib.DataType,
    func_registry: FunctionRegistry | None = None,
) -> None:
    """
    Register a user-defined non-decomposable aggregate function.

    This API is EXPERIMENTAL.

    A non-decomposable aggregation function is a function that executes
    aggregate operations on the whole data that it is aggregating.
    In other words, non-decomposable aggregate function cannot be
    split into consume/merge/finalize steps.

    This is often used with ordered or segmented aggregation where groups
    can be emit before accumulating all of the input data.

    Note that currently the size of any input column cannot exceed 2 GB
    for a single segment (all groups combined).

    Parameters
    ----------
    func : callable
        A callable implementing the user-defined function.
        The first argument is the context argument of type
        UdfContext.
        Then, it must take arguments equal to the number of
        in_types defined. It must return a Scalar matching the
        out_type.
        To define a varargs function, pass a callable that takes
        *args. The in_type needs to match in type of inputs when
        the function gets called.
    function_name : str
        Name of the function. This name must be unique, i.e.,
        there should only be one function registered with
        this name in the function registry.
    function_doc : dict
        A dictionary object with keys "summary" (str),
        and "description" (str).
    in_types : Dict[str, DataType]
        A dictionary mapping function argument names to
        their respective DataType.
        The argument names will be used to generate
        documentation for the function. The number of
        arguments specified here determines the function
        arity.
    out_type : DataType
        Output type of the function.
    func_registry : FunctionRegistry
        Optional function registry to use instead of the default global one.

    Examples
    --------
    >>> import numpy as np
    >>> import pyarrow as pa
    >>> import pyarrow.compute as pc
    >>>
    >>> func_doc = {}
    >>> func_doc["summary"] = "simple median udf"
    >>> func_doc["description"] = "compute median"
    >>>
    >>> def compute_median(ctx, array):
    ...     return pa.scalar(np.median(array))
    >>>
    >>> func_name = "py_compute_median"
    >>> in_types = {"array": pa.int64()}
    >>> out_type = pa.float64()
    >>> pc.register_aggregate_function(compute_median, func_name, func_doc, in_types, out_type)
    >>>
    >>> func = pc.get_function(func_name)
    >>> func.name
    'py_compute_median'
    >>> answer = pc.call_function(func_name, [pa.array([20, 40])])
    >>> answer
    <pyarrow.DoubleScalar: 30.0>
    >>> table = pa.table([pa.array([1, 1, 2, 2]), pa.array([10, 20, 30, 40])], names=["k", "v"])
    >>> result = table.group_by("k").aggregate([("v", "py_compute_median")])
    >>> result
    pyarrow.Table
    k: int64
    v_py_compute_median: double
    ----
    k: [[1,2]]
    v_py_compute_median: [[15,35]]
    """

def register_vector_function(
    func: Callable,
    function_name: str,
    function_doc: _FunctionDoc,
    in_types: dict[str, lib.DataType],
    out_type: lib.DataType,
    func_registry: FunctionRegistry | None = None,
) -> None:
    """
    Register a user-defined vector function.

    This API is EXPERIMENTAL.

    A vector function is a function that executes vector
    operations on arrays. Vector function is often used
    when compute doesn't fit other more specific types of
    functions (e.g., scalar and aggregate).

    Parameters
    ----------
    func : callable
        A callable implementing the user-defined function.
        The first argument is the context argument of type
        UdfContext.
        Then, it must take arguments equal to the number of
        in_types defined. It must return an Array or Scalar
        matching the out_type. It must return a Scalar if
        all arguments are scalar, else it must return an Array.

        To define a varargs function, pass a callable that takes
        *args. The last in_type will be the type of all varargs
        arguments.
    function_name : str
        Name of the function. There should only be one function
        registered with this name in the function registry.
    function_doc : dict
        A dictionary object with keys "summary" (str),
        and "description" (str).
    in_types : Dict[str, DataType]
        A dictionary mapping function argument names to
        their respective DataType.
        The argument names will be used to generate
        documentation for the function. The number of
        arguments specified here determines the function
        arity.
    out_type : DataType
        Output type of the function.
    func_registry : FunctionRegistry
        Optional function registry to use instead of the default global one.

    Examples
    --------
    >>> import pyarrow as pa
    >>> import pyarrow.compute as pc
    >>>
    >>> func_doc = {}
    >>> func_doc["summary"] = "percent rank"
    >>> func_doc["description"] = "compute percent rank"
    >>>
    >>> def list_flatten_udf(ctx, x):
    ...     return pc.list_flatten(x)
    >>>
    >>> func_name = "list_flatten_udf"
    >>> in_types = {"array": pa.list_(pa.int64())}
    >>> out_type = pa.int64()
    >>> pc.register_vector_function(list_flatten_udf, func_name, func_doc, in_types, out_type)
    >>>
    >>> answer = pc.call_function(func_name, [pa.array([[1, 2], [3, 4]])])
    >>> answer
    <pyarrow.lib.Int64Array object at ...>
    [
      1,
      2,
      3,
      4
    ]
    """

class UdfContext:
    """
    Per-invocation function context/state.

    This object will always be the first argument to a user-defined
    function. It should not be used outside of a call to the function.
    """

    @property
    def batch_length(self) -> int:
        """
        The common length of all input arguments (int).

        In the case that all arguments are scalars, this value
        is used to pass the "actual length" of the arguments,
        e.g. because the scalar values are encoding a column
        with a constant value.
        """
    @property
    def memory_pool(self) -> lib.MemoryPool:
        """
        A memory pool for allocations (:class:`MemoryPool`).

        This is the memory pool supplied by the user when they invoked
        the function and it should be used in any calls to arrow that the
        UDF makes if that call accepts a memory_pool.
        """

# ==================== _compute.pyx Expression ====================
class Expression(lib._Weakrefable):
    """
    A logical expression to be evaluated against some input.

    To create an expression:

    - Use the factory function ``pyarrow.compute.scalar()`` to create a
      scalar (not necessary when combined, see example below).
    - Use the factory function ``pyarrow.compute.field()`` to reference
      a field (column in table).
    - Compare fields and scalars with ``<``, ``<=``, ``==``, ``>=``, ``>``.
    - Combine expressions using python operators ``&`` (logical and),
      ``|`` (logical or) and ``~`` (logical not).
      Note: python keywords ``and``, ``or`` and ``not`` cannot be used
      to combine expressions.
    - Create expression predicates using Expression methods such as
      ``pyarrow.compute.Expression.isin()``.

    Examples
    --------

    >>> import pyarrow.compute as pc
    >>> (pc.field("a") < pc.scalar(3)) | (pc.field("b") > 7)
    <pyarrow.compute.Expression ((a < 3) or (b > 7))>
    >>> pc.field("a") != 3
    <pyarrow.compute.Expression (a != 3)>
    >>> pc.field("a").isin([1, 2, 3])
    <pyarrow.compute.Expression is_in(a, {value_set=int64:[
      1,
      2,
      3
    ], null_matching_behavior=MATCH})>
    """

    @staticmethod
    def from_substrait(buffer: bytes | lib.Buffer) -> Expression:
        """
        Deserialize an expression from Substrait

        The serialized message must be an ExtendedExpression message that has
        only a single expression.  The name of the expression and the schema
        the expression was bound to will be ignored.  Use
        pyarrow.substrait.deserialize_expressions if this information is needed
        or if the message might contain multiple expressions.

        Parameters
        ----------
        message : bytes or Buffer or a protobuf Message
            The Substrait message to deserialize

        Returns
        -------
        Expression
            The deserialized expression
        """
    def to_substrait(self, schema: lib.Schema, allow_arrow_extensions: bool = False) -> lib.Buffer:
        """
        Serialize the expression using Substrait

        The expression will be serialized as an ExtendedExpression message that has a
        single expression named "expression"

        Parameters
        ----------
        schema : Schema
            The input schema the expression will be bound to
        allow_arrow_extensions : bool, default False
            If False then only functions that are part of the core Substrait function
            definitions will be allowed.  Set this to True to allow pyarrow-specific functions
            but the result may not be accepted by other compute libraries.

        Returns
        -------
        Buffer
            A buffer containing the serialized Protobuf plan.
        """
    def __invert__(self) -> Expression: ...
    def __and__(self, other) -> Expression: ...
    def __or__(self, other) -> Expression: ...
    def __add__(self, other) -> Expression: ...
    def __mul__(self, other) -> Expression: ...
    def __sub__(self, other) -> Expression: ...
    def __eq__(self, value: object) -> Expression: ...  # type: ignore[override]
    def __ne__(self, value: object) -> Expression: ...  # type: ignore[override]
    def __gt__(self, value: object) -> Expression: ...  # type: ignore[override]
    def __lt__(self, value: object) -> Expression: ...  # type: ignore[override]
    def __ge__(self, value: object) -> Expression: ...  # type: ignore[override]
    def __le__(self, value: object) -> Expression: ...  # type: ignore[override]
    def __truediv__(self, other) -> Expression: ...
    def is_valid(self) -> bool:
        """
        Check whether the expression is not-null (valid).

        This creates a new expression equivalent to calling the
        `is_valid` compute function on this expression.

        Returns
        -------
        is_valid : Expression
        """
    def is_null(self, nan_is_null: bool = False) -> Expression:
        """
        Check whether the expression is null.

        This creates a new expression equivalent to calling the
        `is_null` compute function on this expression.

        Parameters
        ----------
        nan_is_null : boolean, default False
            Whether floating-point NaNs are considered null.

        Returns
        -------
        is_null : Expression
        """
    def is_nan(self) -> Expression:
        """
        Check whether the expression is NaN.

        This creates a new expression equivalent to calling the
        `is_nan` compute function on this expression.

        Returns
        -------
        is_nan : Expression
        """
    def cast(
        self, type: lib.DataType, safe: bool = True, options: CastOptions | None = None
    ) -> Expression:
        """
        Explicitly set or change the expression's data type.

        This creates a new expression equivalent to calling the
        `cast` compute function on this expression.

        Parameters
        ----------
        type : DataType, default None
            Type to cast array to.
        safe : boolean, default True
            Whether to check for conversion errors such as overflow.
        options : CastOptions, default None
            Additional checks pass by CastOptions

        Returns
        -------
        cast : Expression
        """
    def isin(self, values: lib.Array | Iterable) -> Expression:
        """
        Check whether the expression is contained in values.

        This creates a new expression equivalent to calling the
        `is_in` compute function on this expression.

        Parameters
        ----------
        values : Array or iterable
            The values to check for.

        Returns
        -------
        isin : Expression
            A new expression that, when evaluated, checks whether
            this expression's value is contained in `values`.
        """

# ==================== _compute.py ====================
