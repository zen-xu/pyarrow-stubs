import datetime as dt
import sys

from decimal import Decimal

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self
if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias
from typing import (
    Any,
    Collection,
    Generator,
    Generic,
    Iterable,
    Iterator,
    Literal,
    Mapping,
    Sequence,
    TypeVar,
    overload,
)

import numpy as np
import pandas as pd

from numpy.typing import NDArray
from pyarrow._compute import (
    CastOptions,
    CountOptions,
    FunctionOptions,
    ScalarAggregateOptions,
    TDigestOptions,
    VarianceOptions,
)
from pyarrow._stubs_typing import (
    Indices,
    Mask,
    NullEncoding,
    NullSelectionBehavior,
    Order,
    SupportArrowArray,
    SupportArrowDeviceArray,
    SupportArrowStream,
)
from pyarrow.compute import ArrayOrChunkedArray, Expression
from pyarrow.interchange.dataframe import _PyArrowDataFrame
from pyarrow.lib import Device, MemoryManager, MemoryPool, MonthDayNano, Schema
from pyarrow.lib import Field as _Field

from . import array, scalar, types
from .array import Array, NullableCollection, StructArray, _CastAs, _PandasConvertible
from .device import DeviceAllocationType
from .io import Buffer
from .ipc import RecordBatchReader
from .scalar import Int64Scalar, Scalar
from .tensor import Tensor
from .types import DataType, _AsPyType, _BasicDataType, _DataTypeT

Field: TypeAlias = _Field[DataType]
_ScalarT = TypeVar("_ScalarT", bound=Scalar)
_Scalar_co = TypeVar("_Scalar_co", bound=Scalar, covariant=True)

_Aggregation: TypeAlias = Literal[
    "all",
    "any",
    "approximate_median",
    "count",
    "count_all",
    "count_distinct",
    "distinct",
    "first",
    "first_last",
    "last",
    "list",
    "max",
    "mean",
    "min",
    "min_max",
    "one",
    "product",
    "stddev",
    "sum",
    "tdigest",
    "variance",
]
_AggregationPrefixed: TypeAlias = Literal[
    "hash_all",
    "hash_any",
    "hash_approximate_median",
    "hash_count",
    "hash_count_all",
    "hash_count_distinct",
    "hash_distinct",
    "hash_first",
    "hash_first_last",
    "hash_last",
    "hash_list",
    "hash_max",
    "hash_mean",
    "hash_min",
    "hash_min_max",
    "hash_one",
    "hash_product",
    "hash_stddev",
    "hash_sum",
    "hash_tdigest",
    "hash_variance",
]
Aggregation: TypeAlias = _Aggregation | _AggregationPrefixed
AggregateOptions: TypeAlias = (
    ScalarAggregateOptions | CountOptions | TDigestOptions | VarianceOptions | FunctionOptions
)

UnarySelector: TypeAlias = str
NullarySelector: TypeAlias = tuple[()]
NarySelector: TypeAlias = list[str] | tuple[str, ...]
ColumnSelector: TypeAlias = UnarySelector | NullarySelector | NarySelector
MetaData: TypeAlias = dict[bytes, bytes] | dict[bytes, str] | dict[str, str] | dict[str, bytes]

class ChunkedArray(_PandasConvertible[pd.Series], Generic[_Scalar_co]):
    """
    An array-like composed from a (possibly empty) collection of pyarrow.Arrays

    Warnings
    --------
    Do not call this class's constructor directly.

    Examples
    --------
    To construct a ChunkedArray object use :func:`pyarrow.chunked_array`:

    >>> import pyarrow as pa
    >>> pa.chunked_array([], type=pa.int8())
    <pyarrow.lib.ChunkedArray object at ...>
    [
    ...
    ]

    >>> pa.chunked_array([[2, 2, 4], [4, 5, 100]])
    <pyarrow.lib.ChunkedArray object at ...>
    [
      [
        2,
        2,
        4
      ],
      [
        4,
        5,
        100
      ]
    ]
    >>> isinstance(pa.chunked_array([[2, 2, 4], [4, 5, 100]]), pa.ChunkedArray)
    True
    """

    @property
    def data(self) -> Self: ...
    @property
    def type(self: ChunkedArray[Scalar[_DataTypeT]]) -> _DataTypeT:
        """
        Return data type of a ChunkedArray.

        Examples
        --------
        >>> import pyarrow as pa
        >>> n_legs = pa.chunked_array([[2, 2, 4], [4, 5, 100]])
        >>> n_legs.type
        DataType(int64)
        """
    def length(self) -> int:
        """
        Return length of a ChunkedArray.

        Examples
        --------
        >>> import pyarrow as pa
        >>> n_legs = pa.chunked_array([[2, 2, 4], [4, 5, 100]])
        >>> n_legs.length()
        6
        """
    __len__ = length
    def to_string(
        self,
        *,
        indent: int = 0,
        window: int = 5,
        container_window: int = 2,
        skip_new_lines: bool = False,
    ) -> str:
        """
        Render a "pretty-printed" string representation of the ChunkedArray

        Parameters
        ----------
        indent : int
            How much to indent right the content of the array,
            by default ``0``.
        window : int
            How many items to preview within each chunk at the begin and end
            of the chunk when the chunk is bigger than the window.
            The other elements will be ellipsed.
        container_window : int
            How many chunks to preview at the begin and end
            of the array when the array is bigger than the window.
            The other elements will be ellipsed.
            This setting also applies to list columns.
        skip_new_lines : bool
            If the array should be rendered as a single line of text
            or if each element should be on its own line.

        Examples
        --------
        >>> import pyarrow as pa
        >>> n_legs = pa.chunked_array([[2, 2, 4], [4, 5, 100]])
        >>> n_legs.to_string(skip_new_lines=True)
        '[[2,2,4],[4,5,100]]'
        """
    format = to_string
    def validate(self, *, full: bool = False) -> None:
        """
        Perform validation checks.  An exception is raised if validation fails.

        By default only cheap validation checks are run.  Pass `full=True`
        for thorough validation checks (potentially O(n)).

        Parameters
        ----------
        full : bool, default False
            If True, run expensive checks, otherwise cheap checks only.

        Raises
        ------
        ArrowInvalid
        """
    @property
    def null_count(self) -> int:
        """
        Number of null entries

        Returns
        -------
        int

        Examples
        --------
        >>> import pyarrow as pa
        >>> n_legs = pa.chunked_array([[2, 2, 4], [4, None, 100]])
        >>> n_legs.null_count
        1
        """
    @property
    def nbytes(self) -> int:
        """
        Total number of bytes consumed by the elements of the chunked array.

        In other words, the sum of bytes from all buffer ranges referenced.

        Unlike `get_total_buffer_size` this method will account for array
        offsets.

        If buffers are shared between arrays then the shared
        portion will only be counted multiple times.

        The dictionary of dictionary arrays will always be counted in their
        entirety even if the array only references a portion of the dictionary.

        Examples
        --------
        >>> import pyarrow as pa
        >>> n_legs = pa.chunked_array([[2, 2, 4], [4, None, 100]])
        >>> n_legs.nbytes
        49
        """
    def get_total_buffer_size(self) -> int:
        """
        The sum of bytes in each buffer referenced by the chunked array.

        An array may only reference a portion of a buffer.
        This method will overestimate in this case and return the
        byte size of the entire buffer.

        If a buffer is referenced multiple times then it will
        only be counted once.

        Examples
        --------
        >>> import pyarrow as pa
        >>> n_legs = pa.chunked_array([[2, 2, 4], [4, None, 100]])
        >>> n_legs.get_total_buffer_size()
        49
        """
    def __sizeof__(self) -> int: ...
    @overload
    def __getitem__(self, key: slice) -> Self: ...
    @overload
    def __getitem__(self, key: int) -> _Scalar_co: ...
    def __getitem__(self, key):
        """
        Slice or return value at given index

        Parameters
        ----------
        key : integer or slice
            Slices with step not equal to 1 (or None) will produce a copy
            rather than a zero-copy view

        Returns
        -------
        value : Scalar (index) or ChunkedArray (slice)
        """
    def getitem(self, i: int) -> Scalar: ...
    def is_null(self, *, nan_is_null: bool = False) -> ChunkedArray[scalar.BooleanScalar]:
        """
        Return boolean array indicating the null values.

        Parameters
        ----------
        nan_is_null : bool (optional, default False)
            Whether floating-point NaN values should also be considered null.

        Returns
        -------
        array : boolean Array or ChunkedArray

        Examples
        --------
        >>> import pyarrow as pa
        >>> n_legs = pa.chunked_array([[2, 2, 4], [4, None, 100]])
        >>> n_legs.is_null()
        <pyarrow.lib.ChunkedArray object at ...>
        [
          [
            false,
            false,
            false,
            false,
            true,
            false
          ]
        ]
        """
    def is_nan(self) -> ChunkedArray[scalar.BooleanScalar]:
        """
        Return boolean array indicating the NaN values.

        Examples
        --------
        >>> import pyarrow as pa
        >>> import numpy as np
        >>> arr = pa.chunked_array([[2, np.nan, 4], [4, None, 100]])
        >>> arr.is_nan()
        <pyarrow.lib.ChunkedArray object at ...>
        [
          [
            false,
            true,
            false,
            false,
            null,
            false
          ]
        ]
        """
    def is_valid(self) -> ChunkedArray[scalar.BooleanScalar]:
        """
        Return boolean array indicating the non-null values.

        Examples
        --------
        >>> import pyarrow as pa
        >>> n_legs = pa.chunked_array([[2, 2, 4], [4, None, 100]])
        >>> n_legs.is_valid()
        <pyarrow.lib.ChunkedArray object at ...>
        [
          [
            true,
            true,
            true
          ],
          [
            true,
            false,
            true
          ]
        ]
        """
    def fill_null(self, fill_value: Scalar[_DataTypeT]) -> Self:
        """
        Replace each null element in values with fill_value.

        See :func:`pyarrow.compute.fill_null` for full usage.

        Parameters
        ----------
        fill_value : any
            The replacement value for null entries.

        Returns
        -------
        result : Array or ChunkedArray
            A new array with nulls replaced by the given value.

        Examples
        --------
        >>> import pyarrow as pa
        >>> fill_value = pa.scalar(5, type=pa.int8())
        >>> n_legs = pa.chunked_array([[2, 2, 4], [4, None, 100]])
        >>> n_legs.fill_null(fill_value)
        <pyarrow.lib.ChunkedArray object at ...>
        [
          [
            2,
            2,
            4,
            4,
            5,
            100
          ]
        ]
        """
    def equals(self, other: Self) -> bool:
        """
        Return whether the contents of two chunked arrays are equal.

        Parameters
        ----------
        other : pyarrow.ChunkedArray
            Chunked array to compare against.

        Returns
        -------
        are_equal : bool

        Examples
        --------
        >>> import pyarrow as pa
        >>> n_legs = pa.chunked_array([[2, 2, 4], [4, 5, 100]])
        >>> animals = pa.chunked_array(
        ...     (["Flamingo", "Parrot", "Dog"], ["Horse", "Brittle stars", "Centipede"])
        ... )
        >>> n_legs.equals(n_legs)
        True
        >>> n_legs.equals(animals)
        False
        """
    def to_numpy(self, zero_copy_only: bool = False) -> np.ndarray:
        """
        Return a NumPy copy of this array (experimental).

        Parameters
        ----------
        zero_copy_only : bool, default False
            Introduced for signature consistence with pyarrow.Array.to_numpy.
            This must be False here since NumPy arrays' buffer must be contiguous.

        Returns
        -------
        array : numpy.ndarray

        Examples
        --------
        >>> import pyarrow as pa
        >>> n_legs = pa.chunked_array([[2, 2, 4], [4, 5, 100]])
        >>> n_legs.to_numpy()
        array([  2,   2,   4,   4,   5, 100])
        """
    def __array__(self, dtype: np.dtype | None = None, copy: bool | None = None) -> np.ndarray: ...
    @overload
    def cast(
        self,
        target_type: None = None,
        safe: bool | None = None,
        options: CastOptions | None = None,
    ) -> Self: ...
    @overload
    def cast(
        self, target_type: _CastAs, safe: bool | None = None, options: CastOptions | None = None
    ) -> ChunkedArray[Scalar[_CastAs]]: ...
    def cast(self, *args, **kwargs):
        """
        Cast array values to another data type

        See :func:`pyarrow.compute.cast` for usage.

        Parameters
        ----------
        target_type : DataType, None
            Type to cast array to.
        safe : boolean, default True
            Whether to check for conversion errors such as overflow.
        options : CastOptions, default None
            Additional checks pass by CastOptions

        Returns
        -------
        cast : Array or ChunkedArray

        Examples
        --------
        >>> import pyarrow as pa
        >>> n_legs = pa.chunked_array([[2, 2, 4], [4, 5, 100]])
        >>> n_legs.type
        DataType(int64)

        Change the data type of an array:

        >>> n_legs_seconds = n_legs.cast(pa.duration("s"))
        >>> n_legs_seconds.type
        DurationType(duration[s])
        """
    def dictionary_encode(self, null_encoding: NullEncoding = "mask") -> Self:
        """
        Compute dictionary-encoded representation of array.

        See :func:`pyarrow.compute.dictionary_encode` for full usage.

        Parameters
        ----------
        null_encoding : str, default "mask"
            How to handle null entries.

        Returns
        -------
        encoded : ChunkedArray
            A dictionary-encoded version of this array.

        Examples
        --------
        >>> import pyarrow as pa
        >>> animals = pa.chunked_array(
        ...     (["Flamingo", "Parrot", "Dog"], ["Horse", "Brittle stars", "Centipede"])
        ... )
        >>> animals.dictionary_encode()
        <pyarrow.lib.ChunkedArray object at ...>
        [
        ...
          -- dictionary:
            [
              "Flamingo",
              "Parrot",
              "Dog",
              "Horse",
              "Brittle stars",
              "Centipede"
            ]
          -- indices:
            [
              0,
              1,
              2
            ],
        ...
          -- dictionary:
            [
              "Flamingo",
              "Parrot",
              "Dog",
              "Horse",
              "Brittle stars",
              "Centipede"
            ]
          -- indices:
            [
              3,
              4,
              5
            ]
        ]
        """
    def flatten(self, memory_pool: MemoryPool | None = None) -> list[ChunkedArray[Any]]:
        """
        Flatten this ChunkedArray.  If it has a struct type, the column is
        flattened into one array per struct field.

        Parameters
        ----------
        memory_pool : MemoryPool, default None
            For memory allocations, if required, otherwise use default pool

        Returns
        -------
        result : list of ChunkedArray

        Examples
        --------
        >>> import pyarrow as pa
        >>> n_legs = pa.chunked_array([[2, 2, 4], [4, 5, 100]])
        >>> c_arr = pa.chunked_array(n_legs.value_counts())
        >>> c_arr
        <pyarrow.lib.ChunkedArray object at ...>
        [
          -- is_valid: all not null
          -- child 0 type: int64
            [
              2,
              4,
              5,
              100
            ]
          -- child 1 type: int64
            [
              2,
              2,
              1,
              1
            ]
        ]
        >>> c_arr.flatten()
        [<pyarrow.lib.ChunkedArray object at ...>
        [
          [
            2,
            4,
            5,
            100
          ]
        ], <pyarrow.lib.ChunkedArray object at ...>
        [
          [
            2,
            2,
            1,
            1
          ]
        ]]
        >>> c_arr.type
        StructType(struct<values: int64, counts: int64>)
        >>> n_legs.type
        DataType(int64)
        """
    def combine_chunks(self, memory_pool: MemoryPool | None = None) -> Array[_Scalar_co]:
        """
        Flatten this ChunkedArray into a single non-chunked array.

        Parameters
        ----------
        memory_pool : MemoryPool, default None
            For memory allocations, if required, otherwise use default pool

        Returns
        -------
        result : Array

        Examples
        --------
        >>> import pyarrow as pa
        >>> n_legs = pa.chunked_array([[2, 2, 4], [4, 5, 100]])
        >>> n_legs
        <pyarrow.lib.ChunkedArray object at ...>
        [
          [
            2,
            2,
            4
          ],
          [
            4,
            5,
            100
          ]
        ]
        >>> n_legs.combine_chunks()
        <pyarrow.lib.Int64Array object at ...>
        [
          2,
          2,
          4,
          4,
          5,
          100
        ]
        """
    def unique(self) -> ChunkedArray[_Scalar_co]:
        """
        Compute distinct elements in array

        Returns
        -------
        pyarrow.Array

        Examples
        --------
        >>> import pyarrow as pa
        >>> n_legs = pa.chunked_array([[2, 2, 4], [4, 5, 100]])
        >>> n_legs
        <pyarrow.lib.ChunkedArray object at ...>
        [
          [
            2,
            2,
            4
          ],
          [
            4,
            5,
            100
          ]
        ]
        >>> n_legs.unique()
        <pyarrow.lib.Int64Array object at ...>
        [
          2,
          4,
          5,
          100
        ]
        """
    def value_counts(self) -> StructArray:
        """
        Compute counts of unique elements in array.

        Returns
        -------
        An array of  <input type "Values", int64_t "Counts"> structs

        Examples
        --------
        >>> import pyarrow as pa
        >>> n_legs = pa.chunked_array([[2, 2, 4], [4, 5, 100]])
        >>> n_legs
        <pyarrow.lib.ChunkedArray object at ...>
        [
          [
            2,
            2,
            4
          ],
          [
            4,
            5,
            100
          ]
        ]
        >>> n_legs.value_counts()
        <pyarrow.lib.StructArray object at ...>
        -- is_valid: all not null
        -- child 0 type: int64
          [
            2,
            4,
            5,
            100
          ]
        -- child 1 type: int64
          [
            2,
            2,
            1,
            1
          ]
        """
    def slice(self, offset: int = 0, length: int | None = None) -> Self:
        """
        Compute zero-copy slice of this ChunkedArray

        Parameters
        ----------
        offset : int, default 0
            Offset from start of array to slice
        length : int, default None
            Length of slice (default is until end of batch starting from
            offset)

        Returns
        -------
        sliced : ChunkedArray

        Examples
        --------
        >>> import pyarrow as pa
        >>> n_legs = pa.chunked_array([[2, 2, 4], [4, 5, 100]])
        >>> n_legs
        <pyarrow.lib.ChunkedArray object at ...>
        [
          [
            2,
            2,
            4
          ],
          [
            4,
            5,
            100
          ]
        ]
        >>> n_legs.slice(2, 2)
        <pyarrow.lib.ChunkedArray object at ...>
        [
          [
            4
          ],
          [
            4
          ]
        ]
        """
    def filter(self, mask: Mask, null_selection_behavior: NullSelectionBehavior = "drop") -> Self:
        """
        Select values from the chunked array.

        See :func:`pyarrow.compute.filter` for full usage.

        Parameters
        ----------
        mask : Array or array-like
            The boolean mask to filter the chunked array with.
        null_selection_behavior : str, default "drop"
            How nulls in the mask should be handled.

        Returns
        -------
        filtered : Array or ChunkedArray
            An array of the same type, with only the elements selected by
            the boolean mask.

        Examples
        --------
        >>> import pyarrow as pa
        >>> n_legs = pa.chunked_array([[2, 2, 4], [4, 5, 100]])
        >>> n_legs
        <pyarrow.lib.ChunkedArray object at ...>
        [
          [
            2,
            2,
            4
          ],
          [
            4,
            5,
            100
          ]
        ]
        >>> mask = pa.array([True, False, None, True, False, True])
        >>> n_legs.filter(mask)
        <pyarrow.lib.ChunkedArray object at ...>
        [
          [
            2
          ],
          [
            4,
            100
          ]
        ]
        >>> n_legs.filter(mask, null_selection_behavior="emit_null")
        <pyarrow.lib.ChunkedArray object at ...>
        [
          [
            2,
            null
          ],
          [
            4,
            100
          ]
        ]
        """
    @overload
    def index(
        self: ChunkedArray[Scalar[_BasicDataType[_AsPyType]]],
        value: Scalar[_DataTypeT] | _AsPyType,
        start: int | None = None,
        end: int | None = None,
        *,
        memory_pool: MemoryPool | None = None,
    ) -> Int64Scalar: ...
    @overload
    def index(
        self,
        value: Scalar[_DataTypeT],
        start: int | None = None,
        end: int | None = None,
        *,
        memory_pool: MemoryPool | None = None,
    ) -> Int64Scalar: ...
    def index(self, *args, **kwargs):
        """
        Find the first index of a value.

        See :func:`pyarrow.compute.index` for full usage.

        Parameters
        ----------
        value : Scalar or object
            The value to look for in the array.
        start : int, optional
            The start index where to look for `value`.
        end : int, optional
            The end index where to look for `value`.
        memory_pool : MemoryPool, optional
            A memory pool for potential memory allocations.

        Returns
        -------
        index : Int64Scalar
            The index of the value in the array (-1 if not found).

        Examples
        --------
        >>> import pyarrow as pa
        >>> n_legs = pa.chunked_array([[2, 2, 4], [4, 5, 100]])
        >>> n_legs
        <pyarrow.lib.ChunkedArray object at ...>
        [
          [
            2,
            2,
            4
          ],
          [
            4,
            5,
            100
          ]
        ]
        >>> n_legs.index(4)
        <pyarrow.Int64Scalar: 2>
        >>> n_legs.index(4, start=3)
        <pyarrow.Int64Scalar: 3>
        """
    def take(self, indices: Indices) -> Self:
        """
        Select values from the chunked array.

        See :func:`pyarrow.compute.take` for full usage.

        Parameters
        ----------
        indices : Array or array-like
            The indices in the array whose values will be returned.

        Returns
        -------
        taken : Array or ChunkedArray
            An array with the same datatype, containing the taken values.

        Examples
        --------
        >>> import pyarrow as pa
        >>> n_legs = pa.chunked_array([[2, 2, 4], [4, 5, 100]])
        >>> n_legs
        <pyarrow.lib.ChunkedArray object at ...>
        [
          [
            2,
            2,
            4
          ],
          [
            4,
            5,
            100
          ]
        ]
        >>> n_legs.take([1, 4, 5])
        <pyarrow.lib.ChunkedArray object at ...>
        [
          [
            2,
            5,
            100
          ]
        ]
        """
    def drop_null(self) -> Self:
        """
        Remove missing values from a chunked array.
        See :func:`pyarrow.compute.drop_null` for full description.

        Examples
        --------
        >>> import pyarrow as pa
        >>> n_legs = pa.chunked_array([[2, 2, None], [4, 5, 100]])
        >>> n_legs
        <pyarrow.lib.ChunkedArray object at ...>
        [
          [
            2,
            2,
            null
          ],
          [
            4,
            5,
            100
          ]
        ]
        >>> n_legs.drop_null()
        <pyarrow.lib.ChunkedArray object at ...>
        [
          [
            2,
            2
          ],
          [
            4,
            5,
            100
          ]
        ]
        """
    def sort(self, order: Order = "ascending", **kwargs) -> Self:
        """
        Sort the ChunkedArray

        Parameters
        ----------
        order : str, default "ascending"
            Which order to sort values in.
            Accepted values are "ascending", "descending".
        **kwargs : dict, optional
            Additional sorting options.
            As allowed by :class:`SortOptions`

        Returns
        -------
        result : ChunkedArray
        """
    def unify_dictionaries(self, memory_pool: MemoryPool | None = None) -> Self:
        """
        Unify dictionaries across all chunks.

        This method returns an equivalent chunked array, but where all
        chunks share the same dictionary values.  Dictionary indices are
        transposed accordingly.

        If there are no dictionaries in the chunked array, it is returned
        unchanged.

        Parameters
        ----------
        memory_pool : MemoryPool, default None
            For memory allocations, if required, otherwise use default pool

        Returns
        -------
        result : ChunkedArray

        Examples
        --------
        >>> import pyarrow as pa
        >>> arr_1 = pa.array(["Flamingo", "Parrot", "Dog"]).dictionary_encode()
        >>> arr_2 = pa.array(["Horse", "Brittle stars", "Centipede"]).dictionary_encode()
        >>> c_arr = pa.chunked_array([arr_1, arr_2])
        >>> c_arr
        <pyarrow.lib.ChunkedArray object at ...>
        [
        ...
          -- dictionary:
            [
              "Flamingo",
              "Parrot",
              "Dog"
            ]
          -- indices:
            [
              0,
              1,
              2
            ],
        ...
          -- dictionary:
            [
              "Horse",
              "Brittle stars",
              "Centipede"
            ]
          -- indices:
            [
              0,
              1,
              2
            ]
        ]
        >>> c_arr.unify_dictionaries()
        <pyarrow.lib.ChunkedArray object at ...>
        [
        ...
          -- dictionary:
            [
              "Flamingo",
              "Parrot",
              "Dog",
              "Horse",
              "Brittle stars",
              "Centipede"
            ]
          -- indices:
            [
              0,
              1,
              2
            ],
        ...
          -- dictionary:
            [
              "Flamingo",
              "Parrot",
              "Dog",
              "Horse",
              "Brittle stars",
              "Centipede"
            ]
          -- indices:
            [
              3,
              4,
              5
            ]
        ]
        """
    @property
    def num_chunks(self) -> int:
        """
        Number of underlying chunks.

        Returns
        -------
        int

        Examples
        --------
        >>> import pyarrow as pa
        >>> n_legs = pa.chunked_array([[2, 2, None], [4, 5, 100]])
        >>> n_legs.num_chunks
        2
        """
    def chunk(self, i: int) -> ChunkedArray[_Scalar_co]:
        """
        Select a chunk by its index.

        Parameters
        ----------
        i : int

        Returns
        -------
        pyarrow.Array

        Examples
        --------
        >>> import pyarrow as pa
        >>> n_legs = pa.chunked_array([[2, 2, None], [4, 5, 100]])
        >>> n_legs.chunk(1)
        <pyarrow.lib.Int64Array object at ...>
        [
          4,
          5,
          100
        ]
        """
    @property
    def chunks(self) -> list[Array[_Scalar_co]]:
        """
        Convert to a list of single-chunked arrays.

        Examples
        --------
        >>> import pyarrow as pa
        >>> n_legs = pa.chunked_array([[2, 2, None], [4, 5, 100]])
        >>> n_legs
        <pyarrow.lib.ChunkedArray object at ...>
        [
          [
            2,
            2,
            null
          ],
          [
            4,
            5,
            100
          ]
        ]
        >>> n_legs.chunks
        [<pyarrow.lib.Int64Array object at ...>
        [
          2,
          2,
          null
        ], <pyarrow.lib.Int64Array object at ...>
        [
          4,
          5,
          100
        ]]
        """
    @overload
    def iterchunks(
        self: ChunkedArray[scalar.NullScalar],
    ) -> Generator[array.NullArray, None, None]: ...
    @overload
    def iterchunks(
        self: ChunkedArray[scalar.BooleanScalar],
    ) -> Generator[array.BooleanArray, None, None]: ...
    @overload
    def iterchunks(
        self: ChunkedArray[scalar.UInt8Scalar],
    ) -> Generator[array.UInt8Array, None, None]: ...
    @overload
    def iterchunks(
        self: ChunkedArray[scalar.Int8Scalar],
    ) -> Generator[array.Int8Array, None, None]: ...
    @overload
    def iterchunks(
        self: ChunkedArray[scalar.UInt16Scalar],
    ) -> Generator[array.UInt16Array, None, None]: ...
    @overload
    def iterchunks(
        self: ChunkedArray[scalar.Int16Scalar],
    ) -> Generator[array.Int16Array, None, None]: ...
    @overload
    def iterchunks(
        self: ChunkedArray[scalar.UInt32Scalar],
    ) -> Generator[array.UInt32Array, None, None]: ...
    @overload
    def iterchunks(
        self: ChunkedArray[scalar.Int32Scalar],
    ) -> Generator[array.Int32Array, None, None]:
        """
        Convert to an iterator of ChunkArrays.

        Examples
        --------
        >>> import pyarrow as pa
        >>> n_legs = pa.chunked_array([[2, 2, 4], [4, None, 100]])
        >>> for i in n_legs.iterchunks():
        ...     print(i.null_count)
        0
        1

        """
    @overload
    def iterchunks(
        self: ChunkedArray[scalar.UInt64Scalar],
    ) -> Generator[array.UInt64Array, None, None]: ...
    @overload
    def iterchunks(
        self: ChunkedArray[scalar.Int64Scalar],
    ) -> Generator[array.Int64Array, None, None]: ...
    @overload
    def iterchunks(
        self: ChunkedArray[scalar.HalfFloatScalar],
    ) -> Generator[array.HalfFloatArray, None, None]: ...
    @overload
    def iterchunks(
        self: ChunkedArray[scalar.FloatScalar],
    ) -> Generator[array.FloatArray, None, None]: ...
    @overload
    def iterchunks(
        self: ChunkedArray[scalar.DoubleScalar],
    ) -> Generator[array.DoubleArray, None, None]: ...
    @overload
    def iterchunks(
        self: ChunkedArray[scalar.Decimal32Scalar],
    ) -> Generator[array.Decimal32Array, None, None]: ...
    @overload
    def iterchunks(
        self: ChunkedArray[scalar.Decimal64Scalar],
    ) -> Generator[array.Decimal64Array, None, None]: ...
    @overload
    def iterchunks(
        self: ChunkedArray[scalar.Decimal128Scalar],
    ) -> Generator[array.Decimal128Array, None, None]: ...
    @overload
    def iterchunks(
        self: ChunkedArray[scalar.Decimal256Scalar],
    ) -> Generator[array.Decimal256Array, None, None]: ...
    @overload
    def iterchunks(
        self: ChunkedArray[scalar.Date32Scalar],
    ) -> Generator[array.Date32Array, None, None]: ...
    @overload
    def iterchunks(
        self: ChunkedArray[scalar.Date64Scalar],
    ) -> Generator[array.Date64Array, None, None]: ...
    @overload
    def iterchunks(
        self: ChunkedArray[scalar.Time32Scalar[types._Time32Unit]],
    ) -> Generator[array.Time32Array[types._Time32Unit], None, None]: ...
    @overload
    def iterchunks(
        self: ChunkedArray[scalar.Time64Scalar[types._Time64Unit]],
    ) -> Generator[array.Time64Array[types._Time64Unit], None, None]: ...
    @overload
    def iterchunks(
        self: ChunkedArray[scalar.DurationScalar[types._Unit]],
    ) -> Generator[array.DurationArray[types._Unit], None, None]: ...
    @overload
    def iterchunks(
        self: ChunkedArray[scalar.MonthDayNanoIntervalScalar],
    ) -> Generator[array.MonthDayNanoIntervalArray, None, None]: ...
    @overload
    def iterchunks(
        self: ChunkedArray[scalar.BinaryScalar],
    ) -> Generator[array.BinaryArray, None, None]: ...
    @overload
    def iterchunks(
        self: ChunkedArray[scalar.LargeBinaryScalar],
    ) -> Generator[array.LargeBinaryArray, None, None]: ...
    @overload
    def iterchunks(
        self: ChunkedArray[scalar.FixedSizeBinaryScalar],
    ) -> Generator[array.FixedSizeBinaryArray, None, None]: ...
    @overload
    def iterchunks(
        self: ChunkedArray[scalar.StringScalar],
    ) -> Generator[array.StringArray, None, None]: ...
    @overload
    def iterchunks(
        self: ChunkedArray[scalar.LargeStringScalar],
    ) -> Generator[array.LargeStringArray, None, None]: ...
    @overload
    def iterchunks(
        self: ChunkedArray[scalar.BinaryViewScalar],
    ) -> Generator[array.BinaryViewArray, None, None]: ...
    @overload
    def iterchunks(
        self: ChunkedArray[scalar.StringViewScalar],
    ) -> Generator[array.StringViewArray, None, None]: ...
    @overload
    def iterchunks(
        self: ChunkedArray[scalar.ListScalar[_DataTypeT]],
    ) -> Generator[array.ListArray[scalar.ListScalar[_DataTypeT]], None, None]: ...
    @overload
    def iterchunks(
        self: ChunkedArray[scalar.FixedSizeListScalar[_DataTypeT, types._Size]],
    ) -> Generator[array.FixedSizeListArray[_DataTypeT, types._Size], None, None]: ...
    @overload
    def iterchunks(
        self: ChunkedArray[scalar.LargeListScalar[_DataTypeT]],
    ) -> Generator[array.LargeListArray[_DataTypeT], None, None]: ...
    @overload
    def iterchunks(
        self: ChunkedArray[scalar.LargeListViewScalar[_DataTypeT]],
    ) -> Generator[array.LargeListViewArray[_DataTypeT], None, None]: ...
    @overload
    def iterchunks(
        self: ChunkedArray[scalar.StructScalar],
    ) -> Generator[array.StructArray, None, None]: ...
    @overload
    def iterchunks(
        self: ChunkedArray[scalar.MapScalar[array._MapKeyT, array._MapItemT]],
    ) -> Generator[array.MapArray[array._MapKeyT, array._MapItemT], None, None]: ...
    @overload
    def iterchunks(
        self: ChunkedArray[scalar.DictionaryScalar[types._IndexT, types._BasicValueT]],
    ) -> Generator[array.DictionaryArray[types._IndexT, types._BasicValueT], None, None]: ...
    @overload
    def iterchunks(
        self: ChunkedArray[scalar.RunEndEncodedScalar],
    ) -> Generator[array.RunEndEncodedArray, None, None]: ...
    @overload
    def iterchunks(
        self: ChunkedArray[scalar.UnionScalar],
    ) -> Generator[array.UnionArray, None, None]: ...
    @overload
    def iterchunks(
        self: ChunkedArray[scalar.Bool8Scalar],
    ) -> Generator[array.Bool8Array, None, None]: ...
    @overload
    def iterchunks(
        self: ChunkedArray[scalar.UuidScalar],
    ) -> Generator[array.UuidArray, None, None]: ...
    @overload
    def iterchunks(
        self: ChunkedArray[scalar.JsonScalar],
    ) -> Generator[array.JsonArray, None, None]: ...
    @overload
    def iterchunks(
        self: ChunkedArray[scalar.OpaqueScalar],
    ) -> Generator[array.OpaqueArray, None, None]: ...
    def iterchunks(self):
        """
        Convert to an iterator of ChunkArrays.

        Examples
        --------
        >>> import pyarrow as pa
        >>> n_legs = pa.chunked_array([[2, 2, 4], [4, None, 100]])
        >>> for i in n_legs.iterchunks():
        ...     print(i.null_count)
        0
        1

        """
    def __iter__(self) -> Iterator[_Scalar_co]: ...
    def to_pylist(
        self: ChunkedArray[Scalar[_BasicDataType[_AsPyType]]],
        *,
        maps_as_pydicts: Literal["lossy", "strict"] | None = None,
    ) -> list[_AsPyType | None]:
        """
        Convert to a list of native Python objects.

        Parameters
        ----------
        maps_as_pydicts : str, optional, default `None`
            Valid values are `None`, 'lossy', or 'strict'.
            The default behavior (`None`), is to convert Arrow Map arrays to
            Python association lists (list-of-tuples) in the same order as the
            Arrow Map, as in [(key1, value1), (key2, value2), ...].

            If 'lossy' or 'strict', convert Arrow Map arrays to native Python dicts.

            If 'lossy', whenever duplicate keys are detected, a warning will be printed.
            The last seen value of a duplicate key will be in the Python dictionary.
            If 'strict', this instead results in an exception being raised when detected.

        Examples
        --------
        >>> import pyarrow as pa
        >>> n_legs = pa.chunked_array([[2, 2, 4], [4, None, 100]])
        >>> n_legs.to_pylist()
        [2, 2, 4, 4, None, 100]
        """
    def __arrow_c_stream__(self, requested_schema=None) -> Any:
        """
        Export to a C ArrowArrayStream PyCapsule.

        Parameters
        ----------
        requested_schema : PyCapsule, default None
            The schema to which the stream should be casted, passed as a
            PyCapsule containing a C ArrowSchema representation of the
            requested schema.

        Returns
        -------
        PyCapsule
            A capsule containing a C ArrowArrayStream struct.
        """
    @classmethod
    def _import_from_c_capsule(cls, stream) -> Self:
        """
        Import ChunkedArray from a C ArrowArrayStream PyCapsule.

        Parameters
        ----------
        stream: PyCapsule
            A capsule containing a C ArrowArrayStream PyCapsule.

        Returns
        -------
        ChunkedArray
        """
    @property
    def is_cpu(self) -> bool:
        """
        Whether all chunks in the ChunkedArray are CPU-accessible.
        """

@overload
def chunked_array(
    values: Iterable[NullableCollection[bool]],
    type: None = None,
) -> ChunkedArray[scalar.BooleanScalar]: ...
@overload
def chunked_array(
    values: Iterable[NullableCollection[int]],
    type: None = None,
) -> ChunkedArray[scalar.Int64Scalar]: ...
@overload
def chunked_array(
    values: Iterable[NullableCollection[float]],
    type: None = None,
) -> ChunkedArray[scalar.DoubleScalar]: ...
@overload
def chunked_array(
    values: Iterable[NullableCollection[Decimal]],
    type: None = None,
) -> ChunkedArray[scalar.Decimal128Scalar]: ...
@overload
def chunked_array(
    values: Iterable[NullableCollection[dict[str, Any]]],
    type: None = None,
) -> ChunkedArray[scalar.StructScalar]: ...
@overload
def chunked_array(
    values: Iterable[NullableCollection[dt.datetime]],
    type: None = None,
) -> ChunkedArray[scalar.TimestampScalar]: ...
@overload
def chunked_array(
    values: Iterable[NullableCollection[dt.date]],
    type: None = None,
) -> ChunkedArray[scalar.Date32Scalar]: ...
@overload
def chunked_array(
    values: Iterable[NullableCollection[dt.time]],
    type: None = None,
) -> ChunkedArray[scalar.Time64Scalar[Literal["us"]]]: ...
@overload
def chunked_array(
    values: Iterable[NullableCollection[dt.timedelta]],
    type: None = None,
) -> ChunkedArray[scalar.DurationScalar[Literal["us"]]]: ...
@overload
def chunked_array(
    values: Iterable[NullableCollection[MonthDayNano]],
    type: None = None,
) -> ChunkedArray[scalar.MonthDayNanoIntervalScalar]: ...
@overload
def chunked_array(
    values: Iterable[NullableCollection[str]],
    type: None = None,
) -> ChunkedArray[scalar.StringScalar]: ...
@overload
def chunked_array(
    values: Iterable[NullableCollection[bytes]],
    type: None = None,
) -> ChunkedArray[scalar.BinaryScalar]: ...
@overload
def chunked_array(
    values: Iterable[NullableCollection[list[Any]]],
    type: None = None,
) -> ChunkedArray[scalar.ListScalar[Any]]: ...
@overload
def chunked_array(
    values: Iterable[Iterable[Any] | SupportArrowStream | SupportArrowArray],
    type: Literal["null"] | types.NullType,
) -> ChunkedArray[scalar.NullScalar]: ...
@overload
def chunked_array(
    values: Iterable[Iterable[Any] | SupportArrowStream | SupportArrowArray],
    type: Literal["bool", "boolean"] | types.BoolType,
) -> ChunkedArray[scalar.BooleanScalar]: ...
@overload
def chunked_array(
    values: Iterable[Iterable[Any] | SupportArrowStream | SupportArrowArray],
    type: Literal["i1", "int8"] | types.Int8Type,
) -> ChunkedArray[scalar.Int8Scalar]: ...
@overload
def chunked_array(
    values: Iterable[Iterable[Any] | SupportArrowStream | SupportArrowArray],
    type: Literal["i2", "int16"] | types.Int16Type,
) -> ChunkedArray[scalar.Int16Scalar]: ...
@overload
def chunked_array(
    values: Iterable[Iterable[Any] | SupportArrowStream | SupportArrowArray],
    type: Literal["i4", "int32"] | types.Int32Type,
) -> ChunkedArray[scalar.Int32Scalar]: ...
@overload
def chunked_array(
    values: Iterable[Iterable[Any] | SupportArrowStream | SupportArrowArray],
    type: Literal["i8", "int64"] | types.Int64Type,
) -> ChunkedArray[scalar.Int64Scalar]: ...
@overload
def chunked_array(
    values: Iterable[Iterable[Any] | SupportArrowStream | SupportArrowArray],
    type: Literal["u1", "uint8"] | types.UInt8Type,
) -> ChunkedArray[scalar.UInt8Scalar]: ...
@overload
def chunked_array(
    values: Iterable[Iterable[Any] | SupportArrowStream | SupportArrowArray],
    type: Literal["u2", "uint16"] | types.UInt16Type,
) -> ChunkedArray[scalar.UInt16Scalar]: ...
@overload
def chunked_array(
    values: Iterable[Iterable[Any] | SupportArrowStream | SupportArrowArray],
    type: Literal["u4", "uint32"] | types.Uint32Type,
) -> ChunkedArray[scalar.UInt32Scalar]: ...
@overload
def chunked_array(
    values: Iterable[Iterable[Any] | SupportArrowStream | SupportArrowArray],
    type: Literal["u8", "uint64"] | types.UInt64Type,
) -> ChunkedArray[scalar.UInt64Scalar]: ...
@overload
def chunked_array(
    values: Iterable[Iterable[Any] | SupportArrowStream | SupportArrowArray],
    type: Literal["f2", "halffloat", "float16"] | types.Float16Type,
) -> ChunkedArray[scalar.HalfFloatScalar]: ...
@overload
def chunked_array(
    values: Iterable[Iterable[Any] | SupportArrowStream | SupportArrowArray],
    type: Literal["f4", "float", "float32"] | types.Float32Type,
) -> ChunkedArray[scalar.FloatScalar]: ...
@overload
def chunked_array(
    values: Iterable[Iterable[Any] | SupportArrowStream | SupportArrowArray],
    type: Literal["f8", "double", "float64"] | types.Float64Type,
) -> ChunkedArray[scalar.DoubleScalar]: ...
@overload
def chunked_array(
    values: Iterable[Iterable[Any] | SupportArrowStream | SupportArrowArray],
    type: Literal["string", "str", "utf8"] | types.StringType,
) -> ChunkedArray[scalar.StringScalar]: ...
@overload
def chunked_array(
    values: Iterable[Iterable[Any] | SupportArrowStream | SupportArrowArray],
    type: Literal["binary"] | types.BinaryType,
) -> ChunkedArray[scalar.BinaryScalar]: ...
@overload
def chunked_array(
    values: Iterable[Iterable[Any] | SupportArrowStream | SupportArrowArray],
    type: Literal["large_string", "large_str", "large_utf8"] | types.LargeStringType,
) -> ChunkedArray[scalar.LargeStringScalar]: ...
@overload
def chunked_array(
    values: Iterable[Iterable[Any] | SupportArrowStream | SupportArrowArray],
    type: Literal["large_binary"] | types.LargeBinaryType,
) -> ChunkedArray[scalar.LargeBinaryScalar]: ...
@overload
def chunked_array(
    values: Iterable[Iterable[Any] | SupportArrowStream | SupportArrowArray],
    type: Literal["binary_view"] | types.BinaryViewType,
) -> ChunkedArray[scalar.BinaryViewScalar]: ...
@overload
def chunked_array(
    values: Iterable[Iterable[Any] | SupportArrowStream | SupportArrowArray],
    type: Literal["string_view"] | types.StringViewType,
) -> ChunkedArray[scalar.StringViewScalar]: ...
@overload
def chunked_array(
    values: Iterable[Iterable[Any] | SupportArrowStream | SupportArrowArray],
    type: Literal["date32", "date32[day]"] | types.Date32Type,
) -> ChunkedArray[scalar.Date32Scalar]: ...
@overload
def chunked_array(
    values: Iterable[Iterable[Any] | SupportArrowStream | SupportArrowArray],
    type: Literal["date64", "date64[ms]"] | types.Date64Type,
) -> ChunkedArray[scalar.Date64Scalar]: ...
@overload
def chunked_array(
    values: Iterable[Iterable[Any] | SupportArrowStream | SupportArrowArray],
    type: Literal["time32[s]"] | types.Time32Type[Literal["s"]],
) -> ChunkedArray[scalar.Time32Scalar[Literal["s"]]]: ...
@overload
def chunked_array(
    values: Iterable[Iterable[Any] | SupportArrowStream | SupportArrowArray],
    type: Literal["time32[ms]"] | types.Time32Type[Literal["ms"]],
) -> ChunkedArray[scalar.Time32Scalar[Literal["ms"]]]: ...
@overload
def chunked_array(
    values: Iterable[Iterable[Any] | SupportArrowStream | SupportArrowArray],
    type: Literal["time64[us]"] | types.Time64Type[Literal["us"]],
) -> ChunkedArray[scalar.Time64Scalar[Literal["us"]]]: ...
@overload
def chunked_array(
    values: Iterable[Iterable[Any] | SupportArrowStream | SupportArrowArray],
    type: Literal["time64[ns]"] | types.Time64Type[Literal["ns"]],
) -> ChunkedArray[scalar.Time64Scalar[Literal["ns"]]]: ...
@overload
def chunked_array(
    values: Iterable[Iterable[Any] | SupportArrowStream | SupportArrowArray],
    type: Literal["timestamp[s]"] | types.TimestampType[Literal["s"]],
) -> ChunkedArray[scalar.TimestampScalar[Literal["s"]]]: ...
@overload
def chunked_array(
    values: Iterable[Iterable[Any] | SupportArrowStream | SupportArrowArray],
    type: Literal["timestamp[ms]"] | types.TimestampType[Literal["ms"]],
) -> ChunkedArray[scalar.TimestampScalar[Literal["ms"]]]: ...
@overload
def chunked_array(
    values: Iterable[Iterable[Any] | SupportArrowStream | SupportArrowArray],
    type: Literal["timestamp[us]"] | types.TimestampType[Literal["us"]],
) -> ChunkedArray[scalar.TimestampScalar[Literal["us"]]]: ...
@overload
def chunked_array(
    values: Iterable[Iterable[Any] | SupportArrowStream | SupportArrowArray],
    type: Literal["timestamp[ns]"] | types.TimestampType[Literal["ns"]],
) -> ChunkedArray[scalar.TimestampScalar[Literal["ns"]]]: ...
@overload
def chunked_array(
    values: Iterable[Iterable[Any] | SupportArrowStream | SupportArrowArray],
    type: Literal["duration[s]"] | types.DurationType[Literal["s"]],
) -> ChunkedArray[scalar.DurationScalar[Literal["s"]]]: ...
@overload
def chunked_array(
    values: Iterable[Iterable[Any] | SupportArrowStream | SupportArrowArray],
    type: Literal["duration[ms]"] | types.DurationType[Literal["ms"]],
) -> ChunkedArray[scalar.DurationScalar[Literal["ms"]]]: ...
@overload
def chunked_array(
    values: Iterable[Iterable[Any] | SupportArrowStream | SupportArrowArray],
    type: Literal["duration[us]"] | types.DurationType[Literal["us"]],
) -> ChunkedArray[scalar.DurationScalar[Literal["us"]]]: ...
@overload
def chunked_array(
    values: Iterable[Iterable[Any] | SupportArrowStream | SupportArrowArray],
    type: Literal["duration[ns]"] | types.DurationType[Literal["ns"]],
) -> ChunkedArray[scalar.DurationScalar[Literal["ns"]]]: ...
@overload
def chunked_array(
    values: Iterable[Iterable[Any]] | SupportArrowStream | SupportArrowArray,
    type: Literal["month_day_nano_interval"] | types.MonthDayNanoIntervalType,
) -> ChunkedArray[scalar.MonthDayNanoIntervalScalar]: ...
@overload
def chunked_array(
    values: Iterable[Array[_ScalarT]],
    type: None = None,
) -> ChunkedArray[_ScalarT]: ...
def chunked_array(value, type=None):
    """
    Construct chunked array from list of array-like objects

    Parameters
    ----------
    arrays : Array, list of Array, or array-like
        Must all be the same data type. Can be empty only if type also passed.
        Any Arrow-compatible array that implements the Arrow PyCapsule Protocol
        (has an ``__arrow_c_array__`` or ``__arrow_c_stream__`` method) can be
        passed as well.
    type : DataType or string coercible to DataType

    Returns
    -------
    ChunkedArray

    Examples
    --------
    >>> import pyarrow as pa
    >>> pa.chunked_array([], type=pa.int8())
    <pyarrow.lib.ChunkedArray object at ...>
    [
    ...
    ]

    >>> pa.chunked_array([[2, 2, 4], [4, 5, 100]])
    <pyarrow.lib.ChunkedArray object at ...>
    [
      [
        2,
        2,
        4
      ],
      [
        4,
        5,
        100
      ]
    ]
    """

_ColumnT = TypeVar("_ColumnT", bound=ArrayOrChunkedArray[Any])

class _Tabular(_PandasConvertible[pd.DataFrame], Generic[_ColumnT]):
    def __array__(self, dtype: np.dtype | None = None, copy: bool | None = None) -> np.ndarray: ...
    def __dataframe__(
        self, nan_as_null: bool = False, allow_copy: bool = True
    ) -> _PyArrowDataFrame:
        """
        Return the dataframe interchange object implementing the interchange protocol.

        Parameters
        ----------
        nan_as_null : bool, default False
            Whether to tell the DataFrame to overwrite null values in the data
            with ``NaN`` (or ``NaT``).
        allow_copy : bool, default True
            Whether to allow memory copying when exporting. If set to False
            it would cause non-zero-copy exports to fail.

        Returns
        -------
        DataFrame interchange object
            The object which consuming library can use to ingress the dataframe.

        Notes
        -----
        Details on the interchange protocol:
        https://data-apis.org/dataframe-protocol/latest/index.html
        `nan_as_null` currently has no effect; once support for nullable extension
        dtypes is added, this value should be propagated to columns.
        """
    @overload
    def __getitem__(self, key: int | str) -> _ColumnT: ...
    @overload
    def __getitem__(self, key: slice) -> Self: ...
    def __getitem__(self, key):
        """
        Slice or return column at given index or column name

        Parameters
        ----------
        key : integer, str, or slice
            Slices with step not equal to 1 (or None) will produce a copy
            rather than a zero-copy view

        Returns
        -------
        Array (from RecordBatch) or ChunkedArray (from Table) for column input.
        RecordBatch or Table for slice input.
        """
    def __len__(self) -> int: ...
    def column(self, i: int | str) -> _ColumnT:
        """
        Select single column from Table or RecordBatch.

        Parameters
        ----------
        i : int or string
            The index or name of the column to retrieve.

        Returns
        -------
        column : Array (for RecordBatch) or ChunkedArray (for Table)

        Examples
        --------
        Table (works similarly for RecordBatch)

        >>> import pyarrow as pa
        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...     {
        ...         "n_legs": [2, 4, 5, 100],
        ...         "animals": ["Flamingo", "Horse", "Brittle stars", "Centipede"],
        ...     }
        ... )
        >>> table = pa.Table.from_pandas(df)

        Select a column by numeric index:

        >>> table.column(0)
        <pyarrow.lib.ChunkedArray object at ...>
        [
          [
            2,
            4,
            5,
            100
          ]
        ]

        Select a column by its name:

        >>> table.column("animals")
        <pyarrow.lib.ChunkedArray object at ...>
        [
          [
            "Flamingo",
            "Horse",
            "Brittle stars",
            "Centipede"
          ]
        ]
        """
    @property
    def column_names(self) -> list[str]:
        """
        Names of the Table or RecordBatch columns.

        Returns
        -------
        list of str

        Examples
        --------
        Table (works similarly for RecordBatch)

        >>> import pyarrow as pa
        >>> table = pa.Table.from_arrays(
        ...     [[2, 4, 5, 100], ["Flamingo", "Horse", "Brittle stars", "Centipede"]],
        ...     names=["n_legs", "animals"],
        ... )
        >>> table.column_names
        ['n_legs', 'animals']
        """
    @property
    def columns(self) -> list[_ColumnT]:
        """
        List of all columns in numerical order.

        Returns
        -------
        columns : list of Array (for RecordBatch) or list of ChunkedArray (for Table)

        Examples
        --------
        Table (works similarly for RecordBatch)

        >>> import pyarrow as pa
        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...     {"n_legs": [None, 4, 5, None], "animals": ["Flamingo", "Horse", None, "Centipede"]}
        ... )
        >>> table = pa.Table.from_pandas(df)
        >>> table.columns
        [<pyarrow.lib.ChunkedArray object at ...>
        [
          [
            null,
            4,
            5,
            null
          ]
        ], <pyarrow.lib.ChunkedArray object at ...>
        [
          [
            "Flamingo",
            "Horse",
            null,
            "Centipede"
          ]
        ]]
        """
    def drop_null(self) -> Self:
        """
        Remove rows that contain missing values from a Table or RecordBatch.

        See :func:`pyarrow.compute.drop_null` for full usage.

        Returns
        -------
        Table or RecordBatch
            A tabular object with the same schema, with rows containing
            no missing values.

        Examples
        --------
        Table (works similarly for RecordBatch)

        >>> import pyarrow as pa
        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...     {
        ...         "year": [None, 2022, 2019, 2021],
        ...         "n_legs": [2, 4, 5, 100],
        ...         "animals": ["Flamingo", "Horse", None, "Centipede"],
        ...     }
        ... )
        >>> table = pa.Table.from_pandas(df)
        >>> table.drop_null()
        pyarrow.Table
        year: double
        n_legs: int64
        animals: string
        ----
        year: [[2022,2021]]
        n_legs: [[4,100]]
        animals: [["Horse","Centipede"]]
        """
    def field(self, i: int | str) -> Field:
        """
        Select a schema field by its column name or numeric index.

        Parameters
        ----------
        i : int or string
            The index or name of the field to retrieve.

        Returns
        -------
        Field

        Examples
        --------
        Table (works similarly for RecordBatch)

        >>> import pyarrow as pa
        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...     {
        ...         "n_legs": [2, 4, 5, 100],
        ...         "animals": ["Flamingo", "Horse", "Brittle stars", "Centipede"],
        ...     }
        ... )
        >>> table = pa.Table.from_pandas(df)
        >>> table.field(0)
        pyarrow.Field<n_legs: int64>
        >>> table.field(1)
        pyarrow.Field<animals: string>
        """
    @classmethod
    def from_pydict(
        cls,
        mapping: Mapping[str, ArrayOrChunkedArray[Any] | list[Any] | np.ndarray],
        schema: Schema | None = None,
        metadata: MetaData | None = None,
    ) -> Self:
        """
        Construct a Table or RecordBatch from Arrow arrays or columns.

        Parameters
        ----------
        mapping : dict or Mapping
            A mapping of strings to Arrays or Python lists.
        schema : Schema, default None
            If not passed, will be inferred from the Mapping values.
        metadata : dict or Mapping, default None
            Optional metadata for the schema (if inferred).

        Returns
        -------
        Table or RecordBatch

        Examples
        --------
        Table (works similarly for RecordBatch)

        >>> import pyarrow as pa
        >>> n_legs = pa.array([2, 4, 5, 100])
        >>> animals = pa.array(["Flamingo", "Horse", "Brittle stars", "Centipede"])
        >>> pydict = {"n_legs": n_legs, "animals": animals}

        Construct a Table from a dictionary of arrays:

        >>> pa.Table.from_pydict(pydict)
        pyarrow.Table
        n_legs: int64
        animals: string
        ----
        n_legs: [[2,4,5,100]]
        animals: [["Flamingo","Horse","Brittle stars","Centipede"]]
        >>> pa.Table.from_pydict(pydict).schema
        n_legs: int64
        animals: string

        Construct a Table from a dictionary of arrays with metadata:

        >>> my_metadata = {"n_legs": "Number of legs per animal"}
        >>> pa.Table.from_pydict(pydict, metadata=my_metadata).schema
        n_legs: int64
        animals: string
        -- schema metadata --
        n_legs: 'Number of legs per animal'

        Construct a Table from a dictionary of arrays with pyarrow schema:

        >>> my_schema = pa.schema(
        ...     [pa.field("n_legs", pa.int64()), pa.field("animals", pa.string())],
        ...     metadata={"n_legs": "Number of legs per animal"},
        ... )
        >>> pa.Table.from_pydict(pydict, schema=my_schema).schema
        n_legs: int64
        animals: string
        -- schema metadata --
        n_legs: 'Number of legs per animal'
        """
    @classmethod
    def from_pylist(
        cls,
        mapping: Sequence[Mapping[str, Any]],
        schema: Schema | None = None,
        metadata: MetaData | None = None,
    ) -> Self:
        """
        Construct a Table or RecordBatch from list of rows / dictionaries.

        Parameters
        ----------
        mapping : list of dicts of rows
            A mapping of strings to row values.
        schema : Schema, default None
            If not passed, will be inferred from the first row of the
            mapping values.
        metadata : dict or Mapping, default None
            Optional metadata for the schema (if inferred).

        Returns
        -------
        Table or RecordBatch

        Examples
        --------
        Table (works similarly for RecordBatch)

        >>> import pyarrow as pa
        >>> pylist = [{"n_legs": 2, "animals": "Flamingo"}, {"n_legs": 4, "animals": "Dog"}]

        Construct a Table from a list of rows:

        >>> pa.Table.from_pylist(pylist)
        pyarrow.Table
        n_legs: int64
        animals: string
        ----
        n_legs: [[2,4]]
        animals: [["Flamingo","Dog"]]

        Construct a Table from a list of rows with metadata:

        >>> my_metadata = {"n_legs": "Number of legs per animal"}
        >>> pa.Table.from_pylist(pylist, metadata=my_metadata).schema
        n_legs: int64
        animals: string
        -- schema metadata --
        n_legs: 'Number of legs per animal'

        Construct a Table from a list of rows with pyarrow schema:

        >>> my_schema = pa.schema(
        ...     [pa.field("n_legs", pa.int64()), pa.field("animals", pa.string())],
        ...     metadata={"n_legs": "Number of legs per animal"},
        ... )
        >>> pa.Table.from_pylist(pylist, schema=my_schema).schema
        n_legs: int64
        animals: string
        -- schema metadata --
        n_legs: 'Number of legs per animal'
        """
    def itercolumns(self) -> Generator[_ColumnT, None, None]:
        """
        Iterator over all columns in their numerical order.

        Yields
        ------
        Array (for RecordBatch) or ChunkedArray (for Table)

        Examples
        --------
        Table (works similarly for RecordBatch)

        >>> import pyarrow as pa
        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...     {"n_legs": [None, 4, 5, None], "animals": ["Flamingo", "Horse", None, "Centipede"]}
        ... )
        >>> table = pa.Table.from_pandas(df)
        >>> for i in table.itercolumns():
        ...     print(i.null_count)
        2
        1
        """
    @property
    def num_columns(self) -> int: ...
    @property
    def num_rows(self) -> int: ...
    @property
    def shape(self) -> tuple[int, int]:
        """
        Dimensions of the table or record batch: (#rows, #columns).

        Returns
        -------
        (int, int)
            Number of rows and number of columns.

        Examples
        --------
        >>> import pyarrow as pa
        >>> table = pa.table(
        ...     {"n_legs": [None, 4, 5, None], "animals": ["Flamingo", "Horse", None, "Centipede"]}
        ... )
        >>> table.shape
        (4, 2)
        """
    @property
    def schema(self) -> Schema: ...
    @property
    def nbytes(self) -> int: ...
    def sort_by(self, sorting: str | list[tuple[str, Order]], **kwargs) -> Self:
        """
        Sort the Table or RecordBatch by one or multiple columns.

        Parameters
        ----------
        sorting : str or list[tuple(name, order)]
            Name of the column to use to sort (ascending), or
            a list of multiple sorting conditions where
            each entry is a tuple with column name
            and sorting order ("ascending" or "descending")
        **kwargs : dict, optional
            Additional sorting options.
            As allowed by :class:`SortOptions`

        Returns
        -------
        Table or RecordBatch
            A new tabular object sorted according to the sort keys.

        Examples
        --------
        Table (works similarly for RecordBatch)

        >>> import pandas as pd
        >>> import pyarrow as pa
        >>> df = pd.DataFrame(
        ...     {
        ...         "year": [2020, 2022, 2021, 2022, 2019, 2021],
        ...         "n_legs": [2, 2, 4, 4, 5, 100],
        ...         "animal": ["Flamingo", "Parrot", "Dog", "Horse", "Brittle stars", "Centipede"],
        ...     }
        ... )
        >>> table = pa.Table.from_pandas(df)
        >>> table.sort_by("animal")
        pyarrow.Table
        year: int64
        n_legs: int64
        animal: string
        ----
        year: [[2019,2021,2021,2020,2022,2022]]
        n_legs: [[5,100,4,2,4,2]]
        animal: [["Brittle stars","Centipede","Dog","Flamingo","Horse","Parrot"]]
        """
    def take(self, indices: Indices) -> Self:
        """
        Select rows from a Table or RecordBatch.

        See :func:`pyarrow.compute.take` for full usage.

        Parameters
        ----------
        indices : Array or array-like
            The indices in the tabular object whose rows will be returned.

        Returns
        -------
        Table or RecordBatch
            A tabular object with the same schema, containing the taken rows.

        Examples
        --------
        Table (works similarly for RecordBatch)

        >>> import pyarrow as pa
        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...     {
        ...         "year": [2020, 2022, 2019, 2021],
        ...         "n_legs": [2, 4, 5, 100],
        ...         "animals": ["Flamingo", "Horse", "Brittle stars", "Centipede"],
        ...     }
        ... )
        >>> table = pa.Table.from_pandas(df)
        >>> table.take([1, 3])
        pyarrow.Table
        year: int64
        n_legs: int64
        animals: string
        ----
        year: [[2022,2021]]
        n_legs: [[4,100]]
        animals: [["Horse","Centipede"]]
        """
    def filter(
        self, mask: Mask | Expression, null_selection_behavior: NullSelectionBehavior = "drop"
    ) -> Self:
        """
        Select rows from the table or record batch based on a boolean mask.

        The Table can be filtered based on a mask, which will be passed to
        :func:`pyarrow.compute.filter` to perform the filtering, or it can
        be filtered through a boolean :class:`.Expression`

        Parameters
        ----------
        mask : Array or array-like or .Expression
            The boolean mask or the :class:`.Expression` to filter the table with.
        null_selection_behavior : str, default "drop"
            How nulls in the mask should be handled, does nothing if
            an :class:`.Expression` is used.

        Returns
        -------
        filtered : Table or RecordBatch
            A tabular object of the same schema, with only the rows selected
            by applied filtering

        Examples
        --------
        Using a Table (works similarly for RecordBatch):

        >>> import pyarrow as pa
        >>> table = pa.table(
        ...     {
        ...         "year": [2020, 2022, 2019, 2021],
        ...         "n_legs": [2, 4, 5, 100],
        ...         "animals": ["Flamingo", "Horse", "Brittle stars", "Centipede"],
        ...     }
        ... )

        Define an expression and select rows:

        >>> import pyarrow.compute as pc
        >>> expr = pc.field("year") <= 2020
        >>> table.filter(expr)
        pyarrow.Table
        year: int64
        n_legs: int64
        animals: string
        ----
        year: [[2020,2019]]
        n_legs: [[2,5]]
        animals: [["Flamingo","Brittle stars"]]

        Define a mask and select rows:

        >>> mask = [True, True, False, None]
        >>> table.filter(mask)
        pyarrow.Table
        year: int64
        n_legs: int64
        animals: string
        ----
        year: [[2020,2022]]
        n_legs: [[2,4]]
        animals: [["Flamingo","Horse"]]
        >>> table.filter(mask, null_selection_behavior="emit_null")
        pyarrow.Table
        year: int64
        n_legs: int64
        animals: string
        ----
        year: [[2020,2022,null]]
        n_legs: [[2,4,null]]
        animals: [["Flamingo","Horse",null]]
        """
    def to_pydict(
        self, *, maps_as_pydicts: Literal["lossy", "strict"] | None = None
    ) -> dict[str, list[Any]]:
        """
        Convert the Table or RecordBatch to a dict or OrderedDict.

        Parameters
        ----------
        maps_as_pydicts : str, optional, default `None`
            Valid values are `None`, 'lossy', or 'strict'.
            The default behavior (`None`), is to convert Arrow Map arrays to
            Python association lists (list-of-tuples) in the same order as the
            Arrow Map, as in [(key1, value1), (key2, value2), ...].

            If 'lossy' or 'strict', convert Arrow Map arrays to native Python dicts.

            If 'lossy', whenever duplicate keys are detected, a warning will be printed.
            The last seen value of a duplicate key will be in the Python dictionary.
            If 'strict', this instead results in an exception being raised when detected.

        Returns
        -------
        dict

        Examples
        --------
        Table (works similarly for RecordBatch)

        >>> import pyarrow as pa
        >>> n_legs = pa.array([2, 2, 4, 4, 5, 100])
        >>> animals = pa.array(
        ...     ["Flamingo", "Parrot", "Dog", "Horse", "Brittle stars", "Centipede"]
        ... )
        >>> table = pa.Table.from_arrays([n_legs, animals], names=["n_legs", "animals"])
        >>> table.to_pydict()
        {'n_legs': [2, 2, 4, 4, 5, 100], 'animals': ['Flamingo', 'Parrot', ..., 'Centipede']}
        """
    def to_pylist(
        self, *, maps_as_pydicts: Literal["lossy", "strict"] | None = None
    ) -> list[dict[str, Any]]:
        """
        Convert the Table or RecordBatch to a list of rows / dictionaries.

        Parameters
        ----------
        maps_as_pydicts : str, optional, default `None`
            Valid values are `None`, 'lossy', or 'strict'.
            The default behavior (`None`), is to convert Arrow Map arrays to
            Python association lists (list-of-tuples) in the same order as the
            Arrow Map, as in [(key1, value1), (key2, value2), ...].

            If 'lossy' or 'strict', convert Arrow Map arrays to native Python dicts.

            If 'lossy', whenever duplicate keys are detected, a warning will be printed.
            The last seen value of a duplicate key will be in the Python dictionary.
            If 'strict', this instead results in an exception being raised when detected.

        Returns
        -------
        list

        Examples
        --------
        Table (works similarly for RecordBatch)

        >>> import pyarrow as pa
        >>> data = [[2, 4, 5, 100], ["Flamingo", "Horse", "Brittle stars", "Centipede"]]
        >>> table = pa.table(data, names=["n_legs", "animals"])
        >>> table.to_pylist()
        [{'n_legs': 2, 'animals': 'Flamingo'}, {'n_legs': 4, 'animals': 'Horse'}, ...
        """
    def to_string(self, *, show_metadata: bool = False, preview_cols: int = 0) -> str:
        """
        Return human-readable string representation of Table or RecordBatch.

        Parameters
        ----------
        show_metadata : bool, default False
            Display Field-level and Schema-level KeyValueMetadata.
        preview_cols : int, default 0
            Display values of the columns for the first N columns.

        Returns
        -------
        str
        """
    def remove_column(self, i: int) -> Self: ...
    def drop_columns(self, columns: str | list[str]) -> Self:
        """
        Drop one or more columns and return a new Table or RecordBatch.

        Parameters
        ----------
        columns : str or list[str]
            Field name(s) referencing existing column(s).

        Raises
        ------
        KeyError
            If any of the passed column names do not exist.

        Returns
        -------
        Table or RecordBatch
            A tabular object without the column(s).

        Examples
        --------
        Table (works similarly for RecordBatch)

        >>> import pyarrow as pa
        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...     {
        ...         "n_legs": [2, 4, 5, 100],
        ...         "animals": ["Flamingo", "Horse", "Brittle stars", "Centipede"],
        ...     }
        ... )
        >>> table = pa.Table.from_pandas(df)

        Drop one column:

        >>> table.drop_columns("animals")
        pyarrow.Table
        n_legs: int64
        ----
        n_legs: [[2,4,5,100]]

        Drop one or more columns:

        >>> table.drop_columns(["n_legs", "animals"])
        pyarrow.Table
        ...
        ----
        """
    def add_column(
        self, i: int, field_: str | Field, column: ArrayOrChunkedArray[Any] | list[list[Any]]
    ) -> Self: ...
    def append_column(
        self, field_: str | Field, column: ArrayOrChunkedArray[Any] | list[list[Any]]
    ) -> Self:
        """
        Append column at end of columns.

        Parameters
        ----------
        field_ : str or Field
            If a string is passed then the type is deduced from the column
            data.
        column : Array or value coercible to array
            Column data.

        Returns
        -------
        Table or RecordBatch
            New table or record batch with the passed column added.

        Examples
        --------
        >>> import pyarrow as pa
        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...     {
        ...         "n_legs": [2, 4, 5, 100],
        ...         "animals": ["Flamingo", "Horse", "Brittle stars", "Centipede"],
        ...     }
        ... )
        >>> table = pa.Table.from_pandas(df)

        Append column at the end:

        >>> year = [2021, 2022, 2019, 2021]
        >>> table.append_column("year", [year])
        pyarrow.Table
        n_legs: int64
        animals: string
        year: int64
        ----
        n_legs: [[2,4,5,100]]
        animals: [["Flamingo","Horse","Brittle stars","Centipede"]]
        year: [[2021,2022,2019,2021]]
        """

class RecordBatch(_Tabular[Array]):
    """
    Batch of rows of columns of equal length

    Warnings
    --------
    Do not call this class's constructor directly, use one of the
    ``RecordBatch.from_*`` functions instead.

    Examples
    --------
    >>> import pyarrow as pa
    >>> n_legs = pa.array([2, 2, 4, 4, 5, 100])
    >>> animals = pa.array(["Flamingo", "Parrot", "Dog", "Horse", "Brittle stars", "Centipede"])
    >>> names = ["n_legs", "animals"]

    Constructing a RecordBatch from arrays:

    >>> pa.RecordBatch.from_arrays([n_legs, animals], names=names)
    pyarrow.RecordBatch
    n_legs: int64
    animals: string
    ----
    n_legs: [2,2,4,4,5,100]
    animals: ["Flamingo","Parrot","Dog","Horse","Brittle stars","Centipede"]
    >>> pa.RecordBatch.from_arrays([n_legs, animals], names=names).to_pandas()
       n_legs        animals
    0       2       Flamingo
    1       2         Parrot
    2       4            Dog
    3       4          Horse
    4       5  Brittle stars
    5     100      Centipede

    Constructing a RecordBatch from pandas DataFrame:

    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     {
    ...         "year": [2020, 2022, 2021, 2022],
    ...         "month": [3, 5, 7, 9],
    ...         "day": [1, 5, 9, 13],
    ...         "n_legs": [2, 4, 5, 100],
    ...         "animals": ["Flamingo", "Horse", "Brittle stars", "Centipede"],
    ...     }
    ... )
    >>> pa.RecordBatch.from_pandas(df)
    pyarrow.RecordBatch
    year: int64
    month: int64
    day: int64
    n_legs: int64
    animals: string
    ----
    year: [2020,2022,2021,2022]
    month: [3,5,7,9]
    day: [1,5,9,13]
    n_legs: [2,4,5,100]
    animals: ["Flamingo","Horse","Brittle stars","Centipede"]
    >>> pa.RecordBatch.from_pandas(df).to_pandas()
       year  month  day  n_legs        animals
    0  2020      3    1       2       Flamingo
    1  2022      5    5       4          Horse
    2  2021      7    9       5  Brittle stars
    3  2022      9   13     100      Centipede

    Constructing a RecordBatch from pylist:

    >>> pylist = [{"n_legs": 2, "animals": "Flamingo"}, {"n_legs": 4, "animals": "Dog"}]
    >>> pa.RecordBatch.from_pylist(pylist).to_pandas()
       n_legs   animals
    0       2  Flamingo
    1       4       Dog

    You can also construct a RecordBatch using :func:`pyarrow.record_batch`:

    >>> pa.record_batch([n_legs, animals], names=names).to_pandas()
       n_legs        animals
    0       2       Flamingo
    1       2         Parrot
    2       4            Dog
    3       4          Horse
    4       5  Brittle stars
    5     100      Centipede

    >>> pa.record_batch(df)
    pyarrow.RecordBatch
    year: int64
    month: int64
    day: int64
    n_legs: int64
    animals: string
    ----
    year: [2020,2022,2021,2022]
    month: [3,5,7,9]
    day: [1,5,9,13]
    n_legs: [2,4,5,100]
    animals: ["Flamingo","Horse","Brittle stars","Centipede"]
    """

    def validate(self, *, full: bool = False) -> None:
        """
        Perform validation checks.  An exception is raised if validation fails.

        By default only cheap validation checks are run.  Pass `full=True`
        for thorough validation checks (potentially O(n)).

        Parameters
        ----------
        full : bool, default False
            If True, run expensive checks, otherwise cheap checks only.

        Raises
        ------
        ArrowInvalid
        """
    def replace_schema_metadata(self, metadata: MetaData | None = None) -> Self:
        """
        Create shallow copy of record batch by replacing schema
        key-value metadata with the indicated new metadata (which may be None,
        which deletes any existing metadata

        Parameters
        ----------
        metadata : dict, default None

        Returns
        -------
        shallow_copy : RecordBatch

        Examples
        --------
        >>> import pyarrow as pa
        >>> n_legs = pa.array([2, 2, 4, 4, 5, 100])

        Constructing a RecordBatch with schema and metadata:

        >>> my_schema = pa.schema(
        ...     [pa.field("n_legs", pa.int64())], metadata={"n_legs": "Number of legs per animal"}
        ... )
        >>> batch = pa.RecordBatch.from_arrays([n_legs], schema=my_schema)
        >>> batch.schema
        n_legs: int64
        -- schema metadata --
        n_legs: 'Number of legs per animal'

        Shallow copy of a RecordBatch with deleted schema metadata:

        >>> batch.replace_schema_metadata().schema
        n_legs: int64
        """
    @property
    def num_columns(self) -> int:
        """
        Number of columns

        Returns
        -------
        int

        Examples
        --------
        >>> import pyarrow as pa
        >>> n_legs = pa.array([2, 2, 4, 4, 5, 100])
        >>> animals = pa.array(
        ...     ["Flamingo", "Parrot", "Dog", "Horse", "Brittle stars", "Centipede"]
        ... )
        >>> batch = pa.RecordBatch.from_arrays([n_legs, animals], names=["n_legs", "animals"])
        >>> batch.num_columns
        2
        """

    @property
    def num_rows(self) -> int:
        """
        Number of rows

        Due to the definition of a RecordBatch, all columns have the same
        number of rows.

        Returns
        -------
        int

        Examples
        --------
        >>> import pyarrow as pa
        >>> n_legs = pa.array([2, 2, 4, 4, 5, 100])
        >>> animals = pa.array(
        ...     ["Flamingo", "Parrot", "Dog", "Horse", "Brittle stars", "Centipede"]
        ... )
        >>> batch = pa.RecordBatch.from_arrays([n_legs, animals], names=["n_legs", "animals"])
        >>> batch.num_rows
        6
        """
    @property
    def schema(self) -> Schema:
        """
        Schema of the RecordBatch and its columns

        Returns
        -------
        pyarrow.Schema

        Examples
        --------
        >>> import pyarrow as pa
        >>> n_legs = pa.array([2, 2, 4, 4, 5, 100])
        >>> animals = pa.array(
        ...     ["Flamingo", "Parrot", "Dog", "Horse", "Brittle stars", "Centipede"]
        ... )
        >>> batch = pa.RecordBatch.from_arrays([n_legs, animals], names=["n_legs", "animals"])
        >>> batch.schema
        n_legs: int64
        animals: string
        """
    @property
    def nbytes(self) -> int:
        """
        Total number of bytes consumed by the elements of the record batch.

        In other words, the sum of bytes from all buffer ranges referenced.

        Unlike `get_total_buffer_size` this method will account for array
        offsets.

        If buffers are shared between arrays then the shared
        portion will only be counted multiple times.

        The dictionary of dictionary arrays will always be counted in their
        entirety even if the array only references a portion of the dictionary.

        Examples
        --------
        >>> import pyarrow as pa
        >>> n_legs = pa.array([2, 2, 4, 4, 5, 100])
        >>> animals = pa.array(
        ...     ["Flamingo", "Parrot", "Dog", "Horse", "Brittle stars", "Centipede"]
        ... )
        >>> batch = pa.RecordBatch.from_arrays([n_legs, animals], names=["n_legs", "animals"])
        >>> batch.nbytes
        116
        """
    def get_total_buffer_size(self) -> int:
        """
        The sum of bytes in each buffer referenced by the record batch

        An array may only reference a portion of a buffer.
        This method will overestimate in this case and return the
        byte size of the entire buffer.

        If a buffer is referenced multiple times then it will
        only be counted once.

        Examples
        --------
        >>> import pyarrow as pa
        >>> n_legs = pa.array([2, 2, 4, 4, 5, 100])
        >>> animals = pa.array(
        ...     ["Flamingo", "Parrot", "Dog", "Horse", "Brittle stars", "Centipede"]
        ... )
        >>> batch = pa.RecordBatch.from_arrays([n_legs, animals], names=["n_legs", "animals"])
        >>> batch.get_total_buffer_size()
        120
        """

    def __sizeof__(self) -> int: ...
    def add_column(
        self, i: int, field_: str | Field, column: ArrayOrChunkedArray[Any] | list
    ) -> Self:
        """
        Add column to RecordBatch at position i.

        A new record batch is returned with the column added, the original record batch
        object is left unchanged.

        Parameters
        ----------
        i : int
            Index to place the column at.
        field_ : str or Field
            If a string is passed then the type is deduced from the column
            data.
        column : Array or value coercible to array
            Column data.

        Returns
        -------
        RecordBatch
            New record batch with the passed column added.

        Examples
        --------
        >>> import pyarrow as pa
        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...     {
        ...         "n_legs": [2, 4, 5, 100],
        ...         "animals": ["Flamingo", "Horse", "Brittle stars", "Centipede"],
        ...     }
        ... )
        >>> batch = pa.RecordBatch.from_pandas(df)

        Add column:

        >>> year = [2021, 2022, 2019, 2021]
        >>> batch.add_column(0, "year", year)
        pyarrow.RecordBatch
        year: int64
        n_legs: int64
        animals: string
        ----
        year: [2021,2022,2019,2021]
        n_legs: [2,4,5,100]
        animals: ["Flamingo","Horse","Brittle stars","Centipede"]

        Original record batch is left unchanged:

        >>> batch
        pyarrow.RecordBatch
        n_legs: int64
        animals: string
        ----
        n_legs: [2,4,5,100]
        animals: ["Flamingo","Horse","Brittle stars","Centipede"]
        """
    def remove_column(self, i: int) -> Self:
        """
        Create new RecordBatch with the indicated column removed.

        Parameters
        ----------
        i : int
            Index of column to remove.

        Returns
        -------
        Table
            New record batch without the column.

        Examples
        --------
        >>> import pyarrow as pa
        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...     {
        ...         "n_legs": [2, 4, 5, 100],
        ...         "animals": ["Flamingo", "Horse", "Brittle stars", "Centipede"],
        ...     }
        ... )
        >>> batch = pa.RecordBatch.from_pandas(df)
        >>> batch.remove_column(1)
        pyarrow.RecordBatch
        n_legs: int64
        ----
        n_legs: [2,4,5,100]
        """
    def set_column(self, i: int, field_: str | Field, column: Array | list) -> Self:
        """
        Replace column in RecordBatch at position.

        Parameters
        ----------
        i : int
            Index to place the column at.
        field_ : str or Field
            If a string is passed then the type is deduced from the column
            data.
        column : Array or value coercible to array
            Column data.

        Returns
        -------
        RecordBatch
            New record batch with the passed column set.

        Examples
        --------
        >>> import pyarrow as pa
        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...     {
        ...         "n_legs": [2, 4, 5, 100],
        ...         "animals": ["Flamingo", "Horse", "Brittle stars", "Centipede"],
        ...     }
        ... )
        >>> batch = pa.RecordBatch.from_pandas(df)

        Replace a column:

        >>> year = [2021, 2022, 2019, 2021]
        >>> batch.set_column(1, "year", year)
        pyarrow.RecordBatch
        n_legs: int64
        year: int64
        ----
        n_legs: [2,4,5,100]
        year: [2021,2022,2019,2021]
        """
    @overload
    def rename_columns(self, names: list[str]) -> Self: ...
    @overload
    def rename_columns(self, names: dict[str, str]) -> Self: ...
    def rename_columns(self, names):
        """
        Create new record batch with columns renamed to provided names.

        Parameters
        ----------
        names : list[str] or dict[str, str]
            List of new column names or mapping of old column names to new column names.

            If a mapping of old to new column names is passed, then all columns which are
            found to match a provided old column name will be renamed to the new column name.
            If any column names are not found in the mapping, a KeyError will be raised.

        Raises
        ------
        KeyError
            If any of the column names passed in the names mapping do not exist.

        Returns
        -------
        RecordBatch

        Examples
        --------
        >>> import pyarrow as pa
        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...     {
        ...         "n_legs": [2, 4, 5, 100],
        ...         "animals": ["Flamingo", "Horse", "Brittle stars", "Centipede"],
        ...     }
        ... )
        >>> batch = pa.RecordBatch.from_pandas(df)
        >>> new_names = ["n", "name"]
        >>> batch.rename_columns(new_names)
        pyarrow.RecordBatch
        n: int64
        name: string
        ----
        n: [2,4,5,100]
        name: ["Flamingo","Horse","Brittle stars","Centipede"]
        >>> new_names = {"n_legs": "n", "animals": "name"}
        >>> batch.rename_columns(new_names)
        pyarrow.RecordBatch
        n: int64
        name: string
        ----
        n: [2,4,5,100]
        name: ["Flamingo","Horse","Brittle stars","Centipede"]
        """
    def serialize(self, memory_pool: MemoryPool | None = None) -> Buffer:
        """
        Write RecordBatch to Buffer as encapsulated IPC message, which does not
        include a Schema.

        To reconstruct a RecordBatch from the encapsulated IPC message Buffer
        returned by this function, a Schema must be passed separately. See
        Examples.

        Parameters
        ----------
        memory_pool : MemoryPool, default None
            Uses default memory pool if not specified

        Returns
        -------
        serialized : Buffer

        Examples
        --------
        >>> import pyarrow as pa
        >>> n_legs = pa.array([2, 2, 4, 4, 5, 100])
        >>> animals = pa.array(
        ...     ["Flamingo", "Parrot", "Dog", "Horse", "Brittle stars", "Centipede"]
        ... )
        >>> batch = pa.RecordBatch.from_arrays([n_legs, animals], names=["n_legs", "animals"])
        >>> buf = batch.serialize()
        >>> buf
        <pyarrow.Buffer address=0x... size=... is_cpu=True is_mutable=True>

        Reconstruct RecordBatch from IPC message Buffer and original Schema

        >>> pa.ipc.read_record_batch(buf, batch.schema)
        pyarrow.RecordBatch
        n_legs: int64
        animals: string
        ----
        n_legs: [2,2,4,4,5,100]
        animals: ["Flamingo","Parrot","Dog","Horse","Brittle stars","Centipede"]
        """
    def slice(self, offset: int = 0, length: int | None = None) -> Self:
        """
        Compute zero-copy slice of this RecordBatch

        Parameters
        ----------
        offset : int, default 0
            Offset from start of record batch to slice
        length : int, default None
            Length of slice (default is until end of batch starting from
            offset)

        Returns
        -------
        sliced : RecordBatch

        Examples
        --------
        >>> import pyarrow as pa
        >>> n_legs = pa.array([2, 2, 4, 4, 5, 100])
        >>> animals = pa.array(
        ...     ["Flamingo", "Parrot", "Dog", "Horse", "Brittle stars", "Centipede"]
        ... )
        >>> batch = pa.RecordBatch.from_arrays([n_legs, animals], names=["n_legs", "animals"])
        >>> batch.to_pandas()
           n_legs        animals
        0       2       Flamingo
        1       2         Parrot
        2       4            Dog
        3       4          Horse
        4       5  Brittle stars
        5     100      Centipede
        >>> batch.slice(offset=3).to_pandas()
           n_legs        animals
        0       4          Horse
        1       5  Brittle stars
        2     100      Centipede
        >>> batch.slice(length=2).to_pandas()
           n_legs   animals
        0       2  Flamingo
        1       2    Parrot
        >>> batch.slice(offset=3, length=1).to_pandas()
           n_legs animals
        0       4   Horse
        """
    def equals(self, other: Self, check_metadata: bool = False) -> bool:
        """
        Check if contents of two record batches are equal.

        Parameters
        ----------
        other : pyarrow.RecordBatch
            RecordBatch to compare against.
        check_metadata : bool, default False
            Whether schema metadata equality should be checked as well.

        Returns
        -------
        are_equal : bool

        Examples
        --------
        >>> import pyarrow as pa
        >>> n_legs = pa.array([2, 2, 4, 4, 5, 100])
        >>> animals = pa.array(
        ...     ["Flamingo", "Parrot", "Dog", "Horse", "Brittle stars", "Centipede"]
        ... )
        >>> batch = pa.RecordBatch.from_arrays([n_legs, animals], names=["n_legs", "animals"])
        >>> batch_0 = pa.record_batch([])
        >>> batch_1 = pa.RecordBatch.from_arrays(
        ...     [n_legs, animals],
        ...     names=["n_legs", "animals"],
        ...     metadata={"n_legs": "Number of legs per animal"},
        ... )
        >>> batch.equals(batch)
        True
        >>> batch.equals(batch_0)
        False
        >>> batch.equals(batch_1)
        True
        >>> batch.equals(batch_1, check_metadata=True)
        False
        """
    def select(self, columns: Iterable[str] | Iterable[int] | NDArray[np.str_]) -> Self:
        """
        Select columns of the RecordBatch.

        Returns a new RecordBatch with the specified columns, and metadata
        preserved.

        Parameters
        ----------
        columns : list-like
            The column names or integer indices to select.

        Returns
        -------
        RecordBatch

        Examples
        --------
        >>> import pyarrow as pa
        >>> n_legs = pa.array([2, 2, 4, 4, 5, 100])
        >>> animals = pa.array(
        ...     ["Flamingo", "Parrot", "Dog", "Horse", "Brittle stars", "Centipede"]
        ... )
        >>> batch = pa.record_batch([n_legs, animals], names=["n_legs", "animals"])

        Select columns my indices:

        >>> batch.select([1])
        pyarrow.RecordBatch
        animals: string
        ----
        animals: ["Flamingo","Parrot","Dog","Horse","Brittle stars","Centipede"]

        Select columns by names:

        >>> batch.select(["n_legs"])
        pyarrow.RecordBatch
        n_legs: int64
        ----
        n_legs: [2,2,4,4,5,100]
        """
    def cast(
        self, target_schema: Schema, safe: bool | None = None, options: CastOptions | None = None
    ) -> Self:
        """
        Cast record batch values to another schema.

        Parameters
        ----------
        target_schema : Schema
            Schema to cast to, the names and order of fields must match.
        safe : bool, default True
            Check for overflows or other unsafe conversions.
        options : CastOptions, default None
            Additional checks pass by CastOptions

        Returns
        -------
        RecordBatch

        Examples
        --------
        >>> import pyarrow as pa
        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...     {
        ...         "n_legs": [2, 4, 5, 100],
        ...         "animals": ["Flamingo", "Horse", "Brittle stars", "Centipede"],
        ...     }
        ... )
        >>> batch = pa.RecordBatch.from_pandas(df)
        >>> batch.schema
        n_legs: int64
        animals: string
        -- schema metadata --
        pandas: '{"index_columns": [{"kind": "range", "name": null, "start": 0, ...

        Define new schema and cast batch values:

        >>> my_schema = pa.schema(
        ...     [pa.field("n_legs", pa.duration("s")), pa.field("animals", pa.string())]
        ... )
        >>> batch.cast(target_schema=my_schema)
        pyarrow.RecordBatch
        n_legs: duration[s]
        animals: string
        ----
        n_legs: [2,4,5,100]
        animals: ["Flamingo","Horse","Brittle stars","Centipede"]
        """
    @classmethod
    def from_arrays(
        cls,
        arrays: Collection[Array],
        names: list[str] | None = None,
        schema: Schema | None = None,
        metadata: MetaData | None = None,
    ) -> Self:
        """
        Construct a RecordBatch from multiple pyarrow.Arrays

        Parameters
        ----------
        arrays : list of pyarrow.Array
            One for each field in RecordBatch
        names : list of str, optional
            Names for the batch fields. If not passed, schema must be passed
        schema : Schema, default None
            Schema for the created batch. If not passed, names must be passed
        metadata : dict or Mapping, default None
            Optional metadata for the schema (if inferred).

        Returns
        -------
        pyarrow.RecordBatch

        Examples
        --------
        >>> import pyarrow as pa
        >>> n_legs = pa.array([2, 2, 4, 4, 5, 100])
        >>> animals = pa.array(
        ...     ["Flamingo", "Parrot", "Dog", "Horse", "Brittle stars", "Centipede"]
        ... )
        >>> names = ["n_legs", "animals"]

        Construct a RecordBatch from pyarrow Arrays using names:

        >>> pa.RecordBatch.from_arrays([n_legs, animals], names=names)
        pyarrow.RecordBatch
        n_legs: int64
        animals: string
        ----
        n_legs: [2,2,4,4,5,100]
        animals: ["Flamingo","Parrot","Dog","Horse","Brittle stars","Centipede"]
        >>> pa.RecordBatch.from_arrays([n_legs, animals], names=names).to_pandas()
           n_legs        animals
        0       2       Flamingo
        1       2         Parrot
        2       4            Dog
        3       4          Horse
        4       5  Brittle stars
        5     100      Centipede

        Construct a RecordBatch from pyarrow Arrays using schema:

        >>> my_schema = pa.schema(
        ...     [pa.field("n_legs", pa.int64()), pa.field("animals", pa.string())],
        ...     metadata={"n_legs": "Number of legs per animal"},
        ... )
        >>> pa.RecordBatch.from_arrays([n_legs, animals], schema=my_schema).to_pandas()
           n_legs        animals
        0       2       Flamingo
        1       2         Parrot
        2       4            Dog
        3       4          Horse
        4       5  Brittle stars
        5     100      Centipede
        >>> pa.RecordBatch.from_arrays([n_legs, animals], schema=my_schema).schema
        n_legs: int64
        animals: string
        -- schema metadata --
        n_legs: 'Number of legs per animal'
        """
    @classmethod
    def from_pandas(
        cls,
        df: pd.DataFrame,
        schema: Schema | None = None,
        preserve_index: bool | None = None,
        nthreads: int | None = None,
        columns: list[str] | None = None,
    ) -> Self:
        """
        Convert pandas.DataFrame to an Arrow RecordBatch

        Parameters
        ----------
        df : pandas.DataFrame
        schema : pyarrow.Schema, optional
            The expected schema of the RecordBatch. This can be used to
            indicate the type of columns if we cannot infer it automatically.
            If passed, the output will have exactly this schema. Columns
            specified in the schema that are not found in the DataFrame columns
            or its index will raise an error. Additional columns or index
            levels in the DataFrame which are not specified in the schema will
            be ignored.
        preserve_index : bool, optional
            Whether to store the index as an additional column in the resulting
            ``RecordBatch``. The default of None will store the index as a
            column, except for RangeIndex which is stored as metadata only. Use
            ``preserve_index=True`` to force it to be stored as a column.
        nthreads : int, default None
            If greater than 1, convert columns to Arrow in parallel using
            indicated number of threads. By default, this follows
            :func:`pyarrow.cpu_count` (may use up to system CPU count threads).
        columns : list, optional
           List of column to be converted. If None, use all columns.

        Returns
        -------
        pyarrow.RecordBatch


        Examples
        --------
        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...     {
        ...         "year": [2020, 2022, 2021, 2022],
        ...         "month": [3, 5, 7, 9],
        ...         "day": [1, 5, 9, 13],
        ...         "n_legs": [2, 4, 5, 100],
        ...         "animals": ["Flamingo", "Horse", "Brittle stars", "Centipede"],
        ...     }
        ... )

        Convert pandas DataFrame to RecordBatch:

        >>> import pyarrow as pa
        >>> pa.RecordBatch.from_pandas(df)
        pyarrow.RecordBatch
        year: int64
        month: int64
        day: int64
        n_legs: int64
        animals: string
        ----
        year: [2020,2022,2021,2022]
        month: [3,5,7,9]
        day: [1,5,9,13]
        n_legs: [2,4,5,100]
        animals: ["Flamingo","Horse","Brittle stars","Centipede"]

        Convert pandas DataFrame to RecordBatch using schema:

        >>> my_schema = pa.schema(
        ...     [pa.field("n_legs", pa.int64()), pa.field("animals", pa.string())],
        ...     metadata={"n_legs": "Number of legs per animal"},
        ... )
        >>> pa.RecordBatch.from_pandas(df, schema=my_schema)
        pyarrow.RecordBatch
        n_legs: int64
        animals: string
        ----
        n_legs: [2,4,5,100]
        animals: ["Flamingo","Horse","Brittle stars","Centipede"]

        Convert pandas DataFrame to RecordBatch specifying columns:

        >>> pa.RecordBatch.from_pandas(df, columns=["n_legs"])
        pyarrow.RecordBatch
        n_legs: int64
        ----
        n_legs: [2,4,5,100]
        """
    @classmethod
    def from_struct_array(
        cls, struct_array: StructArray | ChunkedArray[scalar.StructScalar]
    ) -> Self:
        """
        Construct a RecordBatch from a StructArray.

        Each field in the StructArray will become a column in the resulting
        ``RecordBatch``.

        Parameters
        ----------
        struct_array : StructArray
            Array to construct the record batch from.

        Returns
        -------
        pyarrow.RecordBatch

        Examples
        --------
        >>> import pyarrow as pa
        >>> struct = pa.array([{"n_legs": 2, "animals": "Parrot"}, {"year": 2022, "n_legs": 4}])
        >>> pa.RecordBatch.from_struct_array(struct).to_pandas()
          animals  n_legs    year
        0  Parrot       2     NaN
        1    None       4  2022.0
        """
    def to_struct_array(self) -> StructArray:
        """
        Convert to a struct array.
        """
    def to_tensor(
        self,
        null_to_nan: bool = False,
        row_major: bool = True,
        memory_pool: MemoryPool | None = None,
    ) -> Tensor:
        """
        Convert to a :class:`~pyarrow.Tensor`.

        RecordBatches that can be converted have fields of type signed or unsigned
        integer or float, including all bit-widths.

        ``null_to_nan`` is ``False`` by default and this method will raise an error in case
        any nulls are present. RecordBatches with nulls can be converted with ``null_to_nan``
        set to ``True``. In this case null values are converted to ``NaN`` and integer type
        arrays are promoted to the appropriate float type.

        Parameters
        ----------
        null_to_nan : bool, default False
            Whether to write null values in the result as ``NaN``.
        row_major : bool, default True
            Whether resulting Tensor is row-major or column-major
        memory_pool : MemoryPool, default None
            For memory allocations, if required, otherwise use default pool

        Examples
        --------
        >>> import pyarrow as pa
        >>> batch = pa.record_batch(
        ...     [
        ...         pa.array([1, 2, 3, 4, None], type=pa.int32()),
        ...         pa.array([10, 20, 30, 40, None], type=pa.float32()),
        ...     ],
        ...     names=["a", "b"],
        ... )

        >>> batch
        pyarrow.RecordBatch
        a: int32
        b: float
        ----
        a: [1,2,3,4,null]
        b: [10,20,30,40,null]

        Convert a RecordBatch to row-major Tensor with null values
        written as ``NaN``s

        >>> batch.to_tensor(null_to_nan=True)
        <pyarrow.Tensor>
        type: double
        shape: (5, 2)
        strides: (16, 8)
        >>> batch.to_tensor(null_to_nan=True).to_numpy()
        array([[ 1., 10.],
               [ 2., 20.],
               [ 3., 30.],
               [ 4., 40.],
               [nan, nan]])

        Convert a RecordBatch to column-major Tensor

        >>> batch.to_tensor(null_to_nan=True, row_major=False)
        <pyarrow.Tensor>
        type: double
        shape: (5, 2)
        strides: (8, 40)
        >>> batch.to_tensor(null_to_nan=True, row_major=False).to_numpy()
        array([[ 1., 10.],
               [ 2., 20.],
               [ 3., 30.],
               [ 4., 40.],
               [nan, nan]])
        """
    def _export_to_c(self, out_ptr: int, out_schema_ptr: int = 0):
        """
        Export to a C ArrowArray struct, given its pointer.

        If a C ArrowSchema struct pointer is also given, the record batch
        schema is exported to it at the same time.

        Parameters
        ----------
        out_ptr: int
            The raw pointer to a C ArrowArray struct.
        out_schema_ptr: int (optional)
            The raw pointer to a C ArrowSchema struct.

        Be careful: if you don't pass the ArrowArray struct to a consumer,
        array memory will leak.  This is a low-level function intended for
        expert users.
        """
    @classmethod
    def _import_from_c(cls, in_ptr: int, schema: Schema) -> Self:
        """
        Import RecordBatch from a C ArrowArray struct, given its pointer
        and the imported schema.

        Parameters
        ----------
        in_ptr: int
            The raw pointer to a C ArrowArray struct.
        type: Schema or int
            Either a Schema object, or the raw pointer to a C ArrowSchema
            struct.

        This is a low-level function intended for expert users.
        """
    def __arrow_c_array__(self, requested_schema=None):
        """
        Get a pair of PyCapsules containing a C ArrowArray representation of the object.

        Parameters
        ----------
        requested_schema : PyCapsule | None
            A PyCapsule containing a C ArrowSchema representation of a requested
            schema. PyArrow will attempt to cast the batch to this schema.
            If None, the batch will be returned as-is, with a schema matching the
            one returned by :meth:`__arrow_c_schema__()`.

        Returns
        -------
        Tuple[PyCapsule, PyCapsule]
            A pair of PyCapsules containing a C ArrowSchema and ArrowArray,
            respectively.
        """
    def __arrow_c_stream__(self, requested_schema=None):
        """
        Export the batch as an Arrow C stream PyCapsule.

        Parameters
        ----------
        requested_schema : PyCapsule, default None
            The schema to which the stream should be casted, passed as a
            PyCapsule containing a C ArrowSchema representation of the
            requested schema.
            Currently, this is not supported and will raise a
            NotImplementedError if the schema doesn't match the current schema.

        Returns
        -------
        PyCapsule
        """
    @classmethod
    def _import_from_c_capsule(cls, schema_capsule, array_capsule) -> Self:
        """
        Import RecordBatch from a pair of PyCapsules containing a C ArrowSchema
        and ArrowArray, respectively.

        Parameters
        ----------
        schema_capsule : PyCapsule
            A PyCapsule containing a C ArrowSchema representation of the schema.
        array_capsule : PyCapsule
            A PyCapsule containing a C ArrowArray representation of the array.

        Returns
        -------
        pyarrow.RecordBatch
        """
    def _export_to_c_device(self, out_ptr: int, out_schema_ptr: int = 0) -> None:
        """
        Export to a C ArrowDeviceArray struct, given its pointer.

        If a C ArrowSchema struct pointer is also given, the record batch
        schema is exported to it at the same time.

        Parameters
        ----------
        out_ptr: int
            The raw pointer to a C ArrowDeviceArray struct.
        out_schema_ptr: int (optional)
            The raw pointer to a C ArrowSchema struct.

        Be careful: if you don't pass the ArrowDeviceArray struct to a consumer,
        array memory will leak.  This is a low-level function intended for
        expert users.
        """
    @classmethod
    def _import_from_c_device(cls, in_ptr: int, schema: Schema) -> Self:
        """
        Import RecordBatch from a C ArrowDeviceArray struct, given its pointer
        and the imported schema.

        Parameters
        ----------
        in_ptr: int
            The raw pointer to a C ArrowDeviceArray struct.
        type: Schema or int
            Either a Schema object, or the raw pointer to a C ArrowSchema
            struct.

        This is a low-level function intended for expert users.
        """
    def __arrow_c_device_array__(self, requested_schema=None, **kwargs):
        """
        Get a pair of PyCapsules containing a C ArrowDeviceArray representation
        of the object.

        Parameters
        ----------
        requested_schema : PyCapsule | None
            A PyCapsule containing a C ArrowSchema representation of a requested
            schema. PyArrow will attempt to cast the batch to this data type.
            If None, the batch will be returned as-is, with a type matching the
            one returned by :meth:`__arrow_c_schema__()`.
        kwargs
            Currently no additional keyword arguments are supported, but
            this method will accept any keyword with a value of ``None``
            for compatibility with future keywords.

        Returns
        -------
        Tuple[PyCapsule, PyCapsule]
            A pair of PyCapsules containing a C ArrowSchema and ArrowDeviceArray,
            respectively.
        """
    @classmethod
    def _import_from_c_device_capsule(cls, schema_capsule, array_capsule) -> Self:
        """
        Import RecordBatch from a pair of PyCapsules containing a
        C ArrowSchema and ArrowDeviceArray, respectively.

        Parameters
        ----------
        schema_capsule : PyCapsule
            A PyCapsule containing a C ArrowSchema representation of the schema.
        array_capsule : PyCapsule
            A PyCapsule containing a C ArrowDeviceArray representation of the array.

        Returns
        -------
        pyarrow.RecordBatch
        """
    @property
    def device_type(self) -> DeviceAllocationType:
        """
        The device type where the arrays in the RecordBatch reside.

        Returns
        -------
        DeviceAllocationType
        """
    @property
    def is_cpu(self) -> bool:
        """
        Whether the RecordBatch's arrays are CPU-accessible.
        """
    def copy_to(self, destination: MemoryManager | Device) -> Self:
        """
        Copy the entire RecordBatch to destination device.

        This copies each column of the record batch to create
        a new record batch where all underlying buffers for the columns have
        been copied to the destination MemoryManager.

        Parameters
        ----------
        destination : pyarrow.MemoryManager or pyarrow.Device
            The destination device to copy the array to.

        Returns
        -------
        RecordBatch
        """

def table_to_blocks(options, table: Table, categories, extension_columns): ...

JoinType: TypeAlias = Literal[
    "left semi",
    "right semi",
    "left anti",
    "right anti",
    "inner",
    "left outer",
    "right outer",
    "full outer",
]

class Table(_Tabular[ChunkedArray[Any]]):
    """
    A collection of top-level named, equal length Arrow arrays.

    Warnings
    --------
    Do not call this class's constructor directly, use one of the ``from_*``
    methods instead.

    Examples
    --------
    >>> import pyarrow as pa
    >>> n_legs = pa.array([2, 4, 5, 100])
    >>> animals = pa.array(["Flamingo", "Horse", "Brittle stars", "Centipede"])
    >>> names = ["n_legs", "animals"]

    Construct a Table from arrays:

    >>> pa.Table.from_arrays([n_legs, animals], names=names)
    pyarrow.Table
    n_legs: int64
    animals: string
    ----
    n_legs: [[2,4,5,100]]
    animals: [["Flamingo","Horse","Brittle stars","Centipede"]]

    Construct a Table from a RecordBatch:

    >>> batch = pa.record_batch([n_legs, animals], names=names)
    >>> pa.Table.from_batches([batch])
    pyarrow.Table
    n_legs: int64
    animals: string
    ----
    n_legs: [[2,4,5,100]]
    animals: [["Flamingo","Horse","Brittle stars","Centipede"]]

    Construct a Table from pandas DataFrame:

    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     {
    ...         "year": [2020, 2022, 2019, 2021],
    ...         "n_legs": [2, 4, 5, 100],
    ...         "animals": ["Flamingo", "Horse", "Brittle stars", "Centipede"],
    ...     }
    ... )
    >>> pa.Table.from_pandas(df)
    pyarrow.Table
    year: int64
    n_legs: int64
    animals: string
    ----
    year: [[2020,2022,2019,2021]]
    n_legs: [[2,4,5,100]]
    animals: [["Flamingo","Horse","Brittle stars","Centipede"]]

    Construct a Table from a dictionary of arrays:

    >>> pydict = {"n_legs": n_legs, "animals": animals}
    >>> pa.Table.from_pydict(pydict)
    pyarrow.Table
    n_legs: int64
    animals: string
    ----
    n_legs: [[2,4,5,100]]
    animals: [["Flamingo","Horse","Brittle stars","Centipede"]]
    >>> pa.Table.from_pydict(pydict).schema
    n_legs: int64
    animals: string

    Construct a Table from a dictionary of arrays with metadata:

    >>> my_metadata = {"n_legs": "Number of legs per animal"}
    >>> pa.Table.from_pydict(pydict, metadata=my_metadata).schema
    n_legs: int64
    animals: string
    -- schema metadata --
    n_legs: 'Number of legs per animal'

    Construct a Table from a list of rows:

    >>> pylist = [{"n_legs": 2, "animals": "Flamingo"}, {"year": 2021, "animals": "Centipede"}]
    >>> pa.Table.from_pylist(pylist)
    pyarrow.Table
    n_legs: int64
    animals: string
    ----
    n_legs: [[2,null]]
    animals: [["Flamingo","Centipede"]]

    Construct a Table from a list of rows with pyarrow schema:

    >>> my_schema = pa.schema(
    ...     [
    ...         pa.field("year", pa.int64()),
    ...         pa.field("n_legs", pa.int64()),
    ...         pa.field("animals", pa.string()),
    ...     ],
    ...     metadata={"year": "Year of entry"},
    ... )
    >>> pa.Table.from_pylist(pylist, schema=my_schema).schema
    year: int64
    n_legs: int64
    animals: string
    -- schema metadata --
    year: 'Year of entry'

    Construct a Table with :func:`pyarrow.table`:

    >>> pa.table([n_legs, animals], names=names)
    pyarrow.Table
    n_legs: int64
    animals: string
    ----
    n_legs: [[2,4,5,100]]
    animals: [["Flamingo","Horse","Brittle stars","Centipede"]]
    """

    def validate(self, *, full: bool = False) -> None:
        """
        Perform validation checks.  An exception is raised if validation fails.

        By default only cheap validation checks are run.  Pass `full=True`
        for thorough validation checks (potentially O(n)).

        Parameters
        ----------
        full : bool, default False
            If True, run expensive checks, otherwise cheap checks only.

        Raises
        ------
        ArrowInvalid
        """
    def slice(self, offset: int = 0, length: int | None = None) -> Self:
        """
        Compute zero-copy slice of this Table.

        Parameters
        ----------
        offset : int, default 0
            Offset from start of table to slice.
        length : int, default None
            Length of slice (default is until end of table starting from
            offset).

        Returns
        -------
        Table

        Examples
        --------
        >>> import pyarrow as pa
        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...     {
        ...         "year": [2020, 2022, 2019, 2021],
        ...         "n_legs": [2, 4, 5, 100],
        ...         "animals": ["Flamingo", "Horse", "Brittle stars", "Centipede"],
        ...     }
        ... )
        >>> table = pa.Table.from_pandas(df)
        >>> table.slice(length=3)
        pyarrow.Table
        year: int64
        n_legs: int64
        animals: string
        ----
        year: [[2020,2022,2019]]
        n_legs: [[2,4,5]]
        animals: [["Flamingo","Horse","Brittle stars"]]
        >>> table.slice(offset=2)
        pyarrow.Table
        year: int64
        n_legs: int64
        animals: string
        ----
        year: [[2019,2021]]
        n_legs: [[5,100]]
        animals: [["Brittle stars","Centipede"]]
        >>> table.slice(offset=2, length=1)
        pyarrow.Table
        year: int64
        n_legs: int64
        animals: string
        ----
        year: [[2019]]
        n_legs: [[5]]
        animals: [["Brittle stars"]]
        """
    def select(self, columns: Iterable[str] | Iterable[int] | NDArray[np.str_]) -> Self:
        """
        Select columns of the Table.

        Returns a new Table with the specified columns, and metadata
        preserved.

        Parameters
        ----------
        columns : list-like
            The column names or integer indices to select.

        Returns
        -------
        Table

        Examples
        --------
        >>> import pyarrow as pa
        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...     {
        ...         "year": [2020, 2022, 2019, 2021],
        ...         "n_legs": [2, 4, 5, 100],
        ...         "animals": ["Flamingo", "Horse", "Brittle stars", "Centipede"],
        ...     }
        ... )
        >>> table = pa.Table.from_pandas(df)
        >>> table.select([0, 1])
        pyarrow.Table
        year: int64
        n_legs: int64
        ----
        year: [[2020,2022,2019,2021]]
        n_legs: [[2,4,5,100]]
        >>> table.select(["year"])
        pyarrow.Table
        year: int64
        ----
        year: [[2020,2022,2019,2021]]
        """
    def replace_schema_metadata(self, metadata: MetaData | None = None) -> Self:
        """
        Create shallow copy of table by replacing schema
        key-value metadata with the indicated new metadata (which may be None),
        which deletes any existing metadata.

        Parameters
        ----------
        metadata : dict, default None

        Returns
        -------
        Table

        Examples
        --------
        >>> import pyarrow as pa
        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...     {
        ...         "year": [2020, 2022, 2019, 2021],
        ...         "n_legs": [2, 4, 5, 100],
        ...         "animals": ["Flamingo", "Horse", "Brittle stars", "Centipede"],
        ...     }
        ... )
        >>> table = pa.Table.from_pandas(df)

        Constructing a Table with pyarrow schema and metadata:

        >>> my_schema = pa.schema(
        ...     [pa.field("n_legs", pa.int64()), pa.field("animals", pa.string())],
        ...     metadata={"n_legs": "Number of legs per animal"},
        ... )
        >>> table = pa.table(df, my_schema)
        >>> table.schema
        n_legs: int64
        animals: string
        -- schema metadata --
        n_legs: 'Number of legs per animal'
        pandas: ...

        Create a shallow copy of a Table with deleted schema metadata:

        >>> table.replace_schema_metadata().schema
        n_legs: int64
        animals: string

        Create a shallow copy of a Table with new schema metadata:

        >>> metadata = {"animals": "Which animal"}
        >>> table.replace_schema_metadata(metadata=metadata).schema
        n_legs: int64
        animals: string
        -- schema metadata --
        animals: 'Which animal'
        """
    def flatten(self, memory_pool: MemoryPool | None = None) -> Self:
        """
        Flatten this Table.

        Each column with a struct type is flattened
        into one column per struct field.  Other columns are left unchanged.

        Parameters
        ----------
        memory_pool : MemoryPool, default None
            For memory allocations, if required, otherwise use default pool

        Returns
        -------
        Table

        Examples
        --------
        >>> import pyarrow as pa
        >>> struct = pa.array([{"n_legs": 2, "animals": "Parrot"}, {"year": 2022, "n_legs": 4}])
        >>> month = pa.array([4, 6])
        >>> table = pa.Table.from_arrays([struct, month], names=["a", "month"])
        >>> table
        pyarrow.Table
        a: struct<animals: string, n_legs: int64, year: int64>
          child 0, animals: string
          child 1, n_legs: int64
          child 2, year: int64
        month: int64
        ----
        a: [
          -- is_valid: all not null
          -- child 0 type: string
        ["Parrot",null]
          -- child 1 type: int64
        [2,4]
          -- child 2 type: int64
        [null,2022]]
        month: [[4,6]]

        Flatten the columns with struct field:

        >>> table.flatten()
        pyarrow.Table
        a.animals: string
        a.n_legs: int64
        a.year: int64
        month: int64
        ----
        a.animals: [["Parrot",null]]
        a.n_legs: [[2,4]]
        a.year: [[null,2022]]
        month: [[4,6]]
        """
    def combine_chunks(self, memory_pool: MemoryPool | None = None) -> Self:
        """
        Make a new table by combining the chunks this table has.

        All the underlying chunks in the ChunkedArray of each column are
        concatenated into zero or one chunk.

        Parameters
        ----------
        memory_pool : MemoryPool, default None
            For memory allocations, if required, otherwise use default pool.

        Returns
        -------
        Table

        Examples
        --------
        >>> import pyarrow as pa
        >>> n_legs = pa.chunked_array([[2, 2, 4], [4, 5, 100]])
        >>> animals = pa.chunked_array(
        ...     [["Flamingo", "Parrot", "Dog"], ["Horse", "Brittle stars", "Centipede"]]
        ... )
        >>> names = ["n_legs", "animals"]
        >>> table = pa.table([n_legs, animals], names=names)
        >>> table
        pyarrow.Table
        n_legs: int64
        animals: string
        ----
        n_legs: [[2,2,4],[4,5,100]]
        animals: [["Flamingo","Parrot","Dog"],["Horse","Brittle stars","Centipede"]]
        >>> table.combine_chunks()
        pyarrow.Table
        n_legs: int64
        animals: string
        ----
        n_legs: [[2,2,4,4,5,100]]
        animals: [["Flamingo","Parrot","Dog","Horse","Brittle stars","Centipede"]]
        """
    def unify_dictionaries(self, memory_pool: MemoryPool | None = None) -> Self:
        """
        Unify dictionaries across all chunks.

        This method returns an equivalent table, but where all chunks of
        each column share the same dictionary values.  Dictionary indices
        are transposed accordingly.

        Columns without dictionaries are returned unchanged.

        Parameters
        ----------
        memory_pool : MemoryPool, default None
            For memory allocations, if required, otherwise use default pool

        Returns
        -------
        Table

        Examples
        --------
        >>> import pyarrow as pa
        >>> arr_1 = pa.array(["Flamingo", "Parrot", "Dog"]).dictionary_encode()
        >>> arr_2 = pa.array(["Horse", "Brittle stars", "Centipede"]).dictionary_encode()
        >>> c_arr = pa.chunked_array([arr_1, arr_2])
        >>> table = pa.table([c_arr], names=["animals"])
        >>> table
        pyarrow.Table
        animals: dictionary<values=string, indices=int32, ordered=0>
        ----
        animals: [  -- dictionary:
        ["Flamingo","Parrot","Dog"]  -- indices:
        [0,1,2],  -- dictionary:
        ["Horse","Brittle stars","Centipede"]  -- indices:
        [0,1,2]]

        Unify dictionaries across both chunks:

        >>> table.unify_dictionaries()
        pyarrow.Table
        animals: dictionary<values=string, indices=int32, ordered=0>
        ----
        animals: [  -- dictionary:
        ["Flamingo","Parrot","Dog","Horse","Brittle stars","Centipede"]  -- indices:
        [0,1,2],  -- dictionary:
        ["Flamingo","Parrot","Dog","Horse","Brittle stars","Centipede"]  -- indices:
        [3,4,5]]
        """
    def equals(self, other: Self, check_metadata: bool = False) -> Self:
        """
        Check if contents of two tables are equal.

        Parameters
        ----------
        other : pyarrow.Table
            Table to compare against.
        check_metadata : bool, default False
            Whether schema metadata equality should be checked as well.

        Returns
        -------
        bool

        Examples
        --------
        >>> import pyarrow as pa
        >>> n_legs = pa.array([2, 2, 4, 4, 5, 100])
        >>> animals = pa.array(
        ...     ["Flamingo", "Parrot", "Dog", "Horse", "Brittle stars", "Centipede"]
        ... )
        >>> names = ["n_legs", "animals"]
        >>> table = pa.Table.from_arrays([n_legs, animals], names=names)
        >>> table_0 = pa.Table.from_arrays([])
        >>> table_1 = pa.Table.from_arrays(
        ...     [n_legs, animals], names=names, metadata={"n_legs": "Number of legs per animal"}
        ... )
        >>> table.equals(table)
        True
        >>> table.equals(table_0)
        False
        >>> table.equals(table_1)
        True
        >>> table.equals(table_1, check_metadata=True)
        False
        """
    def cast(
        self, target_schema: Schema, safe: bool | None = None, options: CastOptions | None = None
    ) -> Self:
        """
        Cast table values to another schema.

        Parameters
        ----------
        target_schema : Schema
            Schema to cast to, the names and order of fields must match.
        safe : bool, default True
            Check for overflows or other unsafe conversions.
        options : CastOptions, default None
            Additional checks pass by CastOptions

        Returns
        -------
        Table

        Examples
        --------
        >>> import pyarrow as pa
        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...     {
        ...         "n_legs": [2, 4, 5, 100],
        ...         "animals": ["Flamingo", "Horse", "Brittle stars", "Centipede"],
        ...     }
        ... )
        >>> table = pa.Table.from_pandas(df)
        >>> table.schema
        n_legs: int64
        animals: string
        -- schema metadata --
        pandas: '{"index_columns": [{"kind": "range", "name": null, "start": 0, ...

        Define new schema and cast table values:

        >>> my_schema = pa.schema(
        ...     [pa.field("n_legs", pa.duration("s")), pa.field("animals", pa.string())]
        ... )
        >>> table.cast(target_schema=my_schema)
        pyarrow.Table
        n_legs: duration[s]
        animals: string
        ----
        n_legs: [[2,4,5,100]]
        animals: [["Flamingo","Horse","Brittle stars","Centipede"]]
        """
    @classmethod
    def from_pandas(
        cls,
        df: pd.DataFrame,
        schema: Schema | None = None,
        preserve_index: bool | None = None,
        nthreads: int | None = None,
        columns: list[str] | None = None,
        safe: bool = True,
    ) -> Self:
        """
        Convert pandas.DataFrame to an Arrow Table.

        The column types in the resulting Arrow Table are inferred from the
        dtypes of the pandas.Series in the DataFrame. In the case of non-object
        Series, the NumPy dtype is translated to its Arrow equivalent. In the
        case of `object`, we need to guess the datatype by looking at the
        Python objects in this Series.

        Be aware that Series of the `object` dtype don't carry enough
        information to always lead to a meaningful Arrow type. In the case that
        we cannot infer a type, e.g. because the DataFrame is of length 0 or
        the Series only contains None/nan objects, the type is set to
        null. This behavior can be avoided by constructing an explicit schema
        and passing it to this function.

        Parameters
        ----------
        df : pandas.DataFrame
        schema : pyarrow.Schema, optional
            The expected schema of the Arrow Table. This can be used to
            indicate the type of columns if we cannot infer it automatically.
            If passed, the output will have exactly this schema. Columns
            specified in the schema that are not found in the DataFrame columns
            or its index will raise an error. Additional columns or index
            levels in the DataFrame which are not specified in the schema will
            be ignored.
        preserve_index : bool, optional
            Whether to store the index as an additional column in the resulting
            ``Table``. The default of None will store the index as a column,
            except for RangeIndex which is stored as metadata only. Use
            ``preserve_index=True`` to force it to be stored as a column.
        nthreads : int, default None
            If greater than 1, convert columns to Arrow in parallel using
            indicated number of threads. By default, this follows
            :func:`pyarrow.cpu_count` (may use up to system CPU count threads).
        columns : list, optional
           List of column to be converted. If None, use all columns.
        safe : bool, default True
           Check for overflows or other unsafe conversions.

        Returns
        -------
        Table

        Examples
        --------
        >>> import pyarrow as pa
        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...     {
        ...         "n_legs": [2, 4, 5, 100],
        ...         "animals": ["Flamingo", "Horse", "Brittle stars", "Centipede"],
        ...     }
        ... )
        >>> pa.Table.from_pandas(df)
        pyarrow.Table
        n_legs: int64
        animals: string
        ----
        n_legs: [[2,4,5,100]]
        animals: [["Flamingo","Horse","Brittle stars","Centipede"]]
        """
    @classmethod
    def from_arrays(
        cls,
        arrays: Collection[ArrayOrChunkedArray[Any]],
        names: list[str] | None = None,
        schema: Schema | None = None,
        metadata: MetaData | None = None,
    ) -> Self:
        """
        Construct a Table from Arrow arrays.

        Parameters
        ----------
        arrays : list of pyarrow.Array or pyarrow.ChunkedArray
            Equal-length arrays that should form the table.
        names : list of str, optional
            Names for the table columns. If not passed, schema must be passed.
        schema : Schema, default None
            Schema for the created table. If not passed, names must be passed.
        metadata : dict or Mapping, default None
            Optional metadata for the schema (if inferred).

        Returns
        -------
        Table

        Examples
        --------
        >>> import pyarrow as pa
        >>> n_legs = pa.array([2, 4, 5, 100])
        >>> animals = pa.array(["Flamingo", "Horse", "Brittle stars", "Centipede"])
        >>> names = ["n_legs", "animals"]

        Construct a Table from arrays:

        >>> pa.Table.from_arrays([n_legs, animals], names=names)
        pyarrow.Table
        n_legs: int64
        animals: string
        ----
        n_legs: [[2,4,5,100]]
        animals: [["Flamingo","Horse","Brittle stars","Centipede"]]

        Construct a Table from arrays with metadata:

        >>> my_metadata = {"n_legs": "Number of legs per animal"}
        >>> pa.Table.from_arrays([n_legs, animals], names=names, metadata=my_metadata)
        pyarrow.Table
        n_legs: int64
        animals: string
        ----
        n_legs: [[2,4,5,100]]
        animals: [["Flamingo","Horse","Brittle stars","Centipede"]]
        >>> pa.Table.from_arrays([n_legs, animals], names=names, metadata=my_metadata).schema
        n_legs: int64
        animals: string
        -- schema metadata --
        n_legs: 'Number of legs per animal'

        Construct a Table from arrays with pyarrow schema:

        >>> my_schema = pa.schema(
        ...     [pa.field("n_legs", pa.int64()), pa.field("animals", pa.string())],
        ...     metadata={"animals": "Name of the animal species"},
        ... )
        >>> pa.Table.from_arrays([n_legs, animals], schema=my_schema)
        pyarrow.Table
        n_legs: int64
        animals: string
        ----
        n_legs: [[2,4,5,100]]
        animals: [["Flamingo","Horse","Brittle stars","Centipede"]]
        >>> pa.Table.from_arrays([n_legs, animals], schema=my_schema).schema
        n_legs: int64
        animals: string
        -- schema metadata --
        animals: 'Name of the animal species'
        """
    @classmethod
    def from_struct_array(
        cls, struct_array: StructArray | ChunkedArray[scalar.StructScalar]
    ) -> Self:
        """
        Construct a Table from a StructArray.

        Each field in the StructArray will become a column in the resulting
        ``Table``.

        Parameters
        ----------
        struct_array : StructArray or ChunkedArray
            Array to construct the table from.

        Returns
        -------
        pyarrow.Table

        Examples
        --------
        >>> import pyarrow as pa
        >>> struct = pa.array([{"n_legs": 2, "animals": "Parrot"}, {"year": 2022, "n_legs": 4}])
        >>> pa.Table.from_struct_array(struct).to_pandas()
          animals  n_legs    year
        0  Parrot       2     NaN
        1    None       4  2022.0
        """
    def to_struct_array(
        self, max_chunksize: int | None = None
    ) -> ChunkedArray[scalar.StructScalar]:
        """
        Convert to a chunked array of struct type.

        Parameters
        ----------
        max_chunksize : int, default None
            Maximum number of rows for ChunkedArray chunks. Individual chunks
            may be smaller depending on the chunk layout of individual columns.

        Returns
        -------
        ChunkedArray
        """
    @classmethod
    def from_batches(cls, batches: Iterable[RecordBatch], schema: Schema | None = None) -> Self:
        """
        Construct a Table from a sequence or iterator of Arrow RecordBatches.

        Parameters
        ----------
        batches : sequence or iterator of RecordBatch
            Sequence of RecordBatch to be converted, all schemas must be equal.
        schema : Schema, default None
            If not passed, will be inferred from the first RecordBatch.

        Returns
        -------
        Table

        Examples
        --------
        >>> import pyarrow as pa
        >>> n_legs = pa.array([2, 4, 5, 100])
        >>> animals = pa.array(["Flamingo", "Horse", "Brittle stars", "Centipede"])
        >>> names = ["n_legs", "animals"]
        >>> batch = pa.record_batch([n_legs, animals], names=names)
        >>> batch.to_pandas()
           n_legs        animals
        0       2       Flamingo
        1       4          Horse
        2       5  Brittle stars
        3     100      Centipede

        Construct a Table from a RecordBatch:

        >>> pa.Table.from_batches([batch])
        pyarrow.Table
        n_legs: int64
        animals: string
        ----
        n_legs: [[2,4,5,100]]
        animals: [["Flamingo","Horse","Brittle stars","Centipede"]]

        Construct a Table from a sequence of RecordBatches:

        >>> pa.Table.from_batches([batch, batch])
        pyarrow.Table
        n_legs: int64
        animals: string
        ----
        n_legs: [[2,4,5,100],[2,4,5,100]]
        animals: [["Flamingo","Horse","Brittle stars","Centipede"],["Flamingo","Horse","Brittle stars","Centipede"]]
        """
    def to_batches(self, max_chunksize: int | None = None) -> list[RecordBatch]:
        """
        Convert Table to a list of RecordBatch objects.

        Note that this method is zero-copy, it merely exposes the same data
        under a different API.

        Parameters
        ----------
        max_chunksize : int, default None
            Maximum number of rows for each RecordBatch chunk. Individual chunks
            may be smaller depending on the chunk layout of individual columns.

        Returns
        -------
        list[RecordBatch]

        Examples
        --------
        >>> import pyarrow as pa
        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...     {
        ...         "n_legs": [2, 4, 5, 100],
        ...         "animals": ["Flamingo", "Horse", "Brittle stars", "Centipede"],
        ...     }
        ... )
        >>> table = pa.Table.from_pandas(df)

        Convert a Table to a RecordBatch:

        >>> table.to_batches()[0].to_pandas()
           n_legs        animals
        0       2       Flamingo
        1       4          Horse
        2       5  Brittle stars
        3     100      Centipede

        Convert a Table to a list of RecordBatches:

        >>> table.to_batches(max_chunksize=2)[0].to_pandas()
           n_legs   animals
        0       2  Flamingo
        1       4     Horse
        >>> table.to_batches(max_chunksize=2)[1].to_pandas()
           n_legs        animals
        0       5  Brittle stars
        1     100      Centipede
        """
    def to_reader(self, max_chunksize: int | None = None) -> RecordBatchReader:
        """
        Convert the Table to a RecordBatchReader.

        Note that this method is zero-copy, it merely exposes the same data
        under a different API.

        Parameters
        ----------
        max_chunksize : int, default None
            Maximum number of rows for each RecordBatch chunk. Individual chunks
            may be smaller depending on the chunk layout of individual columns.

        Returns
        -------
        RecordBatchReader

        Examples
        --------
        >>> import pyarrow as pa
        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...     {
        ...         "n_legs": [2, 4, 5, 100],
        ...         "animals": ["Flamingo", "Horse", "Brittle stars", "Centipede"],
        ...     }
        ... )
        >>> table = pa.Table.from_pandas(df)

        Convert a Table to a RecordBatchReader:

        >>> table.to_reader()
        <pyarrow.lib.RecordBatchReader object at ...>

        >>> reader = table.to_reader()
        >>> reader.schema
        n_legs: int64
        animals: string
        -- schema metadata --
        pandas: '{"index_columns": [{"kind": "range", "name": null, "start": 0, ...
        >>> reader.read_all()
        pyarrow.Table
        n_legs: int64
        animals: string
        ----
        n_legs: [[2,4,5,100]]
        animals: [["Flamingo","Horse","Brittle stars","Centipede"]]
        """
    @property
    def schema(self) -> Schema:
        """
        Schema of the table and its columns.

        Returns
        -------
        Schema

        Examples
        --------
        >>> import pyarrow as pa
        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...     {
        ...         "n_legs": [2, 4, 5, 100],
        ...         "animals": ["Flamingo", "Horse", "Brittle stars", "Centipede"],
        ...     }
        ... )
        >>> table = pa.Table.from_pandas(df)
        >>> table.schema
        n_legs: int64
        animals: string
        -- schema metadata --
        pandas: '{"index_columns": [{"kind": "range", "name": null, "start": 0, "' ...
        """
    @property
    def num_columns(self) -> int:
        """
        Number of columns in this table.

        Returns
        -------
        int

        Examples
        --------
        >>> import pyarrow as pa
        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...     {"n_legs": [None, 4, 5, None], "animals": ["Flamingo", "Horse", None, "Centipede"]}
        ... )
        >>> table = pa.Table.from_pandas(df)
        >>> table.num_columns
        2
        """
    @property
    def num_rows(self) -> int:
        """
        Number of rows in this table.

        Due to the definition of a table, all columns have the same number of
        rows.

        Returns
        -------
        int

        Examples
        --------
        >>> import pyarrow as pa
        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...     {"n_legs": [None, 4, 5, None], "animals": ["Flamingo", "Horse", None, "Centipede"]}
        ... )
        >>> table = pa.Table.from_pandas(df)
        >>> table.num_rows
        4
        """
    @property
    def nbytes(self) -> int:
        """
        Total number of bytes consumed by the elements of the table.

        In other words, the sum of bytes from all buffer ranges referenced.

        Unlike `get_total_buffer_size` this method will account for array
        offsets.

        If buffers are shared between arrays then the shared
        portion will only be counted multiple times.

        The dictionary of dictionary arrays will always be counted in their
        entirety even if the array only references a portion of the dictionary.

        Examples
        --------
        >>> import pyarrow as pa
        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...     {"n_legs": [None, 4, 5, None], "animals": ["Flamingo", "Horse", None, "Centipede"]}
        ... )
        >>> table = pa.Table.from_pandas(df)
        >>> table.nbytes
        72
        """
    def get_total_buffer_size(self) -> int:
        """
        The sum of bytes in each buffer referenced by the table.

        An array may only reference a portion of a buffer.
        This method will overestimate in this case and return the
        byte size of the entire buffer.

        If a buffer is referenced multiple times then it will
        only be counted once.

        Examples
        --------
        >>> import pyarrow as pa
        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...     {"n_legs": [None, 4, 5, None], "animals": ["Flamingo", "Horse", None, "Centipede"]}
        ... )
        >>> table = pa.Table.from_pandas(df)
        >>> table.get_total_buffer_size()
        76
        """
    def __sizeof__(self) -> int: ...
    def add_column(
        self, i: int, field_: str | Field, column: ArrayOrChunkedArray[Any] | list[list[Any]]
    ) -> Self:
        """
        Add column to Table at position.

        A new table is returned with the column added, the original table
        object is left unchanged.

        Parameters
        ----------
        i : int
            Index to place the column at.
        field_ : str or Field
            If a string is passed then the type is deduced from the column
            data.
        column : Array, list of Array, or values coercible to arrays
            Column data.

        Returns
        -------
        Table
            New table with the passed column added.

        Examples
        --------
        >>> import pyarrow as pa
        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...     {
        ...         "n_legs": [2, 4, 5, 100],
        ...         "animals": ["Flamingo", "Horse", "Brittle stars", "Centipede"],
        ...     }
        ... )
        >>> table = pa.Table.from_pandas(df)

        Add column:

        >>> year = [2021, 2022, 2019, 2021]
        >>> table.add_column(0, "year", [year])
        pyarrow.Table
        year: int64
        n_legs: int64
        animals: string
        ----
        year: [[2021,2022,2019,2021]]
        n_legs: [[2,4,5,100]]
        animals: [["Flamingo","Horse","Brittle stars","Centipede"]]

        Original table is left unchanged:

        >>> table
        pyarrow.Table
        n_legs: int64
        animals: string
        ----
        n_legs: [[2,4,5,100]]
        animals: [["Flamingo","Horse","Brittle stars","Centipede"]]
        """
    def remove_column(self, i: int) -> Self:
        """
        Create new Table with the indicated column removed.

        Parameters
        ----------
        i : int
            Index of column to remove.

        Returns
        -------
        Table
            New table without the column.

        Examples
        --------
        >>> import pyarrow as pa
        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...     {
        ...         "n_legs": [2, 4, 5, 100],
        ...         "animals": ["Flamingo", "Horse", "Brittle stars", "Centipede"],
        ...     }
        ... )
        >>> table = pa.Table.from_pandas(df)
        >>> table.remove_column(1)
        pyarrow.Table
        n_legs: int64
        ----
        n_legs: [[2,4,5,100]]
        """
    def set_column(
        self, i: int, field_: str | Field, column: ArrayOrChunkedArray[Any] | list[list[Any]]
    ) -> Self:
        """
        Replace column in Table at position.

        Parameters
        ----------
        i : int
            Index to place the column at.
        field_ : str or Field
            If a string is passed then the type is deduced from the column
            data.
        column : Array, list of Array, or values coercible to arrays
            Column data.

        Returns
        -------
        Table
            New table with the passed column set.

        Examples
        --------
        >>> import pyarrow as pa
        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...     {
        ...         "n_legs": [2, 4, 5, 100],
        ...         "animals": ["Flamingo", "Horse", "Brittle stars", "Centipede"],
        ...     }
        ... )
        >>> table = pa.Table.from_pandas(df)

        Replace a column:

        >>> year = [2021, 2022, 2019, 2021]
        >>> table.set_column(1, "year", [year])
        pyarrow.Table
        n_legs: int64
        year: int64
        ----
        n_legs: [[2,4,5,100]]
        year: [[2021,2022,2019,2021]]
        """
    @overload
    def rename_columns(self, names: list[str]) -> Self: ...
    @overload
    def rename_columns(self, names: dict[str, str]) -> Self: ...
    def rename_columns(self, names):
        """
        Create new table with columns renamed to provided names.

        Parameters
        ----------
        names : list[str] or dict[str, str]
            List of new column names or mapping of old column names to new column names.

            If a mapping of old to new column names is passed, then all columns which are
            found to match a provided old column name will be renamed to the new column name.
            If any column names are not found in the mapping, a KeyError will be raised.

        Raises
        ------
        KeyError
            If any of the column names passed in the names mapping do not exist.

        Returns
        -------
        Table

        Examples
        --------
        >>> import pyarrow as pa
        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...     {
        ...         "n_legs": [2, 4, 5, 100],
        ...         "animals": ["Flamingo", "Horse", "Brittle stars", "Centipede"],
        ...     }
        ... )
        >>> table = pa.Table.from_pandas(df)
        >>> new_names = ["n", "name"]
        >>> table.rename_columns(new_names)
        pyarrow.Table
        n: int64
        name: string
        ----
        n: [[2,4,5,100]]
        name: [["Flamingo","Horse","Brittle stars","Centipede"]]
        >>> new_names = {"n_legs": "n", "animals": "name"}
        >>> table.rename_columns(new_names)
        pyarrow.Table
        n: int64
        name: string
        ----
        n: [[2,4,5,100]]
        name: [["Flamingo","Horse","Brittle stars","Centipede"]]
        """
    def drop(self, columns: str | list[str]) -> Self:
        """
        Drop one or more columns and return a new table.

        Alias of Table.drop_columns, but kept for backwards compatibility.

        Parameters
        ----------
        columns : str or list[str]
            Field name(s) referencing existing column(s).

        Returns
        -------
        Table
            New table without the column(s).
        """
    def group_by(self, keys: str | list[str], use_threads: bool = True) -> TableGroupBy:
        """
        Declare a grouping over the columns of the table.

        Resulting grouping can then be used to perform aggregations
        with a subsequent ``aggregate()`` method.

        Parameters
        ----------
        keys : str or list[str]
            Name of the columns that should be used as the grouping key.
        use_threads : bool, default True
            Whether to use multithreading or not. When set to True (the
            default), no stable ordering of the output is guaranteed.

        Returns
        -------
        TableGroupBy

        See Also
        --------
        TableGroupBy.aggregate

        Examples
        --------
        >>> import pandas as pd
        >>> import pyarrow as pa
        >>> df = pd.DataFrame(
        ...     {
        ...         "year": [2020, 2022, 2021, 2022, 2019, 2021],
        ...         "n_legs": [2, 2, 4, 4, 5, 100],
        ...         "animal": ["Flamingo", "Parrot", "Dog", "Horse", "Brittle stars", "Centipede"],
        ...     }
        ... )
        >>> table = pa.Table.from_pandas(df)
        >>> table.group_by("year").aggregate([("n_legs", "sum")])
        pyarrow.Table
        year: int64
        n_legs_sum: int64
        ----
        year: [[2020,2022,2021,2019]]
        n_legs_sum: [[2,6,104,5]]
        """
    def join(
        self,
        right_table: Self,
        keys: str | list[str],
        right_keys: str | list[str] | None = None,
        join_type: JoinType = "left outer",
        left_suffix: str | None = None,
        right_suffix: str | None = None,
        coalesce_keys: bool = True,
        use_threads: bool = True,
    ) -> Self:
        """
        Perform a join between this table and another one.

        Result of the join will be a new Table, where further
        operations can be applied.

        Parameters
        ----------
        right_table : Table
            The table to join to the current one, acting as the right table
            in the join operation.
        keys : str or list[str]
            The columns from current table that should be used as keys
            of the join operation left side.
        right_keys : str or list[str], default None
            The columns from the right_table that should be used as keys
            on the join operation right side.
            When ``None`` use the same key names as the left table.
        join_type : str, default "left outer"
            The kind of join that should be performed, one of
            ("left semi", "right semi", "left anti", "right anti",
            "inner", "left outer", "right outer", "full outer")
        left_suffix : str, default None
            Which suffix to add to left column names. This prevents confusion
            when the columns in left and right tables have colliding names.
        right_suffix : str, default None
            Which suffix to add to the right column names. This prevents confusion
            when the columns in left and right tables have colliding names.
        coalesce_keys : bool, default True
            If the duplicated keys should be omitted from one of the sides
            in the join result.
        use_threads : bool, default True
            Whether to use multithreading or not.

        Returns
        -------
        Table

        Examples
        --------
        >>> import pandas as pd
        >>> import pyarrow as pa
        >>> df1 = pd.DataFrame({"id": [1, 2, 3], "year": [2020, 2022, 2019]})
        >>> df2 = pd.DataFrame(
        ...     {"id": [3, 4], "n_legs": [5, 100], "animal": ["Brittle stars", "Centipede"]}
        ... )
        >>> t1 = pa.Table.from_pandas(df1)
        >>> t2 = pa.Table.from_pandas(df2)

        Left outer join:

        >>> t1.join(t2, "id").combine_chunks().sort_by("year")
        pyarrow.Table
        id: int64
        year: int64
        n_legs: int64
        animal: string
        ----
        id: [[3,1,2]]
        year: [[2019,2020,2022]]
        n_legs: [[5,null,null]]
        animal: [["Brittle stars",null,null]]

        Full outer join:

        >>> t1.join(t2, "id", join_type="full outer").combine_chunks().sort_by("year")
        pyarrow.Table
        id: int64
        year: int64
        n_legs: int64
        animal: string
        ----
        id: [[3,1,2,4]]
        year: [[2019,2020,2022,null]]
        n_legs: [[5,null,null,100]]
        animal: [["Brittle stars",null,null,"Centipede"]]

        Right outer join:

        >>> t1.join(t2, "id", join_type="right outer").combine_chunks().sort_by("year")
        pyarrow.Table
        year: int64
        id: int64
        n_legs: int64
        animal: string
        ----
        year: [[2019,null]]
        id: [[3,4]]
        n_legs: [[5,100]]
        animal: [["Brittle stars","Centipede"]]

        Right anti join

        >>> t1.join(t2, "id", join_type="right anti")
        pyarrow.Table
        id: int64
        n_legs: int64
        animal: string
        ----
        id: [[4]]
        n_legs: [[100]]
        animal: [["Centipede"]]
        """
    def join_asof(
        self,
        right_table: Self,
        on: str,
        by: str | list[str],
        tolerance: int,
        right_on: str | list[str] | None = None,
        right_by: str | list[str] | None = None,
    ) -> Self:
        """
        Perform an asof join between this table and another one.

        This is similar to a left-join except that we match on nearest key rather
        than equal keys. Both tables must be sorted by the key. This type of join
        is most useful for time series data that are not perfectly aligned.

        Optionally match on equivalent keys with "by" before searching with "on".

        Result of the join will be a new Table, where further
        operations can be applied.

        Parameters
        ----------
        right_table : Table
            The table to join to the current one, acting as the right table
            in the join operation.
        on : str
            The column from current table that should be used as the "on" key
            of the join operation left side.

            An inexact match is used on the "on" key, i.e. a row is considered a
            match if and only if left_on - tolerance <= right_on <= left_on.

            The input dataset must be sorted by the "on" key. Must be a single
            field of a common type.

            Currently, the "on" key must be an integer, date, or timestamp type.
        by : str or list[str]
            The columns from current table that should be used as the keys
            of the join operation left side. The join operation is then done
            only for the matches in these columns.
        tolerance : int
            The tolerance for inexact "on" key matching. A right row is considered
            a match with the left row ``right.on - left.on <= tolerance``. The
            ``tolerance`` may be:

            - negative, in which case a past-as-of-join occurs;
            - or positive, in which case a future-as-of-join occurs;
            - or zero, in which case an exact-as-of-join occurs.

            The tolerance is interpreted in the same units as the "on" key.
        right_on : str or list[str], default None
            The columns from the right_table that should be used as the on key
            on the join operation right side.
            When ``None`` use the same key name as the left table.
        right_by : str or list[str], default None
            The columns from the right_table that should be used as keys
            on the join operation right side.
            When ``None`` use the same key names as the left table.

        Returns
        -------
        Table

        Example
        --------
        >>> import pyarrow as pa
        >>> t1 = pa.table({"id": [1, 3, 2, 3, 3], "year": [2020, 2021, 2022, 2022, 2023]})
        >>> t2 = pa.table(
        ...     {
        ...         "id": [3, 4],
        ...         "year": [2020, 2021],
        ...         "n_legs": [5, 100],
        ...         "animal": ["Brittle stars", "Centipede"],
        ...     }
        ... )

        >>> t1.join_asof(t2, on="year", by="id", tolerance=-2)
        pyarrow.Table
        id: int64
        year: int64
        n_legs: int64
        animal: string
        ----
        id: [[1,3,2,3,3]]
        year: [[2020,2021,2022,2022,2023]]
        n_legs: [[null,5,null,5,null]]
        animal: [[null,"Brittle stars",null,"Brittle stars",null]]
        """
    def __arrow_c_stream__(self, requested_schema=None):
        """
        Export the table as an Arrow C stream PyCapsule.

        Parameters
        ----------
        requested_schema : PyCapsule, default None
            The schema to which the stream should be casted, passed as a
            PyCapsule containing a C ArrowSchema representation of the
            requested schema.
            Currently, this is not supported and will raise a
            NotImplementedError if the schema doesn't match the current schema.

        Returns
        -------
        PyCapsule
        """
    @property
    def is_cpu(self) -> bool:
        """
        Whether all ChunkedArrays are CPU-accessible.
        """

def record_batch(
    data: Mapping[str, Sequence[Any] | Array[Any]]
    | Collection[Array[Any]]
    | pd.DataFrame
    | SupportArrowArray
    | SupportArrowDeviceArray,
    names: list[str] | None = None,
    schema: Schema | None = None,
    metadata: MetaData | None = None,
) -> RecordBatch:
    """
    Create a pyarrow.RecordBatch from another Python data structure or sequence
    of arrays.

    Parameters
    ----------
    data : dict, list, pandas.DataFrame, Arrow-compatible table
        A mapping of strings to Arrays or Python lists, a list of Arrays,
        a pandas DataFame, or any tabular object implementing the
        Arrow PyCapsule Protocol (has an ``__arrow_c_array__`` or
        ``__arrow_c_device_array__`` method).
    names : list, default None
        Column names if list of arrays passed as data. Mutually exclusive with
        'schema' argument.
    schema : Schema, default None
        The expected schema of the RecordBatch. If not passed, will be inferred
        from the data. Mutually exclusive with 'names' argument.
    metadata : dict or Mapping, default None
        Optional metadata for the schema (if schema not passed).

    Returns
    -------
    RecordBatch

    See Also
    --------
    RecordBatch.from_arrays, RecordBatch.from_pandas, table

    Examples
    --------
    >>> import pyarrow as pa
    >>> n_legs = pa.array([2, 2, 4, 4, 5, 100])
    >>> animals = pa.array(["Flamingo", "Parrot", "Dog", "Horse", "Brittle stars", "Centipede"])
    >>> names = ["n_legs", "animals"]

    Construct a RecordBatch from a python dictionary:

    >>> pa.record_batch({"n_legs": n_legs, "animals": animals})
    pyarrow.RecordBatch
    n_legs: int64
    animals: string
    ----
    n_legs: [2,2,4,4,5,100]
    animals: ["Flamingo","Parrot","Dog","Horse","Brittle stars","Centipede"]
    >>> pa.record_batch({"n_legs": n_legs, "animals": animals}).to_pandas()
       n_legs        animals
    0       2       Flamingo
    1       2         Parrot
    2       4            Dog
    3       4          Horse
    4       5  Brittle stars
    5     100      Centipede

    Creating a RecordBatch from a list of arrays with names:

    >>> pa.record_batch([n_legs, animals], names=names)
    pyarrow.RecordBatch
    n_legs: int64
    animals: string
    ----
    n_legs: [2,2,4,4,5,100]
    animals: ["Flamingo","Parrot","Dog","Horse","Brittle stars","Centipede"]

    Creating a RecordBatch from a list of arrays with names and metadata:

    >>> my_metadata = {"n_legs": "How many legs does an animal have?"}
    >>> pa.record_batch([n_legs, animals], names=names, metadata=my_metadata)
    pyarrow.RecordBatch
    n_legs: int64
    animals: string
    ----
    n_legs: [2,2,4,4,5,100]
    animals: ["Flamingo","Parrot","Dog","Horse","Brittle stars","Centipede"]
    >>> pa.record_batch([n_legs, animals], names=names, metadata=my_metadata).schema
    n_legs: int64
    animals: string
    -- schema metadata --
    n_legs: 'How many legs does an animal have?'

    Creating a RecordBatch from a pandas DataFrame:

    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     {
    ...         "year": [2020, 2022, 2021, 2022],
    ...         "month": [3, 5, 7, 9],
    ...         "day": [1, 5, 9, 13],
    ...         "n_legs": [2, 4, 5, 100],
    ...         "animals": ["Flamingo", "Horse", "Brittle stars", "Centipede"],
    ...     }
    ... )
    >>> pa.record_batch(df)
    pyarrow.RecordBatch
    year: int64
    month: int64
    day: int64
    n_legs: int64
    animals: string
    ----
    year: [2020,2022,2021,2022]
    month: [3,5,7,9]
    day: [1,5,9,13]
    n_legs: [2,4,5,100]
    animals: ["Flamingo","Horse","Brittle stars","Centipede"]

    >>> pa.record_batch(df).to_pandas()
       year  month  day  n_legs        animals
    0  2020      3    1       2       Flamingo
    1  2022      5    5       4          Horse
    2  2021      7    9       5  Brittle stars
    3  2022      9   13     100      Centipede

    Creating a RecordBatch from a pandas DataFrame with schema:

    >>> my_schema = pa.schema(
    ...     [pa.field("n_legs", pa.int64()), pa.field("animals", pa.string())],
    ...     metadata={"n_legs": "Number of legs per animal"},
    ... )
    >>> pa.record_batch(df, my_schema).schema
    n_legs: int64
    animals: string
    -- schema metadata --
    n_legs: 'Number of legs per animal'
    pandas: ...
    >>> pa.record_batch(df, my_schema).to_pandas()
       n_legs        animals
    0       2       Flamingo
    1       4          Horse
    2       5  Brittle stars
    3     100      Centipede
    """

@overload
def table(
    data: Mapping[str, Sequence[Any] | Array[Any]],
    schema: Schema | None = None,
    metadata: MetaData | None = None,
    nthreads: int | None = None,
) -> Table: ...
@overload
def table(
    data: Collection[ArrayOrChunkedArray[Any]]
    | pd.DataFrame
    | SupportArrowArray
    | SupportArrowStream
    | SupportArrowDeviceArray,
    names: list[str] | None = None,
    schema: Schema | None = None,
    metadata: MetaData | None = None,
    nthreads: int | None = None,
) -> Table: ...
def table(*args, **kwargs):
    """
    Create a pyarrow.Table from a Python data structure or sequence of arrays.

    Parameters
    ----------
    data : dict, list, pandas.DataFrame, Arrow-compatible table
        A mapping of strings to Arrays or Python lists, a list of arrays or
        chunked arrays, a pandas DataFame, or any tabular object implementing
        the Arrow PyCapsule Protocol (has an ``__arrow_c_array__``,
        ``__arrow_c_device_array__`` or ``__arrow_c_stream__`` method).
    names : list, default None
        Column names if list of arrays passed as data. Mutually exclusive with
        'schema' argument.
    schema : Schema, default None
        The expected schema of the Arrow Table. If not passed, will be inferred
        from the data. Mutually exclusive with 'names' argument.
        If passed, the output will have exactly this schema (raising an error
        when columns are not found in the data and ignoring additional data not
        specified in the schema, when data is a dict or DataFrame).
    metadata : dict or Mapping, default None
        Optional metadata for the schema (if schema not passed).
    nthreads : int, default None
        For pandas.DataFrame inputs: if greater than 1, convert columns to
        Arrow in parallel using indicated number of threads. By default,
        this follows :func:`pyarrow.cpu_count` (may use up to system CPU count
        threads).

    Returns
    -------
    Table

    See Also
    --------
    Table.from_arrays, Table.from_pandas, Table.from_pydict

    Examples
    --------
    >>> import pyarrow as pa
    >>> n_legs = pa.array([2, 4, 5, 100])
    >>> animals = pa.array(["Flamingo", "Horse", "Brittle stars", "Centipede"])
    >>> names = ["n_legs", "animals"]

    Construct a Table from a python dictionary:

    >>> pa.table({"n_legs": n_legs, "animals": animals})
    pyarrow.Table
    n_legs: int64
    animals: string
    ----
    n_legs: [[2,4,5,100]]
    animals: [["Flamingo","Horse","Brittle stars","Centipede"]]

    Construct a Table from arrays:

    >>> pa.table([n_legs, animals], names=names)
    pyarrow.Table
    n_legs: int64
    animals: string
    ----
    n_legs: [[2,4,5,100]]
    animals: [["Flamingo","Horse","Brittle stars","Centipede"]]

    Construct a Table from arrays with metadata:

    >>> my_metadata = {"n_legs": "Number of legs per animal"}
    >>> pa.table([n_legs, animals], names=names, metadata=my_metadata).schema
    n_legs: int64
    animals: string
    -- schema metadata --
    n_legs: 'Number of legs per animal'

    Construct a Table from pandas DataFrame:

    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     {
    ...         "year": [2020, 2022, 2019, 2021],
    ...         "n_legs": [2, 4, 5, 100],
    ...         "animals": ["Flamingo", "Horse", "Brittle stars", "Centipede"],
    ...     }
    ... )
    >>> pa.table(df)
    pyarrow.Table
    year: int64
    n_legs: int64
    animals: string
    ----
    year: [[2020,2022,2019,2021]]
    n_legs: [[2,4,5,100]]
    animals: [["Flamingo","Horse","Brittle stars","Centipede"]]

    Construct a Table from pandas DataFrame with pyarrow schema:

    >>> my_schema = pa.schema(
    ...     [pa.field("n_legs", pa.int64()), pa.field("animals", pa.string())],
    ...     metadata={"n_legs": "Number of legs per animal"},
    ... )
    >>> pa.table(df, my_schema).schema
    n_legs: int64
    animals: string
    -- schema metadata --
    n_legs: 'Number of legs per animal'
    pandas: '{"index_columns": [], "column_indexes": [{"name": null, ...

    Construct a Table from chunked arrays:

    >>> n_legs = pa.chunked_array([[2, 2, 4], [4, 5, 100]])
    >>> animals = pa.chunked_array(
    ...     [["Flamingo", "Parrot", "Dog"], ["Horse", "Brittle stars", "Centipede"]]
    ... )
    >>> table = pa.table([n_legs, animals], names=names)
    >>> table
    pyarrow.Table
    n_legs: int64
    animals: string
    ----
    n_legs: [[2,2,4],[4,5,100]]
    animals: [["Flamingo","Parrot","Dog"],["Horse","Brittle stars","Centipede"]]
    """

def concat_tables(
    tables: Iterable[Table],
    memory_pool: MemoryPool | None = None,
    promote_options: Literal["none", "default", "permissive"] = "none",
    **kwargs: Any,
) -> Table:
    """
    Concatenate pyarrow.Table objects.

    If promote_options="none", a zero-copy concatenation will be performed. The schemas
    of all the Tables must be the same (except the metadata), otherwise an
    exception will be raised. The result Table will share the metadata with the
    first table.

    If promote_options="default", any null type arrays will be casted to the type of other
    arrays in the column of the same name. If a table is missing a particular
    field, null values of the appropriate type will be generated to take the
    place of the missing field. The new schema will share the metadata with the
    first table. Each field in the new schema will share the metadata with the
    first table which has the field defined. Note that type promotions may
    involve additional allocations on the given ``memory_pool``.

    If promote_options="permissive", the behavior of default plus types will be promoted
    to the common denominator that fits all the fields.

    Parameters
    ----------
    tables : iterable of pyarrow.Table objects
        Pyarrow tables to concatenate into a single Table.
    memory_pool : MemoryPool, default None
        For memory allocations, if required, otherwise use default pool.
    promote_options : str, default none
        Accepts strings "none", "default" and "permissive".
    **kwargs : dict, optional

    Examples
    --------
    >>> import pyarrow as pa
    >>> t1 = pa.table(
    ...     [
    ...         pa.array([2, 4, 5, 100]),
    ...         pa.array(["Flamingo", "Horse", "Brittle stars", "Centipede"]),
    ...     ],
    ...     names=["n_legs", "animals"],
    ... )
    >>> t2 = pa.table([pa.array([2, 4]), pa.array(["Parrot", "Dog"])], names=["n_legs", "animals"])
    >>> pa.concat_tables([t1, t2])
    pyarrow.Table
    n_legs: int64
    animals: string
    ----
    n_legs: [[2,4,5,100],[2,4]]
    animals: [["Flamingo","Horse","Brittle stars","Centipede"],["Parrot","Dog"]]

    """

class TableGroupBy:
    """
    A grouping of columns in a table on which to perform aggregations.

    Parameters
    ----------
    table : pyarrow.Table
        Input table to execute the aggregation on.
    keys : str or list[str]
        Name of the grouped columns.
    use_threads : bool, default True
        Whether to use multithreading or not. When set to True (the default),
        no stable ordering of the output is guaranteed.

    Examples
    --------
    >>> import pyarrow as pa
    >>> t = pa.table(
    ...     [
    ...         pa.array(["a", "a", "b", "b", "c"]),
    ...         pa.array([1, 2, 3, 4, 5]),
    ...     ],
    ...     names=["keys", "values"],
    ... )

    Grouping of columns:

    >>> pa.TableGroupBy(t, "keys")
    <pyarrow.lib.TableGroupBy object at ...>

    Perform aggregations:

    >>> pa.TableGroupBy(t, "keys").aggregate([("values", "sum")])
    pyarrow.Table
    keys: string
    values_sum: int64
    ----
    keys: [["a","b","c"]]
    values_sum: [[3,7,5]]
    """

    keys: str | list[str]
    def __init__(self, table: Table, keys: str | list[str], use_threads: bool = True): ...
    def aggregate(
        self,
        aggregations: Iterable[
            tuple[ColumnSelector, Aggregation]
            | tuple[ColumnSelector, Aggregation, AggregateOptions | None]
        ],
    ) -> Table:
        """
        Perform an aggregation over the grouped columns of the table.

        Parameters
        ----------
        aggregations : list[tuple(str, str)] or \
list[tuple(str, str, FunctionOptions)]
            List of tuples, where each tuple is one aggregation specification
            and consists of: aggregation column name followed
            by function name and optionally aggregation function option.
            Pass empty list to get a single row for each group.
            The column name can be a string, an empty list or a list of
            column names, for unary, nullary and n-ary aggregation functions
            respectively.

            For the list of function names and respective aggregation
            function options see :ref:`py-grouped-aggrs`.

        Returns
        -------
        Table
            Results of the aggregation functions.

        Examples
        --------
        >>> import pyarrow as pa
        >>> t = pa.table([
        ...       pa.array(["a", "a", "b", "b", "c"]),
        ...       pa.array([1, 2, 3, 4, 5]),
        ... ], names=["keys", "values"])

        Sum the column "values" over the grouped column "keys":

        >>> t.group_by("keys").aggregate([("values", "sum")])
        pyarrow.Table
        keys: string
        values_sum: int64
        ----
        keys: [["a","b","c"]]
        values_sum: [[3,7,5]]

        Count the rows over the grouped column "keys":

        >>> t.group_by("keys").aggregate([([], "count_all")])
        pyarrow.Table
        keys: string
        count_all: int64
        ----
        keys: [["a","b","c"]]
        count_all: [[2,2,1]]

        Do multiple aggregations:

        >>> t.group_by("keys").aggregate([
        ...    ("values", "sum"),
        ...    ("keys", "count")
        ... ])
        pyarrow.Table
        keys: string
        values_sum: int64
        keys_count: int64
        ----
        keys: [["a","b","c"]]
        values_sum: [[3,7,5]]
        keys_count: [[2,2,1]]

        Count the number of non-null values for column "values"
        over the grouped column "keys":

        >>> import pyarrow.compute as pc
        >>> t.group_by(["keys"]).aggregate([
        ...    ("values", "count", pc.CountOptions(mode="only_valid"))
        ... ])
        pyarrow.Table
        keys: string
        values_count: int64
        ----
        keys: [["a","b","c"]]
        values_count: [[2,2,1]]

        Get a single row for each group in column "keys":

        >>> t.group_by("keys").aggregate([])
        pyarrow.Table
        keys: string
        ----
        keys: [["a","b","c"]]
        """
    def _table(self) -> Table: ...
    @property
    def _use_threads(self) -> bool: ...

def concat_batches(
    recordbatches: Iterable[RecordBatch], memory_pool: MemoryPool | None = None
) -> RecordBatch:
    """
    Concatenate pyarrow.RecordBatch objects.

    All recordbatches must share the same Schema,
    the operation implies a copy of the data to merge
    the arrays of the different RecordBatches.

    Parameters
    ----------
    recordbatches : iterable of pyarrow.RecordBatch objects
        Pyarrow record batches to concatenate into a single RecordBatch.
    memory_pool : MemoryPool, default None
        For memory allocations, if required, otherwise use default pool.

    Examples
    --------
    >>> import pyarrow as pa
    >>> t1 = pa.record_batch(
    ...     [
    ...         pa.array([2, 4, 5, 100]),
    ...         pa.array(["Flamingo", "Horse", "Brittle stars", "Centipede"]),
    ...     ],
    ...     names=["n_legs", "animals"],
    ... )
    >>> t2 = pa.record_batch(
    ...     [pa.array([2, 4]), pa.array(["Parrot", "Dog"])], names=["n_legs", "animals"]
    ... )
    >>> pa.concat_batches([t1, t2])
    pyarrow.RecordBatch
    n_legs: int64
    animals: string
    ----
    n_legs: [2,4,5,100,2,4]
    animals: ["Flamingo","Horse","Brittle stars","Centipede","Parrot","Dog"]

    """

__all__ = [
    "ChunkedArray",
    "chunked_array",
    "_Tabular",
    "RecordBatch",
    "table_to_blocks",
    "Table",
    "record_batch",
    "table",
    "concat_tables",
    "TableGroupBy",
    "concat_batches",
]
