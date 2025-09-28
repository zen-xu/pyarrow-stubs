import datetime as dt
import sys

from collections.abc import Mapping, Sequence
from decimal import Decimal

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

from typing import Any, Generic, Iterable, Iterator, Literal, overload

import numpy as np
import pandas as pd

from pyarrow._stubs_typing import SupportArrowSchema
from pyarrow.lib import (
    Array,
    ChunkedArray,
    ExtensionArray,
    MemoryPool,
    MonthDayNano,
    Table,
)
from typing_extensions import TypeVar, deprecated

from .io import Buffer
from .scalar import ExtensionScalar

_AsPyType = TypeVar("_AsPyType")
_DataTypeT = TypeVar("_DataTypeT", bound=DataType)

class _Weakrefable: ...
class _Metadata(_Weakrefable): ...

class DataType(_Weakrefable):
    """
    Base class of all Arrow data types.

    Each data type is an *instance* of this class.

    Examples
    --------
    Instance of int64 type:

    >>> import pyarrow as pa
    >>> pa.int64()
    DataType(int64)
    """
    def field(self, i: int) -> Field:
        """
        Parameters
        ----------
        i : int

        Returns
        -------
        pyarrow.Field
        """
    @property
    def id(self) -> int: ...
    @property
    def bit_width(self) -> int:
        """
        Bit width for fixed width type.

        Examples
        --------
        >>> import pyarrow as pa
        >>> pa.int64()
        DataType(int64)
        >>> pa.int64().bit_width
        64
        """
    @property
    def byte_width(self) -> int:
        """
        Byte width for fixed width type.

        Examples
        --------
        >>> import pyarrow as pa
        >>> pa.int64()
        DataType(int64)
        >>> pa.int64().byte_width
        8
        """
    @property
    def num_fields(self) -> int:
        """
        The number of child fields.

        Examples
        --------
        >>> import pyarrow as pa
        >>> pa.int64()
        DataType(int64)
        >>> pa.int64().num_fields
        0
        >>> pa.list_(pa.string())
        ListType(list<item: string>)
        >>> pa.list_(pa.string()).num_fields
        1
        >>> struct = pa.struct({"x": pa.int32(), "y": pa.string()})
        >>> struct.num_fields
        2
        """
    @property
    def num_buffers(self) -> int:
        """
        Number of data buffers required to construct Array type
        excluding children.

        Examples
        --------
        >>> import pyarrow as pa
        >>> pa.int64().num_buffers
        2
        >>> pa.string().num_buffers
        3
        """
    def __hash__(self) -> int: ...
    def equals(self, other: DataType | str, *, check_metadata: bool = False) -> bool:
        """
        Return true if type is equivalent to passed value.

        Parameters
        ----------
        other : DataType or string convertible to DataType
        check_metadata : bool
            Whether nested Field metadata equality should be checked as well.

        Returns
        -------
        is_equal : bool

        Examples
        --------
        >>> import pyarrow as pa
        >>> pa.int64().equals(pa.string())
        False
        >>> pa.int64().equals(pa.int64())
        True
        """
    def to_pandas_dtype(self) -> np.generic:
        """
        Return the equivalent NumPy / Pandas dtype.

        Examples
        --------
        >>> import pyarrow as pa
        >>> pa.int64().to_pandas_dtype()
        <class 'numpy.int64'>
        """
    def _export_to_c(self, out_ptr: int) -> None:
        """
        Export to a C ArrowSchema struct, given its pointer.

        Be careful: if you don't pass the ArrowSchema struct to a consumer,
        its memory will leak.  This is a low-level function intended for
        expert users.
        """
    @classmethod
    def _import_from_c(cls, in_ptr: int) -> Self:
        """
        Import DataType from a C ArrowSchema struct, given its pointer.

        This is a low-level function intended for expert users.
        """
    def __arrow_c_schema__(self) -> Any:
        """
        Export to a ArrowSchema PyCapsule

        Unlike _export_to_c, this will not leak memory if the capsule is not used.
        """
    @classmethod
    def _import_from_c_capsule(cls, schema) -> Self:
        """
        Import a DataType from a ArrowSchema PyCapsule

        Parameters
        ----------
        schema : PyCapsule
            A valid PyCapsule with name 'arrow_schema' containing an
            ArrowSchema pointer.
        """

class _BasicDataType(DataType, Generic[_AsPyType]): ...
class NullType(_BasicDataType[None]): ...
class BoolType(_BasicDataType[bool]): ...
class UInt8Type(_BasicDataType[int]): ...
class Int8Type(_BasicDataType[int]): ...
class UInt16Type(_BasicDataType[int]): ...
class Int16Type(_BasicDataType[int]): ...
class Uint32Type(_BasicDataType[int]): ...
class Int32Type(_BasicDataType[int]): ...
class UInt64Type(_BasicDataType[int]): ...
class Int64Type(_BasicDataType[int]): ...
class Float16Type(_BasicDataType[float]): ...
class Float32Type(_BasicDataType[float]): ...
class Float64Type(_BasicDataType[float]): ...
class Date32Type(_BasicDataType[dt.date]): ...
class Date64Type(_BasicDataType[dt.date]): ...
class MonthDayNanoIntervalType(_BasicDataType[MonthDayNano]): ...
class StringType(_BasicDataType[str]): ...
class LargeStringType(_BasicDataType[str]): ...
class StringViewType(_BasicDataType[str]): ...
class BinaryType(_BasicDataType[bytes]): ...
class LargeBinaryType(_BasicDataType[bytes]): ...
class BinaryViewType(_BasicDataType[bytes]): ...

_Unit = TypeVar("_Unit", bound=Literal["s", "ms", "us", "ns"], default=Literal["us"])
_Tz = TypeVar("_Tz", str, None, default=None)

class TimestampType(_BasicDataType[int], Generic[_Unit, _Tz]):
    """
    Concrete class for timestamp data types.

    Examples
    --------
    >>> import pyarrow as pa

    Create an instance of timestamp type:

    >>> pa.timestamp("us")
    TimestampType(timestamp[us])

    Create an instance of timestamp type with timezone:

    >>> pa.timestamp("s", tz="UTC")
    TimestampType(timestamp[s, tz=UTC])
    """
    @property
    def unit(self) -> _Unit:
        """
        The timestamp unit ('s', 'ms', 'us' or 'ns').

        Examples
        --------
        >>> import pyarrow as pa
        >>> t = pa.timestamp("us")
        >>> t.unit
        'us'
        """
    @property
    def tz(self) -> _Tz:
        """
        The timestamp time zone, if any, or None.

        Examples
        --------
        >>> import pyarrow as pa
        >>> t = pa.timestamp("s", tz="UTC")
        >>> t.tz
        'UTC'
        """

_Time32Unit = TypeVar("_Time32Unit", bound=Literal["s", "ms"])

class Time32Type(_BasicDataType[dt.time], Generic[_Time32Unit]):
    """
    Concrete class for time32 data types.

    Supported time unit resolutions are 's' [second]
    and 'ms' [millisecond].

    Examples
    --------
    Create an instance of time32 type:

    >>> import pyarrow as pa
    >>> pa.time32("ms")
    Time32Type(time32[ms])
    """
    @property
    def unit(self) -> _Time32Unit:
        """
        The time unit ('s' or 'ms').

        Examples
        --------
        >>> import pyarrow as pa
        >>> t = pa.time32("ms")
        >>> t.unit
        'ms'
        """

_Time64Unit = TypeVar("_Time64Unit", bound=Literal["us", "ns"])

class Time64Type(_BasicDataType[dt.time], Generic[_Time64Unit]):
    """
    Concrete class for time64 data types.

    Supported time unit resolutions are 'us' [microsecond]
    and 'ns' [nanosecond].

    Examples
    --------
    Create an instance of time64 type:

    >>> import pyarrow as pa
    >>> pa.time64("us")
    Time64Type(time64[us])
    """
    @property
    def unit(self) -> _Time64Unit:
        """
        The time unit ('us' or 'ns').

        Examples
        --------
        >>> import pyarrow as pa
        >>> t = pa.time64("us")
        >>> t.unit
        'us'
        """

class DurationType(_BasicDataType[dt.timedelta], Generic[_Unit]):
    """
    Concrete class for duration data types.

    Examples
    --------
    Create an instance of duration type:

    >>> import pyarrow as pa
    >>> pa.duration("s")
    DurationType(duration[s])
    """
    @property
    def unit(self) -> _Unit:
        """
        The duration unit ('s', 'ms', 'us' or 'ns').

        Examples
        --------
        >>> import pyarrow as pa
        >>> t = pa.duration("s")
        >>> t.unit
        's'
        """

class FixedSizeBinaryType(_BasicDataType[Decimal]):
    """
    Concrete class for fixed-size binary data types.

    Examples
    --------
    Create an instance of fixed-size binary type:

    >>> import pyarrow as pa
    >>> pa.binary(3)
    FixedSizeBinaryType(fixed_size_binary[3])
    """

_Precision = TypeVar("_Precision", default=Any)
_Scale = TypeVar("_Scale", default=Any)

class Decimal32Type(FixedSizeBinaryType, Generic[_Precision, _Scale]):
    """
    Concrete class for decimal32 data types.

    Examples
    --------
    Create an instance of decimal32 type:

    >>> import pyarrow as pa
    >>> pa.decimal32(5, 2)
    Decimal32Type(decimal32(5, 2))
    """
    @property
    def precision(self) -> _Precision:
        """
        The decimal precision, in number of decimal digits (an integer).

        Examples
        --------
        >>> import pyarrow as pa
        >>> t = pa.decimal32(5, 2)
        >>> t.precision
        5
        """
    @property
    def scale(self) -> _Scale:
        """
        The decimal scale (an integer).

        Examples
        --------
        >>> import pyarrow as pa
        >>> t = pa.decimal32(5, 2)
        >>> t.scale
        2
        """

class Decimal64Type(FixedSizeBinaryType, Generic[_Precision, _Scale]):
    """
    Concrete class for decimal64 data types.

    Examples
    --------
    Create an instance of decimal64 type:

    >>> import pyarrow as pa
    >>> pa.decimal64(5, 2)
    Decimal64Type(decimal64(5, 2))
    """
    @property
    def precision(self) -> _Precision:
        """
        The decimal precision, in number of decimal digits (an integer).

        Examples
        --------
        >>> import pyarrow as pa
        >>> t = pa.decimal64(5, 2)
        >>> t.precision
        5
        """
    @property
    def scale(self) -> _Scale:
        """
        The decimal scale (an integer).

        Examples
        --------
        >>> import pyarrow as pa
        >>> t = pa.decimal64(5, 2)
        >>> t.scale
        2
        """

class Decimal128Type(FixedSizeBinaryType, Generic[_Precision, _Scale]):
    """
    Concrete class for decimal128 data types.

    Examples
    --------
    Create an instance of decimal128 type:

    >>> import pyarrow as pa
    >>> pa.decimal128(5, 2)
    Decimal128Type(decimal128(5, 2))
    """
    @property
    def precision(self) -> _Precision:
        """
        The decimal precision, in number of decimal digits (an integer).

        Examples
        --------
        >>> import pyarrow as pa
        >>> t = pa.decimal128(5, 2)
        >>> t.precision
        5
        """
    @property
    def scale(self) -> _Scale:
        """
        The decimal scale (an integer).

        Examples
        --------
        >>> import pyarrow as pa
        >>> t = pa.decimal128(5, 2)
        >>> t.scale
        2
        """

class Decimal256Type(FixedSizeBinaryType, Generic[_Precision, _Scale]):
    """
    Concrete class for decimal256 data types.

    Examples
    --------
    Create an instance of decimal256 type:

    >>> import pyarrow as pa
    >>> pa.decimal256(76, 38)
    Decimal256Type(decimal256(76, 38))
    """
    @property
    def precision(self) -> _Precision:
        """
        The decimal precision, in number of decimal digits (an integer).

        Examples
        --------
        >>> import pyarrow as pa
        >>> t = pa.decimal256(76, 38)
        >>> t.precision
        76
        """
    @property
    def scale(self) -> _Scale:
        """
        The decimal scale (an integer).

        Examples
        --------
        >>> import pyarrow as pa
        >>> t = pa.decimal256(76, 38)
        >>> t.scale
        38
        """

class ListType(DataType, Generic[_DataTypeT]):
    """
    Concrete class for list data types.

    Examples
    --------
    Create an instance of ListType:

    >>> import pyarrow as pa
    >>> pa.list_(pa.string())
    ListType(list<item: string>)
    """
    @property
    def value_field(self) -> Field[_DataTypeT]:
        """
        The field for list values.

        Examples
        --------
        >>> import pyarrow as pa
        >>> pa.list_(pa.string()).value_field
        pyarrow.Field<item: string>
        """
    @property
    def value_type(self) -> _DataTypeT:
        """
        The data type of list values.

        Examples
        --------
        >>> import pyarrow as pa
        >>> pa.list_(pa.string()).value_type
        DataType(string)
        """

class LargeListType(DataType, Generic[_DataTypeT]):
    """
    Concrete class for large list data types
    (like ListType, but with 64-bit offsets).

    Examples
    --------
    Create an instance of LargeListType:

    >>> import pyarrow as pa
    >>> pa.large_list(pa.string())
    LargeListType(large_list<item: string>)
    """
    @property
    def value_field(self) -> Field[_DataTypeT]: ...
    @property
    def value_type(self) -> _DataTypeT:
        """
        The data type of large list values.

        Examples
        --------
        >>> import pyarrow as pa
        >>> pa.large_list(pa.string()).value_type
        DataType(string)
        """

class ListViewType(DataType, Generic[_DataTypeT]):
    """
    Concrete class for list view data types.

    Examples
    --------
    Create an instance of ListViewType:

    >>> import pyarrow as pa
    >>> pa.list_view(pa.string())
    ListViewType(list_view<item: string>)
    """
    @property
    def value_field(self) -> Field[_DataTypeT]:
        """
        The field for list view values.

        Examples
        --------
        >>> import pyarrow as pa
        >>> pa.list_view(pa.string()).value_field
        pyarrow.Field<item: string>
        """
    @property
    def value_type(self) -> _DataTypeT:
        """
        The data type of list view values.

        Examples
        --------
        >>> import pyarrow as pa
        >>> pa.list_view(pa.string()).value_type
        DataType(string)
        """

class LargeListViewType(DataType, Generic[_DataTypeT]):
    """
    Concrete class for large list view data types
    (like ListViewType, but with 64-bit offsets).

    Examples
    --------
    Create an instance of LargeListViewType:

    >>> import pyarrow as pa
    >>> pa.large_list_view(pa.string())
    LargeListViewType(large_list_view<item: string>)
    """
    @property
    def value_field(self) -> Field[_DataTypeT]:
        """
        The field for large list view values.

        Examples
        --------
        >>> import pyarrow as pa
        >>> pa.large_list_view(pa.string()).value_field
        pyarrow.Field<item: string>
        """
    @property
    def value_type(self) -> _DataTypeT:
        """
        The data type of large list view values.

        Examples
        --------
        >>> import pyarrow as pa
        >>> pa.large_list_view(pa.string()).value_type
        DataType(string)
        """

class FixedSizeListType(DataType, Generic[_DataTypeT, _Size]):
    """
    Concrete class for fixed size list data types.

    Examples
    --------
    Create an instance of FixedSizeListType:

    >>> import pyarrow as pa
    >>> pa.list_(pa.int32(), 2)
    FixedSizeListType(fixed_size_list<item: int32>[2])
    """
    @property
    def value_field(self) -> Field[_DataTypeT]:
        """
        The field for list values.

        Examples
        --------
        >>> import pyarrow as pa
        >>> pa.list_(pa.int32(), 2).value_field
        pyarrow.Field<item: int32>
        """
    @property
    def value_type(self) -> _DataTypeT:
        """
        The data type of large list values.

        Examples
        --------
        >>> import pyarrow as pa
        >>> pa.list_(pa.int32(), 2).value_type
        DataType(int32)
        """
    @property
    def list_size(self) -> _Size:
        """
        The size of the fixed size lists.

        Examples
        --------
        >>> import pyarrow as pa
        >>> pa.list_(pa.int32(), 2).list_size
        2
        """

class DictionaryMemo(_Weakrefable):
    """
    Tracking container for dictionary-encoded fields.
    """

_IndexT = TypeVar(
    "_IndexT",
    UInt8Type,
    Int8Type,
    UInt16Type,
    Int16Type,
    Uint32Type,
    Int32Type,
    UInt64Type,
    Int64Type,
)
_BasicValueT = TypeVar("_BasicValueT", bound=_BasicDataType)
_ValueT = TypeVar("_ValueT", bound=DataType)
_Ordered = TypeVar("_Ordered", Literal[True], Literal[False], default=Literal[False])

class DictionaryType(DataType, Generic[_IndexT, _BasicValueT, _Ordered]):
    """
    Concrete class for dictionary data types.

    Examples
    --------
    Create an instance of dictionary type:

    >>> import pyarrow as pa
    >>> pa.dictionary(pa.int64(), pa.utf8())
    DictionaryType(dictionary<values=string, indices=int64, ordered=0>)
    """

    @property
    def ordered(self) -> _Ordered:
        """
        Whether the dictionary is ordered, i.e. whether the ordering of values
        in the dictionary is important.

        Examples
        --------
        >>> import pyarrow as pa
        >>> pa.dictionary(pa.int64(), pa.utf8()).ordered
        False
        """
    @property
    def index_type(self) -> _IndexT:
        """
        The data type of dictionary indices (a signed integer type).

        Examples
        --------
        >>> import pyarrow as pa
        >>> pa.dictionary(pa.int16(), pa.utf8()).index_type
        DataType(int16)
        """
    @property
    def value_type(self) -> _BasicValueT:
        """
        The dictionary value type.

        The dictionary values are found in an instance of DictionaryArray.

        Examples
        --------
        >>> import pyarrow as pa
        >>> pa.dictionary(pa.int16(), pa.utf8()).value_type
        DataType(string)
        """

_K = TypeVar("_K", bound=DataType)

class MapType(DataType, Generic[_K, _ValueT, _Ordered]):
    """
    Concrete class for map data types.

    Examples
    --------
    Create an instance of MapType:

    >>> import pyarrow as pa
    >>> pa.map_(pa.string(), pa.int32())
    MapType(map<string, int32>)
    >>> pa.map_(pa.string(), pa.int32(), keys_sorted=True)
    MapType(map<string, int32, keys_sorted>)
    """

    @property
    def key_field(self) -> Field[_K]:
        """
        The field for keys in the map entries.

        Examples
        --------
        >>> import pyarrow as pa
        >>> pa.map_(pa.string(), pa.int32()).key_field
        pyarrow.Field<key: string not null>
        """
    @property
    def key_type(self) -> _K:
        """
        The data type of keys in the map entries.

        Examples
        --------
        >>> import pyarrow as pa
        >>> pa.map_(pa.string(), pa.int32()).key_type
        DataType(string)
        """
    @property
    def item_field(self) -> Field[_ValueT]:
        """
        The field for items in the map entries.

        Examples
        --------
        >>> import pyarrow as pa
        >>> pa.map_(pa.string(), pa.int32()).item_field
        pyarrow.Field<value: int32>
        """
    @property
    def item_type(self) -> _ValueT:
        """
        The data type of items in the map entries.

        Examples
        --------
        >>> import pyarrow as pa
        >>> pa.map_(pa.string(), pa.int32()).item_type
        DataType(int32)
        """
    @property
    def keys_sorted(self) -> _Ordered:
        """
        Should the entries be sorted according to keys.

        Examples
        --------
        >>> import pyarrow as pa
        >>> pa.map_(pa.string(), pa.int32(), keys_sorted=True).keys_sorted
        True
        """

_Size = TypeVar("_Size", default=int)

class StructType(DataType):
    """
    Concrete class for struct data types.

    ``StructType`` supports direct indexing using ``[...]`` (implemented via
    ``__getitem__``) to access its fields.
    It will return the struct field with the given index or name.

    Examples
    --------
    >>> import pyarrow as pa

    Accessing fields using direct indexing:

    >>> struct_type = pa.struct({"x": pa.int32(), "y": pa.string()})
    >>> struct_type[0]
    pyarrow.Field<x: int32>
    >>> struct_type["y"]
    pyarrow.Field<y: string>

    Accessing fields using ``field()``:

    >>> struct_type.field(1)
    pyarrow.Field<y: string>
    >>> struct_type.field("x")
    pyarrow.Field<x: int32>

    # Creating a schema from the struct type's fields:
    >>> pa.schema(list(struct_type))
    x: int32
    y: string
    """
    def get_field_index(self, name: str) -> int:
        """
        Return index of the unique field with the given name.

        Parameters
        ----------
        name : str
            The name of the field to look up.

        Returns
        -------
        index : int
            The index of the field with the given name; -1 if the
            name isn't found or there are several fields with the given
            name.

        Examples
        --------
        >>> import pyarrow as pa
        >>> struct_type = pa.struct({"x": pa.int32(), "y": pa.string()})

        Index of the field with a name 'y':

        >>> struct_type.get_field_index("y")
        1

        Index of the field that does not exist:

        >>> struct_type.get_field_index("z")
        -1
        """
    def field(self, i: int | str) -> Field:
        """
        Select a field by its column name or numeric index.

        Parameters
        ----------
        i : int or str

        Returns
        -------
        pyarrow.Field

        Examples
        --------

        >>> import pyarrow as pa
        >>> struct_type = pa.struct({"x": pa.int32(), "y": pa.string()})

        Select the second field:

        >>> struct_type.field(1)
        pyarrow.Field<y: string>

        Select the field named 'x':

        >>> struct_type.field("x")
        pyarrow.Field<x: int32>
        """
    def get_all_field_indices(self, name: str) -> list[int]:
        """
        Return sorted list of indices for the fields with the given name.

        Parameters
        ----------
        name : str
            The name of the field to look up.

        Returns
        -------
        indices : List[int]

        Examples
        --------
        >>> import pyarrow as pa
        >>> struct_type = pa.struct({"x": pa.int32(), "y": pa.string()})
        >>> struct_type.get_all_field_indices("x")
        [0]
        """
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[Field]: ...
    __getitem__ = field  # pyright: ignore[reportUnknownVariableType]
    @property
    def names(self) -> list[str]:
        """
        Lists the field names.

        Examples
        --------
        >>> import pyarrow as pa
        >>> struct_type = pa.struct([("a", pa.int64()), ("b", pa.float64()), ("c", pa.string())])
        >>> struct_type.names
        ['a', 'b', 'c']
        """
    @property
    def fields(self) -> list[Field]:
        """
        Lists all fields within the StructType.

        Examples
        --------
        >>> import pyarrow as pa
        >>> struct_type = pa.struct([("a", pa.int64()), ("b", pa.float64()), ("c", pa.string())])
        >>> struct_type.fields
        [pyarrow.Field<a: int64>, pyarrow.Field<b: double>, pyarrow.Field<c: string>]
        """

class UnionType(DataType):
    """
    Base class for union data types.

    Examples
    --------
    Create an instance of a dense UnionType using ``pa.union``:

    >>> import pyarrow as pa
    >>> (
    ...     pa.union(
    ...         [pa.field("a", pa.binary(10)), pa.field("b", pa.string())],
    ...         mode=pa.lib.UnionMode_DENSE,
    ...     ),
    ... )
    (DenseUnionType(dense_union<a: fixed_size_binary[10]=0, b: string=1>),)

    Create an instance of a dense UnionType using ``pa.dense_union``:

    >>> pa.dense_union([pa.field("a", pa.binary(10)), pa.field("b", pa.string())])
    DenseUnionType(dense_union<a: fixed_size_binary[10]=0, b: string=1>)

    Create an instance of a sparse UnionType using ``pa.union``:

    >>> (
    ...     pa.union(
    ...         [pa.field("a", pa.binary(10)), pa.field("b", pa.string())],
    ...         mode=pa.lib.UnionMode_SPARSE,
    ...     ),
    ... )
    (SparseUnionType(sparse_union<a: fixed_size_binary[10]=0, b: string=1>),)

    Create an instance of a sparse UnionType using ``pa.sparse_union``:

    >>> pa.sparse_union([pa.field("a", pa.binary(10)), pa.field("b", pa.string())])
    SparseUnionType(sparse_union<a: fixed_size_binary[10]=0, b: string=1>)
    """
    @property
    def mode(self) -> Literal["sparse", "dense"]:
        """
        The mode of the union ("dense" or "sparse").

        Examples
        --------
        >>> import pyarrow as pa
        >>> union = pa.sparse_union([pa.field("a", pa.binary(10)), pa.field("b", pa.string())])
        >>> union.mode
        'sparse'
        """
    @property
    def type_codes(self) -> list[int]:
        """
        The type code to indicate each data type in this union.

        Examples
        --------
        >>> import pyarrow as pa
        >>> union = pa.sparse_union([pa.field("a", pa.binary(10)), pa.field("b", pa.string())])
        >>> union.type_codes
        [0, 1]
        """
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[Field]: ...
    def field(self, i: int) -> Field:
        """
        Return a child field by its numeric index.

        Parameters
        ----------
        i : int

        Returns
        -------
        pyarrow.Field

        Examples
        --------
        >>> import pyarrow as pa
        >>> union = pa.sparse_union([pa.field("a", pa.binary(10)), pa.field("b", pa.string())])
        >>> union[0]
        pyarrow.Field<a: fixed_size_binary[10]>
        """
    __getitem__ = field  # pyright: ignore[reportUnknownVariableType]

class SparseUnionType(UnionType):
    """
    Concrete class for sparse union types.

    Examples
    --------
    Create an instance of a sparse UnionType using ``pa.union``:

    >>> import pyarrow as pa
    >>> (
    ...     pa.union(
    ...         [pa.field("a", pa.binary(10)), pa.field("b", pa.string())],
    ...         mode=pa.lib.UnionMode_SPARSE,
    ...     ),
    ... )
    (SparseUnionType(sparse_union<a: fixed_size_binary[10]=0, b: string=1>),)

    Create an instance of a sparse UnionType using ``pa.sparse_union``:

    >>> pa.sparse_union([pa.field("a", pa.binary(10)), pa.field("b", pa.string())])
    SparseUnionType(sparse_union<a: fixed_size_binary[10]=0, b: string=1>)
    """
    @property
    def mode(self) -> Literal["sparse"]: ...

class DenseUnionType(UnionType):
    """
    Concrete class for dense union types.

    Examples
    --------
    Create an instance of a dense UnionType using ``pa.union``:

    >>> import pyarrow as pa
    >>> (
    ...     pa.union(
    ...         [pa.field("a", pa.binary(10)), pa.field("b", pa.string())],
    ...         mode=pa.lib.UnionMode_DENSE,
    ...     ),
    ... )
    (DenseUnionType(dense_union<a: fixed_size_binary[10]=0, b: string=1>),)

    Create an instance of a dense UnionType using ``pa.dense_union``:

    >>> pa.dense_union([pa.field("a", pa.binary(10)), pa.field("b", pa.string())])
    DenseUnionType(dense_union<a: fixed_size_binary[10]=0, b: string=1>)
    """

    @property
    def mode(self) -> Literal["dense"]: ...

_RunEndType = TypeVar("_RunEndType", Int16Type, Int32Type, Int64Type)

class RunEndEncodedType(DataType, Generic[_RunEndType, _BasicValueT]):
    """
    Concrete class for run-end encoded types.
    """
    @property
    def run_end_type(self) -> _RunEndType: ...
    @property
    def value_type(self) -> _BasicValueT: ...

_StorageT = TypeVar("_StorageT", bound=Array | ChunkedArray)

class BaseExtensionType(DataType):
    """
    Concrete base class for extension types.
    """
    def __arrow_ext_class__(self) -> type[ExtensionArray]:
        """
        The associated array extension class
        """
    def __arrow_ext_scalar_class__(self) -> type[ExtensionScalar]:
        """
        The associated scalar class
        """
    @property
    def extension_name(self) -> str:
        """
        The extension type name.
        """
    @property
    def storage_type(self) -> DataType:
        """
        The underlying storage type.
        """
    def wrap_array(self, storage: _StorageT) -> _StorageT: ...

class ExtensionType(BaseExtensionType):
    """
    Concrete base class for Python-defined extension types.

    Parameters
    ----------
    storage_type : DataType
        The underlying storage type for the extension type.
    extension_name : str
        A unique name distinguishing this extension type. The name will be
        used when deserializing IPC data.

    Examples
    --------
    Define a RationalType extension type subclassing ExtensionType:

    >>> import pyarrow as pa
    >>> class RationalType(pa.ExtensionType):
    ...     def __init__(self, data_type: pa.DataType):
    ...         if not pa.types.is_integer(data_type):
    ...             raise TypeError(f"data_type must be an integer type not {data_type}")
    ...         super().__init__(
    ...             pa.struct(
    ...                 [
    ...                     ("numer", data_type),
    ...                     ("denom", data_type),
    ...                 ],
    ...             ),
    ...             # N.B. This name does _not_ reference `data_type` so deserialization
    ...             # will work for _any_ integer `data_type` after registration
    ...             "my_package.rational",
    ...         )
    ...     def __arrow_ext_serialize__(self) -> bytes:
    ...         # No parameters are necessary
    ...         return b""
    ...     @classmethod
    ...     def __arrow_ext_deserialize__(cls, storage_type, serialized):
    ...         # return an instance of this subclass
    ...         return RationalType(storage_type[0].type)

    Register the extension type:

    >>> pa.register_extension_type(RationalType(pa.int64()))

    Create an instance of RationalType extension type:

    >>> rational_type = RationalType(pa.int32())

    Inspect the extension type:

    >>> rational_type.extension_name
    'my_package.rational'
    >>> rational_type.storage_type
    StructType(struct<numer: int32, denom: int32>)

    Wrap an array as an extension array:

    >>> storage_array = pa.array(
    ...     [
    ...         {"numer": 10, "denom": 17},
    ...         {"numer": 20, "denom": 13},
    ...     ],
    ...     type=rational_type.storage_type,
    ... )
    >>> rational_array = rational_type.wrap_array(storage_array)
    >>> rational_array
    <pyarrow.lib.ExtensionArray object at ...>
    -- is_valid: all not null
    -- child 0 type: int32
      [
        10,
        20
      ]
    -- child 1 type: int32
      [
        17,
        13
      ]

    Or do the same with creating an ExtensionArray:

    >>> rational_array = pa.ExtensionArray.from_storage(rational_type, storage_array)
    >>> rational_array
    <pyarrow.lib.ExtensionArray object at ...>
    -- is_valid: all not null
    -- child 0 type: int32
      [
        10,
        20
      ]
    -- child 1 type: int32
      [
        17,
        13
      ]

    Unregister the extension type:

    >>> pa.unregister_extension_type("my_package.rational")

    Note that even though we registered the concrete type
    ``RationalType(pa.int64())``, PyArrow will be able to deserialize
    ``RationalType(integer_type)`` for any ``integer_type``, as the deserializer
    will reference the name ``my_package.rational`` and the ``@classmethod``
    ``__arrow_ext_deserialize__``.
    """

    def __init__(self, storage_type: DataType, extension_name: str) -> None: ...
    def __arrow_ext_serialize__(self) -> bytes:
        """
        Serialized representation of metadata to reconstruct the type object.

        This method should return a bytes object, and those serialized bytes
        are stored in the custom metadata of the Field holding an extension
        type in an IPC message.
        The bytes are passed to ``__arrow_ext_deserialize`` and should hold
        sufficient information to reconstruct the data type instance.
        """
    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type: DataType, serialized: bytes) -> Self:
        """
        Return an extension type instance from the storage type and serialized
        metadata.

        This method should return an instance of the ExtensionType subclass
        that matches the passed storage type and serialized metadata (the
        return value of ``__arrow_ext_serialize__``).
        """

class FixedShapeTensorType(BaseExtensionType, Generic[_ValueT]):
    """
    Concrete class for fixed shape tensor extension type.

    Examples
    --------
    Create an instance of fixed shape tensor extension type:

    >>> import pyarrow as pa
    >>> pa.fixed_shape_tensor(pa.int32(), [2, 2])
    FixedShapeTensorType(extension<arrow.fixed_shape_tensor[value_type=int32, shape=[2,2]]>)

    Create an instance of fixed shape tensor extension type with
    permutation:

    >>> tensor_type = pa.fixed_shape_tensor(pa.int8(), (2, 2, 3), permutation=[0, 2, 1])
    >>> tensor_type.permutation
    [0, 2, 1]
    """
    @property
    def value_type(self) -> _ValueT:
        """
        Data type of an individual tensor.
        """
    @property
    def shape(self) -> list[int]:
        """
        Shape of the tensors.
        """
    @property
    def dim_names(self) -> list[str] | None:
        """
        Explicit names of the dimensions.
        """
    @property
    def permutation(self) -> list[int] | None:
        """
        Indices of the dimensions ordering.
        """

class Bool8Type(BaseExtensionType):
    """
    Concrete class for bool8 extension type.

    Bool8 is an alternate representation for boolean
    arrays using 8 bits instead of 1 bit per value. The underlying
    storage type is int8.

    Examples
    --------
    Create an instance of bool8 extension type:

    >>> import pyarrow as pa
    >>> pa.bool8()
    Bool8Type(extension<arrow.bool8>)
    """

class UuidType(BaseExtensionType):
    """
    Concrete class for UUID extension type.
    """

class JsonType(BaseExtensionType):
    """
    Concrete class for JSON extension type.

    Examples
    --------
    Define the extension type for JSON array

    >>> import pyarrow as pa
    >>> json_type = pa.json_(pa.large_utf8())

    Create an extension array

    >>> arr = [None, '{ "id":30, "values":["a", "b"] }']
    >>> storage = pa.array(arr, pa.large_utf8())
    >>> pa.ExtensionArray.from_storage(json_type, storage)
    <pyarrow.lib.JsonArray object at ...>
    [
      null,
      "{ "id":30, "values":["a", "b"] }"
    ]
    """

class OpaqueType(BaseExtensionType):
    """
    Concrete class for opaque extension type.

    Opaque is a placeholder for a type from an external (often non-Arrow)
    system that could not be interpreted.

    Examples
    --------
    Create an instance of opaque extension type:

    >>> import pyarrow as pa
    >>> pa.opaque(pa.int32(), "geometry", "postgis")
    OpaqueType(extension<arrow.opaque[storage_type=int32, type_name=geometry, vendor_name=postgis]>)
    """
    @property
    def type_name(self) -> str:
        """
        The name of the type in the external system.
        """
    @property
    def vendor_name(self) -> str:
        """
        The name of the external system.
        """

@deprecated(
    "This class is deprecated and its deserialization is disabled by default. "
    ":class:`ExtensionType` is recommended instead."
)
class PyExtensionType(ExtensionType):
    """
    Concrete base class for Python-defined extension types based on pickle
    for (de)serialization.

    .. warning::
       This class is deprecated and its deserialization is disabled by default.
       :class:`ExtensionType` is recommended instead.

    Parameters
    ----------
    storage_type : DataType
        The storage type for which the extension is built.
    """
    def __init__(self, storage_type: DataType) -> None: ...
    @classmethod
    def set_auto_load(cls, value: bool) -> None:
        """
        Enable or disable auto-loading of serialized PyExtensionType instances.

        Parameters
        ----------
        value : bool
            Whether to enable auto-loading.
        """

class UnknownExtensionType(PyExtensionType):  # type: ignore
    """
    A concrete class for Python-defined extension types that refer to
    an unknown Python implementation.

    Parameters
    ----------
    storage_type : DataType
        The storage type for which the extension is built.
    serialized : bytes
        The serialised output.
    """
    def __init__(self, storage_type: DataType, serialized: bytes) -> None: ...

def register_extension_type(ext_type: PyExtensionType) -> None:  # type: ignore
    """
    Register a Python extension type.

    Registration is based on the extension name (so different registered types
    need unique extension names). Registration needs an extension type
    instance, but then works for any instance of the same subclass regardless
    of parametrization of the type.

    Parameters
    ----------
    ext_type : BaseExtensionType instance
        The ExtensionType subclass to register.

    Examples
    --------
    Define a RationalType extension type subclassing ExtensionType:

    >>> import pyarrow as pa
    >>> class RationalType(pa.ExtensionType):
    ...     def __init__(self, data_type: pa.DataType):
    ...         if not pa.types.is_integer(data_type):
    ...             raise TypeError(f"data_type must be an integer type not {data_type}")
    ...         super().__init__(
    ...             pa.struct(
    ...                 [
    ...                     ("numer", data_type),
    ...                     ("denom", data_type),
    ...                 ],
    ...             ),
    ...             # N.B. This name does _not_ reference `data_type` so deserialization
    ...             # will work for _any_ integer `data_type` after registration
    ...             "my_package.rational",
    ...         )
    ...     def __arrow_ext_serialize__(self) -> bytes:
    ...         # No parameters are necessary
    ...         return b""
    ...     @classmethod
    ...     def __arrow_ext_deserialize__(cls, storage_type, serialized):
    ...         # return an instance of this subclass
    ...         return RationalType(storage_type[0].type)

    Register the extension type:

    >>> pa.register_extension_type(RationalType(pa.int64()))

    Unregister the extension type:

    >>> pa.unregister_extension_type("my_package.rational")
    """

def unregister_extension_type(type_name: str) -> None:
    """
    Unregister a Python extension type.

    Parameters
    ----------
    type_name : str
        The name of the ExtensionType subclass to unregister.

    Examples
    --------
    Define a RationalType extension type subclassing ExtensionType:

    >>> import pyarrow as pa
    >>> class RationalType(pa.ExtensionType):
    ...     def __init__(self, data_type: pa.DataType):
    ...         if not pa.types.is_integer(data_type):
    ...             raise TypeError(f"data_type must be an integer type not {data_type}")
    ...         super().__init__(
    ...             pa.struct(
    ...                 [
    ...                     ("numer", data_type),
    ...                     ("denom", data_type),
    ...                 ],
    ...             ),
    ...             # N.B. This name does _not_ reference `data_type` so deserialization
    ...             # will work for _any_ integer `data_type` after registration
    ...             "my_package.rational",
    ...         )
    ...     def __arrow_ext_serialize__(self) -> bytes:
    ...         # No parameters are necessary
    ...         return b""
    ...     @classmethod
    ...     def __arrow_ext_deserialize__(cls, storage_type, serialized):
    ...         # return an instance of this subclass
    ...         return RationalType(storage_type[0].type)

    Register the extension type:

    >>> pa.register_extension_type(RationalType(pa.int64()))

    Unregister the extension type:

    >>> pa.unregister_extension_type("my_package.rational")
    """

class KeyValueMetadata(_Metadata, Mapping[bytes, bytes]):
    """
    KeyValueMetadata

    Parameters
    ----------
    __arg0__ : dict
        A dict of the key-value metadata
    **kwargs : optional
        additional key-value metadata
    """
    def __init__(self, __arg0__: Mapping[bytes, bytes] | None = None, **kwargs) -> None: ...
    def equals(self, other: KeyValueMetadata) -> bool: ...
    def __len__(self) -> int: ...
    def __contains__(self, __key: object) -> bool: ...
    def __getitem__(self, __key: Any) -> Any: ...
    def __iter__(self) -> Iterator[bytes]: ...
    def get_all(self, key: str) -> list[bytes]: ...
    def to_dict(self) -> dict[bytes, bytes]:
        """
        Convert KeyValueMetadata to dict. If a key occurs twice, the value for
        the first one is returned
        """

def ensure_metadata(
    meta: Mapping[bytes | str, bytes | str] | KeyValueMetadata | None, allow_none: bool = False
) -> KeyValueMetadata | None: ...

_DataTypeT_co = TypeVar("_DataTypeT_co", bound=DataType, covariant=True)

class Field(_Weakrefable, Generic[_DataTypeT_co]):
    """
    A named field, with a data type, nullability, and optional metadata.

    Notes
    -----
    Do not use this class's constructor directly; use pyarrow.field

    Examples
    --------
    Create an instance of pyarrow.Field:

    >>> import pyarrow as pa
    >>> pa.field("key", pa.int32())
    pyarrow.Field<key: int32>
    >>> pa.field("key", pa.int32(), nullable=False)
    pyarrow.Field<key: int32 not null>
    >>> field = pa.field("key", pa.int32(), metadata={"key": "Something important"})
    >>> field
    pyarrow.Field<key: int32>
    >>> field.metadata
    {b'key': b'Something important'}

    Use the field to create a struct type:

    >>> pa.struct([field])
    StructType(struct<key: int32>)
    """

    def equals(self, other: Field, check_metadata: bool = False) -> bool:
        """
        Test if this field is equal to the other

        Parameters
        ----------
        other : pyarrow.Field
        check_metadata : bool, default False
            Whether Field metadata equality should be checked as well.

        Returns
        -------
        is_equal : bool

        Examples
        --------
        >>> import pyarrow as pa
        >>> f1 = pa.field("key", pa.int32())
        >>> f2 = pa.field("key", pa.int32(), nullable=False)
        >>> f1.equals(f2)
        False
        >>> f1.equals(f1)
        True
        """
    def __hash__(self) -> int: ...
    @property
    def nullable(self) -> bool:
        """
        The field nullability.

        Examples
        --------
        >>> import pyarrow as pa
        >>> f1 = pa.field("key", pa.int32())
        >>> f2 = pa.field("key", pa.int32(), nullable=False)
        >>> f1.nullable
        True
        >>> f2.nullable
        False
        """
    @property
    def name(self) -> str:
        """
        The field name.

        Examples
        --------
        >>> import pyarrow as pa
        >>> field = pa.field("key", pa.int32())
        >>> field.name
        'key'
        """
    @property
    def metadata(self) -> dict[bytes, bytes] | None:
        """
        The field metadata (if any is set).

        Returns
        -------
        metadata : dict or None

        Examples
        --------
        >>> import pyarrow as pa
        >>> field = pa.field("key", pa.int32(), metadata={"key": "Something important"})
        >>> field.metadata
        {b'key': b'Something important'}
        """
    @property
    def type(self: Field[_DataTypeT]) -> _DataTypeT: ...
    def with_metadata(self, metadata: dict[bytes | str, bytes | str]) -> Self:
        """
        Add metadata as dict of string keys and values to Field

        Parameters
        ----------
        metadata : dict
            Keys and values must be string-like / coercible to bytes

        Returns
        -------
        field : pyarrow.Field

        Examples
        --------
        >>> import pyarrow as pa
        >>> field = pa.field("key", pa.int32())

        Create new field by adding metadata to existing one:

        >>> field_new = field.with_metadata({"key": "Something important"})
        >>> field_new
        pyarrow.Field<key: int32>
        >>> field_new.metadata
        {b'key': b'Something important'}
        """
    def remove_metadata(self) -> Self:
        """
        Create new field without metadata, if any

        Returns
        -------
        field : pyarrow.Field

        Examples
        --------
        >>> import pyarrow as pa
        >>> field = pa.field("key", pa.int32(), metadata={"key": "Something important"})
        >>> field.metadata
        {b'key': b'Something important'}

        Create new field by removing the metadata from the existing one:

        >>> field_new = field.remove_metadata()
        >>> field_new.metadata
        """
    def with_type(self, new_type: _DataTypeT) -> Field[_DataTypeT]:
        """
        A copy of this field with the replaced type

        Parameters
        ----------
        new_type : pyarrow.DataType

        Returns
        -------
        field : pyarrow.Field

        Examples
        --------
        >>> import pyarrow as pa
        >>> field = pa.field("key", pa.int32())
        >>> field
        pyarrow.Field<key: int32>

        Create new field by replacing type of an existing one:

        >>> field_new = field.with_type(pa.int64())
        >>> field_new
        pyarrow.Field<key: int64>
        """
    def with_name(self, name: str) -> Self:
        """
        A copy of this field with the replaced name

        Parameters
        ----------
        name : str

        Returns
        -------
        field : pyarrow.Field

        Examples
        --------
        >>> import pyarrow as pa
        >>> field = pa.field("key", pa.int32())
        >>> field
        pyarrow.Field<key: int32>

        Create new field by replacing the name of an existing one:

        >>> field_new = field.with_name("lock")
        >>> field_new
        pyarrow.Field<lock: int32>
        """
    def with_nullable(self: Field[_DataTypeT], nullable: bool) -> Field[_DataTypeT]:
        """
        A copy of this field with the replaced nullability

        Parameters
        ----------
        nullable : bool

        Returns
        -------
        field: pyarrow.Field

        Examples
        --------
        >>> import pyarrow as pa
        >>> field = pa.field("key", pa.int32())
        >>> field
        pyarrow.Field<key: int32>
        >>> field.nullable
        True

        Create new field by replacing the nullability of an existing one:

        >>> field_new = field.with_nullable(False)
        >>> field_new
        pyarrow.Field<key: int32 not null>
        >>> field_new.nullable
        False
        """
    def flatten(self) -> list[Field]:
        """
        Flatten this field.  If a struct field, individual child fields
        will be returned with their names prefixed by the parent's name.

        Returns
        -------
        fields : List[pyarrow.Field]

        Examples
        --------
        >>> import pyarrow as pa
        >>> f1 = pa.field("bar", pa.float64(), nullable=False)
        >>> f2 = pa.field("foo", pa.int32()).with_metadata({"key": "Something important"})
        >>> ff = pa.field("ff", pa.struct([f1, f2]), nullable=False)

        Flatten a struct field:

        >>> ff
        pyarrow.Field<ff: struct<bar: double not null, foo: int32> not null>
        >>> ff.flatten()
        [pyarrow.Field<ff.bar: double not null>, pyarrow.Field<ff.foo: int32>]
        """
    def _export_to_c(self, out_ptr: int) -> None:
        """
        Export to a C ArrowSchema struct, given its pointer.

        Be careful: if you don't pass the ArrowSchema struct to a consumer,
        its memory will leak.  This is a low-level function intended for
        expert users.
        """
    @classmethod
    def _import_from_c(cls, in_ptr: int) -> Self:
        """
        Import Field from a C ArrowSchema struct, given its pointer.

        This is a low-level function intended for expert users.
        """
    def __arrow_c_schema__(self) -> Any:
        """
        Export to a ArrowSchema PyCapsule

        Unlike _export_to_c, this will not leak memory if the capsule is not used.
        """
    @classmethod
    def _import_from_c_capsule(cls, schema) -> Self:
        """
        Import a Field from a ArrowSchema PyCapsule

        Parameters
        ----------
        schema : PyCapsule
            A valid PyCapsule with name 'arrow_schema' containing an
            ArrowSchema pointer.
        """

class Schema(_Weakrefable):
    """
    A named collection of types a.k.a schema. A schema defines the
    column names and types in a record batch or table data structure.
    They also contain metadata about the columns. For example, schemas
    converted from Pandas contain metadata about their original Pandas
    types so they can be converted back to the same types.

    Warnings
    --------
    Do not call this class's constructor directly. Instead use
    :func:`pyarrow.schema` factory function which makes a new Arrow
    Schema object.

    Examples
    --------
    Create a new Arrow Schema object:

    >>> import pyarrow as pa
    >>> pa.schema([("some_int", pa.int32()), ("some_string", pa.string())])
    some_int: int32
    some_string: string

    Create Arrow Schema with metadata:

    >>> pa.schema(
    ...     [pa.field("n_legs", pa.int64()), pa.field("animals", pa.string())],
    ...     metadata={"n_legs": "Number of legs per animal"},
    ... )
    n_legs: int64
    animals: string
    -- schema metadata --
    n_legs: 'Number of legs per animal'
    """

    def __len__(self) -> int: ...
    def __getitem__(self, key: str) -> Field: ...
    _field = __getitem__  # pyright: ignore[reportUnknownVariableType]
    def __iter__(self) -> Iterator[Field]: ...
    def __hash__(self) -> int: ...
    def __sizeof__(self) -> int: ...
    @property
    def pandas_metadata(self) -> dict:
        """
        Return deserialized-from-JSON pandas metadata field (if it exists)

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
        >>> schema = pa.Table.from_pandas(df).schema

        Select pandas metadata field from Arrow Schema:

        >>> schema.pandas_metadata
        {'index_columns': [{'kind': 'range', 'name': None, 'start': 0, 'stop': 4, 'step': 1}], ...
        """
    @property
    def names(self) -> list[str]:
        """
        The schema's field names.

        Returns
        -------
        list of str

        Examples
        --------
        >>> import pyarrow as pa
        >>> schema = pa.schema([pa.field("n_legs", pa.int64()), pa.field("animals", pa.string())])

        Get the names of the schema's fields:

        >>> schema.names
        ['n_legs', 'animals']
        """
    @property
    def types(self) -> list[DataType]:
        """
        The schema's field types.

        Returns
        -------
        list of DataType

        Examples
        --------
        >>> import pyarrow as pa
        >>> schema = pa.schema([pa.field("n_legs", pa.int64()), pa.field("animals", pa.string())])

        Get the types of the schema's fields:

        >>> schema.types
        [DataType(int64), DataType(string)]
        """
    @property
    def metadata(self) -> dict[bytes, bytes]:
        """
        The schema's metadata (if any is set).

        Returns
        -------
        metadata: dict or None

        Examples
        --------
        >>> import pyarrow as pa
        >>> schema = pa.schema(
        ...     [pa.field("n_legs", pa.int64()), pa.field("animals", pa.string())],
        ...     metadata={"n_legs": "Number of legs per animal"},
        ... )

        Get the metadata of the schema's fields:

        >>> schema.metadata
        {b'n_legs': b'Number of legs per animal'}
        """
    def empty_table(self) -> Table:
        """
        Provide an empty table according to the schema.

        Returns
        -------
        table: pyarrow.Table

        Examples
        --------
        >>> import pyarrow as pa
        >>> schema = pa.schema([pa.field("n_legs", pa.int64()), pa.field("animals", pa.string())])

        Create an empty table with schema's fields:

        >>> schema.empty_table()
        pyarrow.Table
        n_legs: int64
        animals: string
        ----
        n_legs: [[]]
        animals: [[]]
        """
    def equals(self, other: Schema, check_metadata: bool = False) -> bool:
        """
        Test if this schema is equal to the other

        Parameters
        ----------
        other :  pyarrow.Schema
        check_metadata : bool, default False
            Key/value metadata must be equal too

        Returns
        -------
        is_equal : bool

        Examples
        --------
        >>> import pyarrow as pa
        >>> schema1 = pa.schema(
        ...     [pa.field("n_legs", pa.int64()), pa.field("animals", pa.string())],
        ...     metadata={"n_legs": "Number of legs per animal"},
        ... )
        >>> schema2 = pa.schema([("some_int", pa.int32()), ("some_string", pa.string())])

        Test two equal schemas:

        >>> schema1.equals(schema1)
        True

        Test two unequal schemas:

        >>> schema1.equals(schema2)
        False
        """
    @classmethod
    def from_pandas(cls, df: pd.DataFrame, preserve_index: bool | None = None) -> Schema:
        """
        Returns implied schema from dataframe

        Parameters
        ----------
        df : pandas.DataFrame
        preserve_index : bool, default True
            Whether to store the index as an additional column (or columns, for
            MultiIndex) in the resulting `Table`.
            The default of None will store the index as a column, except for
            RangeIndex which is stored as metadata only. Use
            ``preserve_index=True`` to force it to be stored as a column.

        Returns
        -------
        pyarrow.Schema

        Examples
        --------
        >>> import pandas as pd
        >>> import pyarrow as pa
        >>> df = pd.DataFrame({"int": [1, 2], "str": ["a", "b"]})

        Create an Arrow Schema from the schema of a pandas dataframe:

        >>> pa.Schema.from_pandas(df)
        int: int64
        str: string
        -- schema metadata --
        pandas: '{"index_columns": [{"kind": "range", "name": null, ...
        """
    def field(self, i: int | str | bytes) -> Field:
        """
        Select a field by its column name or numeric index.

        Parameters
        ----------
        i : int or string

        Returns
        -------
        pyarrow.Field

        Examples
        --------
        >>> import pyarrow as pa
        >>> schema = pa.schema([pa.field("n_legs", pa.int64()), pa.field("animals", pa.string())])

        Select the second field:

        >>> schema.field(1)
        pyarrow.Field<animals: string>

        Select the field of the column named 'n_legs':

        >>> schema.field("n_legs")
        pyarrow.Field<n_legs: int64>
        """
    @deprecated("Use 'field' instead")
    def field_by_name(self, name: str) -> Field:
        """
        DEPRECATED

        Parameters
        ----------
        name : str

        Returns
        -------
        field: pyarrow.Field
        """
    def get_field_index(self, name: str) -> int:
        """
        Return index of the unique field with the given name.

        Parameters
        ----------
        name : str
            The name of the field to look up.

        Returns
        -------
        index : int
            The index of the field with the given name; -1 if the
            name isn't found or there are several fields with the given
            name.

        Examples
        --------
        >>> import pyarrow as pa
        >>> schema = pa.schema([pa.field("n_legs", pa.int64()), pa.field("animals", pa.string())])

        Get the index of the field named 'animals':

        >>> schema.get_field_index("animals")
        1

        Index in case of several fields with the given name:

        >>> schema = pa.schema(
        ...     [
        ...         pa.field("n_legs", pa.int64()),
        ...         pa.field("animals", pa.string()),
        ...         pa.field("animals", pa.bool_()),
        ...     ],
        ...     metadata={"n_legs": "Number of legs per animal"},
        ... )
        >>> schema.get_field_index("animals")
        -1
        """
    def get_all_field_indices(self, name: str) -> list[int]:
        """
        Return sorted list of indices for the fields with the given name.

        Parameters
        ----------
        name : str
            The name of the field to look up.

        Returns
        -------
        indices : List[int]

        Examples
        --------
        >>> import pyarrow as pa
        >>> schema = pa.schema(
        ...     [
        ...         pa.field("n_legs", pa.int64()),
        ...         pa.field("animals", pa.string()),
        ...         pa.field("animals", pa.bool_()),
        ...     ]
        ... )

        Get the indexes of the fields named 'animals':

        >>> schema.get_all_field_indices("animals")
        [1, 2]
        """
    def append(self, field: Field) -> Schema:
        """
        Append a field at the end of the schema.

        In contrast to Python's ``list.append()`` it does return a new
        object, leaving the original Schema unmodified.

        Parameters
        ----------
        field : Field

        Returns
        -------
        schema: Schema
            New object with appended field.

        Examples
        --------
        >>> import pyarrow as pa
        >>> schema = pa.schema([pa.field("n_legs", pa.int64()), pa.field("animals", pa.string())])

        Append a field 'extra' at the end of the schema:

        >>> schema_new = schema.append(pa.field("extra", pa.bool_()))
        >>> schema_new
        n_legs: int64
        animals: string
        extra: bool

        Original schema is unmodified:

        >>> schema
        n_legs: int64
        animals: string
        """
    def insert(self, i: int, field: Field) -> Schema:
        """
        Add a field at position i to the schema.

        Parameters
        ----------
        i : int
        field : Field

        Returns
        -------
        schema: Schema

        Examples
        --------
        >>> import pyarrow as pa
        >>> schema = pa.schema([pa.field("n_legs", pa.int64()), pa.field("animals", pa.string())])

        Insert a new field on the second position:

        >>> schema.insert(1, pa.field("extra", pa.bool_()))
        n_legs: int64
        extra: bool
        animals: string
        """
    def remove(self, i: int) -> Schema:
        """
        Remove the field at index i from the schema.

        Parameters
        ----------
        i : int

        Returns
        -------
        schema: Schema

        Examples
        --------
        >>> import pyarrow as pa
        >>> schema = pa.schema([pa.field("n_legs", pa.int64()), pa.field("animals", pa.string())])

        Remove the second field of the schema:

        >>> schema.remove(1)
        n_legs: int64
        """
    def set(self, i: int, field: Field) -> Schema:
        """
        Replace a field at position i in the schema.

        Parameters
        ----------
        i : int
        field : Field

        Returns
        -------
        schema: Schema

        Examples
        --------
        >>> import pyarrow as pa
        >>> schema = pa.schema([pa.field("n_legs", pa.int64()), pa.field("animals", pa.string())])

        Replace the second field of the schema with a new field 'extra':

        >>> schema.set(1, pa.field("replaced", pa.bool_()))
        n_legs: int64
        replaced: bool
        """
    @deprecated("Use 'with_metadata' instead")
    def add_metadata(self, metadata: dict) -> Schema:
        """
        DEPRECATED

        Parameters
        ----------
        metadata : dict
            Keys and values must be string-like / coercible to bytes
        """
    def with_metadata(self, metadata: dict) -> Schema:
        """
        Add metadata as dict of string keys and values to Schema

        Parameters
        ----------
        metadata : dict
            Keys and values must be string-like / coercible to bytes

        Returns
        -------
        schema : pyarrow.Schema

        Examples
        --------
        >>> import pyarrow as pa
        >>> schema = pa.schema([pa.field("n_legs", pa.int64()), pa.field("animals", pa.string())])

        Add metadata to existing schema field:

        >>> schema.with_metadata({"n_legs": "Number of legs per animal"})
        n_legs: int64
        animals: string
        -- schema metadata --
        n_legs: 'Number of legs per animal'
        """
    def serialize(self, memory_pool: MemoryPool | None = None) -> Buffer:
        """
        Write Schema to Buffer as encapsulated IPC message

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
        >>> schema = pa.schema([pa.field("n_legs", pa.int64()), pa.field("animals", pa.string())])

        Write schema to Buffer:

        >>> schema.serialize()
        <pyarrow.Buffer address=0x... size=... is_cpu=True is_mutable=True>
        """
    def remove_metadata(self) -> Schema:
        """
        Create new schema without metadata, if any

        Returns
        -------
        schema : pyarrow.Schema

        Examples
        --------
        >>> import pyarrow as pa
        >>> schema = pa.schema(
        ...     [pa.field("n_legs", pa.int64()), pa.field("animals", pa.string())],
        ...     metadata={"n_legs": "Number of legs per animal"},
        ... )
        >>> schema
        n_legs: int64
        animals: string
        -- schema metadata --
        n_legs: 'Number of legs per animal'

        Create a new schema with removing the metadata from the original:

        >>> schema.remove_metadata()
        n_legs: int64
        animals: string
        """
    def to_string(
        self,
        truncate_metadata: bool = True,
        show_field_metadata: bool = True,
        show_schema_metadata: bool = True,
    ) -> str:
        """
        Return human-readable representation of Schema

        Parameters
        ----------
        truncate_metadata : boolean, default True
            Limit metadata key/value display to a single line of ~80 characters
            or less
        show_field_metadata : boolean, default True
            Display Field-level KeyValueMetadata
        show_schema_metadata : boolean, default True
            Display Schema-level KeyValueMetadata

        Returns
        -------
        str : the formatted output
        """
    def _export_to_c(self, out_ptr: int) -> None:
        """
        Export to a C ArrowSchema struct, given its pointer.

        Be careful: if you don't pass the ArrowSchema struct to a consumer,
        its memory will leak.  This is a low-level function intended for
        expert users.
        """
    @classmethod
    def _import_from_c(cls, in_ptr: int) -> Schema:
        """
        Import Schema from a C ArrowSchema struct, given its pointer.

        This is a low-level function intended for expert users.
        """
    def __arrow_c_schema__(self) -> Any:
        """
        Export to a ArrowSchema PyCapsule

        Unlike _export_to_c, this will not leak memory if the capsule is not used.
        """
    @staticmethod
    def _import_from_c_capsule(schema: Any) -> Schema:
        """
        Import a Schema from a ArrowSchema PyCapsule

        Parameters
        ----------
        schema : PyCapsule
            A valid PyCapsule with name 'arrow_schema' containing an
            ArrowSchema pointer.
        """

def unify_schemas(
    schemas: list[Schema], *, promote_options: Literal["default", "permissive"] = "default"
) -> Schema:
    """
    Unify schemas by merging fields by name.

    The resulting schema will contain the union of fields from all schemas.
    Fields with the same name will be merged. Note that two fields with
    different types will fail merging by default.

    - The unified field will inherit the metadata from the schema where
        that field is first defined.
    - The first N fields in the schema will be ordered the same as the
        N fields in the first schema.

    The resulting schema will inherit its metadata from the first input
    schema.

    Parameters
    ----------
    schemas : list of Schema
        Schemas to merge into a single one.
    promote_options : str, default default
        Accepts strings "default" and "permissive".
        Default: null and only null can be unified with another type.
        Permissive: types are promoted to the greater common denominator.

    Returns
    -------
    Schema

    Raises
    ------
    ArrowInvalid :
        If any input schema contains fields with duplicate names.
        If Fields of the same name are not mergeable.
    """

@overload
def field(name: SupportArrowSchema) -> Field[Any]: ...
@overload
def field(
    name: str, type: _DataTypeT, nullable: bool = ..., metadata: dict[Any, Any] | None = None
) -> Field[_DataTypeT]: ...
def field(*args, **kwargs):
    """
    Create a pyarrow.Field instance.

    Parameters
    ----------
    name : str or bytes
        Name of the field.
        Alternatively, you can also pass an object that implements the Arrow
        PyCapsule Protocol for schemas (has an ``__arrow_c_schema__`` method).
    type : pyarrow.DataType or str
        Arrow datatype of the field or a string matching one.
    nullable : bool, default True
        Whether the field's values are nullable.
    metadata : dict, default None
        Optional field metadata, the keys and values must be coercible to
        bytes.

    Returns
    -------
    field : pyarrow.Field

    Examples
    --------
    Create an instance of pyarrow.Field:

    >>> import pyarrow as pa
    >>> pa.field("key", pa.int32())
    pyarrow.Field<key: int32>
    >>> pa.field("key", pa.int32(), nullable=False)
    pyarrow.Field<key: int32 not null>

    >>> field = pa.field("key", pa.int32(), metadata={"key": "Something important"})
    >>> field
    pyarrow.Field<key: int32>
    >>> field.metadata
    {b'key': b'Something important'}

    Use the field to create a struct type:

    >>> pa.struct([field])
    StructType(struct<key: int32>)

    A str can also be passed for the type parameter:

    >>> pa.field("key", "int32")
    pyarrow.Field<key: int32>
    """

def null() -> NullType:
    """
    Create instance of null type.

    Examples
    --------
    Create an instance of a null type:

    >>> import pyarrow as pa
    >>> pa.null()
    DataType(null)
    >>> print(pa.null())
    null

    Create a ``Field`` type with a null type and a name:

    >>> pa.field("null_field", pa.null())
    pyarrow.Field<null_field: null>
    """

def bool_() -> BoolType:
    """
    Create instance of boolean type.

    Examples
    --------
    Create an instance of a boolean type:

    >>> import pyarrow as pa
    >>> pa.bool_()
    DataType(bool)
    >>> print(pa.bool_())
    bool

    Create a ``Field`` type with a boolean type
    and a name:

    >>> pa.field("bool_field", pa.bool_())
    pyarrow.Field<bool_field: bool>
    """

def uint8() -> UInt8Type:
    """
    Create instance of unsigned int8 type.

    Examples
    --------
    Create an instance of unsigned int8 type:

    >>> import pyarrow as pa
    >>> pa.uint8()
    DataType(uint8)
    >>> print(pa.uint8())
    uint8

    Create an array with unsigned int8 type:

    >>> pa.array([0, 1, 2], type=pa.uint8())
    <pyarrow.lib.UInt8Array object at ...>
    [
      0,
      1,
      2
    ]
    """

def int8() -> Int8Type:
    """
    Create instance of signed int8 type.

    Examples
    --------
    Create an instance of int8 type:

    >>> import pyarrow as pa
    >>> pa.int8()
    DataType(int8)
    >>> print(pa.int8())
    int8

    Create an array with int8 type:

    >>> pa.array([0, 1, 2], type=pa.int8())
    <pyarrow.lib.Int8Array object at ...>
    [
      0,
      1,
      2
    ]
    """

def uint16() -> UInt16Type:
    """
    Create instance of unsigned uint16 type.

    Examples
    --------
    Create an instance of unsigned int16 type:

    >>> import pyarrow as pa
    >>> pa.uint16()
    DataType(uint16)
    >>> print(pa.uint16())
    uint16

    Create an array with unsigned int16 type:

    >>> pa.array([0, 1, 2], type=pa.uint16())
    <pyarrow.lib.UInt16Array object at ...>
    [
      0,
      1,
      2
    ]
    """

def int16() -> Int16Type:
    """
    Create instance of signed int16 type.

    Examples
    --------
    Create an instance of int16 type:

    >>> import pyarrow as pa
    >>> pa.int16()
    DataType(int16)
    >>> print(pa.int16())
    int16

    Create an array with int16 type:

    >>> pa.array([0, 1, 2], type=pa.int16())
    <pyarrow.lib.Int16Array object at ...>
    [
      0,
      1,
      2
    ]
    """

def uint32() -> Uint32Type:
    """
    Create instance of unsigned uint32 type.

    Examples
    --------
    Create an instance of unsigned int32 type:

    >>> import pyarrow as pa
    >>> pa.uint32()
    DataType(uint32)
    >>> print(pa.uint32())
    uint32

    Create an array with unsigned int32 type:

    >>> pa.array([0, 1, 2], type=pa.uint32())
    <pyarrow.lib.UInt32Array object at ...>
    [
      0,
      1,
      2
    ]
    """

def int32() -> Int32Type:
    """
    Create instance of signed int32 type.

    Examples
    --------
    Create an instance of int32 type:

    >>> import pyarrow as pa
    >>> pa.int32()
    DataType(int32)
    >>> print(pa.int32())
    int32

    Create an array with int32 type:

    >>> pa.array([0, 1, 2], type=pa.int32())
    <pyarrow.lib.Int32Array object at ...>
    [
      0,
      1,
      2
    ]
    """

def int64() -> Int64Type:
    """
    Create instance of signed int64 type.

    Examples
    --------
    Create an instance of int64 type:

    >>> import pyarrow as pa
    >>> pa.int64()
    DataType(int64)
    >>> print(pa.int64())
    int64

    Create an array with int64 type:

    >>> pa.array([0, 1, 2], type=pa.int64())
    <pyarrow.lib.Int64Array object at ...>
    [
      0,
      1,
      2
    ]
    """

def uint64() -> UInt64Type:
    """
    Create instance of unsigned uint64 type.

    Examples
    --------
    Create an instance of unsigned int64 type:

    >>> import pyarrow as pa
    >>> pa.uint64()
    DataType(uint64)
    >>> print(pa.uint64())
    uint64

    Create an array with unsigned uint64 type:

    >>> pa.array([0, 1, 2], type=pa.uint64())
    <pyarrow.lib.UInt64Array object at ...>
    [
      0,
      1,
      2
    ]
    """

def tzinfo_to_string(tz: dt.tzinfo) -> str:
    """
    Converts a time zone object into a string indicating the name of a time
    zone, one of:
    * As used in the Olson time zone database (the "tz database" or
      "tzdata"), such as "America/New_York"
    * An absolute time zone offset of the form +XX:XX or -XX:XX, such as +07:30

    Parameters
    ----------
      tz : datetime.tzinfo
        Time zone object

    Returns
    -------
      name : str
        Time zone name
    """

def string_to_tzinfo(name: str) -> dt.tzinfo:
    """
    Convert a time zone name into a time zone object.

    Supported input strings are:
    * As used in the Olson time zone database (the "tz database" or
      "tzdata"), such as "America/New_York"
    * An absolute time zone offset of the form +XX:XX or -XX:XX, such as +07:30

    Parameters
    ----------
      name: str
        Time zone name.

    Returns
    -------
      tz : datetime.tzinfo
        Time zone object
    """

@overload
def timestamp(unit: _Unit) -> TimestampType[_Unit, _Tz]: ...
@overload
def timestamp(unit: _Unit, tz: _Tz) -> TimestampType[_Unit, _Tz]: ...
def timestamp(*args, **kwargs):
    """
    Create instance of timestamp type with resolution and optional time zone.

    Parameters
    ----------
    unit : str
        one of 's' [second], 'ms' [millisecond], 'us' [microsecond], or 'ns'
        [nanosecond]
    tz : str, default None
        Time zone name. None indicates time zone naive

    Examples
    --------
    Create an instance of timestamp type:

    >>> import pyarrow as pa
    >>> pa.timestamp("us")
    TimestampType(timestamp[us])
    >>> pa.timestamp("s", tz="America/New_York")
    TimestampType(timestamp[s, tz=America/New_York])
    >>> pa.timestamp("s", tz="+07:30")
    TimestampType(timestamp[s, tz=+07:30])

    Use timestamp type when creating a scalar object:

    >>> from datetime import datetime
    >>> pa.scalar(datetime(2012, 1, 1), type=pa.timestamp("s", tz="UTC"))
    <pyarrow.TimestampScalar: '2012-01-01T00:00:00+0000'>
    >>> pa.scalar(datetime(2012, 1, 1), type=pa.timestamp("us"))
    <pyarrow.TimestampScalar: '2012-01-01T00:00:00.000000'>

    Returns
    -------
    timestamp_type : TimestampType
    """

def time32(unit: _Time32Unit) -> Time32Type[_Time32Unit]:
    """
    Create instance of 32-bit time (time of day) type with unit resolution.

    Parameters
    ----------
    unit : str
        one of 's' [second], or 'ms' [millisecond]

    Returns
    -------
    type : pyarrow.Time32Type

    Examples
    --------
    >>> import pyarrow as pa
    >>> pa.time32("s")
    Time32Type(time32[s])
    >>> pa.time32("ms")
    Time32Type(time32[ms])
    """

def time64(unit: _Time64Unit) -> Time64Type[_Time64Unit]:
    """
    Create instance of 64-bit time (time of day) type with unit resolution.

    Parameters
    ----------
    unit : str
        One of 'us' [microsecond], or 'ns' [nanosecond].

    Returns
    -------
    type : pyarrow.Time64Type

    Examples
    --------
    >>> import pyarrow as pa
    >>> pa.time64("us")
    Time64Type(time64[us])
    >>> pa.time64("ns")
    Time64Type(time64[ns])
    """

def duration(unit: _Unit) -> DurationType[_Unit]:
    """
    Create instance of a duration type with unit resolution.

    Parameters
    ----------
    unit : str
        One of 's' [second], 'ms' [millisecond], 'us' [microsecond], or
        'ns' [nanosecond].

    Returns
    -------
    type : pyarrow.DurationType

    Examples
    --------
    Create an instance of duration type:

    >>> import pyarrow as pa
    >>> pa.duration("us")
    DurationType(duration[us])
    >>> pa.duration("s")
    DurationType(duration[s])

    Create an array with duration type:

    >>> pa.array([0, 1, 2], type=pa.duration("s"))
    <pyarrow.lib.DurationArray object at ...>
    [
      0,
      1,
      2
    ]
    """

def month_day_nano_interval() -> MonthDayNanoIntervalType:
    """
    Create instance of an interval type representing months, days and
    nanoseconds between two dates.

    Examples
    --------
    Create an instance of an month_day_nano_interval type:

    >>> import pyarrow as pa
    >>> pa.month_day_nano_interval()
    DataType(month_day_nano_interval)

    Create a scalar with month_day_nano_interval type:

    >>> pa.scalar((1, 15, -30), type=pa.month_day_nano_interval())
    <pyarrow.MonthDayNanoIntervalScalar: MonthDayNano(months=1, days=15, nanoseconds=-30)>
    """

def date32() -> Date32Type:
    """
    Create instance of 32-bit date (days since UNIX epoch 1970-01-01).

    Examples
    --------
    Create an instance of 32-bit date type:

    >>> import pyarrow as pa
    >>> pa.date32()
    DataType(date32[day])

    Create a scalar with 32-bit date type:

    >>> from datetime import date
    >>> pa.scalar(date(2012, 1, 1), type=pa.date32())
    <pyarrow.Date32Scalar: datetime.date(2012, 1, 1)>
    """

def date64() -> Date64Type:
    """
    Create instance of 64-bit date (milliseconds since UNIX epoch 1970-01-01).

    Examples
    --------
    Create an instance of 64-bit date type:

    >>> import pyarrow as pa
    >>> pa.date64()
    DataType(date64[ms])

    Create a scalar with 64-bit date type:

    >>> from datetime import datetime
    >>> pa.scalar(datetime(2012, 1, 1), type=pa.date64())
    <pyarrow.Date64Scalar: datetime.date(2012, 1, 1)>
    """

def float16() -> Float16Type:
    """
    Create half-precision floating point type.

    Examples
    --------
    Create an instance of float16 type:

    >>> import pyarrow as pa
    >>> pa.float16()
    DataType(halffloat)
    >>> print(pa.float16())
    halffloat

    Create an array with float16 type:

    >>> arr = np.array([1.5, np.nan], dtype=np.float16)
    >>> a = pa.array(arr, type=pa.float16())
    >>> a
    <pyarrow.lib.HalfFloatArray object at ...>
    [
      15872,
      32256
    ]

    Note that unlike other float types, if you convert this array
    to a python list, the types of its elements will be ``np.float16``

    >>> [type(val) for val in a.to_pylist()]
    [<class 'numpy.float16'>, <class 'numpy.float16'>]
    """

def float32() -> Float32Type:
    """
    Create single-precision floating point type.

    Examples
    --------
    Create an instance of float32 type:

    >>> import pyarrow as pa
    >>> pa.float32()
    DataType(float)
    >>> print(pa.float32())
    float

    Create an array with float32 type:

    >>> pa.array([0.0, 1.0, 2.0], type=pa.float32())
    <pyarrow.lib.FloatArray object at ...>
    [
      0,
      1,
      2
    ]
    """

def float64() -> Float64Type:
    """
    Create double-precision floating point type.

    Examples
    --------
    Create an instance of float64 type:

    >>> import pyarrow as pa
    >>> pa.float64()
    DataType(double)
    >>> print(pa.float64())
    double

    Create an array with float64 type:

    >>> pa.array([0.0, 1.0, 2.0], type=pa.float64())
    <pyarrow.lib.DoubleArray object at ...>
    [
      0,
      1,
      2
    ]
    """

@overload
def decimal32(precision: _Precision) -> Decimal32Type[_Precision, Literal[0]]: ...
@overload
def decimal32(precision: _Precision, scale: _Scale) -> Decimal32Type[_Precision, _Scale]: ...
def decimal32(*args, **kwargs):
    """
    Create decimal type with precision and scale and 32-bit width.

    Arrow decimals are fixed-point decimal numbers encoded as a scaled
    integer.  The precision is the number of significant digits that the
    decimal type can represent; the scale is the number of digits after
    the decimal point (note the scale can be negative).

    As an example, ``decimal32(7, 3)`` can exactly represent the numbers
    1234.567 and -1234.567 (encoded internally as the 32-bit integers
    1234567 and -1234567, respectively), but neither 12345.67 nor 123.4567.

    ``decimal32(5, -3)`` can exactly represent the number 12345000
    (encoded internally as the 32-bit integer 12345), but neither
    123450000 nor 1234500.

    If you need a precision higher than 9 significant digits, consider
    using ``decimal64``, ``decimal128``, or ``decimal256``.

    Parameters
    ----------
    precision : int
        Must be between 1 and 9
    scale : int

    Returns
    -------
    decimal_type : Decimal32Type

    Examples
    --------
    Create an instance of decimal type:

    >>> import pyarrow as pa
    >>> pa.decimal32(5, 2)
    Decimal32Type(decimal32(5, 2))

    Create an array with decimal type:

    >>> import decimal
    >>> a = decimal.Decimal("123.45")
    >>> pa.array([a], pa.decimal32(5, 2))
    <pyarrow.lib.Decimal32Array object at ...>
    [
      123.45
    ]
    """

@overload
def decimal64(precision: _Precision) -> Decimal64Type[_Precision, Literal[0]]: ...
@overload
def decimal64(precision: _Precision, scale: _Scale) -> Decimal64Type[_Precision, _Scale]: ...
def decimal64(*args, **kwargs):
    """
    Create decimal type with precision and scale and 64-bit width.

    Arrow decimals are fixed-point decimal numbers encoded as a scaled
    integer.  The precision is the number of significant digits that the
    decimal type can represent; the scale is the number of digits after
    the decimal point (note the scale can be negative).

    As an example, ``decimal64(7, 3)`` can exactly represent the numbers
    1234.567 and -1234.567 (encoded internally as the 64-bit integers
    1234567 and -1234567, respectively), but neither 12345.67 nor 123.4567.

    ``decimal64(5, -3)`` can exactly represent the number 12345000
    (encoded internally as the 64-bit integer 12345), but neither
    123450000 nor 1234500.

    If you need a precision higher than 18 significant digits, consider
    using ``decimal128``, or ``decimal256``.

    Parameters
    ----------
    precision : int
        Must be between 1 and 18
    scale : int

    Returns
    -------
    decimal_type : Decimal64Type

    Examples
    --------
    Create an instance of decimal type:

    >>> import pyarrow as pa
    >>> pa.decimal64(5, 2)
    Decimal64Type(decimal64(5, 2))

    Create an array with decimal type:

    >>> import decimal
    >>> a = decimal.Decimal("123.45")
    >>> pa.array([a], pa.decimal64(5, 2))
    <pyarrow.lib.Decimal64Array object at ...>
    [
      123.45
    ]
    """

@overload
def decimal128(precision: _Precision) -> Decimal128Type[_Precision, Literal[0]]: ...
@overload
def decimal128(precision: _Precision, scale: _Scale) -> Decimal128Type[_Precision, _Scale]: ...
def decimal128(*args, **kwargs):
    """
    Create decimal type with precision and scale and 128-bit width.

    Arrow decimals are fixed-point decimal numbers encoded as a scaled
    integer.  The precision is the number of significant digits that the
    decimal type can represent; the scale is the number of digits after
    the decimal point (note the scale can be negative).

    As an example, ``decimal128(7, 3)`` can exactly represent the numbers
    1234.567 and -1234.567 (encoded internally as the 128-bit integers
    1234567 and -1234567, respectively), but neither 12345.67 nor 123.4567.

    ``decimal128(5, -3)`` can exactly represent the number 12345000
    (encoded internally as the 128-bit integer 12345), but neither
    123450000 nor 1234500.

    If you need a precision higher than 38 significant digits, consider
    using ``decimal256``.

    Parameters
    ----------
    precision : int
        Must be between 1 and 38
    scale : int

    Returns
    -------
    decimal_type : Decimal128Type

    Examples
    --------
    Create an instance of decimal type:

    >>> import pyarrow as pa
    >>> pa.decimal128(5, 2)
    Decimal128Type(decimal128(5, 2))

    Create an array with decimal type:

    >>> import decimal
    >>> a = decimal.Decimal("123.45")
    >>> pa.array([a], pa.decimal128(5, 2))
    <pyarrow.lib.Decimal128Array object at ...>
    [
      123.45
    ]
    """

@overload
def decimal256(precision: _Precision) -> Decimal256Type[_Precision, Literal[0]]: ...
@overload
def decimal256(precision: _Precision, scale: _Scale) -> Decimal256Type[_Precision, _Scale]: ...
def decimal256(*args, **kwargs):
    """
    Create decimal type with precision and scale and 256-bit width.

    Arrow decimals are fixed-point decimal numbers encoded as a scaled
    integer.  The precision is the number of significant digits that the
    decimal type can represent; the scale is the number of digits after
    the decimal point (note the scale can be negative).

    For most use cases, the maximum precision offered by ``decimal128``
    is sufficient, and it will result in a more compact and more efficient
    encoding.  ``decimal256`` is useful if you need a precision higher
    than 38 significant digits.

    Parameters
    ----------
    precision : int
        Must be between 1 and 76
    scale : int

    Returns
    -------
    decimal_type : Decimal256Type
    """

def string() -> StringType:
    """
    Create UTF8 variable-length string type.

    Examples
    --------
    Create an instance of a string type:

    >>> import pyarrow as pa
    >>> pa.string()
    DataType(string)

    and use the string type to create an array:

    >>> pa.array(["foo", "bar", "baz"], type=pa.string())
    <pyarrow.lib.StringArray object at ...>
    [
      "foo",
      "bar",
      "baz"
    ]
    """

utf8 = string
"""
Alias for string().

Examples
--------
Create an instance of a string type:

>>> import pyarrow as pa
>>> pa.utf8()
DataType(string)

and use the string type to create an array:

>>> pa.array(['foo', 'bar', 'baz'], type=pa.utf8())
<pyarrow.lib.StringArray object at ...>
[
    "foo",
    "bar",
    "baz"
]
"""

@overload
def binary(length: Literal[-1] = ...) -> BinaryType: ...
@overload
def binary(length: int) -> FixedSizeBinaryType: ...
def binary(length):
    """
    Create variable-length or fixed size binary type.

    Parameters
    ----------
    length : int, optional, default -1
        If length == -1 then return a variable length binary type. If length is
        greater than or equal to 0 then return a fixed size binary type of
        width `length`.

    Examples
    --------
    Create an instance of a variable-length binary type:

    >>> import pyarrow as pa
    >>> pa.binary()
    DataType(binary)

    and use the variable-length binary type to create an array:

    >>> pa.array(["foo", "bar", "baz"], type=pa.binary())
    <pyarrow.lib.BinaryArray object at ...>
    [
      666F6F,
      626172,
      62617A
    ]

    Create an instance of a fixed-size binary type:

    >>> pa.binary(3)
    FixedSizeBinaryType(fixed_size_binary[3])

    and use the fixed-length binary type to create an array:

    >>> pa.array(["foo", "bar", "baz"], type=pa.binary(3))
    <pyarrow.lib.FixedSizeBinaryArray object at ...>
    [
      666F6F,
      626172,
      62617A
    ]
    """

def large_binary() -> LargeBinaryType:
    """
    Create large variable-length binary type.

    This data type may not be supported by all Arrow implementations.  Unless
    you need to represent data larger than 2GB, you should prefer binary().

    Examples
    --------
    Create an instance of large variable-length binary type:

    >>> import pyarrow as pa
    >>> pa.large_binary()
    DataType(large_binary)

    and use the type to create an array:

    >>> pa.array(["foo", "bar", "baz"], type=pa.large_binary())
    <pyarrow.lib.LargeBinaryArray object at ...>
    [
      666F6F,
      626172,
      62617A
    ]
    """

def large_string() -> LargeStringType:
    """
    Create large UTF8 variable-length string type.

    This data type may not be supported by all Arrow implementations.  Unless
    you need to represent data larger than 2GB, you should prefer string().

    Examples
    --------
    Create an instance of large UTF8 variable-length binary type:

    >>> import pyarrow as pa
    >>> pa.large_string()
    DataType(large_string)

    and use the type to create an array:

    >>> pa.array(["foo", "bar"] * 50, type=pa.large_string())
    <pyarrow.lib.LargeStringArray object at ...>
    [
      "foo",
      "bar",
      ...
      "foo",
      "bar"
    ]
    """

large_utf8 = large_string
"""
Alias for large_string().

Examples
--------
Create an instance of large UTF8 variable-length binary type:

>>> import pyarrow as pa
>>> pa.large_utf8()
DataType(large_string)

and use the type to create an array:

>>> pa.array(['foo', 'bar'] * 50, type=pa.large_utf8())
<pyarrow.lib.LargeStringArray object at ...>
[
    "foo",
    "bar",
    ...
    "foo",
    "bar"
]
"""

def binary_view() -> BinaryViewType:
    """
    Create a variable-length binary view type.

    Examples
    --------
    Create an instance of a string type:

    >>> import pyarrow as pa
    >>> pa.binary_view()
    DataType(binary_view)
    """

def string_view() -> StringViewType:
    """
    Create UTF8 variable-length string view type.

    Examples
    --------
    Create an instance of a string type:

    >>> import pyarrow as pa
    >>> pa.string_view()
    DataType(string_view)
    """

@overload
def list_(
    value_type: _DataTypeT | Field[_DataTypeT], list_size: Literal[-1] = ...
) -> ListType[_DataTypeT]: ...
@overload
def list_(
    value_type: _DataTypeT | Field[_DataTypeT], list_size: _Size
) -> FixedSizeListType[_DataTypeT, _Size]: ...
def list_(*args, **kwargs):
    """
    Create ListType instance from child data type or field.

    Parameters
    ----------
    value_type : DataType or Field
    list_size : int, optional, default -1
        If length == -1 then return a variable length list type. If length is
        greater than or equal to 0 then return a fixed size list type.

    Returns
    -------
    list_type : DataType

    Examples
    --------
    Create an instance of ListType:

    >>> import pyarrow as pa
    >>> pa.list_(pa.string())
    ListType(list<item: string>)
    >>> pa.list_(pa.int32(), 2)
    FixedSizeListType(fixed_size_list<item: int32>[2])

    Use the ListType to create a scalar:

    >>> pa.scalar(["foo", None], type=pa.list_(pa.string(), 2))
    <pyarrow.FixedSizeListScalar: ['foo', None]>

    or an array:

    >>> pa.array([[1, 2], [3, 4]], pa.list_(pa.int32(), 2))
    <pyarrow.lib.FixedSizeListArray object at ...>
    [
      [
        1,
        2
      ],
      [
        3,
        4
      ]
    ]
    """

def large_list(value_type: _DataTypeT | Field[_DataTypeT]) -> LargeListType[_DataTypeT]:
    """
    Create LargeListType instance from child data type or field.

    This data type may not be supported by all Arrow implementations.
    Unless you need to represent data larger than 2**31 elements, you should
    prefer list_().

    Parameters
    ----------
    value_type : DataType or Field

    Returns
    -------
    list_type : DataType

    Examples
    --------
    Create an instance of LargeListType:

    >>> import pyarrow as pa
    >>> pa.large_list(pa.int8())
    LargeListType(large_list<item: int8>)

    Use the LargeListType to create an array:

    >>> pa.array([[-1, 3]] * 5, type=pa.large_list(pa.int8()))
    <pyarrow.lib.LargeListArray object at ...>
    [
      [
        -1,
        3
      ],
      [
        -1,
        3
      ],
    ...
    """

def list_view(value_type: _DataTypeT | Field[_DataTypeT]) -> ListViewType[_DataTypeT]:
    """
    Create ListViewType instance from child data type or field.

    This data type may not be supported by all Arrow implementations
    because it is an alternative to the ListType.

    Parameters
    ----------
    value_type : DataType or Field

    Returns
    -------
    list_view_type : DataType

    Examples
    --------
    Create an instance of ListViewType:

    >>> import pyarrow as pa
    >>> pa.list_view(pa.string())
    ListViewType(list_view<item: string>)
    """

def large_list_view(
    value_type: _DataTypeT | Field[_DataTypeT],
) -> LargeListViewType[_DataTypeT]:
    """
    Create LargeListViewType instance from child data type or field.

    This data type may not be supported by all Arrow implementations
    because it is an alternative to the ListType.

    Parameters
    ----------
    value_type : DataType or Field

    Returns
    -------
    list_view_type : DataType

    Examples
    --------
    Create an instance of LargeListViewType:

    >>> import pyarrow as pa
    >>> pa.large_list_view(pa.int8())
    LargeListViewType(large_list_view<item: int8>)
    """

@overload
def map_(key_type: _K, item_type: _ValueT) -> MapType[_K, _ValueT, _Ordered]: ...
@overload
def map_(
    key_type: _K, item_type: _ValueT, key_sorted: _Ordered
) -> MapType[_K, _ValueT, _Ordered]: ...
def map_(*args, **kwargs):
    """
    Create MapType instance from key and item data types or fields.

    Parameters
    ----------
    key_type : DataType or Field
    item_type : DataType or Field
    keys_sorted : bool

    Returns
    -------
    map_type : DataType

    Examples
    --------
    Create an instance of MapType:

    >>> import pyarrow as pa
    >>> pa.map_(pa.string(), pa.int32())
    MapType(map<string, int32>)
    >>> pa.map_(pa.string(), pa.int32(), keys_sorted=True)
    MapType(map<string, int32, keys_sorted>)

    Use MapType to create an array:

    >>> data = [[{"key": "a", "value": 1}, {"key": "b", "value": 2}], [{"key": "c", "value": 3}]]
    >>> pa.array(data, type=pa.map_(pa.string(), pa.int32(), keys_sorted=True))
    <pyarrow.lib.MapArray object at ...>
    [
      keys:
      [
        "a",
        "b"
      ]
      values:
      [
        1,
        2
      ],
      keys:
      [
        "c"
      ]
      values:
      [
        3
      ]
    ]
    """

@overload
def dictionary(
    index_type: _IndexT, value_type: _BasicValueT
) -> DictionaryType[_IndexT, _BasicValueT, _Ordered]: ...
@overload
def dictionary(
    index_type: _IndexT, value_type: _BasicValueT, ordered: _Ordered
) -> DictionaryType[_IndexT, _BasicValueT, _Ordered]: ...
def dictionary(*args, **kwargs):
    """
    Dictionary (categorical, or simply encoded) type.

    Parameters
    ----------
    index_type : DataType
    value_type : DataType
    ordered : bool

    Returns
    -------
    type : DictionaryType

    Examples
    --------
    Create an instance of dictionary type:

    >>> import pyarrow as pa
    >>> pa.dictionary(pa.int64(), pa.utf8())
    DictionaryType(dictionary<values=string, indices=int64, ordered=0>)

    Use dictionary type to create an array:

    >>> pa.array(["a", "b", None, "d"], pa.dictionary(pa.int64(), pa.utf8()))
    <pyarrow.lib.DictionaryArray object at ...>
    ...
    -- dictionary:
      [
        "a",
        "b",
        "d"
      ]
    -- indices:
      [
        0,
        1,
        null,
        2
      ]
    """

def struct(
    fields: Iterable[Field[Any] | tuple[str, Field[Any]] | tuple[str, DataType]]
    | Mapping[str, Field[Any]],
) -> StructType:
    """
    Create StructType instance from fields.

    A struct is a nested type parameterized by an ordered sequence of types
    (which can all be distinct), called its fields.

    Parameters
    ----------
    fields : iterable of Fields or tuples, or mapping of strings to DataTypes
        Each field must have a UTF8-encoded name, and these field names are
        part of the type metadata.

    Examples
    --------
    Create an instance of StructType from an iterable of tuples:

    >>> import pyarrow as pa
    >>> fields = [
    ...     ("f1", pa.int32()),
    ...     ("f2", pa.string()),
    ... ]
    >>> struct_type = pa.struct(fields)
    >>> struct_type
    StructType(struct<f1: int32, f2: string>)

    Retrieve a field from a StructType:

    >>> struct_type[0]
    pyarrow.Field<f1: int32>
    >>> struct_type["f1"]
    pyarrow.Field<f1: int32>

    Create an instance of StructType from an iterable of Fields:

    >>> fields = [
    ...     pa.field("f1", pa.int32()),
    ...     pa.field("f2", pa.string(), nullable=False),
    ... ]
    >>> pa.struct(fields)
    StructType(struct<f1: int32, f2: string not null>)

    Returns
    -------
    type : DataType
    """

def sparse_union(
    child_fields: list[Field[Any]], type_codes: list[int] | None = None
) -> SparseUnionType:
    """
    Create SparseUnionType from child fields.

    A sparse union is a nested type where each logical value is taken from
    a single child.  A buffer of 8-bit type ids indicates which child
    a given logical value is to be taken from.

    In a sparse union, each child array should have the same length as the
    union array, regardless of the actual number of union values that
    refer to it.

    Parameters
    ----------
    child_fields : sequence of Field values
        Each field must have a UTF8-encoded name, and these field names are
        part of the type metadata.
    type_codes : list of integers, default None

    Returns
    -------
    type : SparseUnionType
    """

def dense_union(
    child_fields: list[Field[Any]], type_codes: list[int] | None = None
) -> DenseUnionType:
    """
    Create DenseUnionType from child fields.

    A dense union is a nested type where each logical value is taken from
    a single child, at a specific offset.  A buffer of 8-bit type ids
    indicates which child a given logical value is to be taken from,
    and a buffer of 32-bit offsets indicates at which physical position
    in the given child array the logical value is to be taken from.

    Unlike a sparse union, a dense union allows encoding only the child array
    values which are actually referred to by the union array.  This is
    counterbalanced by the additional footprint of the offsets buffer, and
    the additional indirection cost when looking up values.

    Parameters
    ----------
    child_fields : sequence of Field values
        Each field must have a UTF8-encoded name, and these field names are
        part of the type metadata.
    type_codes : list of integers, default None

    Returns
    -------
    type : DenseUnionType
    """

@overload
def union(
    child_fields: list[Field[Any]], mode: Literal["sparse"], type_codes: list[int] | None = None
) -> SparseUnionType: ...
@overload
def union(
    child_fields: list[Field[Any]], mode: Literal["dense"], type_codes: list[int] | None = None
) -> DenseUnionType: ...
def union(*args, **kwargs):
    """
    Create UnionType from child fields.

    A union is a nested type where each logical value is taken from a
    single child.  A buffer of 8-bit type ids indicates which child
    a given logical value is to be taken from.

    Unions come in two flavors: sparse and dense
    (see also `pyarrow.sparse_union` and `pyarrow.dense_union`).

    Parameters
    ----------
    child_fields : sequence of Field values
        Each field must have a UTF8-encoded name, and these field names are
        part of the type metadata.
    mode : str
        Must be 'sparse' or 'dense'
    type_codes : list of integers, default None

    Returns
    -------
    type : UnionType
    """

def run_end_encoded(
    run_end_type: _RunEndType, value_type: _BasicValueT
) -> RunEndEncodedType[_RunEndType, _BasicValueT]:
    """
    Create RunEndEncodedType from run-end and value types.

    Parameters
    ----------
    run_end_type : pyarrow.DataType
        The integer type of the run_ends array. Must be 'int16', 'int32', or 'int64'.
    value_type : pyarrow.DataType
        The type of the values array.

    Returns
    -------
    type : RunEndEncodedType
    """

def json_(storage_type: DataType = ...) -> JsonType:
    """
    Create instance of JSON extension type.

    Parameters
    ----------
    storage_type : DataType, default pyarrow.string()
        The underlying data type. Can be on of the following types:
        string, large_string, string_view.

    Returns
    -------
    type : JsonType

    Examples
    --------
    Create an instance of JSON extension type:

    >>> import pyarrow as pa
    >>> pa.json_(pa.utf8())
    JsonType(extension<arrow.json>)

    Use the JSON type to create an array:

    >>> pa.array(['{"a": 1}', '{"b": 2}'], type=pa.json_(pa.utf8()))
    <pyarrow.lib.JsonArray object at ...>
    [
      "{"a": 1}",
      "{"b": 2}"
    ]
    """

def uuid() -> UuidType:
    """
    Create UuidType instance.

    Returns
    -------
    type : UuidType
    """

def fixed_shape_tensor(
    value_type: _ValueT,
    shape: Sequence[int],
    dim_names: Sequence[str] | None = None,
    permutation: Sequence[int] | None = None,
) -> FixedShapeTensorType[_ValueT]:
    """
    Create instance of fixed shape tensor extension type with shape and optional
    names of tensor dimensions and indices of the desired logical
    ordering of dimensions.

    Parameters
    ----------
    value_type : DataType
        Data type of individual tensor elements.
    shape : tuple or list of integers
        The physical shape of the contained tensors.
    dim_names : tuple or list of strings, default None
        Explicit names to tensor dimensions.
    permutation : tuple or list integers, default None
        Indices of the desired ordering of the original dimensions.
        The indices contain a permutation of the values ``[0, 1, .., N-1]`` where
        N is the number of dimensions. The permutation indicates which dimension
        of the logical layout corresponds to which dimension of the physical tensor.
        For more information on this parameter see
        :ref:`fixed_shape_tensor_extension`.

    Examples
    --------
    Create an instance of fixed shape tensor extension type:

    >>> import pyarrow as pa
    >>> tensor_type = pa.fixed_shape_tensor(pa.int32(), [2, 2])
    >>> tensor_type
    FixedShapeTensorType(extension<arrow.fixed_shape_tensor[value_type=int32, shape=[2,2]]>)

    Inspect the data type:

    >>> tensor_type.value_type
    DataType(int32)
    >>> tensor_type.shape
    [2, 2]

    Create a table with fixed shape tensor extension array:

    >>> arr = [[1, 2, 3, 4], [10, 20, 30, 40], [100, 200, 300, 400]]
    >>> storage = pa.array(arr, pa.list_(pa.int32(), 4))
    >>> tensor = pa.ExtensionArray.from_storage(tensor_type, storage)
    >>> pa.table([tensor], names=["tensor_array"])
    pyarrow.Table
    tensor_array: extension<arrow.fixed_shape_tensor[value_type=int32, shape=[2,2]]>
    ----
    tensor_array: [[[1,2,3,4],[10,20,30,40],[100,200,300,400]]]

    Create an instance of fixed shape tensor extension type with names
    of tensor dimensions:

    >>> tensor_type = pa.fixed_shape_tensor(pa.int8(), (2, 2, 3), dim_names=["C", "H", "W"])
    >>> tensor_type.dim_names
    ['C', 'H', 'W']

    Create an instance of fixed shape tensor extension type with
    permutation:

    >>> tensor_type = pa.fixed_shape_tensor(pa.int8(), (2, 2, 3), permutation=[0, 2, 1])
    >>> tensor_type.permutation
    [0, 2, 1]

    Returns
    -------
    type : FixedShapeTensorType
    """

def bool8() -> Bool8Type:
    """
    Create instance of bool8 extension type.

    Examples
    --------
    Create an instance of bool8 extension type:

    >>> import pyarrow as pa
    >>> type = pa.bool8()
    >>> type
    Bool8Type(extension<arrow.bool8>)

    Inspect the data type:

    >>> type.storage_type
    DataType(int8)

    Create a table with a bool8 array:

    >>> arr = [-1, 0, 1, 2, None]
    >>> storage = pa.array(arr, pa.int8())
    >>> other = pa.ExtensionArray.from_storage(type, storage)
    >>> pa.table([other], names=["unknown_col"])
    pyarrow.Table
    unknown_col: extension<arrow.bool8>
    ----
    unknown_col: [[-1,0,1,2,null]]

    Returns
    -------
    type : Bool8Type
    """

def opaque(storage_type: DataType, type_name: str, vendor_name: str) -> OpaqueType:
    """
    Create instance of opaque extension type.

    Parameters
    ----------
    storage_type : DataType
        The underlying data type.
    type_name : str
        The name of the type in the external system.
    vendor_name : str
        The name of the external system.

    Examples
    --------
    Create an instance of an opaque extension type:

    >>> import pyarrow as pa
    >>> type = pa.opaque(pa.binary(), "other", "jdbc")
    >>> type
    OpaqueType(extension<arrow.opaque[storage_type=binary, type_name=other, vendor_name=jdbc]>)

    Inspect the data type:

    >>> type.storage_type
    DataType(binary)
    >>> type.type_name
    'other'
    >>> type.vendor_name
    'jdbc'

    Create a table with an opaque array:

    >>> arr = [None, b"foobar"]
    >>> storage = pa.array(arr, pa.binary())
    >>> other = pa.ExtensionArray.from_storage(type, storage)
    >>> pa.table([other], names=["unknown_col"])
    pyarrow.Table
    unknown_col: extension<arrow.opaque[storage_type=binary, type_name=other, vendor_name=jdbc]>
    ----
    unknown_col: [[null,666F6F626172]]

    Returns
    -------
    type : OpaqueType
    """

@overload
def type_for_alias(name: Literal["null"]) -> NullType: ...
@overload
def type_for_alias(name: Literal["bool", "boolean"]) -> BoolType: ...
@overload
def type_for_alias(name: Literal["i1", "int8"]) -> Int8Type: ...
@overload
def type_for_alias(name: Literal["i2", "int16"]) -> Int16Type: ...
@overload
def type_for_alias(name: Literal["i4", "int32"]) -> Int32Type: ...
@overload
def type_for_alias(name: Literal["i8", "int64"]) -> Int64Type: ...
@overload
def type_for_alias(name: Literal["u1", "uint8"]) -> UInt8Type: ...
@overload
def type_for_alias(name: Literal["u2", "uint16"]) -> UInt16Type: ...
@overload
def type_for_alias(name: Literal["u4", "uint32"]) -> Uint32Type: ...
@overload
def type_for_alias(name: Literal["u8", "uint64"]) -> UInt64Type: ...
@overload
def type_for_alias(name: Literal["f2", "halffloat", "float16"]) -> Float16Type: ...
@overload
def type_for_alias(name: Literal["f4", "float", "float32"]) -> Float32Type: ...
@overload
def type_for_alias(name: Literal["f8", "double", "float64"]) -> Float64Type: ...
@overload
def type_for_alias(name: Literal["string", "str", "utf8"]) -> StringType: ...
@overload
def type_for_alias(name: Literal["binary"]) -> BinaryType: ...
@overload
def type_for_alias(
    name: Literal["large_string", "large_str", "large_utf8"],
) -> LargeStringType: ...
@overload
def type_for_alias(name: Literal["large_binary"]) -> LargeBinaryType: ...
@overload
def type_for_alias(name: Literal["binary_view"]) -> BinaryViewType: ...
@overload
def type_for_alias(name: Literal["string_view"]) -> StringViewType: ...
@overload
def type_for_alias(name: Literal["date32", "date32[day]"]) -> Date32Type: ...
@overload
def type_for_alias(name: Literal["date64", "date64[ms]"]) -> Date64Type: ...
@overload
def type_for_alias(name: Literal["time32[s]"]) -> Time32Type[Literal["s"]]: ...
@overload
def type_for_alias(name: Literal["time32[ms]"]) -> Time32Type[Literal["ms"]]: ...
@overload
def type_for_alias(name: Literal["time64[us]"]) -> Time64Type[Literal["us"]]: ...
@overload
def type_for_alias(name: Literal["time64[ns]"]) -> Time64Type[Literal["ns"]]: ...
@overload
def type_for_alias(name: Literal["timestamp[s]"]) -> TimestampType[Literal["s"], Any]: ...
@overload
def type_for_alias(name: Literal["timestamp[ms]"]) -> TimestampType[Literal["ms"], Any]: ...
@overload
def type_for_alias(name: Literal["timestamp[us]"]) -> TimestampType[Literal["us"], Any]: ...
@overload
def type_for_alias(name: Literal["timestamp[ns]"]) -> TimestampType[Literal["ns"], Any]: ...
@overload
def type_for_alias(name: Literal["duration[s]"]) -> DurationType[Literal["s"]]: ...
@overload
def type_for_alias(name: Literal["duration[ms]"]) -> DurationType[Literal["ms"]]: ...
@overload
def type_for_alias(name: Literal["duration[us]"]) -> DurationType[Literal["us"]]: ...
@overload
def type_for_alias(name: Literal["duration[ns]"]) -> DurationType[Literal["ns"]]: ...
@overload
def type_for_alias(name: Literal["month_day_nano_interval"]) -> MonthDayNanoIntervalType: ...
def type_for_alias(name):
    """
    Return DataType given a string alias if one exists.

    Parameters
    ----------
    name : str
        The alias of the DataType that should be retrieved.

    Returns
    -------
    type : DataType
    """

@overload
def ensure_type(ty: None, allow_none: Literal[True]) -> None: ...
@overload
def ensure_type(ty: _DataTypeT) -> _DataTypeT: ...
@overload
def ensure_type(ty: Literal["null"]) -> NullType: ...
@overload
def ensure_type(ty: Literal["bool", "boolean"]) -> BoolType: ...
@overload
def ensure_type(ty: Literal["i1", "int8"]) -> Int8Type: ...
@overload
def ensure_type(ty: Literal["i2", "int16"]) -> Int16Type: ...
@overload
def ensure_type(ty: Literal["i4", "int32"]) -> Int32Type: ...
@overload
def ensure_type(ty: Literal["i8", "int64"]) -> Int64Type: ...
@overload
def ensure_type(ty: Literal["u1", "uint8"]) -> UInt8Type: ...
@overload
def ensure_type(ty: Literal["u2", "uint16"]) -> UInt16Type: ...
@overload
def ensure_type(ty: Literal["u4", "uint32"]) -> Uint32Type: ...
@overload
def ensure_type(ty: Literal["u8", "uint64"]) -> UInt64Type: ...
@overload
def ensure_type(ty: Literal["f2", "halffloat", "float16"]) -> Float16Type: ...
@overload
def ensure_type(ty: Literal["f4", "float", "float32"]) -> Float32Type: ...
@overload
def ensure_type(ty: Literal["f8", "double", "float64"]) -> Float64Type: ...
@overload
def ensure_type(ty: Literal["string", "str", "utf8"]) -> StringType: ...
@overload
def ensure_type(ty: Literal["binary"]) -> BinaryType: ...
@overload
def ensure_type(
    ty: Literal["large_string", "large_str", "large_utf8"],
) -> LargeStringType: ...
@overload
def ensure_type(ty: Literal["large_binary"]) -> LargeBinaryType: ...
@overload
def ensure_type(ty: Literal["binary_view"]) -> BinaryViewType: ...
@overload
def ensure_type(ty: Literal["string_view"]) -> StringViewType: ...
@overload
def ensure_type(ty: Literal["date32", "date32[day]"]) -> Date32Type: ...
@overload
def ensure_type(ty: Literal["date64", "date64[ms]"]) -> Date64Type: ...
@overload
def ensure_type(ty: Literal["time32[s]"]) -> Time32Type[Literal["s"]]: ...
@overload
def ensure_type(ty: Literal["time32[ms]"]) -> Time32Type[Literal["ms"]]: ...
@overload
def ensure_type(ty: Literal["time64[us]"]) -> Time64Type[Literal["us"]]: ...
@overload
def ensure_type(ty: Literal["time64[ns]"]) -> Time64Type[Literal["ns"]]: ...
@overload
def ensure_type(ty: Literal["timestamp[s]"]) -> TimestampType[Literal["s"], Any]: ...
@overload
def ensure_type(ty: Literal["timestamp[ms]"]) -> TimestampType[Literal["ms"], Any]: ...
@overload
def ensure_type(ty: Literal["timestamp[us]"]) -> TimestampType[Literal["us"], Any]: ...
@overload
def ensure_type(ty: Literal["timestamp[ns]"]) -> TimestampType[Literal["ns"], Any]: ...
@overload
def ensure_type(ty: Literal["duration[s]"]) -> DurationType[Literal["s"]]: ...
@overload
def ensure_type(ty: Literal["duration[ms]"]) -> DurationType[Literal["ms"]]: ...
@overload
def ensure_type(ty: Literal["duration[us]"]) -> DurationType[Literal["us"]]: ...
@overload
def ensure_type(ty: Literal["duration[ns]"]) -> DurationType[Literal["ns"]]: ...
@overload
def ensure_type(ty: Literal["month_day_nano_interval"]) -> MonthDayNanoIntervalType: ...
def schema(
    fields: Iterable[Field[Any]] | Iterable[tuple[str, DataType]] | Mapping[str, DataType],
    metadata: dict[bytes | str, bytes | str] | None = None,
) -> Schema:
    """
    Construct pyarrow.Schema from collection of fields.

    Parameters
    ----------
    fields : iterable of Fields or tuples, or mapping of strings to DataTypes
        Can also pass an object that implements the Arrow PyCapsule Protocol
        for schemas (has an ``__arrow_c_schema__`` method).
    metadata : dict, default None
        Keys and values must be coercible to bytes.

    Examples
    --------
    Create a Schema from iterable of tuples:

    >>> import pyarrow as pa
    >>> pa.schema(
    ...     [
    ...         ("some_int", pa.int32()),
    ...         ("some_string", pa.string()),
    ...         pa.field("some_required_string", pa.string(), nullable=False),
    ...     ]
    ... )
    some_int: int32
    some_string: string
    some_required_string: string not null

    Create a Schema from iterable of Fields:

    >>> pa.schema([pa.field("some_int", pa.int32()), pa.field("some_string", pa.string())])
    some_int: int32
    some_string: string

    DataTypes can also be passed as strings. The following is equivalent to the
    above example:

    >>> pa.schema([pa.field("some_int", "int32"), pa.field("some_string", "string")])
    some_int: int32
    some_string: string

    Or more concisely:

    >>> pa.schema([("some_int", "int32"), ("some_string", "string")])
    some_int: int32
    some_string: string

    Returns
    -------
    schema : pyarrow.Schema
    """

def from_numpy_dtype(dtype: np.dtype[Any]) -> DataType:
    """
    Convert NumPy dtype to pyarrow.DataType.

    Parameters
    ----------
    dtype : the numpy dtype to convert


    Examples
    --------
    Create a pyarrow DataType from NumPy dtype:

    >>> import pyarrow as pa
    >>> import numpy as np
    >>> pa.from_numpy_dtype(np.dtype("float16"))
    DataType(halffloat)
    >>> pa.from_numpy_dtype("U")
    DataType(string)
    >>> pa.from_numpy_dtype(bool)
    DataType(bool)
    >>> pa.from_numpy_dtype(np.str_)
    DataType(string)
    """

def is_boolean_value(obj: Any) -> bool:
    """
    Check if the object is a boolean.

    Parameters
    ----------
    obj : object
        The object to check
    """

def is_integer_value(obj: Any) -> bool:
    """
    Check if the object is an integer.

    Parameters
    ----------
    obj : object
        The object to check
    """

def is_float_value(obj: Any) -> bool:
    """
    Check if the object is a float.

    Parameters
    ----------
    obj : object
        The object to check
    """

__all__ = [
    "_Weakrefable",
    "_Metadata",
    "DataType",
    "_BasicDataType",
    "NullType",
    "BoolType",
    "UInt8Type",
    "Int8Type",
    "UInt16Type",
    "Int16Type",
    "Uint32Type",
    "Int32Type",
    "UInt64Type",
    "Int64Type",
    "Float16Type",
    "Float32Type",
    "Float64Type",
    "Date32Type",
    "Date64Type",
    "MonthDayNanoIntervalType",
    "StringType",
    "LargeStringType",
    "StringViewType",
    "BinaryType",
    "LargeBinaryType",
    "BinaryViewType",
    "TimestampType",
    "Time32Type",
    "Time64Type",
    "DurationType",
    "FixedSizeBinaryType",
    "Decimal32Type",
    "Decimal64Type",
    "Decimal128Type",
    "Decimal256Type",
    "ListType",
    "LargeListType",
    "ListViewType",
    "LargeListViewType",
    "FixedSizeListType",
    "DictionaryMemo",
    "DictionaryType",
    "MapType",
    "StructType",
    "UnionType",
    "SparseUnionType",
    "DenseUnionType",
    "RunEndEncodedType",
    "BaseExtensionType",
    "ExtensionType",
    "FixedShapeTensorType",
    "Bool8Type",
    "UuidType",
    "JsonType",
    "OpaqueType",
    "PyExtensionType",
    "UnknownExtensionType",
    "register_extension_type",
    "unregister_extension_type",
    "KeyValueMetadata",
    "ensure_metadata",
    "Field",
    "Schema",
    "unify_schemas",
    "field",
    "null",
    "bool_",
    "uint8",
    "int8",
    "uint16",
    "int16",
    "uint32",
    "int32",
    "int64",
    "uint64",
    "tzinfo_to_string",
    "string_to_tzinfo",
    "timestamp",
    "time32",
    "time64",
    "duration",
    "month_day_nano_interval",
    "date32",
    "date64",
    "float16",
    "float32",
    "float64",
    "decimal32",
    "decimal64",
    "decimal128",
    "decimal256",
    "string",
    "utf8",
    "binary",
    "large_binary",
    "large_string",
    "large_utf8",
    "binary_view",
    "string_view",
    "list_",
    "large_list",
    "list_view",
    "large_list_view",
    "map_",
    "dictionary",
    "struct",
    "sparse_union",
    "dense_union",
    "union",
    "run_end_encoded",
    "json_",
    "uuid",
    "fixed_shape_tensor",
    "bool8",
    "opaque",
    "type_for_alias",
    "ensure_type",
    "schema",
    "from_numpy_dtype",
    "is_boolean_value",
    "is_integer_value",
    "is_float_value",
]
