from typing import Any, Protocol, TypeAlias

from pyarrow.lib import Array, Buffer, DataType, DictionaryArray, RecordBatch, Table

from .column import (
    ColumnBuffers,
    ColumnNullType,
    Dtype,
    DtypeKind,
)

class DataFrameObject(Protocol):
    def __dataframe__(self, nan_as_null: bool = False, allow_copy: bool = True) -> Any: ...

ColumnObject: TypeAlias = Any

def from_dataframe(df: DataFrameObject, allow_copy=True) -> Table:
    """
    Build a ``pa.Table`` from any DataFrame supporting the interchange protocol.

    Parameters
    ----------
    df : DataFrameObject
        Object supporting the interchange protocol, i.e. `__dataframe__`
        method.
    allow_copy : bool, default: True
        Whether to allow copying the memory to perform the conversion
        (if false then zero-copy approach is requested).

    Returns
    -------
    pa.Table

    Examples
    --------
    >>> import pyarrow
    >>> from pyarrow.interchange import from_dataframe

    Convert a pandas dataframe to a pyarrow table:

    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     {
    ...         "n_attendees": [100, 10, 1],
    ...         "country": ["Italy", "Spain", "Slovenia"],
    ...     }
    ... )
    >>> df
       n_attendees   country
    0          100     Italy
    1           10     Spain
    2            1  Slovenia
    >>> from_dataframe(df)
    pyarrow.Table
    n_attendees: int64
    country: large_string
    ----
    n_attendees: [[100,10,1]]
    country: [["Italy","Spain","Slovenia"]]
    """

def protocol_df_chunk_to_pyarrow(df: DataFrameObject, allow_copy: bool = True) -> RecordBatch:
    """
    Convert interchange protocol chunk to ``pa.RecordBatch``.

    Parameters
    ----------
    df : DataFrameObject
        Object supporting the interchange protocol, i.e. `__dataframe__`
        method.
    allow_copy : bool, default: True
        Whether to allow copying the memory to perform the conversion
        (if false then zero-copy approach is requested).

    Returns
    -------
    pa.RecordBatch
    """

def column_to_array(col: ColumnObject, allow_copy: bool = True) -> Array:
    """
    Convert a column holding one of the primitive dtypes to a PyArrow array.
    A primitive type is one of: int, uint, float, bool (1 bit).

    Parameters
    ----------
    col : ColumnObject
    allow_copy : bool, default: True
        Whether to allow copying the memory to perform the conversion
        (if false then zero-copy approach is requested).

    Returns
    -------
    pa.Array
    """

def bool_column_to_array(col: ColumnObject, allow_copy: bool = True) -> Array:
    """
    Convert a column holding boolean dtype to a PyArrow array.

    Parameters
    ----------
    col : ColumnObject
    allow_copy : bool, default: True
        Whether to allow copying the memory to perform the conversion
        (if false then zero-copy approach is requested).

    Returns
    -------
    pa.Array
    """

def categorical_column_to_dictionary(
    col: ColumnObject, allow_copy: bool = True
) -> DictionaryArray:
    """
    Convert a column holding categorical data to a pa.DictionaryArray.

    Parameters
    ----------
    col : ColumnObject
    allow_copy : bool, default: True
        Whether to allow copying the memory to perform the conversion
        (if false then zero-copy approach is requested).

    Returns
    -------
    pa.DictionaryArray
    """

def parse_datetime_format_str(format_str: str) -> tuple[str, str]:
    """Parse datetime `format_str` to interpret the `data`."""

def map_date_type(data_type: tuple[DtypeKind, int, str, str]) -> DataType:
    """Map column date type to pyarrow date type."""

def buffers_to_array(
    buffers: ColumnBuffers,
    data_type: tuple[DtypeKind, int, str, str],
    length: int,
    describe_null: ColumnNullType,
    offset: int = 0,
    allow_copy: bool = True,
) -> Array:
    """
    Build a PyArrow array from the passed buffer.

    Parameters
    ----------
    buffer : ColumnBuffers
        Dictionary containing tuples of underlying buffers and
        their associated dtype.
    data_type : Tuple[DtypeKind, int, str, str],
        Dtype description of the column as a tuple ``(kind, bit-width, format string,
        endianness)``.
    length : int
        The number of values in the array.
    describe_null: ColumnNullType
        Null representation the column dtype uses,
        as a tuple ``(kind, value)``
    offset : int, default: 0
        Number of elements to offset from the start of the buffer.
    allow_copy : bool, default: True
        Whether to allow copying the memory to perform the conversion
        (if false then zero-copy approach is requested).

    Returns
    -------
    pa.Array

    Notes
    -----
    The returned array doesn't own the memory. The caller of this function
    is responsible for keeping the memory owner object alive as long as
    the returned PyArrow array is being used.
    """

def validity_buffer_from_mask(
    validity_buff: Buffer,
    validity_dtype: Dtype,
    describe_null: ColumnNullType,
    length: int,
    offset: int = 0,
    allow_copy: bool = True,
) -> Buffer:
    """
    Build a PyArrow buffer from the passed mask buffer.

    Parameters
    ----------
    validity_buff : BufferObject
        Tuple of underlying validity buffer and associated dtype.
    validity_dtype : Dtype
        Dtype description as a tuple ``(kind, bit-width, format string,
        endianness)``.
    describe_null : ColumnNullType
        Null representation the column dtype uses,
        as a tuple ``(kind, value)``
    length : int
        The number of values in the array.
    offset : int, default: 0
        Number of elements to offset from the start of the buffer.
    allow_copy : bool, default: True
        Whether to allow copying the memory to perform the conversion
        (if false then zero-copy approach is requested).

    Returns
    -------
    pa.Buffer
    """

def validity_buffer_nan_sentinel(
    data_pa_buffer: Buffer,
    data_type: Dtype,
    describe_null: ColumnNullType,
    length: int,
    offset: int = 0,
    allow_copy: bool = True,
) -> Buffer:
    """
    Build a PyArrow buffer from NaN or sentinel values.

    Parameters
    ----------
    data_pa_buffer : pa.Buffer
        PyArrow buffer for the column data.
    data_type : Dtype
        Dtype description as a tuple ``(kind, bit-width, format string,
        endianness)``.
    describe_null : ColumnNullType
        Null representation the column dtype uses,
        as a tuple ``(kind, value)``
    length : int
        The number of values in the array.
    offset : int, default: 0
        Number of elements to offset from the start of the buffer.
    allow_copy : bool, default: True
        Whether to allow copying the memory to perform the conversion
        (if false then zero-copy approach is requested).

    Returns
    -------
    pa.Buffer
    """
