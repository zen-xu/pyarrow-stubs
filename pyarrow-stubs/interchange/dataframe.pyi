import sys

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self
from typing import Any, Iterable, Sequence

from pyarrow.interchange.column import _PyArrowColumn
from pyarrow.lib import RecordBatch, Table

class _PyArrowDataFrame:
    """
    A data frame class, with only the methods required by the interchange
    protocol defined.

    A "data frame" represents an ordered collection of named columns.
    A column's "name" must be a unique string.
    Columns may be accessed by name or by position.

    This could be a public data frame class, or an object with the methods and
    attributes defined on this DataFrame class could be returned from the
    ``__dataframe__`` method of a public data frame class in a library adhering
    to the dataframe interchange protocol specification.
    """

    def __init__(
        self, df: Table | RecordBatch, nan_as_null: bool = False, allow_copy: bool = True
    ) -> None: ...
    def __dataframe__(
        self, nan_as_null: bool = False, allow_copy: bool = True
    ) -> _PyArrowDataFrame:
        """
        Construct a new exchange object, potentially changing the parameters.
        ``nan_as_null`` is a keyword intended for the consumer to tell the
        producer to overwrite null values in the data with ``NaN``.
        It is intended for cases where the consumer does not support the bit
        mask or byte mask that is the producer's native representation.
        ``allow_copy`` is a keyword that defines whether or not the library is
        allowed to make a copy of the data. For example, copying data would be
        necessary if a library supports strided buffers, given that this
        protocol specifies contiguous buffers.
        """
    @property
    def metadata(self) -> dict[str, Any]:
        """
        The metadata for the data frame, as a dictionary with string keys. The
        contents of `metadata` may be anything, they are meant for a library
        to store information that it needs to, e.g., roundtrip losslessly or
        for two implementations to share data that is not (yet) part of the
        interchange protocol specification. For avoiding collisions with other
        entries, please add name the keys with the name of the library
        followed by a period and the desired name, e.g, ``pandas.indexcol``.
        """
    def num_columns(self) -> int:
        """
        Return the number of columns in the DataFrame.
        """
    def num_rows(self) -> int:
        """
        Return the number of rows in the DataFrame, if available.
        """
    def num_chunks(self) -> int:
        """
        Return the number of chunks the DataFrame consists of.
        """
    def column_names(self) -> Iterable[str]:
        """
        Return an iterator yielding the column names.
        """
    def get_column(self, i: int) -> _PyArrowColumn:
        """
        Return the column at the indicated position.
        """
    def get_column_by_name(self, name: str) -> _PyArrowColumn:
        """
        Return the column whose name is the indicated name.
        """
    def get_columns(self) -> Iterable[_PyArrowColumn]:
        """
        Return an iterator yielding the columns.
        """
    def select_columns(self, indices: Sequence[int]) -> Self:
        """
        Create a new DataFrame by selecting a subset of columns by index.
        """
    def select_columns_by_name(self, names: Sequence[str]) -> Self:
        """
        Create a new DataFrame by selecting a subset of columns by name.
        """
    def get_chunks(self, n_chunks: int | None = None) -> Iterable[Self]:
        """
        Return an iterator yielding the chunks.

        By default (None), yields the chunks that the data is stored as by the
        producer. If given, ``n_chunks`` must be a multiple of
        ``self.num_chunks()``, meaning the producer must subdivide each chunk
        before yielding it.

        Note that the producer must ensure that all columns are chunked the
        same way.
        """
