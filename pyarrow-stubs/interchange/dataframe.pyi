import sys

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self
from typing import Any, Iterable, Sequence

from pyarrow.interchange.column import _PyArrowColumn
from pyarrow.lib import RecordBatch, Table

class _PyArrowDataFrame:
    def __init__(
        self, df: Table | RecordBatch, nan_as_null: bool = False, allow_copy: bool = True
    ) -> None: ...
    def __dataframe__(
        self, nan_as_null: bool = False, allow_copy: bool = True
    ) -> _PyArrowDataFrame: ...
    @property
    def metadata(self) -> dict[str, Any]: ...
    def num_columns(self) -> int: ...
    def num_rows(self) -> int: ...
    def num_chunks(self) -> int: ...
    def column_names(self) -> Iterable[str]: ...
    def get_column(self, i: int) -> _PyArrowColumn: ...
    def get_column_by_name(self, name: str) -> _PyArrowColumn: ...
    def get_columns(self) -> Iterable[_PyArrowColumn]: ...
    def select_columns(self, indices: Sequence[int]) -> Self: ...
    def select_columns_by_name(self, names: Sequence[str]) -> Self: ...
    def get_chunks(self, n_chunks: int | None = None) -> Iterable[Self]: ...
