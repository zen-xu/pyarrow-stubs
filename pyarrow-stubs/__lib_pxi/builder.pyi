from typing import Iterable

from pyarrow.lib import MemoryPool, _Weakrefable

from .array import StringArray, StringViewArray

class StringBuilder(_Weakrefable):
    """
    Builder class for UTF8 strings.

    This class exposes facilities for incrementally adding string values and
    building the null bitmap for a pyarrow.Array (type='string').
    """
    def __init__(self, memory_pool: MemoryPool | None = None) -> None: ...
    def append(self, value: str | bytes | None):
        """
        Append a single value to the builder.

        The value can either be a string/bytes object or a null value
        (np.nan or None).

        Parameters
        ----------
        value : string/bytes or np.nan/None
            The value to append to the string array builder.
        """
    def append_values(self, values: Iterable[str | bytes | None]):
        """
        Append all the values from an iterable.

        Parameters
        ----------
        values : iterable of string/bytes or np.nan/None values
            The values to append to the string array builder.
        """
    def finish(self) -> StringArray:
        """
        Return result of builder as an Array object; also resets the builder.

        Returns
        -------
        array : pyarrow.Array
        """
    @property
    def null_count(self) -> int: ...
    def __len__(self) -> int: ...

class StringViewBuilder(_Weakrefable):
    """
    Builder class for UTF8 string views.

    This class exposes facilities for incrementally adding string values and
    building the null bitmap for a pyarrow.Array (type='string_view').
    """
    def __init__(self, memory_pool: MemoryPool | None = None) -> None: ...
    def append(self, value: str | bytes | None):
        """
        Append a single value to the builder.

        The value can either be a string/bytes object or a null value
        (np.nan or None).

        Parameters
        ----------
        value : string/bytes or np.nan/None
            The value to append to the string array builder.
        """
    def append_values(self, values: Iterable[str | bytes | None]):
        """
        Append all the values from an iterable.

        Parameters
        ----------
        values : iterable of string/bytes or np.nan/None values
            The values to append to the string array builder.
        """
    def finish(self) -> StringViewArray:
        """
        Return result of builder as an Array object; also resets the builder.

        Returns
        -------
        array : pyarrow.Array
        """
    @property
    def null_count(self) -> int: ...
    def __len__(self) -> int: ...

__all__ = ["StringBuilder", "StringViewBuilder"]
