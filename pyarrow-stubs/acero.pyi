import sys

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self
if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias
from typing import Literal

from . import lib
from .compute import Expression, FunctionOptions

_StrOrExpr: TypeAlias = str | Expression

class Declaration(lib._Weakrefable):
    def __init__(
        self,
        factory_name: str,
        options: ExecNodeOptions,
        inputs: list[Declaration] | None = None,
    ) -> None: ...
    @classmethod
    def from_sequence(cls, decls: list[Declaration]) -> Self: ...
    def to_reader(self, use_threads: bool = True) -> lib.RecordBatchReader: ...
    def to_table(self, use_threads: bool = True) -> lib.Table: ...

class ExecNodeOptions(lib._Weakrefable): ...

class TableSourceNodeOptions(ExecNodeOptions):
    def __init__(self, table: lib.Table) -> None: ...

class FilterNodeOptions(ExecNodeOptions):
    def __init__(self, filter_expression: Expression) -> None: ...

class ProjectNodeOptions(ExecNodeOptions):
    def __init__(self, expressions: list[Expression], names: list[str] | None = None) -> None: ...

class AggregateNodeOptions(ExecNodeOptions):
    def __init__(
        self,
        aggregates: list[tuple[list[str], str, FunctionOptions, str]],
        keys: list[_StrOrExpr] | None = None,
    ) -> None: ...

class OrderByNodeOptions(ExecNodeOptions):
    def __init__(
        self,
        sort_keys: tuple[tuple[str, Literal["ascending", "descending"]], ...] = (),
        *,
        null_placement: Literal["at_start", "at_end"] = "at_end",
    ) -> None: ...

class HashJoinNodeOptions(ExecNodeOptions):
    def __init__(
        self,
        join_type: Literal[
            "left semi",
            "right semi",
            "left anti",
            "right anti",
            "inner",
            "left outer",
            "right outer",
            "full outer",
        ],
        left_keys: _StrOrExpr | list[_StrOrExpr],
        right_keys: _StrOrExpr | list[_StrOrExpr],
        left_output: list[_StrOrExpr] | None = None,
        right_output: list[_StrOrExpr] | None = None,
        output_suffix_for_left: str = "",
        output_suffix_for_right: str = "",
    ) -> None: ...

class AsofJoinNodeOptions(ExecNodeOptions):
    def __init__(
        self,
        left_on: _StrOrExpr,
        left_by: _StrOrExpr | list[_StrOrExpr],
        right_on: _StrOrExpr,
        right_by: _StrOrExpr | list[_StrOrExpr],
        tolerance: int,
    ) -> None: ...
