from typing import Literal
from typing import Self
from typing import TypeAlias

from . import lib

# TODO: str | pyarrow.compute.Expression
_StrOrExpr: TypeAlias = str

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
    # TODO: filter_expression: pyarrow.compute.Expression
    def __init__(self, filter_expression) -> None: ...

class ProjectNodeOptions(ExecNodeOptions):
    # TODO: expressions: list[pyarrow.compute.Expression]
    def __init__(self, expressions: list, names: list[str] | None = None) -> None: ...

class AggregateNodeOptions(ExecNodeOptions):
    # TODO: object: pyarrow.compute.FunctionOptions
    #         keys: list[str | pyarrow.compute.Expression]
    def __init__(
        self,
        aggregates: list[tuple[list[str], str, object, str]],
        keys: list[_StrOrExpr] | None = None,
    ) -> None: ...

class OrderByNodeOptions(ExecNodeOptions):
    def __init__(
        self,
        sort_keys: tuple[tuple[str, Literal["ascending", "descending"]], ...] = (),
        *,
        null_placement: str = "at_end",
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
