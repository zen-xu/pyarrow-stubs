from typing import (
    Any,
    ClassVar,
)

import pyarrow._dataset

class InMemoryDataset(pyarrow._dataset.Dataset):
    def __init__(self, *args, **kwargs) -> None: ...
    def __reduce__(self) -> Any: ...
    def __setstate__(self, state) -> Any: ...

def _filter_table(table, expression, output_type=...) -> Any: ...
def _perform_join(
    join_type,
    left_operand,
    left_keys,
    right_operand,
    right_keys,
    left_suffix=...,
    right_suffix=...,
    use_threads=...,
    coalesce_keys=...,
    output_type=...,
) -> Any: ...
def tobytes(o) -> Any: ...
