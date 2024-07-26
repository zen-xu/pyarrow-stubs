from typing import (
    Any,
)

import pyarrow._dataset

class OrcFileFormat(pyarrow._dataset.FileFormat):
    default_extname: Any
    def __init__(self, *args, **kwargs) -> None: ...
    def equals(self, OrcFileFormatother) -> Any: ...
    def __reduce__(self) -> Any: ...
