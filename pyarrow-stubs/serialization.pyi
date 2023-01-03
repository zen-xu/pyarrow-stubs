from pyarrow.lib import (
    SerializationContext as SerializationContext,
    builtin_pickle as builtin_pickle,
    py_buffer as py_buffer,
)

try:
    import cloudpickle  # type: ignore
except ImportError:
    cloudpickle = builtin_pickle

def register_torch_serialization_handlers(
    serialization_context: SerializationContext,
): ...
def register_default_serialization_handlers(
    serialization_context: SerializationContext,
) -> None: ...
def default_serialization_context() -> SerializationContext: ...
