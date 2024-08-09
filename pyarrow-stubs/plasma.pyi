from collections.abc import Generator
from subprocess import Popen
from types import ModuleType

from pyarrow._plasma import ObjectID as ObjectID
from pyarrow._plasma import ObjectNotAvailable as ObjectNotAvailable
from pyarrow._plasma import PlasmaBuffer as PlasmaBuffer
from pyarrow._plasma import PlasmaClient as PlasmaClient
from pyarrow._plasma import PlasmaObjectExists as PlasmaObjectExists
from pyarrow._plasma import PlasmaObjectNotFound as PlasmaObjectNotFound
from pyarrow._plasma import PlasmaStoreFull as PlasmaStoreFull
from pyarrow._plasma import connect as connect

TF_PLASMA_OP_PATH: str
tf_plasma_op: ModuleType | None

def load_plasma_tensorflow_op() -> None: ...
def build_plasma_tensorflow_op() -> None: ...
def start_plasma_store(
    plasma_store_memory: int,
    use_valgrind: bool = ...,
    use_profiler: bool = ...,
    plasma_directory: str | None = ...,
    use_hugepages: bool = ...,
    external_store: str | None = ...,
) -> Generator[tuple[str, Popen[str]], None, None]: ...
