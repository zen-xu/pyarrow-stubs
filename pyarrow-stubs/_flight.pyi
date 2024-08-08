import enum
import importlib._bootstrap  # type: ignore
import re
from typing import (
    Any,
    ClassVar,
)

import pyarrow.lib
from pyarrow.lib import Schema

_FLIGHT_SERVER_ERROR_REGEX: re.Pattern
_get_legacy_format_default: function

class Action(pyarrow.lib._Weakrefable):
    __hash__: ClassVar[None] = ...  # type: ignore
    body: Any
    type: Any
    def __init__(self, *args, **kwargs) -> None: ...
    @classmethod
    def deserialize(cls, typecls, serialized) -> Any: ...
    def serialize(self) -> Any: ...
    def __eq__(self, other) -> Any: ...
    def __ge__(self, other) -> Any: ...
    def __gt__(self, other) -> Any: ...
    def __le__(self, other) -> Any: ...
    def __lt__(self, other) -> Any: ...
    def __ne__(self, other) -> Any: ...
    def __reduce__(self) -> Any: ...
    def __setstate__(self, state) -> Any: ...

class ActionType(importlib._bootstrap._ActionType):
    def make_action(self, buf) -> Any: ...

class ArrowCancelled(pyarrow.lib.ArrowException):
    def __init__(self, message, signum=...) -> None: ...

class ArrowException(Exception): ...
class ArrowInvalid(ValueError, pyarrow.lib.ArrowException): ...

class BasicAuth(pyarrow.lib._Weakrefable):
    __hash__: ClassVar[None] = ...  # type: ignore
    password: Any
    username: Any
    def __init__(self, *args, **kwargs) -> None: ...
    def deserialize(self, serialized) -> Any: ...
    def serialize(self) -> Any: ...
    def __eq__(self, other) -> Any: ...
    def __ge__(self, other) -> Any: ...
    def __gt__(self, other) -> Any: ...
    def __le__(self, other) -> Any: ...
    def __lt__(self, other) -> Any: ...
    def __ne__(self, other) -> Any: ...
    def __reduce__(self) -> Any: ...
    def __setstate__(self, state) -> Any: ...

class CallInfo(importlib._bootstrap._CallInfo): ...
class CertKeyPair(importlib._bootstrap._CertKeyPair): ...

class ClientAuthHandler(pyarrow.lib._Weakrefable):
    def __init__(self, *args, **kwargs) -> None: ...
    def authenticate(self, outgoing, incoming) -> Any: ...
    def get_token(self) -> Any: ...
    def __reduce__(self) -> Any: ...
    def __setstate__(self, state) -> Any: ...

class ClientAuthReader(pyarrow.lib._Weakrefable):
    def __init__(self, *args, **kwargs) -> None: ...
    def read(self) -> Any: ...
    def __reduce__(self) -> Any: ...
    def __setstate__(self, state) -> Any: ...

class ClientAuthSender(pyarrow.lib._Weakrefable):
    def __init__(self, *args, **kwargs) -> None: ...
    def write(self, message) -> Any: ...
    def __reduce__(self) -> Any: ...
    def __setstate__(self, state) -> Any: ...

class ClientMiddleware(pyarrow.lib._Weakrefable):
    def __init__(self, *args, **kwargs) -> None: ...
    def call_completed(self, exception) -> Any: ...
    def received_headers(self, headers) -> Any: ...
    def sending_headers(self) -> Any: ...
    def __reduce__(self) -> Any: ...
    def __setstate__(self, state) -> Any: ...

class ClientMiddlewareFactory(pyarrow.lib._Weakrefable):
    def start_call(self, info) -> Any: ...
    def __reduce__(self) -> Any: ...
    def __setstate__(self, state) -> Any: ...

class DescriptorType(enum.Enum):
    CMD: ClassVar[DescriptorType] = ...
    PATH: ClassVar[DescriptorType] = ...
    UNKNOWN: ClassVar[DescriptorType] = ...

class FlightCallOptions(pyarrow.lib._Weakrefable):
    def __init__(self, *args, **kwargs) -> None: ...
    def __reduce__(self) -> Any: ...
    def __setstate__(self, state) -> Any: ...

class FlightCancelledError(FlightError, pyarrow.lib.ArrowCancelled):
    def __init__(self, *args, **kwargs) -> None: ...
    def __reduce_cython__(self) -> Any: ...
    def __setstate_cython__(self, __pyx_state) -> Any: ...

class FlightClient(pyarrow.lib._Weakrefable):
    def __init__(self, *args, **kwargs) -> None: ...
    def authenticate(
        self, auth_handler, FlightCallOptionsoptions: FlightCallOptions = ...
    ) -> Any: ...
    def authenticate_basic_token(
        self, username, password, FlightCallOptionsoptions: FlightCallOptions = ...
    ) -> Any: ...
    def close(self) -> Any: ...
    @classmethod
    def connect(
        cls,
        typecls,
        location,
        tls_root_certs=...,
        cert_chain=...,
        private_key=...,
        override_hostname=...,
        disable_server_verification=...,
    ) -> Any: ...
    def do_action(
        self, action, FlightCallOptionsoptions: FlightCallOptions = ...
    ) -> Any: ...
    def do_exchange(
        self,
        FlightDescriptordescriptor: FlightDescriptor,
        FlightCallOptionsoptions: FlightCallOptions = ...,
    ) -> Any: ...
    def do_get(
        self, Ticketticket: Ticket, FlightCallOptionsoptions: FlightCallOptions = ...
    ) -> Any: ...
    def do_put(
        self,
        FlightDescriptordescriptor: FlightDescriptor,
        Schemaschema: Schema,
        FlightCallOptionsoptions: FlightCallOptions = ...,
    ) -> Any: ...
    def get_flight_info(
        self,
        FlightDescriptordescriptor: FlightDescriptor,
        FlightCallOptionsoptions: FlightCallOptions = ...,
    ) -> Any: ...
    def get_schema(
        self,
        FlightDescriptordescriptor: FlightDescriptor,
        FlightCallOptionsoptions: FlightCallOptions = ...,
    ) -> Any: ...
    def list_actions(
        self, FlightCallOptionsoptions: FlightCallOptions = ...
    ) -> Any: ...
    def list_flights(
        self,
        bytescriteria: bytes = ...,
        FlightCallOptionsoptions: FlightCallOptions = ...,
    ) -> Any: ...
    def wait_for_available(self, timeout=...) -> Any: ...
    def __del__(self) -> Any: ...
    def __enter__(self) -> Any: ...
    def __exit__(self, exc_type, exc_value, traceback) -> Any: ...
    def __reduce__(self) -> Any: ...
    def __setstate__(self, state) -> Any: ...

class FlightDataStream(pyarrow.lib._Weakrefable):
    def __init__(self, *args, **kwargs) -> None: ...
    def __reduce__(self) -> Any: ...
    def __setstate__(self, state) -> Any: ...

class FlightDescriptor(pyarrow.lib._Weakrefable):
    __hash__: ClassVar[None] = ...  # type: ignore
    command: bytes | None
    descriptor_type: DescriptorType
    path: list[bytes] | None
    def __init__(self, *args, **kwargs) -> None: ...
    @classmethod
    def deserialize(cls, serialized: str | bytes) -> FlightDescriptor: ...
    @staticmethod
    def for_command(command: str | bytes) -> FlightDescriptor: ...
    @staticmethod
    def for_path(*path: str | bytes) -> FlightDescriptor: ...
    def serialize(self) -> bytes: ...
    def __eq__(self, other) -> Any: ...
    def __ge__(self, other) -> Any: ...
    def __gt__(self, other) -> Any: ...
    def __le__(self, other) -> Any: ...
    def __lt__(self, other) -> Any: ...
    def __ne__(self, other) -> Any: ...
    def __reduce__(self) -> Any: ...
    def __setstate__(self, state) -> Any: ...

class FlightEndpoint(pyarrow.lib._Weakrefable):
    __hash__: ClassVar[None] = ...  # type: ignore
    locations: Any
    ticket: Any
    def __init__(self, *args, **kwargs) -> None: ...
    @classmethod
    def deserialize(cls, typecls, serialized) -> Any: ...
    def serialize(self) -> Any: ...
    def __eq__(self, other) -> Any: ...
    def __ge__(self, other) -> Any: ...
    def __gt__(self, other) -> Any: ...
    def __le__(self, other) -> Any: ...
    def __lt__(self, other) -> Any: ...
    def __ne__(self, other) -> Any: ...
    def __reduce__(self) -> Any: ...
    def __setstate__(self, state) -> Any: ...

class FlightError(Exception):
    def __init__(self, *args, **kwargs) -> None: ...
    def __reduce_cython__(self) -> Any: ...
    def __setstate_cython__(self, __pyx_state) -> Any: ...

class FlightInfo(pyarrow.lib._Weakrefable):
    descriptor: Any
    endpoints: Any
    schema: Any
    total_bytes: Any
    total_records: Any
    def __init__(self, *args, **kwargs) -> None: ...
    @classmethod
    def deserialize(cls, typecls, serialized) -> Any: ...
    def serialize(self) -> Any: ...
    def __reduce__(self) -> Any: ...
    def __setstate__(self, state) -> Any: ...

class FlightInternalError(FlightError, pyarrow.lib.ArrowException):
    @classmethod
    def __init__(self, *args, **kwargs) -> None: ...
    def __reduce_cython__(self) -> Any: ...
    def __setstate_cython__(self, __pyx_state) -> Any: ...

class FlightMetadataReader(pyarrow.lib._Weakrefable):
    def __init__(self, *args, **kwargs) -> None: ...
    def read(self) -> Any: ...
    def __reduce__(self) -> Any: ...
    def __setstate__(self, state) -> Any: ...

class FlightMetadataWriter(pyarrow.lib._Weakrefable):
    def __init__(self, *args, **kwargs) -> None: ...
    def write(self, message) -> Any: ...
    def __reduce__(self) -> Any: ...
    def __setstate__(self, state) -> Any: ...

class FlightMethod(enum.Enum):
    DO_ACTION: ClassVar[FlightMethod] = ...
    DO_EXCHANGE: ClassVar[FlightMethod] = ...
    DO_GET: ClassVar[FlightMethod] = ...
    DO_PUT: ClassVar[FlightMethod] = ...
    GET_FLIGHT_INFO: ClassVar[FlightMethod] = ...
    GET_SCHEMA: ClassVar[FlightMethod] = ...
    HANDSHAKE: ClassVar[FlightMethod] = ...
    INVALID: ClassVar[FlightMethod] = ...
    LIST_ACTIONS: ClassVar[FlightMethod] = ...
    LIST_FLIGHTS: ClassVar[FlightMethod] = ...

class FlightServerBase(pyarrow.lib._Weakrefable):
    port: Any
    def __init__(self, *args, **kwargs) -> None: ...
    def do_action(self, context, action) -> Any: ...
    def do_exchange(self, context, descriptor, reader, writer) -> Any: ...
    def do_get(self, context, ticket) -> Any: ...
    def do_put(
        self,
        context,
        descriptor,
        MetadataRecordBatchReaderreader: MetadataRecordBatchReader,
        FlightMetadataWriterwriter: FlightMetadataWriter,
    ) -> Any: ...
    def get_flight_info(self, context, descriptor) -> Any: ...
    def get_schema(self, context, descriptor) -> Any: ...
    def list_actions(self, context) -> Any: ...
    def list_flights(self, context, criteria) -> Any: ...
    def run(self) -> Any: ...
    def serve(self) -> Any: ...
    def shutdown(self) -> Any: ...
    def wait(self) -> Any: ...
    def __enter__(self) -> Any: ...
    def __exit__(self, exc_type, exc_value, traceback) -> Any: ...
    def __reduce__(self) -> Any: ...
    def __setstate__(self, state) -> Any: ...

class FlightServerError(FlightError, pyarrow.lib.ArrowException):
    def __init__(self, *args, **kwargs) -> None: ...
    def __reduce_cython__(self) -> Any: ...
    def __setstate_cython__(self, __pyx_state) -> Any: ...

class FlightStreamChunk(pyarrow.lib._Weakrefable):
    app_metadata: Any
    data: Any

    def __init__(self, *args, **kwargs) -> None: ...
    def __iter__(self) -> Any: ...
    def __reduce__(self) -> Any: ...
    def __setstate__(self, state) -> Any: ...

class FlightStreamReader(MetadataRecordBatchReader):
    def __init__(self, *args, **kwargs) -> None: ...
    def cancel(self) -> Any: ...
    def read_all(self) -> Any: ...
    def __reduce__(self) -> Any: ...
    def __setstate__(self, state) -> Any: ...

class FlightStreamWriter(MetadataRecordBatchWriter):
    def __init__(self, *args, **kwargs) -> None: ...
    def done_writing(self) -> Any: ...
    def __reduce__(self) -> Any: ...
    def __setstate__(self, state) -> Any: ...

class FlightTimedOutError(FlightError, pyarrow.lib.ArrowException):
    def __init__(self, *args, **kwargs) -> None: ...
    def __reduce_cython__(self) -> Any: ...
    def __setstate_cython__(self, __pyx_state) -> Any: ...

class FlightUnauthenticatedError(FlightError, pyarrow.lib.ArrowException):
    def __init__(self, *args, **kwargs) -> None: ...
    def __reduce_cython__(self) -> Any: ...
    def __setstate_cython__(self, __pyx_state) -> Any: ...

class FlightUnauthorizedError(FlightError, pyarrow.lib.ArrowException):
    def __init__(self, *args, **kwargs) -> None: ...
    def __reduce_cython__(self) -> Any: ...
    def __setstate_cython__(self, __pyx_state) -> Any: ...

class FlightUnavailableError(FlightError, pyarrow.lib.ArrowException):
    def __init__(self, *args, **kwargs) -> None: ...
    def __reduce_cython__(self) -> Any: ...
    def __setstate_cython__(self, __pyx_state) -> Any: ...

class FlightWriteSizeExceededError(pyarrow.lib.ArrowInvalid):
    def __init__(self, message, limit, actual) -> None: ...

class GeneratorStream(FlightDataStream):
    def __init__(self, *args, **kwargs) -> None: ...
    def __reduce__(self) -> Any: ...
    def __setstate__(self, state) -> Any: ...

class Location(pyarrow.lib._Weakrefable):
    __hash__: ClassVar[None] = ...  # type: ignore
    uri: Any
    def __init__(self, *args, **kwargs) -> None: ...
    def equals(self, Locationother) -> Any: ...
    def for_grpc_tcp(self, host, port) -> Any: ...
    def for_grpc_tls(self, host, port) -> Any: ...
    def for_grpc_unix(self, path) -> Any: ...
    def __eq__(self, other) -> Any: ...
    def __ge__(self, other) -> Any: ...
    def __gt__(self, other) -> Any: ...
    def __le__(self, other) -> Any: ...
    def __lt__(self, other) -> Any: ...
    def __ne__(self, other) -> Any: ...
    def __reduce__(self) -> Any: ...
    def __setstate__(self, state) -> Any: ...

class MetadataRecordBatchReader(_MetadataRecordBatchReader):
    def __init__(self, *args, **kwargs) -> None: ...
    def __reduce__(self) -> Any: ...
    def __setstate__(self, state) -> Any: ...

class MetadataRecordBatchWriter(pyarrow.lib._CRecordBatchWriter):
    def __init__(self, *args, **kwargs) -> None: ...
    def begin(self, schema: Schema, options=...) -> Any: ...
    def close(self) -> Any: ...
    def write_metadata(self, buf) -> Any: ...
    def write_with_metadata(self, RecordBatchbatch, buf) -> Any: ...
    def __reduce__(self) -> Any: ...
    def __setstate__(self, state) -> Any: ...

class RecordBatchStream(FlightDataStream):
    def __init__(self, *args, **kwargs) -> None: ...
    def __reduce__(self) -> Any: ...
    def __setstate__(self, state) -> Any: ...

class Result(pyarrow.lib._Weakrefable):
    __hash__: ClassVar[None] = ...  # type: ignore
    body: Any
    def __init__(self, *args, **kwargs) -> None: ...
    @classmethod
    def deserialize(cls, typecls, serialized) -> Any: ...
    def serialize(self) -> Any: ...
    def __eq__(self, other) -> Any: ...
    def __ge__(self, other) -> Any: ...
    def __gt__(self, other) -> Any: ...
    def __le__(self, other) -> Any: ...
    def __lt__(self, other) -> Any: ...
    def __ne__(self, other) -> Any: ...
    def __reduce__(self) -> Any: ...
    def __setstate__(self, state) -> Any: ...

class SchemaResult(pyarrow.lib._Weakrefable):
    __hash__: ClassVar[None] = ...  # type: ignore
    schema: Any
    def __init__(self, *args, **kwargs) -> None: ...
    @classmethod
    def deserialize(cls, typecls, serialized) -> Any: ...
    def serialize(self) -> Any: ...
    def __eq__(self, other) -> Any: ...
    def __ge__(self, other) -> Any: ...
    def __gt__(self, other) -> Any: ...
    def __le__(self, other) -> Any: ...
    def __lt__(self, other) -> Any: ...
    def __ne__(self, other) -> Any: ...
    def __reduce__(self) -> Any: ...
    def __setstate__(self, state) -> Any: ...

class ServerAuthHandler(pyarrow.lib._Weakrefable):
    def __init__(self, *args, **kwargs) -> None: ...
    def authenticate(self, outgoing, incoming) -> Any: ...
    def is_valid(self, token) -> Any: ...
    def __reduce__(self) -> Any: ...
    def __setstate__(self, state) -> Any: ...

class ServerAuthReader(pyarrow.lib._Weakrefable):
    def __init__(self, *args, **kwargs) -> None: ...
    def read(self) -> Any: ...
    def __reduce__(self) -> Any: ...
    def __setstate__(self, state) -> Any: ...

class ServerAuthSender(pyarrow.lib._Weakrefable):
    def __init__(self, *args, **kwargs) -> None: ...
    def write(self, message) -> Any: ...
    def __reduce__(self) -> Any: ...
    def __setstate__(self, state) -> Any: ...

class ServerCallContext(pyarrow.lib._Weakrefable):
    def __init__(self, *args, **kwargs) -> None: ...
    def get_middleware(self, key) -> Any: ...
    def is_cancelled(self) -> Any: ...
    def peer(self) -> Any: ...
    def peer_identity(self) -> Any: ...
    def __reduce__(self) -> Any: ...
    def __setstate__(self, state) -> Any: ...

class ServerMiddleware(pyarrow.lib._Weakrefable):
    def __init__(self, *args, **kwargs) -> None: ...
    def call_completed(self, exception) -> Any: ...
    def sending_headers(self) -> Any: ...
    def __reduce__(self) -> Any: ...
    def __setstate__(self, state) -> Any: ...

class ServerMiddlewareFactory(pyarrow.lib._Weakrefable):
    def __init__(self, *args, **kwargs) -> None: ...
    def start_call(self, info, headers) -> Any: ...
    def __reduce__(self) -> Any: ...
    def __setstate__(self, state) -> Any: ...

class SignalStopHandler:
    stop_token: Any
    def __init__(self, *args, **kwargs) -> None: ...
    def _init_signals(self) -> Any: ...
    def __enter__(self) -> Any: ...
    def __exit__(self, exc_type, exc_value, exc_tb) -> Any: ...
    def __reduce__(self) -> Any: ...
    def __setstate__(self, state) -> Any: ...

class Ticket(pyarrow.lib._Weakrefable):
    __hash__: ClassVar[None] = ...  # type: ignore
    ticket: Any
    def __init__(self, *args, **kwargs) -> None: ...
    @classmethod
    def deserialize(cls, typecls, serialized) -> Any: ...
    def serialize(self) -> Any: ...
    def __eq__(self, other) -> Any: ...
    def __ge__(self, other) -> Any: ...
    def __gt__(self, other) -> Any: ...
    def __le__(self, other) -> Any: ...
    def __lt__(self, other) -> Any: ...
    def __ne__(self, other) -> Any: ...
    def __reduce__(self) -> Any: ...
    def __setstate__(self, state) -> Any: ...

class TracingServerMiddleware(ServerMiddleware):
    __slots__: ClassVar[list] = ...
    trace_context: Any
    def __init__(self, trace_context) -> None: ...

class TracingServerMiddlewareFactory(ServerMiddlewareFactory):
    def __init__(self, *args, **kwargs) -> None: ...
    def __reduce__(self) -> Any: ...
    def __setstate__(self, state) -> Any: ...

class _ActionType(tuple):
    _asdict: ClassVar[function] = ...
    _field_defaults: ClassVar[dict] = ...
    _fields: ClassVar[tuple] = ...
    _replace: ClassVar[function] = ...
    __getnewargs__: ClassVar[function] = ...
    __match_args__: ClassVar[tuple] = ...
    __slots__: ClassVar[tuple] = ...
    description: Any
    type: Any
    def __init__(self, *args, **kwargs) -> None: ...
    @classmethod
    def _make(cls, *args, **kwargs) -> Any: ...

class _CallInfo(tuple):
    _asdict: ClassVar[function] = ...
    _field_defaults: ClassVar[dict] = ...
    _fields: ClassVar[tuple] = ...
    _replace: ClassVar[function] = ...
    __getnewargs__: ClassVar[function] = ...
    __match_args__: ClassVar[tuple] = ...
    __slots__: ClassVar[tuple] = ...
    method: Any
    def __init__(self, *args, **kwargs) -> None: ...
    @classmethod
    def _make(cls, *args, **kwargs) -> Any: ...

class _CertKeyPair(tuple):
    _asdict: ClassVar[function] = ...
    _field_defaults: ClassVar[dict] = ...
    _fields: ClassVar[tuple] = ...
    _replace: ClassVar[function] = ...
    __getnewargs__: ClassVar[function] = ...
    __match_args__: ClassVar[tuple] = ...
    __slots__: ClassVar[tuple] = ...
    cert: Any
    key: Any
    def __init__(self, *args, **kwargs) -> None: ...
    @classmethod
    def _make(cls, *args, **kwargs) -> Any: ...

class _FlightServerFinalizer(pyarrow.lib._Weakrefable):
    @classmethod
    def __init__(self, *args, **kwargs) -> None: ...
    def finalize(self) -> Any: ...
    def __reduce__(self) -> Any: ...
    def __setstate__(self, state) -> Any: ...

class _MetadataRecordBatchReader(
    pyarrow.lib._Weakrefable, pyarrow.lib._ReadPandasMixin
):
    schema: Any
    @classmethod
    def __init__(self, *args, **kwargs) -> None: ...
    def read_all(self) -> Any: ...
    def read_chunk(self) -> Any: ...
    def to_reader(self) -> Any: ...
    def __iter__(self) -> Any: ...
    def __reduce__(self) -> Any: ...
    def __setstate__(self, state) -> Any: ...

class _ReadPandasMixin:
    def read_pandas(self, **options) -> Any: ...

class _ServerMiddlewareFactoryWrapper(ServerMiddlewareFactory):
    def __init__(self, *args, **kwargs) -> None: ...
    def start_call(self, info, headers) -> Any: ...
    def __reduce__(self) -> Any: ...
    def __setstate__(self, state) -> Any: ...

class _ServerMiddlewareWrapper(ServerMiddleware):
    def __init__(self, *args, **kwargs) -> None: ...
    def call_completed(self, exception) -> Any: ...
    def sending_headers(self) -> Any: ...
    def __reduce__(self) -> Any: ...
    def __setstate__(self, state) -> Any: ...

def __pyx_unpickle_ClientAuthHandler(
    __pyx_type, long__pyx_checksum, __pyx_state
) -> Any: ...
def __pyx_unpickle_ClientMiddleware(
    __pyx_type, long__pyx_checksum, __pyx_state
) -> Any: ...
def __pyx_unpickle_ClientMiddlewareFactory(
    __pyx_type, long__pyx_checksum, __pyx_state
) -> Any: ...
def __pyx_unpickle_FlightCancelledError(
    __pyx_type, long__pyx_checksum, __pyx_state
) -> Any: ...
def __pyx_unpickle_FlightDataStream(
    __pyx_type, long__pyx_checksum, __pyx_state
) -> Any: ...
def __pyx_unpickle_FlightError(__pyx_type, long__pyx_checksum, __pyx_state) -> Any: ...
def __pyx_unpickle_FlightInternalError(
    __pyx_type, long__pyx_checksum, __pyx_state
) -> Any: ...
def __pyx_unpickle_FlightServerError(
    __pyx_type, long__pyx_checksum, __pyx_state
) -> Any: ...
def __pyx_unpickle_FlightTimedOutError(
    __pyx_type, long__pyx_checksum, __pyx_state
) -> Any: ...
def __pyx_unpickle_FlightUnauthenticatedError(
    __pyx_type, long__pyx_checksum, __pyx_state
) -> Any: ...
def __pyx_unpickle_FlightUnauthorizedError(
    __pyx_type, long__pyx_checksum, __pyx_state
) -> Any: ...
def __pyx_unpickle_FlightUnavailableError(
    __pyx_type, long__pyx_checksum, __pyx_state
) -> Any: ...
def __pyx_unpickle_ServerAuthHandler(
    __pyx_type, long__pyx_checksum, __pyx_state
) -> Any: ...
def __pyx_unpickle_ServerMiddleware(
    __pyx_type, long__pyx_checksum, __pyx_state
) -> Any: ...
def __pyx_unpickle_ServerMiddlewareFactory(
    __pyx_type, long__pyx_checksum, __pyx_state
) -> Any: ...
def __pyx_unpickle_TracingServerMiddlewareFactory(
    __pyx_type, long__pyx_checksum, __pyx_state
) -> Any: ...
def __pyx_unpickle__ServerMiddlewareFactoryWrapper(
    __pyx_type, long__pyx_checksum, __pyx_state
) -> Any: ...
def __pyx_unpickle__ServerMiddlewareWrapper(
    __pyx_type, long__pyx_checksum, __pyx_state
) -> Any: ...
def _munge_grpc_python_error(message) -> Any: ...
def as_buffer(o) -> Any: ...
def connect(location, **kwargs) -> Any: ...
def frombytes(*args, **kwargs) -> Any: ...
def tobytes(o) -> Any: ...
