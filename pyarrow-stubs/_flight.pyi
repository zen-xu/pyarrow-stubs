import asyncio
import enum
import sys

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self
from typing import Generator, Generic, Iterable, Iterator, NamedTuple, TypeVar

from typing_extensions import deprecated

from .ipc import _ReadPandasMixin
from .lib import (
    ArrowCancelled,
    ArrowException,
    ArrowInvalid,
    Buffer,
    IpcReadOptions,
    IpcWriteOptions,
    RecordBatch,
    RecordBatchReader,
    Schema,
    Table,
    TimestampScalar,
    _CRecordBatchWriter,
    _Weakrefable,
)

_T = TypeVar("_T")

class FlightCallOptions(_Weakrefable):
    """RPC-layer options for a Flight call."""

    def __init__(
        self,
        timeout: float | None = None,
        write_options: IpcWriteOptions | None = None,
        headers: list[tuple[str, str]] | None = None,
        read_options: IpcReadOptions | None = None,
    ) -> None:
        """Create call options.

        Parameters
        ----------
        timeout : float, None
            A timeout for the call, in seconds. None means that the
            timeout defaults to an implementation-specific value.
        write_options : pyarrow.ipc.IpcWriteOptions, optional
            IPC write options. The default options can be controlled
            by environment variables (see pyarrow.ipc).
        headers : List[Tuple[str, str]], optional
            A list of arbitrary headers as key, value tuples
        read_options : pyarrow.ipc.IpcReadOptions, optional
            Serialization options for reading IPC format.
        """

class CertKeyPair(NamedTuple):
    """A TLS certificate and key for use in Flight."""

    cert: str
    key: str

class FlightError(Exception):
    """
    The base class for Flight-specific errors.

    A server may raise this class or one of its subclasses to provide
    a more detailed error to clients.

    Parameters
    ----------
    message : str, optional
        The error message.
    extra_info : bytes, optional
        Extra binary error details that were provided by the
        server/will be sent to the client.

    Attributes
    ----------
    extra_info : bytes
        Extra binary error details that were provided by the
        server/will be sent to the client.
    """

    extra_info: bytes

class FlightInternalError(FlightError, ArrowException):
    """An error internal to the Flight server occurred."""

class FlightTimedOutError(FlightError, ArrowException):
    """The Flight RPC call timed out."""

class FlightCancelledError(FlightError, ArrowCancelled):
    """The operation was cancelled."""

class FlightServerError(FlightError, ArrowException):
    """A server error occurred."""

class FlightUnauthenticatedError(FlightError, ArrowException):
    """The client is not authenticated."""

class FlightUnauthorizedError(FlightError, ArrowException):
    """The client is not authorized to perform the given operation."""

class FlightUnavailableError(FlightError, ArrowException):
    """The server is not reachable or available."""

class FlightWriteSizeExceededError(ArrowInvalid):
    """A write operation exceeded the client-configured limit."""

    limit: int
    actual: int

class Action(_Weakrefable):
    """An action executable on a Flight service."""

    def __init__(self, action_type: bytes | str, buf: Buffer | bytes) -> None:
        """Create an action from a type and a buffer.

        Parameters
        ----------
        action_type : bytes or str
        buf : Buffer or bytes-like object
        """
    @property
    def type(self) -> str:
        """The action type."""
    @property
    def body(self) -> Buffer:
        """The action body (arguments for the action)."""
    def serialize(self) -> bytes:
        """Get the wire-format representation of this type.

        Useful when interoperating with non-Flight systems (e.g. REST
        services) that may want to return Flight types.

        """
    @classmethod
    def deserialize(cls, serialized: bytes) -> Self:
        """Parse the wire-format representation of this type.

        Useful when interoperating with non-Flight systems (e.g. REST
        services) that may want to return Flight types.

        """

class ActionType(NamedTuple):
    """A type of action that is executable on a Flight service."""

    type: str
    description: str

    def make_action(self, buf: Buffer | bytes) -> Action:
        """Create an Action with this type.

        Parameters
        ----------
        buf : obj
            An Arrow buffer or Python bytes or bytes-like object.
        """

class Result(_Weakrefable):
    """A result from executing an Action."""
    def __init__(self, buf: Buffer | bytes) -> None:
        """Create a new result.

        Parameters
        ----------
        buf : Buffer or bytes-like object
        """
    @property
    def body(self) -> Buffer:
        """Get the Buffer containing the result."""
    def serialize(self) -> bytes:
        """Get the wire-format representation of this type.

        Useful when interoperating with non-Flight systems (e.g. REST
        services) that may want to return Flight types.

        """
    @classmethod
    def deserialize(cls, serialized: bytes) -> Self:
        """Parse the wire-format representation of this type.

        Useful when interoperating with non-Flight systems (e.g. REST
        services) that may want to return Flight types.

        """

class BasicAuth(_Weakrefable):
    """A container for basic auth."""
    def __init__(
        self, username: str | bytes | None = None, password: str | bytes | None = None
    ) -> None:
        """Create a new basic auth object.

        Parameters
        ----------
        username : string
        password : string
        """
    @property
    def username(self) -> bytes: ...
    @property
    def password(self) -> bytes: ...
    def serialize(self) -> str: ...
    @staticmethod
    def deserialize(serialized: str | bytes) -> BasicAuth: ...

class DescriptorType(enum.Enum):
    """
    The type of a FlightDescriptor.

    Attributes
    ----------

    UNKNOWN
        An unknown descriptor type.

    PATH
        A Flight stream represented by a path.

    CMD
        A Flight stream represented by an application-defined command.

    """

    UNKNOWN = 0
    PATH = 1
    CMD = 2

class FlightMethod(enum.Enum):
    """The implemented methods in Flight."""

    INVALID = 0
    HANDSHAKE = 1
    LIST_FLIGHTS = 2
    GET_FLIGHT_INFO = 3
    GET_SCHEMA = 4
    DO_GET = 5
    DO_PUT = 6
    DO_ACTION = 7
    LIST_ACTIONS = 8
    DO_EXCHANGE = 9

class FlightDescriptor(_Weakrefable):
    """A description of a data stream available from a Flight service."""
    @staticmethod
    def for_path(*path: str | bytes) -> FlightDescriptor:
        """Create a FlightDescriptor for a resource path."""

    @staticmethod
    def for_command(command: str | bytes) -> FlightDescriptor:
        """Create a FlightDescriptor for an opaque command."""
    @property
    def descriptor_type(self) -> DescriptorType:
        """Get the type of this descriptor."""
    @property
    def path(self) -> list[bytes] | None:
        """Get the path for this descriptor."""
    @property
    def command(self) -> bytes | None:
        """Get the command for this descriptor."""
    def serialize(self) -> bytes: ...
    @classmethod
    def deserialize(cls, serialized: bytes) -> Self: ...

class Ticket(_Weakrefable):
    """A ticket for requesting a Flight stream."""
    def __init__(self, ticket: str | bytes) -> None: ...
    @property
    def ticket(self) -> bytes: ...
    def serialize(self) -> bytes: ...
    @classmethod
    def deserialize(cls, serialized: bytes) -> Self: ...

class Location(_Weakrefable):
    """The location of a Flight service."""
    def __init__(self, uri: str | bytes) -> None: ...
    @property
    def uri(self) -> bytes: ...
    def equals(self, other: Location) -> bool: ...
    @staticmethod
    def for_grpc_tcp(host: str | bytes, port: int) -> Location:
        """Create a Location for a TCP-based gRPC service."""
    @staticmethod
    def for_grpc_tls(host: str | bytes, port: int) -> Location:
        """Create a Location for a TLS-based gRPC service."""
    @staticmethod
    def for_grpc_unix(path: str | bytes) -> Location:
        """Create a Location for a domain socket-based gRPC service."""

class FlightEndpoint(_Weakrefable):
    """A Flight stream, along with the ticket and locations to access it."""
    def __init__(
        self,
        ticket: Ticket | str | bytes,
        locations: list[str | Location],
        expiration_time: TimestampScalar | None = ...,
        app_metadata: bytes | str = ...,
    ):
        """Create a FlightEndpoint from a ticket and list of locations.

        Parameters
        ----------
        ticket : Ticket or bytes
            the ticket needed to access this flight
        locations : list of string URIs
            locations where this flight is available
        expiration_time : TimestampScalar, default None
            Expiration time of this stream. If present, clients may assume
            they can retry DoGet requests. Otherwise, clients should avoid
            retrying DoGet requests.
        app_metadata : bytes or str, default ""
            Application-defined opaque metadata.

        Raises
        ------
        ArrowException
            If one of the location URIs is not a valid URI.
        """
    @property
    def ticket(self) -> Ticket:
        """Get the ticket in this endpoint."""
    @property
    def locations(self) -> list[Location]:
        """Get locations where this flight is available."""
    def serialize(self) -> bytes: ...
    @property
    def expiration_time(self) -> TimestampScalar | None:
        """Get the expiration time of this stream.

        If present, clients may assume they can retry DoGet requests.
        Otherwise, clients should avoid retrying DoGet requests.

        """
    @property
    def app_metadata(self) -> bytes | str:
        """Get application-defined opaque metadata."""
    @classmethod
    def deserialize(cls, serialized: bytes) -> Self: ...

class SchemaResult(_Weakrefable):
    """The serialized schema returned from a GetSchema request."""
    def __init__(self, schema: Schema) -> None:
        """Create a SchemaResult from a schema.

        Parameters
        ----------
        schema: Schema
            the schema of the data in this flight.
        """
    @property
    def schema(self) -> Schema:
        """The schema of the data in this flight."""
    def serialize(self) -> bytes: ...
    @classmethod
    def deserialize(cls, serialized: bytes) -> Self: ...

class FlightInfo(_Weakrefable):
    """A description of a Flight stream."""
    def __init__(
        self,
        schema: Schema,
        descriptor: FlightDescriptor,
        endpoints: list[FlightEndpoint],
        total_records: int = ...,
        total_bytes: int = ...,
        ordered: bool = ...,
        app_metadata: bytes | str = ...,
    ) -> None:
        """Create a FlightInfo object from a schema, descriptor, and endpoints.

        Parameters
        ----------
        schema : Schema
            the schema of the data in this flight.
        descriptor : FlightDescriptor
            the descriptor for this flight.
        endpoints : list of FlightEndpoint
            a list of endpoints where this flight is available.
        total_records : int, default None
            the total records in this flight, -1 or None if unknown.
        total_bytes : int, default None
            the total bytes in this flight, -1 or None if unknown.
        ordered : boolean, default False
            Whether endpoints are in the same order as the data.
        app_metadata : bytes or str, default ""
            Application-defined opaque metadata.
        """
    @property
    def schema(self) -> Schema:
        """The schema of the data in this flight."""
    @property
    def descriptor(self) -> FlightDescriptor:
        """The descriptor of the data in this flight."""
    @property
    def endpoints(self) -> list[FlightEndpoint]:
        """The endpoints where this flight is available."""
    @property
    def total_records(self) -> int:
        """The total record count of this flight, or -1 if unknown."""
    @property
    def total_bytes(self) -> int:
        """The size in bytes of the data in this flight, or -1 if unknown."""
    @property
    def ordered(self) -> bool:
        """Whether endpoints are in the same order as the data."""
    @property
    def app_metadata(self) -> bytes | str:
        """
        Application-defined opaque metadata.

        There is no inherent or required relationship between this and the
        app_metadata fields in the FlightEndpoints or resulting FlightData
        messages. Since this metadata is application-defined, a given
        application could define there to be a relationship, but there is
        none required by the spec.

        """
    def serialize(self) -> bytes: ...
    @classmethod
    def deserialize(cls, serialized: bytes) -> Self: ...

class FlightStreamChunk(_Weakrefable):
    """A RecordBatch with application metadata on the side."""
    @property
    def data(self) -> RecordBatch | None: ...
    @property
    def app_metadata(self) -> Buffer | None: ...
    def __iter__(self): ...

class _MetadataRecordBatchReader(_Weakrefable, _ReadPandasMixin):
    """A reader for Flight streams."""

    # Needs to be separate class so the "real" class can subclass the
    # pure-Python mixin class

    def __iter__(self) -> Self: ...
    def __next__(self) -> FlightStreamChunk: ...
    @property
    def schema(self) -> Schema:
        """Get the schema for this reader."""
    def read_all(self) -> Table:
        """Read the entire contents of the stream as a Table."""
    def read_chunk(self) -> FlightStreamChunk:
        """Read the next FlightStreamChunk along with any metadata.

        Returns
        -------
        chunk : FlightStreamChunk
            The next FlightStreamChunk in the stream.

        Raises
        ------
        StopIteration
            when the stream is finished
        """
    def to_reader(self) -> RecordBatchReader:
        """Convert this reader into a regular RecordBatchReader.

        This may fail if the schema cannot be read from the remote end.

        Returns
        -------
        RecordBatchReader
        """

class MetadataRecordBatchReader(_MetadataRecordBatchReader):
    """The base class for readers for Flight streams.

    See Also
    --------
    FlightStreamReader
    """

class FlightStreamReader(MetadataRecordBatchReader):
    """A reader that can also be canceled."""
    def cancel(self) -> None:
        """Cancel the read operation."""
    def read_all(self) -> Table:
        """Read the entire contents of the stream as a Table."""

class MetadataRecordBatchWriter(_CRecordBatchWriter):
    """A RecordBatchWriter that also allows writing application metadata.

    This class is a context manager; on exit, close() will be called.
    """

    def begin(self, schema: Schema, options: IpcWriteOptions | None = None) -> None:
        """Prepare to write data to this stream with the given schema."""
    def write_metadata(self, buf: Buffer) -> None:
        """Write Flight metadata by itself."""
    def write_batch(self, batch: RecordBatch) -> None:  # type: ignore[override]
        """
        Write RecordBatch to stream.

        Parameters
        ----------
        batch : RecordBatch
        """
    def write_table(self, table: Table, max_chunksize: int | None = None, **kwargs) -> None:
        """
        Write Table to stream in (contiguous) RecordBatch objects.

        Parameters
        ----------
        table : Table
        max_chunksize : int, default None
            Maximum number of rows for RecordBatch chunks. Individual chunks may
            be smaller depending on the chunk layout of individual columns.
        """
    def close(self) -> None:
        """
        Close stream and write end-of-stream 0 marker.
        """
    def write_with_metadata(self, batch: RecordBatch, buf: Buffer) -> None:
        """Write a RecordBatch along with Flight metadata.

        Parameters
        ----------
        batch : RecordBatch
            The next RecordBatch in the stream.
        buf : Buffer
            Application-specific metadata for the batch as defined by
            Flight.
        """

class FlightStreamWriter(MetadataRecordBatchWriter):
    """A writer that also allows closing the write side of a stream."""
    def done_writing(self) -> None:
        """Indicate that the client is done writing, but not done reading."""

class FlightMetadataReader(_Weakrefable):
    """A reader for Flight metadata messages sent during a DoPut."""
    def read(self) -> Buffer | None:
        """Read the next metadata message."""

class FlightMetadataWriter(_Weakrefable):
    """A sender for Flight metadata messages during a DoPut."""
    def write(self, message: Buffer) -> None:
        """Write the next metadata message.

        Parameters
        ----------
        message : Buffer
        """

class AsyncioCall(Generic[_T]):
    """State for an async RPC using asyncio."""

    _future: asyncio.Future[_T]

    def as_awaitable(self) -> asyncio.Future[_T]: ...
    def wakeup(self, result_or_exception: BaseException | _T) -> None: ...

class AsyncioFlightClient:
    """
    A FlightClient with an asyncio-based async interface.

    This interface is EXPERIMENTAL.
    """

    def __init__(self, client: FlightClient) -> None: ...
    async def get_flight_info(
        self,
        descriptor: FlightDescriptor,
        *,
        options: FlightCallOptions | None = None,
    ): ...

class FlightClient(_Weakrefable):
    """A client to a Flight service.

    Connect to a Flight service on the given host and port.

    Parameters
    ----------
    location : str, tuple or Location
        Location to connect to. Either a gRPC URI like `grpc://localhost:port`,
        a tuple of (host, port) pair, or a Location instance.
    tls_root_certs : bytes or None
        PEM-encoded
    cert_chain: bytes or None
        Client certificate if using mutual TLS
    private_key: bytes or None
        Client private key for cert_chain is using mutual TLS
    override_hostname : str or None
        Override the hostname checked by TLS. Insecure, use with caution.
    middleware : list optional, default None
        A list of ClientMiddlewareFactory instances.
    write_size_limit_bytes : int optional, default None
        A soft limit on the size of a data payload sent to the
        server. Enabled if positive. If enabled, writing a record
        batch that (when serialized) exceeds this limit will raise an
        exception; the client can retry the write with a smaller
        batch.
    disable_server_verification : boolean optional, default False
        A flag that indicates that, if the client is connecting
        with TLS, that it skips server verification. If this is
        enabled, all other TLS settings are overridden.
    generic_options : list optional, default None
        A list of generic (string, int or string) option tuples passed
        to the underlying transport. Effect is implementation
        dependent.
    """
    def __init__(
        self,
        location: str | tuple[str, int] | Location,
        *,
        tls_root_certs: str | None = None,
        cert_chain: str | None = None,
        private_key: str | None = None,
        override_hostname: str | None = None,
        middleware: list[ClientMiddlewareFactory] | None = None,
        write_size_limit_bytes: int | None = None,
        disable_server_verification: bool = False,
        generic_options: list[tuple[str, int | str]] | None = None,
    ): ...
    @property
    def supports_async(self) -> bool: ...
    def as_async(self) -> AsyncioFlightClient: ...
    def wait_for_available(self, timeout: int = 5) -> None:
        """Block until the server can be contacted.

        Parameters
        ----------
        timeout : int, default 5
            The maximum seconds to wait.
        """
    @deprecated(
        "Use the ``FlightClient`` constructor or ``pyarrow.flight.connect`` function instead."
    )
    @classmethod
    def connect(
        cls,
        location: str | tuple[str, int] | Location,
        tls_root_certs: str | None = None,
        cert_chain: str | None = None,
        private_key: str | None = None,
        override_hostname: str | None = None,
        disable_server_verification: bool = False,
    ) -> FlightClient:
        """Connect to a Flight server.

        .. deprecated:: 0.15.0
            Use the ``FlightClient`` constructor or ``pyarrow.flight.connect`` function instead.
        """
    def authenticate(
        self, auth_handler: ClientAuthHandler, options: FlightCallOptions | None = None
    ) -> None:
        """Authenticate to the server.

        Parameters
        ----------
        auth_handler : ClientAuthHandler
            The authentication mechanism to use.
        options : FlightCallOptions
            Options for this call.
        """
    def authenticate_basic_token(
        self, username: str, password: str, options: FlightCallOptions | None = None
    ) -> tuple[str, str]:
        """Authenticate to the server with HTTP basic authentication.

        Parameters
        ----------
        username : string
            Username to authenticate with
        password : string
            Password to authenticate with
        options  : FlightCallOptions
            Options for this call

        Returns
        -------
        tuple : Tuple[str, str]
            A tuple representing the FlightCallOptions authorization
            header entry of a bearer token.
        """
    def list_actions(self, options: FlightCallOptions | None = None) -> list[Action]:
        """List the actions available on a service."""
    def do_action(
        self, action: Action, options: FlightCallOptions | None = None
    ) -> Iterator[Result]:
        """
        Execute an action on a service.

        Parameters
        ----------
        action : str, tuple, or Action
            Can be action type name (no body), type and body, or any Action
            object
        options : FlightCallOptions
            RPC options

        Returns
        -------
        results : iterator of Result values
        """
    def list_flights(
        self, criteria: str | None = None, options: FlightCallOptions | None = None
    ) -> Generator[FlightInfo, None, None]:
        """List the flights available on a service."""
    def get_flight_info(
        self, descriptor: FlightDescriptor, options: FlightCallOptions | None = None
    ) -> FlightInfo:
        """Request information about an available flight."""
    def get_schema(
        self, descriptor: FlightDescriptor, options: FlightCallOptions | None = None
    ) -> Schema:
        """Request schema for an available flight."""
    def do_get(
        self, ticket: Ticket, options: FlightCallOptions | None = None
    ) -> FlightStreamReader:
        """Request the data for a flight.

        Returns
        -------
        reader : FlightStreamReader
        """
    def do_put(
        self,
        descriptor: FlightDescriptor,
        schema: Schema,
        options: FlightCallOptions | None = None,
    ) -> tuple[FlightStreamWriter, FlightStreamReader]:
        """Upload data to a flight.

        Returns
        -------
        writer : FlightStreamWriter
        reader : FlightMetadataReader
        """
    def do_exchange(
        self, descriptor: FlightDescriptor, options: FlightCallOptions | None = None
    ) -> tuple[FlightStreamWriter, FlightStreamReader]:
        """Start a bidirectional data exchange with a server.

        Parameters
        ----------
        descriptor : FlightDescriptor
            A descriptor for the flight.
        options : FlightCallOptions
            RPC options.

        Returns
        -------
        writer : FlightStreamWriter
        reader : FlightStreamReader
        """
    def close(self) -> None:
        """Close the client and disconnect."""
    def __enter__(self) -> Self: ...
    def __exit__(self, exc_type, exc_value, traceback) -> None: ...

class FlightDataStream(_Weakrefable):
    """
    Abstract base class for Flight data streams.

    See Also
    --------
    RecordBatchStream
    GeneratorStream
    """

class RecordBatchStream(FlightDataStream):
    """A Flight data stream backed by RecordBatches.

    The remainder of this DoGet request will be handled in C++,
    without having to acquire the GIL.

    """
    def __init__(
        self, data_source: RecordBatchReader | Table, options: IpcWriteOptions | None = None
    ) -> None:
        """Create a RecordBatchStream from a data source.

        Parameters
        ----------
        data_source : RecordBatchReader or Table
            The data to stream to the client.
        options : pyarrow.ipc.IpcWriteOptions, optional
            Optional IPC options to control how to write the data.
        """

class GeneratorStream(FlightDataStream):
    """A Flight data stream backed by a Python generator."""
    def __init__(
        self,
        schema: Schema,
        generator: Iterable[FlightDataStream | Table | RecordBatch | RecordBatchReader],
        options: IpcWriteOptions | None = None,
    ) -> None:
        """Create a GeneratorStream from a Python generator.

        Parameters
        ----------
        schema : Schema
            The schema for the data to be returned.

        generator : iterator or iterable
            The generator should yield other FlightDataStream objects,
            Tables, RecordBatches, or RecordBatchReaders.

        options : pyarrow.ipc.IpcWriteOptions, optional
        """

class ServerCallContext(_Weakrefable):
    """Per-call state/context."""
    def peer_identity(self) -> bytes:
        """Get the identity of the authenticated peer.

        May be the empty string.
        """
    def peer(self) -> str:
        """Get the address of the peer."""
        # Set safe=True as gRPC on Windows sometimes gives garbage bytes
    def is_cancelled(self) -> bool:
        """Check if the current RPC call has been canceled by the client."""
    def add_header(self, key: str, value: str) -> None:
        """Add a response header."""
    def add_trailer(self, key: str, value: str) -> None:
        """Add a response trailer."""
    def get_middleware(self, key: str) -> ServerMiddleware | None:
        """
        Get a middleware instance by key.

        Returns None if the middleware was not found.
        """

class ServerAuthReader(_Weakrefable):
    """A reader for messages from the client during an auth handshake."""
    def read(self) -> str: ...

class ServerAuthSender(_Weakrefable):
    """A writer for messages to the client during an auth handshake."""
    def write(self, message: str) -> None: ...

class ClientAuthReader(_Weakrefable):
    """A reader for messages from the server during an auth handshake."""
    def read(self) -> str: ...

class ClientAuthSender(_Weakrefable):
    """A writer for messages to the server during an auth handshake."""
    def write(self, message: str) -> None: ...

class ServerAuthHandler(_Weakrefable):
    """Authentication middleware for a server.

    To implement an authentication mechanism, subclass this class and
    override its methods.

    """
    def authenticate(self, outgoing: ServerAuthSender, incoming: ServerAuthReader):
        """Conduct the handshake with the client.

        May raise an error if the client cannot authenticate.

        Parameters
        ----------
        outgoing : ServerAuthSender
            A channel to send messages to the client.
        incoming : ServerAuthReader
            A channel to read messages from the client.
        """
    def is_valid(self, token: str) -> bool:
        """Validate a client token, returning their identity.

        May return an empty string (if the auth mechanism does not
        name the peer) or raise an exception (if the token is
        invalid).

        Parameters
        ----------
        token : bytes
            The authentication token from the client.

        """

class ClientAuthHandler(_Weakrefable):
    """Authentication plugin for a client."""
    def authenticate(self, outgoing: ClientAuthSender, incoming: ClientAuthReader):
        """Conduct the handshake with the server.

        Parameters
        ----------
        outgoing : ClientAuthSender
            A channel to send messages to the server.
        incoming : ClientAuthReader
            A channel to read messages from the server.
        """
    def get_token(self) -> str:
        """Get the auth token for a call."""

class CallInfo(NamedTuple):
    """Information about a particular RPC for Flight middleware."""

    method: FlightMethod

class ClientMiddlewareFactory(_Weakrefable):
    """A factory for new middleware instances.

    All middleware methods will be called from the same thread as the
    RPC method implementation. That is, thread-locals set in the
    client are accessible from the middleware itself.

    """
    def start_call(self, info: CallInfo) -> ClientMiddleware | None:
        """Called at the start of an RPC.

        This must be thread-safe and must not raise exceptions.

        Parameters
        ----------
        info : CallInfo
            Information about the call.

        Returns
        -------
        instance : ClientMiddleware
            An instance of ClientMiddleware (the instance to use for
            the call), or None if this call is not intercepted.

        """

class ClientMiddleware(_Weakrefable):
    """Client-side middleware for a call, instantiated per RPC.

    Methods here should be fast and must be infallible: they should
    not raise exceptions or stall indefinitely.

    """

    def sending_headers(self) -> dict[str, list[str] | list[bytes]]:
        """A callback before headers are sent.

        Returns
        -------
        headers : dict
            A dictionary of header values to add to the request, or
            None if no headers are to be added. The dictionary should
            have string keys and string or list-of-string values.

            Bytes values are allowed, but the underlying transport may
            not support them or may restrict them. For gRPC, binary
            values are only allowed on headers ending in "-bin".

            Header names must be lowercase ASCII.

        """

    def received_headers(self, headers: dict[str, list[str] | list[bytes]]):
        """A callback when headers are received.

        The default implementation does nothing.

        Parameters
        ----------
        headers : dict
            A dictionary of headers from the server. Keys are strings
            and values are lists of strings (for text headers) or
            bytes (for binary headers).

        """

    def call_completed(self, exception: ArrowException):
        """A callback when the call finishes.

        The default implementation does nothing.

        Parameters
        ----------
        exception : ArrowException
            If the call errored, this is the equivalent
            exception. Will be None if the call succeeded.

        """

class ServerMiddlewareFactory(_Weakrefable):
    """A factory for new middleware instances.

    All middleware methods will be called from the same thread as the
    RPC method implementation. That is, thread-locals set in the
    middleware are accessible from the method itself.

    """

    def start_call(
        self, info: CallInfo, headers: dict[str, list[str] | list[bytes]]
    ) -> ServerMiddleware | None:
        """Called at the start of an RPC.

        This must be thread-safe.

        Parameters
        ----------
        info : CallInfo
            Information about the call.
        headers : dict
            A dictionary of headers from the client. Keys are strings
            and values are lists of strings (for text headers) or
            bytes (for binary headers).

        Returns
        -------
        instance : ServerMiddleware
            An instance of ServerMiddleware (the instance to use for
            the call), or None if this call is not intercepted.

        Raises
        ------
        exception : pyarrow.ArrowException
            If an exception is raised, the call will be rejected with
            the given error.

        """

class TracingServerMiddlewareFactory(ServerMiddlewareFactory):
    """A factory for tracing middleware instances.

    This enables OpenTelemetry support in Arrow (if Arrow was compiled
    with OpenTelemetry support enabled). A new span will be started on
    each RPC call. The TracingServerMiddleware instance can then be
    retrieved within an RPC handler to get the propagated context,
    which can be used to start a new span on the Python side.

    Because the Python/C++ OpenTelemetry libraries do not
    interoperate, spans on the C++ side are not directly visible to
    the Python side and vice versa.

    """

class ServerMiddleware(_Weakrefable):
    """Server-side middleware for a call, instantiated per RPC.

    Methods here should be fast and must be infallible: they should
    not raise exceptions or stall indefinitely.

    """

    def sending_headers(self) -> dict[str, list[str] | list[bytes]]:
        """A callback before headers are sent.

        Returns
        -------
        headers : dict
            A dictionary of header values to add to the response, or
            None if no headers are to be added. The dictionary should
            have string keys and string or list-of-string values.

            Bytes values are allowed, but the underlying transport may
            not support them or may restrict them. For gRPC, binary
            values are only allowed on headers ending in "-bin".

            Header names must be lowercase ASCII.

        """
    def call_completed(self, exception: ArrowException):
        """A callback when the call finishes.

        Parameters
        ----------
        exception : pyarrow.ArrowException
            If the call errored, this is the equivalent
            exception. Will be None if the call succeeded.

        """

class TracingServerMiddleware(ServerMiddleware):
    trace_context: dict
    def __init__(self, trace_context: dict) -> None: ...

class _ServerMiddlewareFactoryWrapper(ServerMiddlewareFactory):
    """Wrapper to bundle server middleware into a single C++ one."""

    def __init__(self, factories: dict[str, ServerMiddlewareFactory]) -> None: ...
    def start_call(  # type: ignore[override]
        self, info: CallInfo, headers: dict[str, list[str] | list[bytes]]
    ) -> _ServerMiddlewareFactoryWrapper | None: ...

class _ServerMiddlewareWrapper(ServerMiddleware):
    def __init__(self, middleware: dict[str, ServerMiddleware]) -> None: ...
    def send_headers(self) -> dict[str, dict[str, list[str] | list[bytes]]]: ...
    def call_completed(self, exception: ArrowException) -> None: ...

class _FlightServerFinalizer(_Weakrefable):
    """
    A finalizer that shuts down the server on destruction.

    See ARROW-16597. If the server is still active at interpreter
    exit, the process may segfault.
    """

    def finalize(self) -> None: ...

class FlightServerBase(_Weakrefable):
    """A Flight service definition.

    To start the server, create an instance of this class with an
    appropriate location. The server will be running as soon as the
    instance is created; it is not required to call :meth:`serve`.

    Override methods to define your Flight service.

    Parameters
    ----------
    location : str, tuple or Location optional, default None
        Location to serve on. Either a gRPC URI like `grpc://localhost:port`,
        a tuple of (host, port) pair, or a Location instance.
        If None is passed then the server will be started on localhost with a
        system provided random port.
    auth_handler : ServerAuthHandler optional, default None
        An authentication mechanism to use. May be None.
    tls_certificates : list optional, default None
        A list of (certificate, key) pairs.
    verify_client : boolean optional, default False
        If True, then enable mutual TLS: require the client to present
        a client certificate, and validate the certificate.
    root_certificates : bytes optional, default None
        If enabling mutual TLS, this specifies the PEM-encoded root
        certificate used to validate client certificates.
    middleware : dict optional, default None
        A dictionary of :class:`ServerMiddlewareFactory` instances. The
        string keys can be used to retrieve the middleware instance within
        RPC handlers (see :meth:`ServerCallContext.get_middleware`).

    """
    def __init__(
        self,
        location: str | tuple[str, int] | Location | None = None,
        auth_handler: ServerAuthHandler | None = None,
        tls_certificates: list[tuple[str, str]] | None = None,
        verify_client: bool = False,
        root_certificates: str | None = None,
        middleware: dict[str, ServerMiddlewareFactory] | None = None,
    ): ...
    @property
    def port(self) -> int:
        """
        Get the port that this server is listening on.

        Returns a non-positive value if the operation is invalid
        (e.g. init() was not called or server is listening on a domain
        socket).
        """
    def list_flights(self, context: ServerCallContext, criteria: str) -> Iterator[FlightInfo]:
        """List flights available on this service.

        Applications should override this method to implement their
        own behavior. The default method raises a NotImplementedError.

        Parameters
        ----------
        context : ServerCallContext
            Common contextual information.
        criteria : bytes
            Filter criteria provided by the client.

        Returns
        -------
        iterator of FlightInfo

        """
    def get_flight_info(
        self, context: ServerCallContext, descriptor: FlightDescriptor
    ) -> FlightInfo:
        """Get information about a flight.

        Applications should override this method to implement their
        own behavior. The default method raises a NotImplementedError.

        Parameters
        ----------
        context : ServerCallContext
            Common contextual information.
        descriptor : FlightDescriptor
            The descriptor for the flight provided by the client.

        Returns
        -------
        FlightInfo

        """
    def get_schema(self, context: ServerCallContext, descriptor: FlightDescriptor) -> Schema:
        """Get the schema of a flight.

        Applications should override this method to implement their
        own behavior. The default method raises a NotImplementedError.

        Parameters
        ----------
        context : ServerCallContext
            Common contextual information.
        descriptor : FlightDescriptor
            The descriptor for the flight provided by the client.

        Returns
        -------
        Schema

        """
    def do_put(
        self,
        context: ServerCallContext,
        descriptor: FlightDescriptor,
        reader: MetadataRecordBatchReader,
        writer: FlightMetadataWriter,
    ) -> None:
        """Write data to a flight.

        Applications should override this method to implement their
        own behavior. The default method raises a NotImplementedError.

        Parameters
        ----------
        context : ServerCallContext
            Common contextual information.
        descriptor : FlightDescriptor
            The descriptor for the flight provided by the client.
        reader : MetadataRecordBatchReader
            A reader for data uploaded by the client.
        writer : FlightMetadataWriter
            A writer to send responses to the client.

        """
    def do_get(self, context: ServerCallContext, ticket: Ticket) -> FlightDataStream:
        """Write data to a flight.

        Applications should override this method to implement their
        own behavior. The default method raises a NotImplementedError.

        Parameters
        ----------
        context : ServerCallContext
            Common contextual information.
        ticket : Ticket
            The ticket for the flight.

        Returns
        -------
        FlightDataStream
            A stream of data to send back to the client.

        """
    def do_exchange(
        self,
        context: ServerCallContext,
        descriptor: FlightDescriptor,
        reader: MetadataRecordBatchReader,
        writer: MetadataRecordBatchWriter,
    ) -> None:
        """Write data to a flight.

        Applications should override this method to implement their
        own behavior. The default method raises a NotImplementedError.

        Parameters
        ----------
        context : ServerCallContext
            Common contextual information.
        descriptor : FlightDescriptor
            The descriptor for the flight provided by the client.
        reader : MetadataRecordBatchReader
            A reader for data uploaded by the client.
        writer : MetadataRecordBatchWriter
            A writer to send responses to the client.

        """
    def list_actions(self, context: ServerCallContext) -> Iterable[Action]:
        """List custom actions available on this server.

        Applications should override this method to implement their
        own behavior. The default method raises a NotImplementedError.

        Parameters
        ----------
        context : ServerCallContext
            Common contextual information.

        Returns
        -------
        iterator of ActionType or tuple

        """
    def do_action(self, context: ServerCallContext, action: Action) -> Iterable[bytes]:
        """Execute a custom action.

        This method should return an iterator, or it should be a
        generator. Applications should override this method to
        implement their own behavior. The default method raises a
        NotImplementedError.

        Parameters
        ----------
        context : ServerCallContext
            Common contextual information.
        action : Action
            The action to execute.

        Returns
        -------
        iterator of bytes

        """
    def serve(self) -> None:
        """Block until the server shuts down.

        This method only returns if shutdown() is called or a signal is
        received.
        """
    def run(self) -> None:
        """Block until the server shuts down.

        .. deprecated:: 0.15.0
            Use the ``FlightServer.serve`` method instead
        """
    def shutdown(self) -> None:
        """Shut down the server, blocking until current requests finish.

        Do not call this directly from the implementation of a Flight
        method, as then the server will block forever waiting for that
        request to finish. Instead, call this method from a background
        thread.

        This method should only be called once.
        """
    def wait(self) -> None:
        """Block until server is terminated with shutdown."""
    def __enter__(self) -> Self: ...
    def __exit__(self, exc_type, exc_value, traceback): ...

def connect(
    location: str | tuple[str, int] | Location,
    *,
    tls_root_certs: str | None = None,
    cert_chain: str | None = None,
    private_key: str | None = None,
    override_hostname: str | None = None,
    middleware: list[ClientMiddlewareFactory] | None = None,
    write_size_limit_bytes: int | None = None,
    disable_server_verification: bool = False,
    generic_options: list[tuple[str, int | str]] | None = None,
) -> FlightClient:
    """
    Connect to a Flight server.

    Parameters
    ----------
    location : str, tuple, or Location
        Location to connect to. Either a URI like "grpc://localhost:port",
        a tuple of (host, port), or a Location instance.
    tls_root_certs : bytes or None
        PEM-encoded.
    cert_chain: str or None
        If provided, enables TLS mutual authentication.
    private_key: str or None
        If provided, enables TLS mutual authentication.
    override_hostname : str or None
        Override the hostname checked by TLS. Insecure, use with caution.
    middleware : list or None
        A list of ClientMiddlewareFactory instances to apply.
    write_size_limit_bytes : int or None
        A soft limit on the size of a data payload sent to the
        server. Enabled if positive. If enabled, writing a record
        batch that (when serialized) exceeds this limit will raise an
        exception; the client can retry the write with a smaller
        batch.
    disable_server_verification : boolean or None
        Disable verifying the server when using TLS.
        Insecure, use with caution.
    generic_options : list or None
        A list of generic (string, int or string) options to pass to
        the underlying transport.

    Returns
    -------
    client : FlightClient
    """
