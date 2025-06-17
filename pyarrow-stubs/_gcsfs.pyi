import datetime as dt

from ._fs import FileSystem
from .lib import KeyValueMetadata

class GcsFileSystem(FileSystem):
    """
    Google Cloud Storage (GCS) backed FileSystem implementation

    By default uses the process described in https://google.aip.dev/auth/4110
    to resolve credentials. If not running on Google Cloud Platform (GCP),
    this generally requires the environment variable
    GOOGLE_APPLICATION_CREDENTIALS to point to a JSON file
    containing credentials.

    Note: GCS buckets are special and the operations available on them may be
    limited or more expensive than expected compared to local file systems.

    Note: When pickling a GcsFileSystem that uses default credentials, resolution
    credentials are not stored in the serialized data. Therefore, when unpickling
    it is assumed that the necessary credentials are in place for the target
    process.

    Parameters
    ----------
    anonymous : boolean, default False
        Whether to connect anonymously.
        If true, will not attempt to look up credentials using standard GCP
        configuration methods.
    access_token : str, default None
        GCP access token.  If provided, temporary credentials will be fetched by
        assuming this role; also, a `credential_token_expiration` must be
        specified as well.
    target_service_account : str, default None
        An optional service account to try to impersonate when accessing GCS. This
        requires the specified credential user or service account to have the necessary
        permissions.
    credential_token_expiration : datetime, default None
        Expiration for credential generated with an access token. Must be specified
        if `access_token` is specified.
    default_bucket_location : str, default 'US'
        GCP region to create buckets in.
    scheme : str, default 'https'
        GCS connection transport scheme.
    endpoint_override : str, default None
        Override endpoint with a connect string such as "localhost:9000"
    default_metadata : mapping or pyarrow.KeyValueMetadata, default None
        Default metadata for `open_output_stream`.  This will be ignored if
        non-empty metadata is passed to `open_output_stream`.
    retry_time_limit : timedelta, default None
        Set the maximum amount of time the GCS client will attempt to retry
        transient errors. Subsecond granularity is ignored.
    project_id : str, default None
        The GCP project identifier to use for creating buckets.
        If not set, the library uses the GOOGLE_CLOUD_PROJECT environment
        variable. Most I/O operations do not need a project id, only applications
        that create new buckets need a project id.
    """

    def __init__(
        self,
        *,
        anonymous: bool = False,
        access_token: str | None = None,
        target_service_account: str | None = None,
        credential_token_expiration: dt.datetime | None = None,
        default_bucket_location: str = "US",
        scheme: str = "https",
        endpoint_override: str | None = None,
        default_metadata: dict | KeyValueMetadata | None = None,
        retry_time_limit: dt.timedelta | None = None,
        project_id: str | None = None,
    ): ...
    @property
    def default_bucket_location(self) -> str:
        """
        The GCP location this filesystem will write to.
        """
    @property
    def project_id(self) -> str:
        """
        The GCP project id this filesystem will use.
        """
