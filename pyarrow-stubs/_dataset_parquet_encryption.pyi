from ._dataset_parquet import ParquetFileWriteOptions, ParquetFragmentScanOptions
from ._parquet import FileDecryptionProperties
from ._parquet_encryption import CryptoFactory, EncryptionConfiguration, KmsConnectionConfig
from .lib import _Weakrefable

class ParquetEncryptionConfig(_Weakrefable):
    """
    Core configuration class encapsulating parameters for high-level encryption
    within the Parquet framework.

    The ParquetEncryptionConfig class serves as a bridge for passing encryption-related
    parameters to the appropriate components within the Parquet library. It maintains references
    to objects that define the encryption strategy, Key Management Service (KMS) configuration,
    and specific encryption configurations for Parquet data.

    Parameters
    ----------
    crypto_factory : pyarrow.parquet.encryption.CryptoFactory
        Shared pointer to a `CryptoFactory` object. The `CryptoFactory` is responsible for
        creating cryptographic components, such as encryptors and decryptors.
    kms_connection_config : pyarrow.parquet.encryption.KmsConnectionConfig
        Shared pointer to a `KmsConnectionConfig` object. This object holds the configuration
        parameters necessary for connecting to a Key Management Service (KMS).
    encryption_config : pyarrow.parquet.encryption.EncryptionConfiguration
        Shared pointer to an `EncryptionConfiguration` object. This object defines specific
        encryption settings for Parquet data, including the keys assigned to different columns.

    Raises
    ------
    ValueError
        Raised if `encryption_config` is None.
    """
    def __init__(
        self,
        crypto_factory: CryptoFactory,
        kms_connection_config: KmsConnectionConfig,
        encryption_config: EncryptionConfiguration,
    ) -> None: ...

class ParquetDecryptionConfig(_Weakrefable):
    """
    Core configuration class encapsulating parameters for high-level decryption
    within the Parquet framework.

    ParquetDecryptionConfig is designed to pass decryption-related parameters to
    the appropriate decryption components within the Parquet library. It holds references to
    objects that define the decryption strategy, Key Management Service (KMS) configuration,
    and specific decryption configurations for reading encrypted Parquet data.

    Parameters
    ----------
    crypto_factory : pyarrow.parquet.encryption.CryptoFactory
        Shared pointer to a `CryptoFactory` object, pivotal in creating cryptographic
        components for the decryption process.
    kms_connection_config : pyarrow.parquet.encryption.KmsConnectionConfig
        Shared pointer to a `KmsConnectionConfig` object, containing parameters necessary
        for connecting to a Key Management Service (KMS) during decryption.
    decryption_config : pyarrow.parquet.encryption.DecryptionConfiguration
        Shared pointer to a `DecryptionConfiguration` object, specifying decryption settings
        for reading encrypted Parquet data.

    Raises
    ------
    ValueError
        Raised if `decryption_config` is None.
    """
    def __init__(
        self,
        crypto_factory: CryptoFactory,
        kms_connection_config: KmsConnectionConfig,
        encryption_config: EncryptionConfiguration,
    ) -> None: ...

def set_encryption_config(
    opts: ParquetFileWriteOptions,
    config: ParquetEncryptionConfig,
) -> None: ...
def set_decryption_properties(
    opts: ParquetFragmentScanOptions,
    config: FileDecryptionProperties,
): ...
def set_decryption_config(
    opts: ParquetFragmentScanOptions,
    config: ParquetDecryptionConfig,
): ...
