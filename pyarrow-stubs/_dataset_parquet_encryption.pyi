from ._dataset_parquet import ParquetFileWriteOptions, ParquetFragmentScanOptions
from ._parquet import FileDecryptionProperties
from ._parquet_encryption import CryptoFactory, EncryptionConfiguration, KmsConnectionConfig
from .lib import _Weakrefable

class ParquetEncryptionConfig(_Weakrefable):
    def __init__(
        self,
        crypto_factory: CryptoFactory,
        kms_connection_config: KmsConnectionConfig,
        encryption_config: EncryptionConfiguration,
    ) -> None: ...

class ParquetDecryptionConfig(_Weakrefable):
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
