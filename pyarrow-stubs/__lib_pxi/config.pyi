from typing import NamedTuple

class VersionInfo(NamedTuple):
    major: int
    minor: int
    patch: int

class BuildInfo(NamedTuple):
    version: str
    version_info: VersionInfo
    so_version: str
    full_so_version: str
    compiler_id: str
    compiler_version: str
    compiler_flags: str
    git_id: str
    git_description: str
    package_kind: str
    build_type: str

class RuntimeInfo(NamedTuple):
    simd_level: str
    detected_simd_level: str

cpp_build_info: BuildInfo
cpp_version: str
cpp_version_info: VersionInfo

def runtime_info() -> RuntimeInfo: ...
def set_timezone_db_path(path: str) -> None: ...

__all__ = [
    "VersionInfo",
    "BuildInfo",
    "RuntimeInfo",
    "cpp_build_info",
    "cpp_version",
    "cpp_version_info",
    "runtime_info",
    "set_timezone_db_path",
]
