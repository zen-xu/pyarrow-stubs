import datetime as dt
import enum
import sys

from abc import ABC, abstractmethod

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self
if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

from typing import Union, overload

from fsspec import AbstractFileSystem  # type: ignore[import-untyped]

from .lib import NativeFile, _Weakrefable

SupportedFileSystem: TypeAlias = Union[AbstractFileSystem, FileSystem]

class FileType(enum.IntFlag):
    NotFound = enum.auto()
    Unknown = enum.auto()
    File = enum.auto()
    Directory = enum.auto()

class FileInfo(_Weakrefable):
    """
    FileSystem entry info.

    Parameters
    ----------
    path : str
        The full path to the filesystem entry.
    type : FileType
        The type of the filesystem entry.
    mtime : datetime or float, default None
        If given, the modification time of the filesystem entry.
        If a float is given, it is the number of seconds since the
        Unix epoch.
    mtime_ns : int, default None
        If given, the modification time of the filesystem entry,
        in nanoseconds since the Unix epoch.
        `mtime` and `mtime_ns` are mutually exclusive.
    size : int, default None
        If given, the filesystem entry size in bytes.  This should only
        be given if `type` is `FileType.File`.

    Examples
    --------
    Generate a file:

    >>> from pyarrow import fs
    >>> local = fs.LocalFileSystem()
    >>> path_fs = local_path + "/pyarrow-fs-example.dat"
    >>> with local.open_output_stream(path_fs) as stream:
    ...     stream.write(b"data")
    4

    Get FileInfo object using ``get_file_info()``:

    >>> file_info = local.get_file_info(path_fs)
    >>> file_info
    <FileInfo for '.../pyarrow-fs-example.dat': type=FileType.File, size=4>

    Inspect FileInfo attributes:

    >>> file_info.type
    <FileType.File: 2>

    >>> file_info.is_file
    True

    >>> file_info.path
    '/.../pyarrow-fs-example.dat'

    >>> file_info.base_name
    'pyarrow-fs-example.dat'

    >>> file_info.size
    4

    >>> file_info.extension
    'dat'

    >>> file_info.mtime  # doctest: +SKIP
    datetime.datetime(2022, 6, 29, 7, 56, 10, 873922, tzinfo=datetime.timezone.utc)

    >>> file_info.mtime_ns  # doctest: +SKIP
    1656489370873922073
    """

    def __init__(
        self,
        path: str,
        type: FileType = FileType.Unknown,
        *,
        mtime: dt.datetime | float | None = None,
        mtime_ns: int | None = None,
        size: int | None = None,
    ): ...
    @property
    def type(self) -> FileType:
        """
        Type of the file.

        The returned enum values can be the following:

        - FileType.NotFound: target does not exist
        - FileType.Unknown: target exists but its type is unknown (could be a
          special file such as a Unix socket or character device, or
          Windows NUL / CON / ...)
        - FileType.File: target is a regular file
        - FileType.Directory: target is a regular directory

        Returns
        -------
        type : FileType
        """
    @property
    def is_file(self) -> bool: ...
    @property
    def path(self) -> str:
        """
        The full file path in the filesystem.

        Examples
        --------
        >>> file_info = local.get_file_info(path)
        >>> file_info.path
        '/.../pyarrow-fs-example.dat'
        """
    @property
    def base_name(self) -> str:
        """
        The file base name.

        Component after the last directory separator.

        Examples
        --------
        >>> file_info = local.get_file_info(path)
        >>> file_info.base_name
        'pyarrow-fs-example.dat'
        """
    @property
    def size(self) -> int:
        """
        The size in bytes, if available.

        Only regular files are guaranteed to have a size.

        Returns
        -------
        size : int or None
        """
    @property
    def extension(self) -> str:
        """
        The file extension.

        Examples
        --------
        >>> file_info = local.get_file_info(path)
        >>> file_info.extension
        'dat'
        """
    @property
    def mtime(self) -> dt.datetime | None:
        """
        The time of last modification, if available.

        Returns
        -------
        mtime : datetime.datetime or None

        Examples
        --------
        >>> file_info = local.get_file_info(path)
        >>> file_info.mtime  # doctest: +SKIP
        datetime.datetime(2022, 6, 29, 7, 56, 10, 873922, tzinfo=datetime.timezone.utc)
        """
    @property
    def mtime_ns(self) -> int | None:
        """
        The time of last modification, if available, expressed in nanoseconds
        since the Unix epoch.

        Returns
        -------
        mtime_ns : int or None

        Examples
        --------
        >>> file_info = local.get_file_info(path)
        >>> file_info.mtime_ns  # doctest: +SKIP
        1656489370873922073
        """

class FileSelector(_Weakrefable):
    """
    File and directory selector.

    It contains a set of options that describes how to search for files and
    directories.

    Parameters
    ----------
    base_dir : str
        The directory in which to select files. Relative paths also work, use
        '.' for the current directory and '..' for the parent.
    allow_not_found : bool, default False
        The behavior if `base_dir` doesn't exist in the filesystem.
        If false, an error is returned.
        If true, an empty selection is returned.
    recursive : bool, default False
        Whether to recurse into subdirectories.

    Examples
    --------
    List the contents of a directory and subdirectories:

    >>> selector_1 = fs.FileSelector(local_path, recursive=True)
    >>> local.get_file_info(selector_1)  # doctest: +SKIP
    [<FileInfo for 'tmp/alphabet/example.dat': type=FileType.File, size=4>,
    <FileInfo for 'tmp/alphabet/subdir': type=FileType.Directory>,
    <FileInfo for 'tmp/alphabet/subdir/example_copy.dat': type=FileType.File, size=4>]

    List only the contents of the base directory:

    >>> selector_2 = fs.FileSelector(local_path)
    >>> local.get_file_info(selector_2)  # doctest: +SKIP
    [<FileInfo for 'tmp/alphabet/example.dat': type=FileType.File, size=4>,
    <FileInfo for 'tmp/alphabet/subdir': type=FileType.Directory>]

    Return empty selection if the directory doesn't exist:

    >>> selector_not_found = fs.FileSelector(
    ...     local_path + "/missing", recursive=True, allow_not_found=True
    ... )
    >>> local.get_file_info(selector_not_found)
    []
    """

    base_dir: str
    allow_not_found: bool
    recursive: bool
    def __init__(self, base_dir: str, allow_not_found: bool = False, recursive: bool = False): ...

class FileSystem(_Weakrefable):
    """
    Abstract file system API.
    """

    @classmethod
    def from_uri(cls, uri: str) -> tuple[Self, str]:
        """
        Create a new FileSystem from URI or Path.

        Recognized URI schemes are "file", "mock", "s3fs", "gs", "gcs", "hdfs" and "viewfs".
        In addition, the argument can be a pathlib.Path object, or a string
        describing an absolute local path.

        Parameters
        ----------
        uri : string
            URI-based path, for example: file:///some/local/path.

        Returns
        -------
        tuple of (FileSystem, str path)
            With (filesystem, path) tuple where path is the abstract path
            inside the FileSystem instance.

        Examples
        --------
        Create a new FileSystem subclass from a URI:

        >>> uri = "file:///{}/pyarrow-fs-example.dat".format(local_path)
        >>> local_new, path_new = fs.FileSystem.from_uri(uri)
        >>> local_new
        <pyarrow._fs.LocalFileSystem object at ...
        >>> path_new
        '/.../pyarrow-fs-example.dat'

        Or from a s3 bucket:

        >>> fs.FileSystem.from_uri("s3://usgs-landsat/collection02/")
        (<pyarrow._s3fs.S3FileSystem object at ...>, 'usgs-landsat/collection02')
        """
    def equals(self, other: FileSystem) -> bool:
        """
        Parameters
        ----------
        other : pyarrow.fs.FileSystem

        Returns
        -------
        bool
        """
    @property
    def type_name(self) -> str:
        """
        The filesystem's type name.
        """
    @overload
    def get_file_info(self, paths_or_selector: str) -> FileInfo: ...
    @overload
    def get_file_info(self, paths_or_selector: FileSelector | list[str]) -> list[FileInfo]: ...
    def get_file_info(self, paths_or_selector):
        """
        Get info for the given files.

        Any symlink is automatically dereferenced, recursively. A non-existing
        or unreachable file returns a FileStat object and has a FileType of
        value NotFound. An exception indicates a truly exceptional condition
        (low-level I/O error, etc.).

        Parameters
        ----------
        paths_or_selector : FileSelector, path-like or list of path-likes
            Either a selector object, a path-like object or a list of
            path-like objects. The selector's base directory will not be
            part of the results, even if it exists. If it doesn't exist,
            use `allow_not_found`.

        Returns
        -------
        FileInfo or list of FileInfo
            Single FileInfo object is returned for a single path, otherwise
            a list of FileInfo objects is returned.

        Examples
        --------
        >>> local
        <pyarrow._fs.LocalFileSystem object at ...>
        >>> local.get_file_info("/{}/pyarrow-fs-example.dat".format(local_path))
        <FileInfo for '/.../pyarrow-fs-example.dat': type=FileType.File, size=4>
        """
    def create_dir(self, path: str, *, recursive: bool = True) -> None:
        """
        Create a directory and subdirectories.

        This function succeeds if the directory already exists.

        Parameters
        ----------
        path : str
            The path of the new directory.
        recursive : bool, default True
            Create nested directories as well.
        """
    def delete_dir(self, path: str) -> None:
        """
        Delete a directory and its contents, recursively.

        Parameters
        ----------
        path : str
            The path of the directory to be deleted.
        """
    def delete_dir_contents(
        self, path: str, *, accept_root_dir: bool = False, missing_dir_ok: bool = False
    ) -> None:
        """
        Delete a directory's contents, recursively.

        Like delete_dir, but doesn't delete the directory itself.

        Parameters
        ----------
        path : str
            The path of the directory to be deleted.
        accept_root_dir : boolean, default False
            Allow deleting the root directory's contents
            (if path is empty or "/")
        missing_dir_ok : boolean, default False
            If False then an error is raised if path does
            not exist
        """
    def move(self, src: str, dest: str) -> None:
        """
        Move / rename a file or directory.

        If the destination exists:
        - if it is a non-empty directory, an error is returned
        - otherwise, if it has the same type as the source, it is replaced
        - otherwise, behavior is unspecified (implementation-dependent).

        Parameters
        ----------
        src : str
            The path of the file or the directory to be moved.
        dest : str
            The destination path where the file or directory is moved to.

        Examples
        --------
        Create a new folder with a file:

        >>> local.create_dir("/tmp/other_dir")
        >>> local.copy_file(path, "/tmp/move_example.dat")

        Move the file:

        >>> local.move("/tmp/move_example.dat", "/tmp/other_dir/move_example_2.dat")

        Inspect the file info:

        >>> local.get_file_info("/tmp/other_dir/move_example_2.dat")
        <FileInfo for '/tmp/other_dir/move_example_2.dat': type=FileType.File, size=4>
        >>> local.get_file_info("/tmp/move_example.dat")
        <FileInfo for '/tmp/move_example.dat': type=FileType.NotFound>

        Delete the folder:
        >>> local.delete_dir("/tmp/other_dir")
        """
    def copy_file(self, src: str, dest: str) -> None:
        """
        Copy a file.

        If the destination exists and is a directory, an error is returned.
        Otherwise, it is replaced.

        Parameters
        ----------
        src : str
            The path of the file to be copied from.
        dest : str
            The destination path where the file is copied to.

        Examples
        --------
        >>> local.copy_file(path, local_path + "/pyarrow-fs-example_copy.dat")

        Inspect the file info:

        >>> local.get_file_info(local_path + "/pyarrow-fs-example_copy.dat")
        <FileInfo for '/.../pyarrow-fs-example_copy.dat': type=FileType.File, size=4>
        >>> local.get_file_info(path)
        <FileInfo for '/.../pyarrow-fs-example.dat': type=FileType.File, size=4>
        """
    def delete_file(self, path: str) -> None:
        """
        Delete a file.

        Parameters
        ----------
        path : str
            The path of the file to be deleted.
        """
    def open_input_file(self, path: str) -> NativeFile:
        """
        Open an input file for random access reading.

        Parameters
        ----------
        path : str
            The source to open for reading.

        Returns
        -------
        stream : NativeFile

        Examples
        --------
        Print the data from the file with `open_input_file()`:

        >>> with local.open_input_file(path) as f:
        ...     print(f.readall())
        b'data'
        """
    def open_input_stream(
        self, path: str, compression: str | None = "detect", buffer_size: int | None = None
    ) -> NativeFile:
        """
        Open an input stream for sequential reading.

        Parameters
        ----------
        path : str
            The source to open for reading.
        compression : str optional, default 'detect'
            The compression algorithm to use for on-the-fly decompression.
            If "detect" and source is a file path, then compression will be
            chosen based on the file extension.
            If None, no compression will be applied. Otherwise, a well-known
            algorithm name must be supplied (e.g. "gzip").
        buffer_size : int optional, default None
            If None or 0, no buffering will happen. Otherwise the size of the
            temporary read buffer.

        Returns
        -------
        stream : NativeFile

        Examples
        --------
        Print the data from the file with `open_input_stream()`:

        >>> with local.open_input_stream(path) as f:
        ...     print(f.readall())
        b'data'
        """
    def open_output_stream(
        self,
        path: str,
        compression: str | None = "detect",
        buffer_size: int | None = None,
        metadata: dict[str, str] | None = None,
    ) -> NativeFile:
        """
        Open an output stream for sequential writing.

        If the target already exists, existing data is truncated.

        Parameters
        ----------
        path : str
            The source to open for writing.
        compression : str optional, default 'detect'
            The compression algorithm to use for on-the-fly compression.
            If "detect" and source is a file path, then compression will be
            chosen based on the file extension.
            If None, no compression will be applied. Otherwise, a well-known
            algorithm name must be supplied (e.g. "gzip").
        buffer_size : int optional, default None
            If None or 0, no buffering will happen. Otherwise the size of the
            temporary write buffer.
        metadata : dict optional, default None
            If not None, a mapping of string keys to string values.
            Some filesystems support storing metadata along the file
            (such as "Content-Type").
            Unsupported metadata keys will be ignored.

        Returns
        -------
        stream : NativeFile

        Examples
        --------
        >>> local = fs.LocalFileSystem()
        >>> with local.open_output_stream(path) as stream:
        ...     stream.write(b"data")
        4
        """
    def open_append_stream(
        self,
        path: str,
        compression: str | None = "detect",
        buffer_size: int | None = None,
        metadata: dict[str, str] | None = None,
    ):
        """
        Open an output stream for appending.

        If the target doesn't exist, a new empty file is created.

        .. note::
            Some filesystem implementations do not support efficient
            appending to an existing file, in which case this method will
            raise NotImplementedError.
            Consider writing to multiple files (using e.g. the dataset layer)
            instead.

        Parameters
        ----------
        path : str
            The source to open for writing.
        compression : str optional, default 'detect'
            The compression algorithm to use for on-the-fly compression.
            If "detect" and source is a file path, then compression will be
            chosen based on the file extension.
            If None, no compression will be applied. Otherwise, a well-known
            algorithm name must be supplied (e.g. "gzip").
        buffer_size : int optional, default None
            If None or 0, no buffering will happen. Otherwise the size of the
            temporary write buffer.
        metadata : dict optional, default None
            If not None, a mapping of string keys to string values.
            Some filesystems support storing metadata along the file
            (such as "Content-Type").
            Unsupported metadata keys will be ignored.

        Returns
        -------
        stream : NativeFile

        Examples
        --------
        Append new data to a FileSystem subclass with nonempty file:

        >>> with local.open_append_stream(path) as f:
        ...     f.write(b"+newly added")
        12

        Print out the content to the file:

        >>> with local.open_input_file(path) as f:
        ...     print(f.readall())
        b'data+newly added'
        """
    def normalize_path(self, path: str) -> str:
        """
        Normalize filesystem path.

        Parameters
        ----------
        path : str
            The path to normalize

        Returns
        -------
        normalized_path : str
            The normalized path
        """

class LocalFileSystem(FileSystem):
    """
    A FileSystem implementation accessing files on the local machine.

    Details such as symlinks are abstracted away (symlinks are always followed,
    except when deleting an entry).

    Parameters
    ----------
    use_mmap : bool, default False
        Whether open_input_stream and open_input_file should return
        a mmap'ed file or a regular file.

    Examples
    --------
    Create a FileSystem object with LocalFileSystem constructor:

    >>> from pyarrow import fs
    >>> local = fs.LocalFileSystem()
    >>> local
    <pyarrow._fs.LocalFileSystem object at ...>

    and write data on to the file:

    >>> with local.open_output_stream("/tmp/local_fs.dat") as stream:
    ...     stream.write(b"data")
    4
    >>> with local.open_input_stream("/tmp/local_fs.dat") as stream:
    ...     print(stream.readall())
    b'data'

    Create a FileSystem object inferred from a URI of the saved file:

    >>> local_new, path = fs.LocalFileSystem().from_uri("/tmp/local_fs.dat")
    >>> local_new
    <pyarrow._fs.LocalFileSystem object at ...
    >>> path
    '/tmp/local_fs.dat'

    Check if FileSystems `local` and `local_new` are equal:

    >>> local.equals(local_new)
    True

    Compare two different FileSystems:

    >>> local2 = fs.LocalFileSystem(use_mmap=True)
    >>> local.equals(local2)
    False

    Copy a file and print out the data:

    >>> local.copy_file("/tmp/local_fs.dat", "/tmp/local_fs-copy.dat")
    >>> with local.open_input_stream("/tmp/local_fs-copy.dat") as stream:
    ...     print(stream.readall())
    b'data'

    Open an output stream for appending, add text and print the new data:

    >>> with local.open_append_stream("/tmp/local_fs-copy.dat") as f:
    ...     f.write(b"+newly added")
    12

    >>> with local.open_input_stream("/tmp/local_fs-copy.dat") as f:
    ...     print(f.readall())
    b'data+newly added'

    Create a directory, copy a file into it and then delete the whole directory:

    >>> local.create_dir("/tmp/new_folder")
    >>> local.copy_file("/tmp/local_fs.dat", "/tmp/new_folder/local_fs.dat")
    >>> local.get_file_info("/tmp/new_folder")
    <FileInfo for '/tmp/new_folder': type=FileType.Directory>
    >>> local.delete_dir("/tmp/new_folder")
    >>> local.get_file_info("/tmp/new_folder")
    <FileInfo for '/tmp/new_folder': type=FileType.NotFound>

    Create a directory, copy a file into it and then delete
    the content of the directory:

    >>> local.create_dir("/tmp/new_folder")
    >>> local.copy_file("/tmp/local_fs.dat", "/tmp/new_folder/local_fs.dat")
    >>> local.get_file_info("/tmp/new_folder/local_fs.dat")
    <FileInfo for '/tmp/new_folder/local_fs.dat': type=FileType.File, size=4>
    >>> local.delete_dir_contents("/tmp/new_folder")
    >>> local.get_file_info("/tmp/new_folder")
    <FileInfo for '/tmp/new_folder': type=FileType.Directory>
    >>> local.get_file_info("/tmp/new_folder/local_fs.dat")
    <FileInfo for '/tmp/new_folder/local_fs.dat': type=FileType.NotFound>

    Create a directory, copy a file into it and then delete
    the file from the directory:

    >>> local.create_dir("/tmp/new_folder")
    >>> local.copy_file("/tmp/local_fs.dat", "/tmp/new_folder/local_fs.dat")
    >>> local.delete_file("/tmp/new_folder/local_fs.dat")
    >>> local.get_file_info("/tmp/new_folder/local_fs.dat")
    <FileInfo for '/tmp/new_folder/local_fs.dat': type=FileType.NotFound>
    >>> local.get_file_info("/tmp/new_folder")
    <FileInfo for '/tmp/new_folder': type=FileType.Directory>

    Move the file:

    >>> local.move("/tmp/local_fs-copy.dat", "/tmp/new_folder/local_fs-copy.dat")
    >>> local.get_file_info("/tmp/new_folder/local_fs-copy.dat")
    <FileInfo for '/tmp/new_folder/local_fs-copy.dat': type=FileType.File, size=16>
    >>> local.get_file_info("/tmp/local_fs-copy.dat")
    <FileInfo for '/tmp/local_fs-copy.dat': type=FileType.NotFound>

    To finish delete the file left:
    >>> local.delete_file("/tmp/local_fs.dat")
    """

    def __init__(self, *, use_mmap: bool = False) -> None: ...

class SubTreeFileSystem(FileSystem):
    """
    Delegates to another implementation after prepending a fixed base path.

    This is useful to expose a logical view of a subtree of a filesystem,
    for example a directory in a LocalFileSystem.

    Note, that this makes no security guarantee. For example, symlinks may
    allow to "escape" the subtree and access other parts of the underlying
    filesystem.

    Parameters
    ----------
    base_path : str
        The root of the subtree.
    base_fs : FileSystem
        FileSystem object the operations delegated to.

    Examples
    --------
    Create a LocalFileSystem instance:

    >>> from pyarrow import fs
    >>> local = fs.LocalFileSystem()
    >>> with local.open_output_stream("/tmp/local_fs.dat") as stream:
    ...     stream.write(b"data")
    4

    Create a directory and a SubTreeFileSystem instance:

    >>> local.create_dir("/tmp/sub_tree")
    >>> subtree = fs.SubTreeFileSystem("/tmp/sub_tree", local)

    Write data into the existing file:

    >>> with subtree.open_append_stream("sub_tree_fs.dat") as f:
    ...     f.write(b"+newly added")
    12

    Print out the attributes:

    >>> subtree.base_fs
    <pyarrow._fs.LocalFileSystem object at ...>
    >>> subtree.base_path
    '/tmp/sub_tree/'

    Get info for the given directory or given file:

    >>> subtree.get_file_info("")
    <FileInfo for '': type=FileType.Directory>
    >>> subtree.get_file_info("sub_tree_fs.dat")
    <FileInfo for 'sub_tree_fs.dat': type=FileType.File, size=12>

    Delete the file and directory:

    >>> subtree.delete_file("sub_tree_fs.dat")
    >>> local.delete_dir("/tmp/sub_tree")
    >>> local.delete_file("/tmp/local_fs.dat")

    For usage of the methods see examples for :func:`~pyarrow.fs.LocalFileSystem`.
    """
    def __init__(self, base_path: str, base_fs: FileSystem): ...
    @property
    def base_path(self) -> str: ...
    @property
    def base_fs(self) -> FileSystem: ...

class _MockFileSystem(FileSystem):
    def __init__(self, current_time: dt.datetime | None = None) -> None: ...

class PyFileSystem(FileSystem):
    """
    A FileSystem with behavior implemented in Python.

    Parameters
    ----------
    handler : FileSystemHandler
        The handler object implementing custom filesystem behavior.

    Examples
    --------
    Create an fsspec-based filesystem object for GitHub:

    >>> from fsspec.implementations import github
    >>> gfs = github.GithubFileSystem("apache", "arrow")  # doctest: +SKIP

    Get a PyArrow FileSystem object:

    >>> from pyarrow.fs import PyFileSystem, FSSpecHandler
    >>> pa_fs = PyFileSystem(FSSpecHandler(gfs))  # doctest: +SKIP

    Use :func:`~pyarrow.fs.FileSystem` functionality ``get_file_info()``:

    >>> pa_fs.get_file_info("README.md")  # doctest: +SKIP
    <FileInfo for 'README.md': type=FileType.File, size=...>
    """
    def __init__(self, handler: FileSystemHandler) -> None: ...
    @property
    def handler(self) -> FileSystemHandler:
        """
        The filesystem's underlying handler.

        Returns
        -------
        handler : FileSystemHandler
        """

class FileSystemHandler(ABC):
    """
    An abstract class exposing methods to implement PyFileSystem's behavior.
    """
    @abstractmethod
    def get_type_name(self) -> str:
        """
        Implement PyFileSystem.type_name.
        """
    @abstractmethod
    def get_file_info(self, paths: str | list[str]) -> FileInfo | list[FileInfo]:
        """
        Implement PyFileSystem.get_file_info(paths).

        Parameters
        ----------
        paths : list of str
            paths for which we want to retrieve the info.
        """
    @abstractmethod
    def get_file_info_selector(self, selector: FileSelector) -> list[FileInfo]:
        """
        Implement PyFileSystem.get_file_info(selector).

        Parameters
        ----------
        selector : FileSelector
            selector for which we want to retrieve the info.
        """

    @abstractmethod
    def create_dir(self, path: str, recursive: bool) -> None:
        """
        Implement PyFileSystem.create_dir(...).

        Parameters
        ----------
        path : str
            path of the directory.
        recursive : bool
            if the parent directories should be created too.
        """
    @abstractmethod
    def delete_dir(self, path: str) -> None:
        """
        Implement PyFileSystem.delete_dir(...).

        Parameters
        ----------
        path : str
            path of the directory.
        """
    @abstractmethod
    def delete_dir_contents(self, path: str, missing_dir_ok: bool = False) -> None:
        """
        Implement PyFileSystem.delete_dir_contents(...).

        Parameters
        ----------
        path : str
            path of the directory.
        missing_dir_ok : bool
            if False an error should be raised if path does not exist
        """
    @abstractmethod
    def delete_root_dir_contents(self) -> None:
        """
        Implement PyFileSystem.delete_dir_contents("/", accept_root_dir=True).
        """
    @abstractmethod
    def delete_file(self, path: str) -> None:
        """
        Implement PyFileSystem.delete_file(...).

        Parameters
        ----------
        path : str
            path of the file.
        """
    @abstractmethod
    def move(self, src: str, dest: str) -> None:
        """
        Implement PyFileSystem.move(...).

        Parameters
        ----------
        src : str
            path of what should be moved.
        dest : str
            path of where it should be moved to.
        """

    @abstractmethod
    def copy_file(self, src: str, dest: str) -> None:
        """
        Implement PyFileSystem.copy_file(...).

        Parameters
        ----------
        src : str
            path of what should be copied.
        dest : str
            path of where it should be copied to.
        """
    @abstractmethod
    def open_input_stream(self, path: str) -> NativeFile:
        """
        Implement PyFileSystem.open_input_stream(...).

        Parameters
        ----------
        path : str
            path of what should be opened.
        """
    @abstractmethod
    def open_input_file(self, path: str) -> NativeFile:
        """
        Implement PyFileSystem.open_input_file(...).

        Parameters
        ----------
        path : str
            path of what should be opened.
        """
    @abstractmethod
    def open_output_stream(self, path: str, metadata: dict[str, str]) -> NativeFile:
        """
        Implement PyFileSystem.open_output_stream(...).

        Parameters
        ----------
        path : str
            path of what should be opened.
        metadata :  mapping
            Mapping of string keys to string values.
            Some filesystems support storing metadata along the file
            (such as "Content-Type").
        """

    @abstractmethod
    def open_append_stream(self, path: str, metadata: dict[str, str]) -> NativeFile:
        """
        Implement PyFileSystem.open_append_stream(...).

        Parameters
        ----------
        path : str
            path of what should be opened.
        metadata :  mapping
            Mapping of string keys to string values.
            Some filesystems support storing metadata along the file
            (such as "Content-Type").
        """
    @abstractmethod
    def normalize_path(self, path: str) -> str:
        """
        Implement PyFileSystem.normalize_path(...).

        Parameters
        ----------
        path : str
            path of what should be normalized.
        """
