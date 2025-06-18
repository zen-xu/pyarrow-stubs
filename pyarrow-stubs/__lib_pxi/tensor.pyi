import sys

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

import numpy as np

from pyarrow.lib import _Weakrefable
from scipy.sparse import coo_matrix, csr_matrix
from sparse import COO

class Tensor(_Weakrefable):
    """
    A n-dimensional array a.k.a Tensor.

    Examples
    --------
    >>> import pyarrow as pa
    >>> import numpy as np
    >>> x = np.array([[2, 2, 4], [4, 5, 100]], np.int32)
    >>> pa.Tensor.from_numpy(x, dim_names=["dim1", "dim2"])
    <pyarrow.Tensor>
    type: int32
    shape: (2, 3)
    strides: (12, 4)
    """

    @classmethod
    def from_numpy(cls, obj: np.ndarray, dim_names: list[str] | None = None) -> Self:
        """
        Create a Tensor from a numpy array.

        Parameters
        ----------
        obj : numpy.ndarray
            The source numpy array
        dim_names : list, optional
            Names of each dimension of the Tensor.

        Examples
        --------
        >>> import pyarrow as pa
        >>> import numpy as np
        >>> x = np.array([[2, 2, 4], [4, 5, 100]], np.int32)
        >>> pa.Tensor.from_numpy(x, dim_names=["dim1", "dim2"])
        <pyarrow.Tensor>
        type: int32
        shape: (2, 3)
        strides: (12, 4)
        """
    def to_numpy(self) -> np.ndarray:
        """
        Convert arrow::Tensor to numpy.ndarray with zero copy

        Examples
        --------
        >>> import pyarrow as pa
        >>> import numpy as np
        >>> x = np.array([[2, 2, 4], [4, 5, 100]], np.int32)
        >>> tensor = pa.Tensor.from_numpy(x, dim_names=["dim1", "dim2"])
        >>> tensor.to_numpy()
        array([[  2,   2,   4],
               [  4,   5, 100]], dtype=int32)
        """
    def equals(self, other: Tensor) -> bool:
        """
        Return true if the tensors contains exactly equal data.

        Parameters
        ----------
        other : Tensor
            The other tensor to compare for equality.

        Examples
        --------
        >>> import pyarrow as pa
        >>> import numpy as np
        >>> x = np.array([[2, 2, 4], [4, 5, 100]], np.int32)
        >>> tensor = pa.Tensor.from_numpy(x, dim_names=["dim1", "dim2"])
        >>> y = np.array([[2, 2, 4], [4, 5, 10]], np.int32)
        >>> tensor2 = pa.Tensor.from_numpy(y, dim_names=["a", "b"])
        >>> tensor.equals(tensor)
        True
        >>> tensor.equals(tensor2)
        False
        """
    def dim_name(self, i: int) -> str:
        """
        Returns the name of the i-th tensor dimension.

        Parameters
        ----------
        i : int
            The physical index of the tensor dimension.

        Examples
        --------
        >>> import pyarrow as pa
        >>> import numpy as np
        >>> x = np.array([[2, 2, 4], [4, 5, 100]], np.int32)
        >>> tensor = pa.Tensor.from_numpy(x, dim_names=["dim1", "dim2"])
        >>> tensor.dim_name(0)
        'dim1'
        >>> tensor.dim_name(1)
        'dim2'
        """
    @property
    def dim_names(self) -> list[str]:
        """
        Names of this tensor dimensions.

        Examples
        --------
        >>> import pyarrow as pa
        >>> import numpy as np
        >>> x = np.array([[2, 2, 4], [4, 5, 100]], np.int32)
        >>> tensor = pa.Tensor.from_numpy(x, dim_names=["dim1", "dim2"])
        >>> tensor.dim_names
        ['dim1', 'dim2']
        """
    @property
    def is_mutable(self) -> bool:
        """
        Is this tensor mutable or immutable.

        Examples
        --------
        >>> import pyarrow as pa
        >>> import numpy as np
        >>> x = np.array([[2, 2, 4], [4, 5, 100]], np.int32)
        >>> tensor = pa.Tensor.from_numpy(x, dim_names=["dim1", "dim2"])
        >>> tensor.is_mutable
        True
        """
    @property
    def is_contiguous(self) -> bool:
        """
        Is this tensor contiguous in memory.

        Examples
        --------
        >>> import pyarrow as pa
        >>> import numpy as np
        >>> x = np.array([[2, 2, 4], [4, 5, 100]], np.int32)
        >>> tensor = pa.Tensor.from_numpy(x, dim_names=["dim1", "dim2"])
        >>> tensor.is_contiguous
        True
        """
    @property
    def ndim(self) -> int:
        """
        The dimension (n) of this tensor.

        Examples
        --------
        >>> import pyarrow as pa
        >>> import numpy as np
        >>> x = np.array([[2, 2, 4], [4, 5, 100]], np.int32)
        >>> tensor = pa.Tensor.from_numpy(x, dim_names=["dim1", "dim2"])
        >>> tensor.ndim
        2
        """
    @property
    def size(self) -> str:
        """
        The size of this tensor.

        Examples
        --------
        >>> import pyarrow as pa
        >>> import numpy as np
        >>> x = np.array([[2, 2, 4], [4, 5, 100]], np.int32)
        >>> tensor = pa.Tensor.from_numpy(x, dim_names=["dim1", "dim2"])
        >>> tensor.size
        6
        """
    @property
    def shape(self) -> tuple[int, ...]:
        """
        The shape of this tensor.

        Examples
        --------
        >>> import pyarrow as pa
        >>> import numpy as np
        >>> x = np.array([[2, 2, 4], [4, 5, 100]], np.int32)
        >>> tensor = pa.Tensor.from_numpy(x, dim_names=["dim1", "dim2"])
        >>> tensor.shape
        (2, 3)
        """
    @property
    def strides(self) -> tuple[int, ...]:
        """
        Strides of this tensor.

        Examples
        --------
        >>> import pyarrow as pa
        >>> import numpy as np
        >>> x = np.array([[2, 2, 4], [4, 5, 100]], np.int32)
        >>> tensor = pa.Tensor.from_numpy(x, dim_names=["dim1", "dim2"])
        >>> tensor.strides
        (12, 4)
        """

class SparseCOOTensor(_Weakrefable):
    @classmethod
    def from_dense_numpy(cls, obj: np.ndarray, dim_names: list[str] | None = None) -> Self:
        """
        Convert numpy.ndarray to arrow::SparseCOOTensor

        Parameters
        ----------
        obj : numpy.ndarray
            Data used to populate the rows.
        dim_names : list[str], optional
            Names of the dimensions.

        Returns
        -------
        pyarrow.SparseCOOTensor
        """

    @classmethod
    def from_numpy(
        cls,
        data: np.ndarray,
        coords: np.ndarray,
        shape: tuple[int, ...],
        dim_names: list[str] | None = None,
    ) -> Self:
        """
        Create arrow::SparseCOOTensor from numpy.ndarrays

        Parameters
        ----------
        data : numpy.ndarray
            Data used to populate the rows.
        coords : numpy.ndarray
            Coordinates of the data.
        shape : tuple
            Shape of the tensor.
        dim_names : list, optional
            Names of the dimensions.
        """
    @classmethod
    def from_scipy(cls, obj: csr_matrix, dim_names: list[str] | None = None) -> Self:
        """
        Convert scipy.sparse.coo_matrix to arrow::SparseCOOTensor

        Parameters
        ----------
        obj : scipy.sparse.csr_matrix
            The scipy matrix that should be converted.
        dim_names : list, optional
            Names of the dimensions.
        """
    @classmethod
    def from_pydata_sparse(cls, obj: COO, dim_names: list[str] | None = None) -> Self:
        """
        Convert pydata/sparse.COO to arrow::SparseCOOTensor.

        Parameters
        ----------
        obj : pydata.sparse.COO
            The sparse multidimensional array that should be converted.
        dim_names : list, optional
            Names of the dimensions.
        """
    @classmethod
    def from_tensor(cls, obj: Tensor) -> Self:
        """
        Convert arrow::Tensor to arrow::SparseCOOTensor.

        Parameters
        ----------
        obj : Tensor
            The tensor that should be converted.
        """
    def to_numpy(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert arrow::SparseCOOTensor to numpy.ndarrays with zero copy.
        """
    def to_scipy(self) -> coo_matrix:
        """
        Convert arrow::SparseCOOTensor to scipy.sparse.coo_matrix.
        """
    def to_pydata_sparse(self) -> COO:
        """
        Convert arrow::SparseCOOTensor to pydata/sparse.COO.
        """
    def to_tensor(self) -> Tensor:
        """
        Convert arrow::SparseCOOTensor to arrow::Tensor.
        """
    def equals(self, other: Self) -> bool:
        """
        Return true if sparse tensors contains exactly equal data.

        Parameters
        ----------
        other : SparseCOOTensor
            The other tensor to compare for equality.
        """
    @property
    def is_mutable(self) -> bool: ...
    @property
    def ndim(self) -> int: ...
    @property
    def size(self) -> str: ...
    @property
    def shape(self) -> tuple[int, ...]: ...
    def dim_name(self, i: int) -> str:
        """
        Returns the name of the i-th tensor dimension.

        Parameters
        ----------
        i : int
            The physical index of the tensor dimension.

        Returns
        -------
        str
        """
    @property
    def dim_names(self) -> list[str]: ...
    @property
    def non_zero_length(self) -> int: ...
    @property
    def has_canonical_format(self) -> bool: ...

class SparseCSRMatrix(_Weakrefable):
    """
    A sparse CSR matrix.
    """

    @classmethod
    def from_dense_numpy(cls, obj: np.ndarray, dim_names: list[str] | None = None) -> Self:
        """
        Convert numpy.ndarray to arrow::SparseCSRMatrix

        Parameters
        ----------
        obj : numpy.ndarray
            The dense numpy array that should be converted.
        dim_names : list, optional
            The names of the dimensions.

        Returns
        -------
        pyarrow.SparseCSRMatrix
        """
    @classmethod
    def from_numpy(
        cls,
        data: np.ndarray,
        indptr: np.ndarray,
        indices: np.ndarray,
        shape: tuple[int, ...],
        dim_names: list[str] | None = None,
    ) -> Self:
        """
        Create arrow::SparseCSRMatrix from numpy.ndarrays.

        Parameters
        ----------
        data : numpy.ndarray
            Data used to populate the sparse matrix.
        indptr : numpy.ndarray
            Range of the rows,
            The i-th row spans from `indptr[i]` to `indptr[i+1]` in the data.
        indices : numpy.ndarray
            Column indices of the corresponding non-zero values.
        shape : tuple
            Shape of the matrix.
        dim_names : list, optional
            Names of the dimensions.
        """
    @classmethod
    def from_scipy(cls, obj: csr_matrix, dim_names: list[str] | None = None) -> Self:
        """
        Convert scipy.sparse.csr_matrix to arrow::SparseCSRMatrix.

        Parameters
        ----------
        obj : scipy.sparse.csr_matrix
            The scipy matrix that should be converted.
        dim_names : list, optional
            Names of the dimensions.
        """
    @classmethod
    def from_tensor(cls, obj: Tensor) -> Self:
        """
        Convert arrow::Tensor to arrow::SparseCSRMatrix.

        Parameters
        ----------
        obj : Tensor
            The dense tensor that should be converted.
        """
    def to_numpy(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert arrow::SparseCSRMatrix to numpy.ndarrays with zero copy.
        """
    def to_scipy(self) -> csr_matrix:
        """
        Convert arrow::SparseCSRMatrix to scipy.sparse.csr_matrix.
        """
    def to_tensor(self) -> Tensor:
        """
        Convert arrow::SparseCSRMatrix to arrow::Tensor.
        """
    def equals(self, other: Self) -> bool:
        """
        Return true if sparse tensors contains exactly equal data.

        Parameters
        ----------
        other : SparseCSRMatrix
            The other tensor to compare for equality.
        """
    @property
    def is_mutable(self) -> bool: ...
    @property
    def ndim(self) -> int: ...
    @property
    def size(self) -> str: ...
    @property
    def shape(self) -> tuple[int, ...]: ...
    def dim_name(self, i: int) -> str:
        """
        Returns the name of the i-th tensor dimension.

        Parameters
        ----------
        i : int
            The physical index of the tensor dimension.

        Returns
        -------
        str
        """
    @property
    def dim_names(self) -> list[str]: ...
    @property
    def non_zero_length(self) -> int: ...

class SparseCSCMatrix(_Weakrefable):
    """
    A sparse CSC matrix.
    """

    @classmethod
    def from_dense_numpy(cls, obj: np.ndarray, dim_names: list[str] | None = None) -> Self:
        """
        Convert numpy.ndarray to arrow::SparseCSCMatrix

        Parameters
        ----------
        obj : numpy.ndarray
            Data used to populate the rows.
        dim_names : list[str], optional
            Names of the dimensions.

        Returns
        -------
        pyarrow.SparseCSCMatrix
        """
    @classmethod
    def from_numpy(
        cls,
        data: np.ndarray,
        indptr: np.ndarray,
        indices: np.ndarray,
        shape: tuple[int, ...],
        dim_names: list[str] | None = None,
    ) -> Self:
        """
        Create arrow::SparseCSCMatrix from numpy.ndarrays

        Parameters
        ----------
        data : numpy.ndarray
            Data used to populate the sparse matrix.
        indptr : numpy.ndarray
            Range of the rows,
            The i-th row spans from `indptr[i]` to `indptr[i+1]` in the data.
        indices : numpy.ndarray
            Column indices of the corresponding non-zero values.
        shape : tuple
            Shape of the matrix.
        dim_names : list, optional
            Names of the dimensions.
        """
    @classmethod
    def from_scipy(cls, obj: csr_matrix, dim_names: list[str] | None = None) -> Self:
        """
        Convert scipy.sparse.csc_matrix to arrow::SparseCSCMatrix

        Parameters
        ----------
        obj : scipy.sparse.csc_matrix
            The scipy matrix that should be converted.
        dim_names : list, optional
            Names of the dimensions.
        """
    @classmethod
    def from_tensor(cls, obj: Tensor) -> Self:
        """
        Convert arrow::Tensor to arrow::SparseCSCMatrix

        Parameters
        ----------
        obj : Tensor
            The dense tensor that should be converted.
        """
    def to_numpy(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert arrow::SparseCSCMatrix to numpy.ndarrays with zero copy
        """
    def to_scipy(self) -> csr_matrix:
        """
        Convert arrow::SparseCSCMatrix to scipy.sparse.csc_matrix
        """
    def to_tensor(self) -> Tensor:
        """
        Convert arrow::SparseCSCMatrix to arrow::Tensor
        """
    def equals(self, other: Self) -> bool:
        """
        Return true if sparse tensors contains exactly equal data

        Parameters
        ----------
        other : SparseCSCMatrix
            The other tensor to compare for equality.
        """
    @property
    def is_mutable(self) -> bool: ...
    @property
    def ndim(self) -> int: ...
    @property
    def size(self) -> str: ...
    @property
    def shape(self) -> tuple[int, ...]: ...
    def dim_name(self, i: int) -> str:
        """
        Returns the name of the i-th tensor dimension.

        Parameters
        ----------
        i : int
            The physical index of the tensor dimension.

        Returns
        -------
        str
        """
    @property
    def dim_names(self) -> list[str]: ...
    @property
    def non_zero_length(self) -> int: ...

class SparseCSFTensor(_Weakrefable):
    """
    A sparse CSF tensor.

    CSF is a generalization of compressed sparse row (CSR) index.

    CSF index recursively compresses each dimension of a tensor into a set
    of prefix trees. Each path from a root to leaf forms one tensor
    non-zero index. CSF is implemented with two arrays of buffers and one
    arrays of integers.
    """

    @classmethod
    def from_dense_numpy(cls, obj: np.ndarray, dim_names: list[str] | None = None) -> Self:
        """
        Convert numpy.ndarray to arrow::SparseCSFTensor

        Parameters
        ----------
        obj : numpy.ndarray
            Data used to populate the rows.
        dim_names : list[str], optional
            Names of the dimensions.

        Returns
        -------
        pyarrow.SparseCSFTensor
        """
    @classmethod
    def from_numpy(
        cls,
        data: np.ndarray,
        indptr: np.ndarray,
        indices: np.ndarray,
        shape: tuple[int, ...],
        dim_names: list[str] | None = None,
    ) -> Self:
        """
        Create arrow::SparseCSFTensor from numpy.ndarrays

        Parameters
        ----------
        data : numpy.ndarray
            Data used to populate the sparse tensor.
        indptr : numpy.ndarray
            The sparsity structure.
            Each two consecutive dimensions in a tensor correspond to
            a buffer in indices.
            A pair of consecutive values at `indptr[dim][i]`
            `indptr[dim][i + 1]` signify a range of nodes in
            `indices[dim + 1]` who are children of `indices[dim][i]` node.
        indices : numpy.ndarray
            Stores values of nodes.
            Each tensor dimension corresponds to a buffer in indptr.
        shape : tuple
            Shape of the matrix.
        axis_order : list, optional
            the sequence in which dimensions were traversed to
            produce the prefix tree.
        dim_names : list, optional
            Names of the dimensions.
        """
    @classmethod
    def from_tensor(cls, obj: Tensor) -> Self:
        """
        Convert arrow::Tensor to arrow::SparseCSFTensor

        Parameters
        ----------
        obj : Tensor
            The dense tensor that should be converted.
        """
    def to_numpy(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert arrow::SparseCSFTensor to numpy.ndarrays with zero copy
        """
    def to_tensor(self) -> Tensor:
        """
        Convert arrow::SparseCSFTensor to arrow::Tensor
        """
    def equals(self, other: Self) -> bool:
        """
        Return true if sparse tensors contains exactly equal data

        Parameters
        ----------
        other : SparseCSFTensor
            The other tensor to compare for equality.
        """
    @property
    def is_mutable(self) -> bool: ...
    @property
    def ndim(self) -> int: ...
    @property
    def size(self) -> str: ...
    @property
    def shape(self) -> tuple[int, ...]: ...
    def dim_name(self, i: int) -> str:
        """
        Returns the name of the i-th tensor dimension.

        Parameters
        ----------
        i : int
            The physical index of the tensor dimension.

        Returns
        -------
        str
        """
    @property
    def dim_names(self) -> list[str]: ...
    @property
    def non_zero_length(self) -> int: ...

__all__ = [
    "Tensor",
    "SparseCOOTensor",
    "SparseCSRMatrix",
    "SparseCSCMatrix",
    "SparseCSFTensor",
]
