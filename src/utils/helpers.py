from collections.abc import Callable
import numpy as np


def fid(mu1: np.ndarray, mu2: np.ndarray,
        cov1: np.ndarray, cov2: np.ndarray) -> np.floating:
    """
    Fréchet distance between two Gaussian distributions:
        N(mu1, cov1) and N(mu2, cov2).

    FID(mu1,cov1; mu2,cov2) =
        ||mu1 - mu2||_2^2 + Tr(cov1 + cov2 - 2 * (cov1^{1/2} cov2 cov1^{1/2})^{1/2})

    Args:
        mu1 (np.ndarray): Mean of the first distribution, shape (d,) or (d,1).
        mu2 (np.ndarray): Mean of the second distribution, shape (d,) or (d,1).
        cov1 (np.ndarray): Covariance matrix of the first distribution, shape (d,d).
        cov2 (np.ndarray): Covariance matrix of the second distribution, shape (d,d).

    Returns:
        float: The Fréchet distance between the two distributions.
    """
    mu1 = np.asarray(mu1).ravel()
    mu2 = np.asarray(mu2).ravel()
    cov1 = np.asarray(cov1)
    cov2 = np.asarray(cov2)

    # ---------- 1D fast path ----------
    if np.asarray(mu1).size == 1 and np.asarray(mu2).size == 1:
        m1 = float(np.asarray(mu1).ravel()[0])
        m2 = float(np.asarray(mu2).ravel()[0])

        # Accept cov as scalar or 1x1 array
        c1 = float(np.asarray(cov1).squeeze())
        c2 = float(np.asarray(cov2).squeeze())
        fid_value = (m1 - m2) ** 2 + c1 + c2 - 2.0 * np.sqrt(np.abs(c1 * c2))
        return fid_value

    diff = mu1 - mu2
    diff_term = np.dot(diff, diff)

    # Helper: symmetric eigendecomposition-based matrix square root for (near-)SPD matrices
    def _sym_mat_sqrt(mat: np.ndarray) -> np.ndarray:
        mat = (mat + mat.T) * 0.5
        w, big_v = np.linalg.eigh(mat)
        w_max = np.max(w) if w.size else 0.0
        tol = np.finfo(w.dtype).eps * max(mat.shape) * max(w_max, 1.0)
        w = np.where(w < 0.0, np.where(np.abs(w) <= tol, 0.0, w), w)
        if np.any(w < 0.0):
            w = np.maximum(w, 0.0)
        sqrt_w = np.sqrt(w, dtype=mat.dtype)
        return (big_v * sqrt_w) @ big_v.T

    cov1_sqrt = _sym_mat_sqrt(cov1)
    big_a = cov1_sqrt @ cov2 @ cov1_sqrt
    big_a = (big_a + big_a.T) * 0.5

    w_a, _ = np.linalg.eigh(big_a)
    w_a_max = np.max(w_a) if w_a.size else 0.0
    tol_a = np.finfo(w_a.dtype).eps * max(big_a.shape) * max(w_a_max, 1.0)
    w_a = np.where(w_a < 0.0, np.where(np.abs(w_a) <= tol_a, 0.0, w_a), w_a)
    if np.any(w_a < 0.0):
        w_a = np.maximum(w_a, 0.0)

    trace_sqrt = float(np.sum(np.sqrt(w_a, dtype=big_a.dtype)))

    tr1 = float(np.trace(cov1))
    tr2 = float(np.trace(cov2))
    fid_value = diff_term + tr1 + tr2 - 2.0 * trace_sqrt
    return fid_value


def minimize_scalar(func: Callable) -> float:
    """
    Minimize a scalar function f: R -> R

    Args:
        func (Callable): The scalar function to minimize.

    Returns:
        float: The x value that minimizes the function.
    """
    raise NotImplementedError("Scalar minimization not implemented.")
