"""Cluster phasor coordinates.

The ``phasorpy.cluster`` module provides functions to:

- fit elliptical clusters to phasor coordinates using a
  Gaussian Mixture Model (GMM):

  - :py:func:`phasor_cluster_gmm`

- fit elliptical clusters to phasor coordinates using k-means clustering:

  - :py:func:`phasor_cluster_kmeans`

- fit elliptical clusters to phasor coordinates using HDBSCAN
  density-based clustering:

  - :py:func:`phasor_cluster_hdbscan`

"""

from __future__ import annotations

__all__ = [
    'phasor_cluster_gmm',
    'phasor_cluster_hdbscan',
    'phasor_cluster_kmeans',
]

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._typing import Any, ArrayLike, Literal, NDArray

import math

import numpy


def phasor_cluster_gmm(
    real: ArrayLike,
    imag: ArrayLike,
    /,
    *,
    sigma: float = 2.0,
    clusters: int = 1,
    sort: Literal['polar', 'phasor', 'area'] | None = None,
    **kwargs: Any,
) -> tuple[
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
]:
    """Return elliptical clusters in phasor coordinates using GMM.

    Fit a Gaussian Mixture Model (GMM) to the provided phasor coordinates and
    extract the parameters of ellipses that represent each cluster according
    to [1]_.

    Parameters
    ----------
    real : array_like
        Real component of phasor coordinates.
    imag : array_like
        Imaginary component of phasor coordinates.
    sigma : float, optional, default: 2
        Scaling factor for radii of major and minor axes.
        The default 2.0 is commonly used for visualization of confidence
        ellipses (~98.2%).
    clusters : int, optional, default: 1
        Number of Gaussian distributions to fit to phasor coordinates.
    sort : {'polar', 'phasor', 'area'}, optional
        Sorting method for output clusters.
        By default, use 'polar' sorting.

        - 'polar': Sort by polar coordinates (phase, then modulation).
        - 'phasor': Sort by phasor coordinates (imaginary, then real).
        - 'area': Sort by inverse area of ellipse (-major * minor).

    **kwargs
        Optional arguments passed to
        :py:class:`sklearn.mixture.GaussianMixture`.

        Common options include:

        - covariance_type : {'full', 'tied', 'diag', 'spherical'}
        - max_iter : int, maximum number of EM iterations
        - random_state : int, for reproducible results

    Returns
    -------
    center_real : tuple of float
        Real component of ellipse centers.
    center_imag : tuple of float
        Imaginary component of ellipse centers.
    radius_major : tuple of float
        Major radii of ellipses.
    radius_minor : tuple of float
        Minor radii of ellipses.
    angle : tuple of float
        Rotation angles of major axes in radians, within range [0, pi].

    Raises
    ------
    ValueError
        If `sigma` is not positive.
        If `clusters` is less than 1.
        If the array shapes of `real` and `imag` do not match.
        If the number of valid (non-NaN) data points is less than `clusters`.
        If `sort` is not a valid sorting method.

    References
    ----------
    .. [1] Vallmitjana A, Torrado B, and Gratton E.
       `Phasor-based image segmentation: machine learning clustering techniques
       <https://doi.org/10.1364/BOE.422766>`_.
       *Biomed Opt Express*, 12(6): 3410-3422 (2021)

    Examples
    --------
    Recover the clusters from a synthetic distribution of phasor coordinates
    with two clusters:

    >>> real1, imag1 = numpy.random.multivariate_normal(
    ...     [0.2, 0.3], [[3e-3, 1e-3], [1e-3, 2e-3]], 100
    ... ).T
    >>> real2, imag2 = numpy.random.multivariate_normal(
    ...     [0.4, 0.5], [[2e-3, -1e-3], [-1e-3, 3e-3]], 100
    ... ).T
    >>> real = numpy.concatenate([real1, real2])
    >>> imag = numpy.concatenate([imag1, imag2])
    >>> center_real, center_imag, radius_major, radius_minor, angle = (
    ...     phasor_cluster_gmm(real, imag, clusters=2)
    ... )
    >>> center_real  # doctest: +SKIP
    (0.2, 0.4)

    """
    from sklearn.mixture import GaussianMixture

    if sigma <= 0.0:
        msg = f'{sigma=} <= 0'
        raise ValueError(msg)
    sigma = float(sigma)

    if clusters < 1:
        msg = f'{clusters=} < 1'
        raise ValueError(msg)

    coords = numpy.stack([real, imag], axis=-1).reshape((-1, 2))

    valid_data = ~numpy.isnan(coords).any(axis=1)
    coords = coords[valid_data]

    if coords.shape[0] < clusters:
        msg = f'number of valid data points ({coords.shape[0]}) < {clusters=}'
        raise ValueError(msg)

    kwargs.pop('n_components', None)

    gmm = GaussianMixture(n_components=clusters, **kwargs)
    gmm.fit(coords)

    center_real = []
    center_imag = []
    radius_major = []
    radius_minor = []
    angle = []

    for i in range(clusters):
        center_real.append(float(gmm.means_[i, 0]))
        center_imag.append(float(gmm.means_[i, 1]))

        match gmm.covariance_type:
            case 'full':
                cov = gmm.covariances_[i]
            case 'tied':
                cov = gmm.covariances_
            case 'diag':
                cov = numpy.diag(gmm.covariances_[i])
            case 'spherical':
                cov = numpy.eye(2) * gmm.covariances_[i]
            case _:
                msg = f'unknown {gmm.covariance_type=!r}'
                raise ValueError(msg)

        major, minor, current_angle = _ellipse_from_covariance(cov, sigma)
        radius_major.append(major)
        radius_minor.append(minor)
        angle.append(current_angle)

    argsort = _sort_clusters(
        center_real, center_imag, radius_major, radius_minor, sort
    )

    return (
        tuple(center_real[i] for i in argsort),
        tuple(center_imag[i] for i in argsort),
        tuple(radius_major[i] for i in argsort),
        tuple(radius_minor[i] for i in argsort),
        tuple(angle[i] for i in argsort),
    )


def phasor_cluster_kmeans(
    real: ArrayLike,
    imag: ArrayLike,
    /,
    *,
    sigma: float = 2.0,
    clusters: int = 1,
    sort: Literal['polar', 'phasor', 'area'] | None = None,
    **kwargs: Any,
) -> tuple[
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
]:
    """Return elliptical clusters in phasor coordinates using k-means.

    Partition the provided phasor coordinates into a fixed number of clusters
    using the k-means algorithm and extract the parameters of ellipses that
    represent each cluster. The ellipse axes and orientation are derived from
    the empirical covariance of the coordinates assigned to each cluster.

    Parameters
    ----------
    real : array_like
        Real component of phasor coordinates.
    imag : array_like
        Imaginary component of phasor coordinates.
    sigma : float, optional, default: 2
        Scaling factor for radii of major and minor axes.
        The default 2.0 is commonly used for visualization of confidence
        ellipses (~98.2%).
    clusters : int, optional, default: 1
        Number of clusters to partition phasor coordinates into.
    sort : {'polar', 'phasor', 'area'}, optional
        Sorting method for output clusters.
        By default, use 'polar' sorting.

        - 'polar': Sort by polar coordinates (phase, then modulation).
        - 'phasor': Sort by phasor coordinates (imaginary, then real).
        - 'area': Sort by inverse area of ellipse (-major * minor).

    **kwargs
        Optional arguments passed to :py:class:`sklearn.cluster.KMeans`.

        Common options include:

        - n_init : int, number of times the algorithm is run.
        - max_iter : int, maximum number of iterations.
        - random_state : int, for reproducible results.

    Returns
    -------
    center_real : tuple of float
        Real component of ellipse centers.
    center_imag : tuple of float
        Imaginary component of ellipse centers.
    radius_major : tuple of float
        Major radii of ellipses.
    radius_minor : tuple of float
        Minor radii of ellipses.
    angle : tuple of float
        Rotation angles of major axes in radians, within range [0, pi].

    Raises
    ------
    ValueError
        If `sigma` is not positive.
        If `clusters` is less than 1.
        If the array shapes of `real` and `imag` do not match.
        If the number of valid (non-NaN) data points is less than `clusters`.
        If `sort` is not a valid sorting method.

    References
    ----------
    .. [1] Vallmitjana A, Torrado B, and Gratton E.
       `Phasor-based image segmentation: machine learning clustering techniques
       <https://doi.org/10.1364/BOE.422766>`_.
       *Biomed Opt Express*, 12(6): 3410-3422 (2021)

    Examples
    --------
    Recover the clusters from a synthetic distribution of phasor coordinates
    with two clusters:

    >>> real1, imag1 = numpy.random.multivariate_normal(
    ...     [0.2, 0.3], [[3e-3, 1e-3], [1e-3, 2e-3]], 100
    ... ).T
    >>> real2, imag2 = numpy.random.multivariate_normal(
    ...     [0.4, 0.5], [[2e-3, -1e-3], [-1e-3, 3e-3]], 100
    ... ).T
    >>> real = numpy.concatenate([real1, real2])
    >>> imag = numpy.concatenate([imag1, imag2])
    >>> center_real, center_imag, radius_major, radius_minor, angle = (
    ...     phasor_cluster_kmeans(real, imag, clusters=2)
    ... )
    >>> center_real  # doctest: +SKIP
    (0.2, 0.4)

    """
    from sklearn.cluster import KMeans

    if sigma <= 0.0:
        msg = f'{sigma=} <= 0'
        raise ValueError(msg)
    sigma = float(sigma)

    if clusters < 1:
        msg = f'{clusters=} < 1'
        raise ValueError(msg)

    coords = numpy.stack([real, imag], axis=-1).reshape((-1, 2))

    valid_data = ~numpy.isnan(coords).any(axis=1)
    coords = coords[valid_data]

    if coords.shape[0] < clusters:
        msg = f'number of valid data points ({coords.shape[0]}) < {clusters=}'
        raise ValueError(msg)

    kwargs.pop('n_clusters', None)

    kmeans = KMeans(n_clusters=clusters, **kwargs)
    labels = kmeans.fit_predict(coords)

    center_real = []
    center_imag = []
    radius_major = []
    radius_minor = []
    angle = []

    for i in range(clusters):
        center_real.append(float(kmeans.cluster_centers_[i, 0]))
        center_imag.append(float(kmeans.cluster_centers_[i, 1]))

        cov = _cluster_covariance(coords[labels == i])
        major, minor, current_angle = _ellipse_from_covariance(cov, sigma)
        radius_major.append(major)
        radius_minor.append(minor)
        angle.append(current_angle)

    argsort = _sort_clusters(
        center_real, center_imag, radius_major, radius_minor, sort
    )

    return (
        tuple(center_real[i] for i in argsort),
        tuple(center_imag[i] for i in argsort),
        tuple(radius_major[i] for i in argsort),
        tuple(radius_minor[i] for i in argsort),
        tuple(angle[i] for i in argsort),
    )


def phasor_cluster_hdbscan(
    real: ArrayLike,
    imag: ArrayLike,
    /,
    *,
    sigma: float = 2.0,
    sort: Literal['polar', 'phasor', 'area'] | None = None,
    **kwargs: Any,
) -> tuple[
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
]:
    """Return elliptical clusters in phasor coordinates using HDBSCAN.

    Cluster the provided phasor coordinates using the density-based HDBSCAN
    algorithm and extract the parameters of ellipses that represent each
    cluster. Unlike :py:func:`phasor_cluster_gmm` and
    :py:func:`phasor_cluster_kmeans`, the number of clusters is determined
    automatically from the density of the data, and points classified as
    noise are excluded. The ellipse axes and orientation are derived from the
    empirical covariance of the coordinates assigned to each cluster.

    Parameters
    ----------
    real : array_like
        Real component of phasor coordinates.
    imag : array_like
        Imaginary component of phasor coordinates.
    sigma : float, optional, default: 2
        Scaling factor for radii of major and minor axes.
        The default 2.0 is commonly used for visualization of confidence
        ellipses (~98.2%).
    sort : {'polar', 'phasor', 'area'}, optional
        Sorting method for output clusters.
        By default, use 'polar' sorting.

        - 'polar': Sort by polar coordinates (phase, then modulation).
        - 'phasor': Sort by phasor coordinates (imaginary, then real).
        - 'area': Sort by inverse area of ellipse (-major * minor).

    **kwargs
        Optional arguments passed to :py:class:`sklearn.cluster.HDBSCAN`.

        Common options include:

        - min_cluster_size : int, minimum number of samples in a cluster.
        - min_samples : int, number of samples in a neighborhood for a
          point to be considered a core point.
        - cluster_selection_epsilon : float, distance threshold.

    Returns
    -------
    center_real : tuple of float
        Real component of ellipse centers.
    center_imag : tuple of float
        Imaginary component of ellipse centers.
    radius_major : tuple of float
        Major radii of ellipses.
    radius_minor : tuple of float
        Minor radii of ellipses.
    angle : tuple of float
        Rotation angles of major axes in radians, within range [0, pi].

    Raises
    ------
    ValueError
        If `sigma` is not positive.
        If the array shapes of `real` and `imag` do not match.
        If there are no valid (non-NaN) data points.
        If `sort` is not a valid sorting method.

    Notes
    -----
    If HDBSCAN does not find any cluster, that is, all points are classified
    as noise, empty tuples are returned.

    References
    ----------
    .. [1] Campello RJGB, Moulavi D, and Sander J.
       `Density-based clustering based on hierarchical density estimates
       <https://doi.org/10.1007/978-3-642-37456-2_14>`_.
       *Advances in Knowledge Discovery and Data Mining*, 160-172 (2013)

    Examples
    --------
    Recover the clusters from a synthetic distribution of phasor coordinates
    with two clusters:

    >>> real1, imag1 = numpy.random.multivariate_normal(
    ...     [0.2, 0.3], [[3e-3, 1e-3], [1e-3, 2e-3]], 100
    ... ).T
    >>> real2, imag2 = numpy.random.multivariate_normal(
    ...     [0.4, 0.5], [[2e-3, -1e-3], [-1e-3, 3e-3]], 100
    ... ).T
    >>> real = numpy.concatenate([real1, real2])
    >>> imag = numpy.concatenate([imag1, imag2])
    >>> center_real, center_imag, radius_major, radius_minor, angle = (
    ...     phasor_cluster_hdbscan(real, imag)
    ... )
    >>> center_real  # doctest: +SKIP
    (0.2, 0.4)

    """
    from sklearn.cluster import HDBSCAN

    if sigma <= 0.0:
        msg = f'{sigma=} <= 0'
        raise ValueError(msg)
    sigma = float(sigma)

    coords = numpy.stack([real, imag], axis=-1).reshape((-1, 2))

    valid_data = ~numpy.isnan(coords).any(axis=1)
    coords = coords[valid_data]

    if coords.shape[0] < 1:
        msg = 'number of valid data points is 0'
        raise ValueError(msg)

    hdbscan = HDBSCAN(**kwargs)
    labels = hdbscan.fit_predict(coords)

    center_real = []
    center_imag = []
    radius_major = []
    radius_minor = []
    angle = []

    # exclude noise points, labeled as -1
    for label in sorted(set(labels.tolist()) - {-1}):
        points = coords[labels == label]
        center_real.append(float(points[:, 0].mean()))
        center_imag.append(float(points[:, 1].mean()))

        cov = _cluster_covariance(points)
        major, minor, current_angle = _ellipse_from_covariance(cov, sigma)
        radius_major.append(major)
        radius_minor.append(minor)
        angle.append(current_angle)

    argsort = _sort_clusters(
        center_real, center_imag, radius_major, radius_minor, sort
    )

    return (
        tuple(center_real[i] for i in argsort),
        tuple(center_imag[i] for i in argsort),
        tuple(radius_major[i] for i in argsort),
        tuple(radius_minor[i] for i in argsort),
        tuple(angle[i] for i in argsort),
    )


def _cluster_covariance(points: NDArray[Any]) -> NDArray[Any]:
    """Return 2x2 covariance matrix of cluster points.

    Return a zero matrix if the cluster contains fewer than two points,
    in which case the covariance is undefined.

    """
    if points.shape[0] < 2:
        return numpy.zeros((2, 2))
    return numpy.cov(points, rowvar=False)


def _ellipse_from_covariance(
    cov: NDArray[Any], sigma: float, /
) -> tuple[float, float, float]:
    """Return major radius, minor radius, and angle of confidence ellipse.

    The ellipse parameters are derived from the eigen-decomposition of the
    2x2 covariance matrix `cov`, scaled by `sigma`.

    """
    eigenvalues, eigenvectors = numpy.linalg.eigh(cov[:2, :2])

    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    major_vector = eigenvectors[:, 0]
    angle = math.atan2(major_vector[1], major_vector[0])

    if angle < 0:
        angle += math.pi

    # clip to avoid negative eigenvalues from numerical errors
    eigenvalues = numpy.clip(eigenvalues, 0.0, None)

    radius_major = sigma * math.sqrt(2 * eigenvalues[0])
    radius_minor = sigma * math.sqrt(2 * eigenvalues[1])

    return radius_major, radius_minor, float(angle)


def _sort_clusters(
    center_real: list[float],
    center_imag: list[float],
    radius_major: list[float],
    radius_minor: list[float],
    sort: Literal['polar', 'phasor', 'area'] | None,
    /,
) -> list[int]:
    """Return indices that sort clusters according to `sort` method."""
    if len(center_real) <= 1:
        return list(range(len(center_real)))

    match sort:
        case 'polar' | None:

            def sort_key(i: int) -> Any:
                return (
                    math.atan2(center_imag[i], center_real[i]),
                    math.hypot(center_real[i], center_imag[i]),
                )

        case 'phasor':

            def sort_key(i: int) -> Any:
                return center_imag[i], center_real[i]

        case 'area':

            def sort_key(i: int) -> Any:
                return -radius_major[i] * radius_minor[i]

        case _:
            msg = (  # type: ignore[unreachable]
                f"{sort=!r} not in {{'phasor', 'polar', or 'area'}}"
            )
            raise ValueError(msg)

    return sorted(range(len(center_real)), key=sort_key)
