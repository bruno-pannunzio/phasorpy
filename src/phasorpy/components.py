"""Component analysis of phasor coordinates.

The ``phasorpy.components`` module provides functions to:

- calculate fractions of two components of known location by projecting to
  line between components:

  - :py:func:`two_fractions_from_phasor`

- calculate phasor coordinates of second component if only one is
  known (not implemented)

- calculate fractions of two, three or four known components (uses second
  harmonic information for four components):

  - :py:func:`fractions_from_phasor`

- calculate fractions of two or three components of known location by
  resolving graphically with histogram (not implemented)

- blindly resolve fractions of n components by using harmonic
  information (not implemented)

"""

from __future__ import annotations

__all__ = ['two_fractions_from_phasor', 'fractions_from_phasor']

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._typing import Any, ArrayLike, NDArray

import math

import numpy
import numpy as np

from ._utils import project_phasor_to_line


def two_fractions_from_phasor(
    real: ArrayLike,
    imag: ArrayLike,
    real_components: ArrayLike,
    imag_components: ArrayLike,
    /,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Return fractions of two components from phasor coordinates.

    Parameters
    ----------
    real : array_like
        Real component of phasor coordinates.
    imag : array_like
        Imaginary component of phasor coordinates.
    real_components: array_like
        Real coordinates of the first and second components.
    imag_components: array_like
        Imaginary coordinates of the first and second components.

    Returns
    -------
    fraction_of_first_component : ndarray
        Fractions of the first component.
    fraction_of_second_component : ndarray
        Fractions of the second component.

    Notes
    -----
    For the moment, calculation of fraction of components from different
    channels or frequencies is not supported. Only one pair of components can
    be analyzed and will be broadcasted to all channels/frequencies.

    Raises
    ------
    ValueError
        If the real and/or imaginary coordinates of the known components are
        not of size 2.
        If the two components have the same coordinates.

    Examples
    --------
    >>> two_fractions_from_phasor(
    ...     [0.6, 0.5, 0.4], [0.4, 0.3, 0.2], [0.2, 0.9], [0.4, 0.3]
    ... ) # doctest: +NUMBER
    (array([0.44, 0.56, 0.68]), array([0.56, 0.44, 0.32]))

    """
    real_components = numpy.asarray(real_components)
    imag_components = numpy.asarray(imag_components)
    if real_components.shape != (2,):
        raise ValueError(f'{real_components.shape=} != (2,)')
    if imag_components.shape != (2,):
        raise ValueError(f'{imag_components.shape=} != (2,)')
    first_component_phasor = numpy.array(
        [real_components[0], imag_components[0]]
    )
    second_component_phasor = numpy.array(
        [real_components[1], imag_components[1]]
    )
    total_distance_between_components = math.hypot(
        (second_component_phasor[0] - first_component_phasor[0]),
        (second_component_phasor[1] - first_component_phasor[1]),
    )
    if math.isclose(total_distance_between_components, 0, abs_tol=1e-6):
        raise ValueError('components must have different coordinates')
    projected_real, projected_imag = project_phasor_to_line(
        real, imag, real_components, imag_components
    )
    distances_to_first_component = numpy.hypot(
        numpy.asarray(projected_real) - first_component_phasor[0],
        numpy.asarray(projected_imag) - first_component_phasor[1],
    )
    fraction_of_second_component = (
        distances_to_first_component / total_distance_between_components
    )
    return tuple(
        numpy.array(
            [1 - fraction_of_second_component, fraction_of_second_component]
        )
    )


def fractions_from_phasor(
    real: ArrayLike,
    imag: ArrayLike,
    real_components: ArrayLike,
    imag_components: ArrayLike,
    /,
    *,
    axis: int = 0,
) -> tuple[NDArray[Any], ...]:
    """Return fractions of two, three or four components from phasor 
    coordinates.

    Parameters
    ----------
    real : array_like
        Real component of phasor coordinates.
    imag : array_like
        Imaginary component of phasor coordinates.
    real_components: array_like
        Real coordinates of the components.
    imag_components: array_like
        Imaginary coordinates of the components.
    axis : int, optional
        Axis corresponding to harmonics.
        The default is the first axis (0).

    Returns
    -------
    fraction_of_first_component : ndarray
        Fractions of the first component.
    fraction_of_second_component : ndarray
        Fractions of the second component.

    Notes
    -----
    For the moment, calculation of fraction of components from different
    channels or frequencies is not supported.
    
    For the analysis of four components, the phasor coordinates of the first
    and second harmonic must be given along `axis`.

    Raises
    ------
    ValueError
        If the shape of real and imaginary coordinates do not match.

    Examples
    --------
    >>> two_fractions_from_phasor(
    ...     [0.6, 0.5, 0.4], [0.4, 0.3, 0.2], [0.2, 0.9], [0.4, 0.3]
    ... ) # doctest: +NUMBER
    (array([0.44, 0.56, 0.68]), array([0.56, 0.44, 0.32]))

    """
    if real.shape != imag.shape:
        raise ValueError(f'{real.shape=} != {imag.shape=}')
    if real_components.shape != imag_components.shape:
        raise ValueError(
            f'{real_components.shape=} != {imag_components.shape=}'
        )
    real = numpy.atleast_1d(real)
    imag = numpy.atleast_1d(imag)
    real_components = numpy.atleast_1d(real_components)
    imag_components = numpy.atleast_1d(imag_components)
    if real_components.size == 2:
        return two_fractions_from_phasor(
            real, imag, real_components, imag_components
        )
    if real_components.ndim == 1:
        real_components = numpy.expand_dims(real_components, axis=axis)
        imag_components = numpy.expand_dims(imag_components, axis=axis)
    components_coordinates = numpy.concatenate(
        [real_components, imag_components], axis=axis
    )
    if components_coordinates.shape[0] != components_coordinates.shape[1]:
        components_coordinates = numpy.concatenate(
            (real_components, imag_components, numpy.ones(real_components.shape)), axis=axis
        )
        real = numpy.expand_dims(real, axis=axis)
        imag = numpy.expand_dims(imag, axis=axis)
        independent_arrays = numpy.concatenate(
            [real, imag, numpy.ones(real.shape)], axis=axis
        )
    else:
        independent_arrays = numpy.concatenate([real, imag], axis=axis)
    independent_arrays = numpy.tile(
        independent_arrays.T[..., None], (components_coordinates.shape[axis],)
    )
    fractions = numpy.linalg.solve(components_coordinates, independent_arrays)
    return tuple(fractions[(..., 0)].T)

def fractions_with_lstsq(real, imag, real_comp, imag_comp, axis = 0):
    real = numpy.atleast_1d(real)
    imag = numpy.atleast_1d(imag)
    real_comp = numpy.atleast_1d(real_comp)
    imag_comp = numpy.atleast_1d(imag_comp)
    if real_comp.ndim < 2:
        real_comp = np.expand_dims(real_comp, axis = axis)
        imag_comp = np.expand_dims(imag_comp, axis = axis)
        real = np.expand_dims(real, axis = axis)
        imag = np.expand_dims(imag, axis = axis)
    # ones_shape = [1 if i == axis else dim for i, dim in enumerate(real.shape)]
    ones_array = np.ones(real[axis].shape)
    A = np.concatenate([real, imag, ones_array], axis=axis)
    # ones_comp_shape = [1 if i == axis else dim for i, dim in enumerate(real_comp.shape)]
    ones_comp = np.ones(real_comp[axis].shape)
    b = np.concatenate([real_comp, imag_comp, ones_comp], axis=axis)
    if A.ndim > 2:
        print(A)
        print(A.shape)
        A = A.reshape(A.shape[0], -1) 
        print(A.shape)
        print(b)
        print(b.shape)
        b = b.reshape(b.shape[0], -1) 
        print(b.shape)
    
    x, _, _, _ = np.linalg.lstsq(b, A, rcond=None)

    if A.ndim > 2:
        x = x[axis].reshape(real[axis].shape)
    return x