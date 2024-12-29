import numpy as np
from ._find_perimeter_cy import _get_perimeter

def find_perimeter(image, level,
                  *,
                  mask=None):
    """Find the perimeter of the iso-valued contours in a 2D array for a given level value.

    Uses the "marching squares" method to compute the iso-valued contours of
    the input 2D array for a particular level value. As segments are computed, 
    their lengths are added to a total perimeter value which is eventually returned.

    Parameters
    ----------
    image : 2D ndarray of double
        Input image in which to find contours.
    level : float
        Value along which to find contours in the array.

    Returns
    -------
    perimeter : float
        The perimeter of the computed contours, using the distance between two adjacent 
        image array points as the base unit.

    See Also
    --------
    skimage.measure.find_contours

    Notes
    -----
    The marching squares algorithm is a special case of the marching cubes
    algorithm [1]_.  A simple explanation is available here:

    http://users.polytech.unice.fr/~lingrand/MarchingCubes/algo.html

    .. warning::

       Array coordinates/values are assumed to refer to the *center* of the
       array element. Take a simple example input: ``[0, 1]``. The interpolated
       position of 0.5 in this array is midway between the 0-element (at
       ``x=0``) and the 1-element (at ``x=1``), and thus would fall at
       ``x=0.5``.

    This means that to find reasonable contours, it is best to find contours
    midway between the expected "light" and "dark" values. In particular,
    given a binarized array, *do not* choose to find contours at the low or
    high value of the array. This will often yield degenerate contours,
    especially around structures that are a single array element wide. Instead
    choose a middle value, as above.

    References
    ----------
    .. [1] Lorensen, William and Harvey E. Cline. Marching Cubes: A High
           Resolution 3D Surface Construction Algorithm. Computer Graphics
           (SIGGRAPH 87 Proceedings) 21(4) July 1987, p. 163-170).
           :DOI:`10.1145/37401.37422`

    Examples
    --------
    >>> a = np.zeros((3, 3))
    >>> a[0, 0] = 1
    >>> a
    array([[1., 0., 0.],
           [0., 0., 0.],
           [0., 0., 0.]])
    >>> find_perimeter(a, 0.5)
    0.7071067811865476
    """
    if image.shape[0] < 2 or image.shape[1] < 2:
        raise ValueError("Input array must be at least 2x2.")
    if image.ndim != 2:
        raise ValueError('Only 2D arrays are supported.')
    return _get_perimeter(image, float(level))
    