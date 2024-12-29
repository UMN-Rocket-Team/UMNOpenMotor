#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
from cython.parallel import prange
from libc.stdio cimport printf
cimport numpy as cnp
from libc.math cimport sqrt

# helper function to calculate length of diagonals without python
cdef inline cnp.float64_t hypot(cnp.float64_t x, cnp.float64_t y) nogil:
    return sqrt((x * x) + (y * y))

# lerp helper function
cdef inline cnp.float64_t _get_fraction(cnp.float64_t from_value,
                                        cnp.float64_t to_value,
                                        cnp.float64_t level) nogil:
    if (to_value == from_value):
        return 0
    return ((level - from_value) / (to_value - from_value))

def _get_perimeter(cnp.float64_t[:, :] array,
                          cnp.float64_t level):
    """Iterate across the given array in a marching-squares fashion,
    looking for segments that cross 'level'. If such a segment is
    found, its length is added to the total perimeter,
    which is returned by the function.  if vertex_connect_high is
    nonzero, high-values pixels are considered to be face+vertex
    connected into objects; otherwise low-valued pixels are.
    """

    # The plan is to iterate a 2x2 square across the input array. This means
    # that the upper-left corner of the square needs to iterate across a
    # sub-array that's one-less-large in each direction (so that the square
    # never steps out of bounds). The square is represented by four pointers:
    # ul, ur, ll, and lr (for 'upper left', etc.). We also maintain the current
    # 2D coordinates for the position of the upper-left pointer. Note that we
    # ensured that the array is of type 'float64' and is C-contiguous (last
    # index varies the fastest).
    #
    # There are sixteen different possible square types, diagramed below.
    # A + indicates that the vertex is above the contour value, and a -
    # indicates that the vertex is below or equal to the contour value.
    # The vertices of each square are:
    # ul ur
    # ll lr
    # and can be treated as a binary value with the bits in that order. Thus
    # each square case can be numbered:
    #  0--   1+-   2-+   3++   4--   5+-   6-+   7++
    #   --    --    --    --    +-    +-    +-    +-
    #
    #  8--   9+-  10-+  11++  12--  13+-  14-+  15++
    #   -+    -+    -+    -+    ++    ++    ++    ++
    #
    # The position of the line segment that cuts through (or
    # doesn't, in case 0 and 15) each square is clear, except in
    # cases 6 and 9. In this case, the segments are assumed to connect the 
    # negative sections. Lines like \\ are drawn through square 6, and 
    # lines like // are drawn through square 9.

    cdef cnp.float64_t perimeter
    cdef cnp.float64_t addVar # temp accumulator to store the perimeter inside

    cdef unsigned char square_case
    cdef cnp.float64_t top, bottom, left, right
    cdef cnp.float64_t ul, ur, ll, lr
    cdef Py_ssize_t r0, r1, c0, c1

    perimeter = 0
    #not simulating 3 closest pixels to the edge, since those would get discarded anyways
    for r0 in prange(3,array.shape[0] - 4, nogil=True):
        for c0 in prange(3,array.shape[1] - 4):

            r1, c1 = r0 + 1, c0 + 1

            ul = array[r0, c0]
            ur = array[r0, c1]
            ll = array[r1, c0]
            lr = array[r1, c1]

            square_case = (ul > level) + ((ur > level) *2) + ((ll > level) * 4) + ((lr > level) * 8)

            if square_case in [0, 15]:
                # only do anything if there's a line passing through the
                # square. Cases 0 and 15 are entirely below/above the contour.
                continue

            if hypot(r0 +0.5 -(array.shape[0]/2),c0 +0.5 -(array.shape[1]/2)) > (array.shape[0] / 2) - 3:
                # skips this square if outside the motor radius 
                # tolerance of 3 adapted from geometry.length function
                continue

            top = _get_fraction(ul, ur, level)
            bottom = _get_fraction(ll, lr, level)
            left = _get_fraction(ll, ul, level)
            right = _get_fraction(lr, ur, level)

            addVar = 0
            if (square_case == 1 or square_case == 14):
                # top to left
                addVar = hypot(top,1-left)
            elif (square_case == 2 or square_case == 13):
                # right to top
                addVar = hypot(1-top,1-right)
            elif (square_case == 3 or square_case == 12):
                # right to left
                addVar = hypot(right-left,1)
            elif (square_case == 4 or square_case == 11):
                # left to bottom
                addVar = hypot(left,bottom)
            elif (square_case == 5 or square_case == 10):
                # top to bottom
                addVar = hypot(top-bottom,1)
            elif (square_case == 6):
                # \\ case
                addVar = hypot(1-top,1-right) + hypot(left,bottom)
            elif (square_case == 7 or square_case == 8):
                # right to bottom
                addVar = hypot(1-bottom,right)
            else: #assume case 9
                # // case
                addVar = hypot(top,1-left) + hypot(1-bottom,right)
            perimeter += addVar
    return perimeter