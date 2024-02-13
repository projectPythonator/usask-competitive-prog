from point2d import Pt2d, Numeric
from math import isclose, dist, sin, cos
from typing import TypeVar


class GeometryAlgorithms:
  def __init__(self):
    pass

  def compare_ab(self, a: Numeric, b: Numeric) -> int:
    """Compare a b, for floats and ints. It is useful when you want set values to observe.
    paste directly into code and drop isclose for runtime speedup."""
    return 0 if isclose(a, b) else -1 if a < b else 1

  def dot_product(self, left_vector: Pt2d, right_vector: Pt2d) -> Numeric:
    """Compute the scalar product a.b of a,b equivalent to: a . b"""
    return left_vector.x*right_vector.x + left_vector.y*right_vector.y

  def cross_product(self, left_vector: Pt2d, right_vector: Pt2d) -> Numeric:
    """Computes the scalar value perpendicular to a,b equivalent to: a x b"""
    return left_vector.x*right_vector.y - left_vector.y*right_vector.x

  def distance_normalized(self, left_point: Pt2d, right_point: Pt2d) -> float:
    """Normalized distance between two points a, b equivalent to: sqrt(a^2 + b^2) = distance."""
    return dist(left_point.get_tup(), right_point.get_tup())

  def distance(self, left_point: Pt2d, right_point: Pt2d) -> Numeric:
    """Squared distance between two points a, b equivalent to: a^2 + b^2 = distance."""
    return self.dot_product(left_point - right_point, left_point - right_point)

  def rotate_cw_90_wrt_origin(self, point: Pt2d) -> Pt2d:
    """Compute a point rotation on pt. Just swap x and y and negate x."""
    return Pt2d(point.y, -point.x)

  def rotate_ccw_90_wrt_origin(self, point: Pt2d) -> Pt2d:
    """Compute a point rotation on pt. Just swap x and y and negate y."""
    return Pt2d(-point.y, point.x)

  def rotate_ccw_rad_wrt_origin(self, point: Pt2d, degree_in_radians: float) -> Pt2d:
    """Compute a counterclockwise point rotation on pt. Accurate only for floating point cords.
    formula: x = (x cos(rad) - y sin(rad)), y = (x sin(rad) + y cos (rad)).

    Complexity per call: Time: O(1), Space: O(1).
    Optimizations: calculate cos and sin outside the return, so you don't double call each.
    """
    return Pt2d(point.x * cos(degree_in_radians) - point.y * sin(degree_in_radians),
                point.x * sin(degree_in_radians) + point.y * cos(degree_in_radians))
