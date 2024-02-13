from point2d import Pt2d
from geometry_utility_2d import GeometryAlgorithms as basicPointFunctions
from math import sqrt, acos


class GeometryAlgorithms(basicPointFunctions):
  def point_c_rotation_wrt_line_ab(self, a_point: Pt2d, b_point: Pt2d, c_point: Pt2d) -> int:
    """Determine orientation of c wrt line ab, in terms of collinear clockwise counterclockwise.
    Since 2d cross-product is the area of the parallelogram, we can use this to accomplish this.

    Complexity per call: Time: O(1), Space: O(1).
    Returns collinear(cl): 0, counterclockwise(ccw): 1, clockwise(cw): -1
    Optimizations: if x,y are ints, use 0 instead of 0.0 or just paste the code here directly.
    """
    return self.compare_ab(self.cross_product(b_point - a_point, c_point - a_point), 0.0)

  def angle_point_c_wrt_line_ab(self, a_point: Pt2d, b_point: Pt2d, c_point: Pt2d) -> float:
    """For a line ab and point c, determine the angle of a to b to c in radians.
    formula: arc-cos(dot(vec_ba, vec_bc) / sqrt(dist_sq(vec_ba) * dist_sq(vec_bc))) = angle

    Complexity per call: Time: O(1), Space: O(1).
    Optimizations: for accuracy we sqrt both distances can remove if distances are ints.
    """
    vector_ba, vector_bc = a_point - b_point, c_point - b_point
    dot_ba_bc = self.dot_product(vector_ba, vector_bc)
    dist_sq_ba = self.dot_product(vector_ba, vector_ba)
    dist_sq_bc = self.dot_product(vector_bc, vector_bc)
    return acos(dot_ba_bc / (sqrt(dist_sq_ba) * sqrt(dist_sq_bc)))
    # return acos(dot_ba_bc / sqrt(dist_sq_ba * dist_sq_bc))

  def pt_on_line_segment_ab(self, point_a: Pt2d, point_b: Pt2d, point: Pt2d) -> bool:
    """Logic is cross == 0 mean parallel, and dot being <= 0 means different directions."""
    vec_pa, vec_pb = point_a - point, point_b - point
    return (self.compare_ab(self.cross_product(vec_pa, vec_pb), 0) == 0
            and self.compare_ab(self.dot_product(vec_pa, vec_pb), 0) <= 0)

  def project_pt_c_to_line_ab(self, a_point: Pt2d, b_point: Pt2d, c_point: Pt2d) -> Pt2d:
    """Compute the point closest to c on the line ab.
    formula: pt = a + u x vector_ba, where u is the scalar projection of vector_ca onto
    vector_ba via dot-product
    # TODO RETEST
    Complexity per call: Time: O(1), Space: O(1).
    """
    vec_ab, vec_ac = b_point - a_point, c_point - a_point
    translation = vec_ab * (self.dot_product(vec_ac, vec_ab) / self.dot_product(vec_ab, vec_ab))
    return a_point + translation

  def project_pt_c_to_line_seg_ab(self, a_point: Pt2d, b_point: Pt2d, c_point: Pt2d) -> Pt2d:
    """Compute the point closest to c on the line segment ab.
    Rule if a==b, then if c closer to a or b, otherwise we can just use the line version.
    # TODO RETEST
    Complexity per call: Time: O(1), Space: O(1).
    Optimizations: use compare_ab on the last line if needed better accuracy.
    """
    if a_point == b_point:  # base case, closest point is either, avoids division by 0 below
      return a_point
    vec_ab, vec_ac = b_point - a_point, c_point - a_point
    u = self.dot_product(vec_ac, vec_ab) / self.dot_product(vec_ab, vec_ab)
    return (a_point if u < 0.0 else b_point if u > 1.0           # closer to a or b
            else self.project_pt_c_to_line_ab(a_point, b_point, c_point))  # inbetween a and b

  def distance_pt_c_to_line_ab(self, a_point: Pt2d, b_point: Pt2d, c_point: Pt2d) -> float:
    """Just return the distance between c and the projected point :)."""
    closest_point: Pt2d = self.project_pt_c_to_line_ab(a_point, b_point, c_point)
    return self.distance_normalized(c_point, closest_point)

  def distance_pt_c_to_line_seg_ab(self, a_point: Pt2d, b_point: Pt2d, c_point: Pt2d) -> float:
    """Same as above, just return the distance between c and the projected point :)."""
    # TODO RETEST
    closest_point: Pt2d = self.project_pt_c_to_line_seg_ab(a_point, b_point, c_point)
    return self.distance_normalized(c_point, closest_point)
